"""Tracing with OpenTelemetry: spans go to Langfuse and/or a JSONL file.

Tracing is off until setup_tracing() finds somewhere to send spans.
"""

import base64
import json
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, Status, StatusCode, Tracer  # noqa: F401 (for span())

LANGFUSE_PATH = "/api/public/otel/v1/traces"

# Our own provider, not OpenTelemetry's global one (that one can only be set once per process).
# None means tracing is off.
_provider: TracerProvider | None = None


def tracer() -> Tracer:
    """The object that starts spans. When tracing is off, its spans do nothing."""
    if _provider is None:
        return trace.NoOpTracer()
    return _provider.get_tracer("coursemate")


def use_provider(provider: TracerProvider | None) -> TracerProvider | None:
    """Swap in a provider (None turns tracing off) and return the previous one."""
    global _provider
    previous = _provider
    _provider = provider
    return previous


def attrs(**values: object) -> dict[str, str | bool | int | float]:
    """Turn keyword arguments into span attributes.

    A double underscore in a name becomes a dot, None values are dropped,
    str/bool/int/float stay as they are, and anything else becomes JSON.
    """
    result: dict[str, str | bool | int | float] = {}
    for name, value in values.items():
        if value is None:
            continue
        if not isinstance(value, str | bool | int | float):
            value = json.dumps(value, default=str, ensure_ascii=False)
        result[name.replace("__", ".")] = value
    return result


def iso_time(epoch_ns: int) -> str:
    """Nanoseconds since 1970 as an ISO 8601 UTC time, to the microsecond."""
    seconds, nanos = divmod(epoch_ns, 1_000_000_000)
    moment = datetime.fromtimestamp(seconds, UTC).replace(microsecond=nanos // 1000)
    return moment.isoformat(timespec="microseconds")


class JsonlSpanExporter(SpanExporter):
    """Append each finished span to a file as one line of JSON."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        lines = "".join(json.dumps(self._record(s), ensure_ascii=False) + "\n" for s in spans)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(lines)
        except OSError:
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    @staticmethod
    def _record(span: ReadableSpan) -> dict[str, object]:
        return {
            "name": span.name,
            "trace_id": trace.format_trace_id(span.context.trace_id),
            "span_id": trace.format_span_id(span.context.span_id),
            "parent_id": trace.format_span_id(span.parent.span_id) if span.parent else None,
            "start": iso_time(span.start_time),
            "duration_ms": (span.end_time - span.start_time) / 1_000_000,
            "status": span.status.status_code.name,
            "attributes": dict(span.attributes or {}),
        }

    def shutdown(self) -> None:
        """Nothing to close: the file is closed after every export."""


def langfuse_exporter(environ: Mapping[str, str]) -> SpanExporter | None:
    """An OTLP/HTTP exporter for Langfuse, or None unless all three settings are set."""
    base_url = environ.get("LANGFUSE_BASE_URL")
    public_key = environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = environ.get("LANGFUSE_SECRET_KEY")
    if not (base_url and public_key and secret_key):
        return None

    # Imported here so that runs without Langfuse never load the HTTP and protobuf code.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return OTLPSpanExporter(
        endpoint=base_url.rstrip("/") + LANGFUSE_PATH,
        headers={
            "Authorization": f"Basic {token}",
            "x-langfuse-ingestion-version": "4",
        },
    )


def setup_tracing(
    span_log: str | Path | None, environ: Mapping[str, str] | None = None
) -> list[str]:
    """Turn tracing on and return where spans will go (empty list: tracing stays off)."""
    environ = os.environ if environ is None else environ
    shutdown_tracing()  # calling this twice must not leave the first provider running

    destinations: list[str] = []
    exporters: list[SpanExporter] = []
    langfuse = langfuse_exporter(environ)
    if langfuse is not None:
        exporters.append(langfuse)
        destinations.append(f"Langfuse at {environ['LANGFUSE_BASE_URL']}")
    if span_log:
        exporters.append(JsonlSpanExporter(span_log))
        destinations.append(f"span log {span_log}")

    if exporters:
        provider = TracerProvider(resource=Resource.create({"service.name": "coursemate"}))
        for exporter in exporters:
            # Batches spans and sends them from a background thread, so answers never wait.
            provider.add_span_processor(BatchSpanProcessor(exporter))
        use_provider(provider)
    return destinations


def shutdown_tracing() -> None:
    """Send any waiting spans and turn tracing off. Safe to call more than once."""
    provider = use_provider(None)
    if provider is not None:
        provider.shutdown()


@contextmanager
def span(name: str, parent: Span | None = None, **attributes: Any) -> Iterator[Span]:
    """A span that starts now and ends when the block ends, even after an error.

    It is a child of `parent` (or a new trace when parent is None). It is not made the
    "current" span, so it is safe inside async generators that pause at every yield.
    """
    context = trace.set_span_in_context(parent) if parent is not None else None
    current = tracer().start_span(name, context=context, attributes=attrs(**attributes))
    try:
        yield current
    except Exception as exc:
        current.record_exception(exc)
        current.set_status(Status(StatusCode.ERROR, str(exc)))
        raise
    finally:
        current.end()
