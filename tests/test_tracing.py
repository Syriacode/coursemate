import base64
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.trace import StatusCode

from coursemate.tracing import (
    attrs,
    iso_time,
    langfuse_exporter,
    setup_tracing,
    shutdown_tracing,
    span,
    tracer,
)

SETTINGS = {
    "LANGFUSE_BASE_URL": "http://127.0.0.1:3000",
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
}


@pytest.fixture(autouse=True)
def tracing_off_afterwards():
    yield
    shutdown_tracing()


def test_attrs_names_values_and_json():
    result = attrs(
        gen_ai__request__model="qwen2.5:3b",
        coursemate__found=False,
        coursemate__k=0,
        coursemate__best_score=0.5,
        langfuse__session__id=None,
        langfuse__observation__input=[{"role": "user", "content": "¿Lab 4?"}],
    )
    assert result == {
        "gen_ai.request.model": "qwen2.5:3b",
        "coursemate.found": False,
        "coursemate.k": 0,
        "coursemate.best_score": 0.5,
        "langfuse.observation.input": '[{"role": "user", "content": "¿Lab 4?"}]',
    }


def test_iso_time():
    assert iso_time(1_500_000) == "1970-01-01T00:00:00.001500+00:00"


def test_nothing_configured_keeps_tracing_off():
    assert setup_tracing(None, environ={}) == []
    assert tracer().start_span("x").is_recording() is False


def test_span_log_writes_one_json_line_per_span(tmp_path):
    log = tmp_path / "logs" / "spans.jsonl"
    assert setup_tracing(log, environ={}) == [f"span log {log}"]

    root = tracer().start_span("chat")
    child = tracer().start_span(
        "retrieve",
        context=trace.set_span_in_context(root),
        attributes=attrs(coursemate__k=4),
    )
    child.end()
    root.end()
    shutdown_tracing()  # flushes the batch

    lines = log.read_text(encoding="utf-8").splitlines()
    first, second = (json.loads(line) for line in lines)
    assert (first["name"], second["name"]) == ("retrieve", "chat")
    assert first["trace_id"] == second["trace_id"]
    assert len(first["trace_id"]) == 32
    assert len(first["span_id"]) == 16
    assert first["parent_id"] == second["span_id"]
    assert second["parent_id"] is None
    assert first["start"].endswith("+00:00")
    assert first["duration_ms"] >= 0
    assert first["status"] == "UNSET"
    assert first["attributes"] == {"coursemate.k": 4}


@pytest.mark.parametrize("missing", list(SETTINGS))
def test_langfuse_needs_all_three_settings(missing):
    environ = {name: value for name, value in SETTINGS.items() if name != missing}
    assert langfuse_exporter(environ) is None
    assert setup_tracing(None, environ=environ) == []


def test_langfuse_with_all_three_settings():
    exporter = langfuse_exporter(SETTINGS)
    assert exporter is not None
    exporter.shutdown()


class _Collector(BaseHTTPRequestHandler):
    """A pretend Langfuse: remember each POST and answer 200."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        self.server.requests.append((self.path, self.headers, body))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args):
        pass  # keep the test output quiet


@pytest.fixture
def pretend_langfuse(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")  # never route this through a proxy
    server = HTTPServer(("127.0.0.1", 0), _Collector)  # port 0: any free port
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


def test_spans_reach_a_pretend_langfuse(pretend_langfuse):
    base_url = f"http://127.0.0.1:{pretend_langfuse.server_port}"
    environ = {**SETTINGS, "LANGFUSE_BASE_URL": base_url + "/"}
    assert setup_tracing(None, environ=environ) == [f"Langfuse at {base_url}/"]

    tracer().start_span("chat").end()
    shutdown_tracing()  # flushes the batch

    [(path, headers, body)] = pretend_langfuse.requests
    assert path == "/api/public/otel/v1/traces"
    expected = base64.b64encode(b"pk-lf-test:sk-lf-test").decode()
    assert headers["Authorization"] == f"Basic {expected}"
    assert headers["x-langfuse-ingestion-version"] == "4"
    message = ExportTraceServiceRequest.FromString(body)
    names = [s.name for r in message.resource_spans for ss in r.scope_spans for s in ss.spans]
    assert names == ["chat"]


# The four span() tests (guide section 4.1). They fail until span() is written (Step 3).


def test_children_share_the_trace_and_point_at_the_root(spans):
    with span("chat") as root:
        for name in ("retrieve", "llm", "check"):
            with span(name, parent=root):
                pass

    finished = {s.name: s for s in spans.get_finished_spans()}
    assert set(finished) == {"chat", "retrieve", "llm", "check"}
    chat = finished["chat"]
    assert chat.parent is None
    for name in ("retrieve", "llm", "check"):
        assert finished[name].context.trace_id == chat.context.trace_id
        assert finished[name].parent.span_id == chat.context.span_id


def test_a_child_lies_inside_its_parent_in_time(spans):
    with span("chat") as root:
        with span("llm", parent=root):
            pass

    llm, chat = spans.get_finished_spans()  # the order in which they ended
    assert (llm.name, chat.name) == ("llm", "chat")
    assert chat.start_time <= llm.start_time <= llm.end_time <= chat.end_time


def test_an_error_marks_the_span_and_still_ends_it(spans):
    with pytest.raises(ValueError, match="model is down"):
        with span("llm"):
            raise ValueError("model is down")

    [llm] = spans.get_finished_spans()
    assert llm.end_time is not None
    assert llm.status.status_code is StatusCode.ERROR
    assert llm.status.description == "model is down"
    assert "exception" in [event.name for event in llm.events]


def test_a_span_without_a_parent_starts_a_new_trace(spans):
    with span("chat", langfuse__observation__input={"q": "Lab 4?"}):
        pass
    with span("chat"):
        pass

    first, second = spans.get_finished_spans()
    assert first.parent is None and second.parent is None
    assert first.context.trace_id != second.context.trace_id
    assert first.attributes["langfuse.observation.input"] == '{"q": "Lab 4?"}'
