import os
from collections.abc import Iterator

import psycopg
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from coursemate.tracing import use_provider

TEST_DB = os.environ.get(
    "DATABASE_URL", "postgresql://coursemate:coursemate@127.0.0.1:5433/coursemate"
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    """Collect finished spans in memory instead of sending them anywhere."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # Exports each span the moment it ends, so a test never waits for a batch.
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = use_provider(provider)
    yield exporter
    use_provider(previous)
    provider.shutdown()


def db_available() -> bool:
    try:
        psycopg.connect(TEST_DB, connect_timeout=2).close()
    except psycopg.OperationalError:
        return False
    return True


def require_db() -> None:
    if db_available():
        return
    if os.environ.get("COURSEMATE_REQUIRE_DB") == "1":
        pytest.fail(f"Postgres is required but not reachable at {TEST_DB}")
    pytest.skip("Postgres is not running (start it with: docker compose up -d)")
