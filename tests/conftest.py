import os

import psycopg
import pytest

TEST_DB = os.environ.get(
    "DATABASE_URL", "postgresql://coursemate:coursemate@127.0.0.1:5433/coursemate"
)


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
