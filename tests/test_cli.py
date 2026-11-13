import io
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from coursemate.cli import main
from tests.conftest import TEST_DB, require_db

SAMPLE = Path(__file__).parents[1] / "notes" / "sample"


def run(*args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(list(args), stdout, stderr)
    return code, stdout.getvalue(), stderr.getvalue()


@pytest.fixture
def cli_table():
    require_db()
    table = f"cli_{uuid.uuid4().hex[:8]}"
    yield table
    with psycopg.connect(TEST_DB, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {t}").format(t=sql.Identifier(table)))
        conn.execute("DELETE FROM index_meta WHERE table_name = %s", (table,))


def fake_args(table):
    return ["--provider", "fake", "--db-url", TEST_DB, "--table", table]


def test_bad_usage_returns_2():
    code, _, err = run("explode")

    assert code == 2
    assert "invalid choice" in err


def test_unreachable_database_is_a_clear_error():
    code, _, err = run(
        "--provider", "fake", "--db-url", "postgresql://x:y@localhost:1/none", "search", "q"
    )

    assert code == 1
    assert "docker compose up -d" in err


@pytest.mark.db
def test_ingest_then_search_with_fake_models(cli_table):
    code, out, _ = run(*fake_args(cli_table), "ingest", str(SAMPLE), "--reset")

    assert code == 0
    assert "ingested 8 files" in out
    assert "syllabus.md" in out

    code, out, _ = run(*fake_args(cli_table), "search", "late work grace days", "-k", "2")

    assert code == 0
    assert out.splitlines()[0].split()[1] == "syllabus.md"

    code, _, err = run(
        "--provider", "ollama", "--db-url", TEST_DB, "--table", cli_table, "search", "x"
    )

    assert code == 1
    assert "--reset" in err


@pytest.mark.db
def test_ask_with_fake_models(cli_table):
    code, _, _ = run(*fake_args(cli_table), "ingest", str(SAMPLE), "--reset")
    assert code == 0

    code, out, err = run(
        *fake_args(cli_table), "ask", "What is the late work policy?", "--min-score", "0.1"
    )

    assert code == 0
    assert "Fake provider: no model was called." in out
    assert "[1] syllabus.md" in out
    assert "tokens: 100 in, 20 out" in err


def test_eval_retrieval_with_fake_models():
    golden = Path(__file__).parents[1] / "evals" / "retrieval_golden.jsonl"

    code, out, _ = run("--provider", "fake", "eval-retrieval", str(SAMPLE), str(golden))

    assert code == 0
    assert "embedding model: fake" in out
    assert "recall@3" in out
    assert "questions: 13 keyword, 12 paraphrase" in out
