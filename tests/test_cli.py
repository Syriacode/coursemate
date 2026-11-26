import io
import json
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from coursemate.agent_tools import Outbox
from coursemate.cli import build_parser, main
from coursemate.embeddings import FakeEmbedder
from coursemate.llm import FakeClient, ToolCall
from coursemate.tracing import tracer
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
def test_search_puts_the_lab_4_section_first_in_every_mode(cli_table):
    code, _, _ = run(*fake_args(cli_table), "ingest", str(SAMPLE), "--reset")
    assert code == 0

    for mode in ("vector", "keyword", "hybrid"):
        code, out, _ = run(
            *fake_args(cli_table), "search", "When is Lab 4 due?", "-k", "3", "--mode", mode
        )

        first_line = out.splitlines()[0]
        assert code == 0, mode
        assert "lecture-02-hash-tables.md" in first_line, mode
        assert first_line.endswith("> Lab 4"), mode


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


def test_serve_with_a_hosted_llm_and_no_key_is_a_clear_error(monkeypatch):
    # "openai" now has a streamer, but its default URL is a hosted API that costs money, so
    # without a key it refuses. The check runs before the database is touched: no Postgres needed.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    code, _, err = run("--provider", "fake", "serve", "--llm", "openai")

    assert code == 1
    assert "costs money" in err
    assert "tracing: off" in err


def test_serve_shuts_tracing_down_when_it_fails_early(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    log = tmp_path / "spans.jsonl"

    code, _, err = run("--provider", "fake", "--span-log", str(log), "serve", "--llm", "openai")

    assert code == 1
    assert f"tracing: span log {log}" in err
    assert not tracer().start_span("x").is_recording()


def test_span_log_defaults_to_the_environment_variable(monkeypatch):
    assert build_parser().parse_args(["serve"]).span_log is None

    monkeypatch.setenv("COURSEMATE_SPAN_LOG", "logs/spans.jsonl")

    assert build_parser().parse_args(["serve"]).span_log == Path("logs/spans.jsonl")
    chosen = build_parser().parse_args(["--span-log", "other.jsonl", "serve"])
    assert chosen.span_log == Path("other.jsonl")


@pytest.mark.db
def test_serve_traces_while_the_server_runs(cli_table, tmp_path, monkeypatch):
    code, _, _ = run(*fake_args(cli_table), "ingest", str(SAMPLE), "--reset")
    assert code == 0
    log = tmp_path / "spans.jsonl"

    def fake_run(app, host, port):
        tracer().start_span("while-serving").end()

    monkeypatch.setattr("uvicorn.run", fake_run)

    code, _, err = run(*fake_args(cli_table), "--span-log", str(log), "serve")

    assert code == 0, err
    assert f"tracing: span log {log}" in err
    # the span reached the file, so shutdown flushed the batch; and tracing is off again
    assert json.loads(log.read_text(encoding="utf-8"))["name"] == "while-serving"
    assert not tracer().start_span("x").is_recording()


@pytest.mark.db
def test_serve_starts_the_app_with_defaults(cli_table, monkeypatch):
    code, _, _ = run(*fake_args(cli_table), "ingest", str(SAMPLE), "--reset")
    assert code == 0
    started = {}

    def fake_run(app, host, port):
        started.update(app=app, host=host, port=port)

    monkeypatch.setattr("uvicorn.run", fake_run)

    code, _, err = run(*fake_args(cli_table), "serve")

    assert code == 0, err
    assert (started["host"], started["port"]) == ("127.0.0.1", 8000)
    assert started["app"].title == "CourseMate"


@pytest.mark.db
def test_serve_fails_with_the_usual_message_for_a_different_embedding_model(cli_table):
    code, _, _ = run(*fake_args(cli_table), "ingest", str(SAMPLE), "--reset")
    assert code == 0

    code, _, err = run(
        "--provider", "ollama", "--db-url", TEST_DB, "--table", cli_table, "serve", "--llm", "fake"
    )

    assert code == 1
    assert "--reset" in err


def test_eval_retrieval_with_fake_models():
    golden = Path(__file__).parents[1] / "evals" / "retrieval_golden.jsonl"

    code, out, _ = run("--provider", "fake", "eval-retrieval", str(SAMPLE), str(golden))

    assert code == 0
    assert "embedding model: fake" in out
    assert "recall@3" in out
    for mode in ("vector", "keyword", "hybrid"):
        assert any(line.split()[0] == mode for line in out.splitlines())
    assert "questions: 13 keyword, 12 paraphrase" in out


def test_eval_retrieval_can_save_the_run(tmp_path):
    golden = Path(__file__).parents[1] / "evals" / "retrieval_golden.jsonl"
    saved = tmp_path / "runs" / "fake.json"

    code, out, _ = run(
        "--provider", "fake", "eval-retrieval", str(SAMPLE), str(golden), "--save", str(saved)
    )

    assert code == 0
    assert "saved run to" in out
    run_data = json.loads(saved.read_text(encoding="utf-8"))
    assert run_data["model"] == "fake"
    assert list(run_data["modes"]) == ["vector", "keyword", "hybrid"]


@pytest.mark.db
def test_agent_shows_the_draft_and_a_no_sends_nothing(cli_table, tmp_path, monkeypatch):
    code, _, _ = run(*fake_args(cli_table), "ingest", str(SAMPLE), "--reset")
    assert code == 0
    scripted = FakeClient(
        [
            [ToolCall("draft_announcement", {"title": "Lab 4", "body": "Lab 4 is due Friday."})],
            [ToolCall("send_announcement", {"draft_id": 1})],
            "I drafted the Lab 4 reminder but it was not sent.",
        ]
    )
    monkeypatch.setattr("coursemate.cli.make_models", lambda args: (FakeEmbedder(), scripted))
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "n")

    code, out, err = run(
        *fake_args(cli_table),
        "agent",
        "Write an announcement reminding students when Lab 4 is due, and send it.",
        "--outbox",
        str(tmp_path / "outbox"),
    )

    assert code == 0, err
    assert "The agent wants to SEND this announcement:" in out
    assert "# Lab 4" in out
    assert "Lab 4 is due Friday." in out
    assert "error: not approved" in out
    assert "I drafted the Lab 4 reminder but it was not sent." in out
    assert asked == ["Send it? [y/N] "]
    assert "(stopped: answer, 3 steps, 300 input tokens)" in err
    assert "(outbox: drafts [1], sent [])" in err


def test_approve_asks_and_only_y_approves(tmp_path, monkeypatch):
    outbox = Outbox(tmp_path / "outbox")
    outbox.save_draft("Lab 4", "Lab 4 is due Friday of week 5.")
    answers = iter(["", "y"])
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or next(answers))

    code, out, _ = run("approve", "1", "--outbox", str(tmp_path / "outbox"))

    assert code == 1
    assert "# Lab 4" in out
    assert outbox.approved_ids() == []

    code, _, _ = run("approve", "1", "--outbox", str(tmp_path / "outbox"))

    assert code == 0
    assert outbox.approved_ids() == [1]
    assert asked == ["Approve draft 1 for sending, exactly as shown? [y/N] "] * 2


def test_approve_an_unknown_draft_says_so(tmp_path, monkeypatch):
    def no_input(prompt=""):
        raise AssertionError("there is nothing to approve, so nothing should be asked")

    monkeypatch.setattr("builtins.input", no_input)

    code, _, err = run("approve", "7", "--outbox", str(tmp_path / "outbox"))

    assert code == 1
    assert "no draft 7; drafts: none" in err
