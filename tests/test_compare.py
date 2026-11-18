import io
from pathlib import Path

import pytest

from coursemate.cli import main
from coursemate.compare import (
    ProviderRun,
    average_tokens,
    endpoint_host,
    format_costs,
    format_runs,
    run_provider,
)
from coursemate.embeddings import FakeEmbedder
from coursemate.retrieval_eval import build_index, load_golden
from coursemate.streaming import (
    AnthropicStreamer,
    FakeStreamer,
    OllamaStreamer,
    OpenAIStreamer,
)

ROOT = Path(__file__).parents[1]
SAMPLE = ROOT / "notes" / "sample"
GOLDEN = ROOT / "evals" / "retrieval_golden.jsonl"

GOOD = "The late penalty is 10% per day [1]."


def run_cli(*args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(list(args), stdout, stderr)
    return code, stdout.getvalue(), stderr.getvalue()


@pytest.fixture(scope="module")
def index():
    embedder = FakeEmbedder()
    return embedder, build_index(SAMPLE, embedder), load_golden(GOLDEN)


@pytest.mark.anyio
async def test_the_warm_up_is_asked_but_not_counted(index):
    embedder, store, golden = index
    # replies in order: warm-up, question 1, question 2
    llm = FakeStreamer([GOOD, GOOD, "I don't know."])

    run = await run_provider(llm, golden[:2], embedder, store, min_score=-1.0)

    assert len(llm.calls) == 3  # the warm-up really ran...
    assert run.n == 2  # ...but is not one of the questions
    assert run.found == 1  # (it would be 2 if the warm-up counted)
    assert len(run.total_ms) == 2


@pytest.mark.anyio
async def test_times_and_tokens_come_from_the_stream_end(index):
    embedder, store, golden = index

    run = await run_provider(FakeStreamer(), golden[:2], embedder, store, min_score=-1.0)

    # FakeStreamer always ends with StreamEnd("fake", 100, 20, 1.0, 2.0)
    assert (run.median_first_ms, run.median_total_ms) == (1.0, 2.0)
    assert (run.mean_input, run.mean_output) == (100, 20)
    assert (run.provider, run.model, run.host) == ("fake", "fake", "-")


@pytest.mark.anyio
async def test_a_refused_question_counts_in_n_but_has_no_time(index):
    embedder, store, golden = index
    llm = FakeStreamer()

    # a cosine score can't reach 2.0, so the retrieval gate refuses everything
    run = await run_provider(llm, golden[:2], embedder, store, min_score=2.0)

    assert llm.calls == []  # the model was never asked, not even for the warm-up
    assert (run.n, run.found) == (2, 0)
    assert run.total_ms == []
    assert run.median_total_ms is None
    assert "-" in format_runs([run]).splitlines()[1]


@pytest.mark.anyio
async def test_no_questions_is_an_error(index):
    embedder, store, _ = index

    with pytest.raises(ValueError, match="no questions"):
        await run_provider(FakeStreamer(), [], embedder, store)


def test_cost_table_for_700_in_and_60_out():
    table = format_costs(700, 60)

    sonnet = next(line for line in table.splitlines() if line.startswith("claude-sonnet-5-5"))
    assert "$0.002000" in sonnet
    assert "$2.00" in sonnet
    for model in ("claude-haiku-5-5", "gpt-6-luna", "gpt-6.1-sol"):
        assert model in table
    assert "local (Ollama)" in table
    assert "prices checked 2026-11-15" in table
    assert "700 in / 60 out" in table
    assert "tokenizers" in table


def test_runs_table_shows_one_row_per_provider():
    run = ProviderRun("openai", "qwen2.5:3b", "localhost:11434", n=3, found=2)
    table = format_runs([run])

    header, row = table.splitlines()
    assert "provider" in header
    assert "openai" in row
    assert "qwen2.5:3b" in row
    assert "localhost:11434" in row
    assert "2/3" in row


def test_average_tokens_over_all_providers_or_none():
    a = ProviderRun("a", "m", "-", 1, 1, [1.0], [2.0], [100], [20])
    b = ProviderRun("b", "m", "-", 1, 1, [1.0], [2.0], [300], [40])
    empty = ProviderRun("c", "m", "-", 1, 0)

    assert average_tokens([a, b, empty]) == (200, 30)
    assert average_tokens([empty]) is None


def test_endpoint_host_shows_host_and_port_only(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    local = "http://localhost:11434"

    assert endpoint_host(OllamaStreamer(model="m")) == "localhost:11434"
    assert endpoint_host(OpenAIStreamer(model="m", base_url=local + "/v1")) == "localhost:11434"
    assert endpoint_host(AnthropicStreamer(model="m", base_url=local)) == "localhost:11434"
    hosted = OpenAIStreamer(model="m", base_url="https://api.openai.com/v1", api_key="test-key")
    assert endpoint_host(hosted) == "api.openai.com"
    assert endpoint_host(FakeStreamer()) == "-"


def test_compare_command_prints_the_runs_table_and_a_cost_answer():
    code, out, err = run_cli(
        "--provider", "fake", "compare", str(SAMPLE),
        "--llm", "fake", "-n", "2", "--golden", str(GOLDEN),
    )  # fmt: skip

    assert code == 0, err
    first_table = out.split("\n\n")[0]
    assert "provider" in first_table
    assert "fake" in first_table
    assert "/2" in first_table  # found/n
    # the fake embedder may score below the default gate, so either ending is valid
    assert "prices checked" in out or "nothing to price" in out


def test_compare_with_n_zero_is_a_clear_error():
    code, _, err = run_cli(
        "--provider", "fake", "compare", str(SAMPLE),
        "--llm", "fake", "-n", "0", "--golden", str(GOLDEN),
    )  # fmt: skip

    assert code == 1
    assert "no golden questions" in err


def test_compare_with_a_hosted_llm_and_no_key_stops_before_spending(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    code, out, err = run_cli(
        "--provider", "fake", "compare", str(SAMPLE),
        "--llm", "openai", "--golden", str(GOLDEN),
    )  # fmt: skip

    assert code == 1
    assert "costs money" in err
    assert out == ""  # nothing was printed or measured
