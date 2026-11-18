import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

from coursemate.chat import Final, stream_answer
from coursemate.embeddings import Embedder
from coursemate.llm import LLMError
from coursemate.pricing import PRICES, PRICES_CHECKED, cost_usd
from coursemate.rag import DEFAULT_MIN_SCORE
from coursemate.retrieval import Mode
from coursemate.retrieval_eval import GoldenQuestion
from coursemate.store import VectorStore
from coursemate.streaming import StreamingLLM


@dataclass(frozen=True)
class ProviderRun:
    """What one provider did with the questions. The lists have one entry per answer that
    reached the model: an answer refused by the retrieval gate never calls it, so it has no time."""

    provider: str
    model: str
    host: str
    n: int  # questions asked (the warm-up is not one of them)
    found: int  # answers that passed the checks
    first_token_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)

    @property
    def median_first_ms(self) -> float | None:
        return statistics.median(self.first_token_ms) if self.first_token_ms else None

    @property
    def median_total_ms(self) -> float | None:
        return statistics.median(self.total_ms) if self.total_ms else None

    @property
    def mean_input(self) -> float | None:
        return statistics.mean(self.input_tokens) if self.input_tokens else None

    @property
    def mean_output(self) -> float | None:
        return statistics.mean(self.output_tokens) if self.output_tokens else None


def endpoint_host(llm: StreamingLLM) -> str:
    """Where the requests go, e.g. localhost:11434 or api.openai.com ("-" when nowhere)."""
    if not llm.base_url:
        return "-"
    url = httpx.URL(llm.base_url)
    return f"{url.host}:{url.port}" if url.port else url.host


async def _final_answer(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    llm: StreamingLLM,
    k: int,
    min_score: float,
    mode: Mode,
) -> Final:
    """Run one question through the real chat code and keep only the last event."""
    final: Final | None = None
    async for event in stream_answer(
        question, [], embedder, store, llm, k=k, min_score=min_score, mode=mode
    ):
        if isinstance(event, Final):
            final = event
    if final is None:
        raise LLMError("the answer ended without a final event")
    return final


async def run_provider(
    llm: StreamingLLM,
    questions: Sequence[GoldenQuestion],
    embedder: Embedder,
    store: VectorStore,
    *,
    k: int = 4,
    min_score: float = DEFAULT_MIN_SCORE,
    mode: Mode = "vector",
) -> ProviderRun:
    if not questions:
        raise ValueError("there are no questions to ask")

    # Warm-up: ask the first question once and throw the answer away. The first request makes
    # Ollama load the model into memory, which is slow and would be counted as that provider's time.
    await _final_answer(questions[0].question, embedder, store, llm, k, min_score, mode)

    finals = [
        await _final_answer(q.question, embedder, store, llm, k, min_score, mode) for q in questions
    ]
    reached_model = [f for f in finals if f.total_ms > 0]
    return ProviderRun(
        provider=llm.provider,
        model=llm.model,
        host=endpoint_host(llm),
        n=len(finals),
        found=sum(f.found for f in finals),
        first_token_ms=[f.first_token_ms for f in reached_model],
        total_ms=[f.total_ms for f in reached_model],
        input_tokens=[f.input_tokens for f in reached_model],
        output_tokens=[f.output_tokens for f in reached_model],
    )


async def compare(
    llms: Sequence[StreamingLLM],
    questions: Sequence[GoldenQuestion],
    embedder: Embedder,
    store: VectorStore,
    *,
    k: int = 4,
    min_score: float = DEFAULT_MIN_SCORE,
    mode: Mode = "vector",
) -> list[ProviderRun]:
    """One provider after another, not at the same time: they would slow each other down."""
    runs = []
    for llm in llms:
        runs.append(
            await run_provider(llm, questions, embedder, store, k=k, min_score=min_score, mode=mode)
        )
    return runs


def average_tokens(runs: Sequence[ProviderRun]) -> tuple[int, int] | None:
    """Mean tokens in and out over every answer that reached a model, or None if none did."""
    inputs = [t for run in runs for t in run.input_tokens]
    outputs = [t for run in runs for t in run.output_tokens]
    if not inputs:
        return None
    return round(statistics.mean(inputs)), round(statistics.mean(outputs))


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(line[i]) for line in [header, *rows]) for i in range(len(header))]
    lines = [
        " | ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True))
        for line in [header, *rows]
    ]
    return "\n".join(lines)


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def format_runs(runs: Sequence[ProviderRun]) -> str:
    """One row per provider."""
    header = ["provider", "model", "endpoint", "found", "1st token ms", "total ms", "tokens in/out"]
    rows = [
        [
            run.provider,
            run.model,
            run.host,
            f"{run.found}/{run.n}",
            _number(run.median_first_ms),
            _number(run.median_total_ms),
            "-"
            if run.mean_input is None
            else f"{_number(run.mean_input)} / {_number(run.mean_output)}",
        ]
        for run in runs
    ]
    return _table(header, rows)


def format_costs(input_tokens: int, output_tokens: int) -> str:
    """What one answer of this size costs on each priced model."""
    rows = [["local (Ollama)", "$0", "$0"]]
    for model, price in PRICES.items():
        cost = cost_usd(input_tokens, output_tokens, price)
        rows.append([model, f"${cost:.6f}", f"${cost * 1000:.2f}"])
    title = (
        f"prices checked {PRICES_CHECKED} · one answer = {input_tokens} in / {output_tokens} out"
    )
    return "\n".join(
        [
            title,
            _table(["model", "per answer", "per 1,000 answers"], rows),
            "note: hosted models use their own tokenizers, so the same text counts a bit "
            "differently there. Treat these as estimates.",
        ]
    )
