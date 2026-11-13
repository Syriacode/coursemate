import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from coursemate.embeddings import Embedder
from coursemate.ingest import ingest
from coursemate.retrieval import MODES, Mode, retrieve
from coursemate.store import Hit, MemoryStore


class GoldenSetError(Exception):
    """The golden question file is malformed, has duplicate ids, or is not fully reviewed."""


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str  # file name, e.g. "syllabus.md"
    section: str  # the last heading, e.g. "Late work"


class GoldenQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expected: list[Expected] = Field(min_length=1)
    kind: Literal["keyword", "paraphrase"]
    reviewed: bool


def load_golden(path: str | Path) -> list[GoldenQuestion]:
    """Read a JSONL golden set; refuse bad rows, duplicate ids and unreviewed rows."""
    questions: list[GoldenQuestion] = []
    first_line_of: dict[str, int] = {}

    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                question = GoldenQuestion.model_validate_json(line)
            except ValidationError as exc:
                raise GoldenSetError(f"{path} line {line_number}: {exc}") from exc

            if question.id in first_line_of:
                raise GoldenSetError(
                    f"{path} line {line_number}: duplicate id {question.id!r} "
                    f"(first on line {first_line_of[question.id]})"
                )
            first_line_of[question.id] = line_number
            questions.append(question)

    unreviewed = [q.id for q in questions if not q.reviewed]
    if unreviewed:
        raise GoldenSetError(f"{path}: not reviewed yet: {', '.join(unreviewed)}")
    return questions


def is_relevant(hit: Hit, expected: list[Expected]) -> bool:
    """True if the hit comes from one of the expected sources and sections."""
    return any(
        hit.source == e.source
        and (hit.heading == e.section or hit.heading.endswith(" > " + e.section))
        for e in expected
    )


def first_relevant_rank(hits: list[Hit], expected: list[Expected]) -> int | None:
    """1-based rank of the first relevant hit, or None if no hit is relevant."""
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, expected):
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Share of questions whose right section is somewhere in the top k."""
    if not ranks:
        return 0.0
    found = sum(1 for rank in ranks if rank is not None and rank <= k)
    return found / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Average of 1/rank over all questions; a miss (None) counts as 0."""
    if not ranks:
        return 0.0
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)


@dataclass(frozen=True)
class ModeResult:
    """Ranks of the first right hit for one search mode: question id -> rank, or None if missed."""

    mode: str
    ranks: dict[str, int | None]

    def recall(self, k: int) -> float:
        return recall_at_k(list(self.ranks.values()), k)

    def mrr(self) -> float:
        return mean_reciprocal_rank(list(self.ranks.values()))


def build_index(notes: Path | str, embedder: Embedder) -> MemoryStore:
    """Chunk and embed the notes into an in-memory store (no database needed)."""
    store = MemoryStore()
    ingest(notes, embedder, store)
    return store


def evaluate(
    golden: list[GoldenQuestion],
    embedder: Embedder,
    store: MemoryStore,
    depth: int = 5,
    modes: tuple[Mode, ...] = MODES,
) -> list[ModeResult]:
    """For each mode, search every question `depth` hits deep and record the right rank."""
    results = []
    for mode in modes:
        ranks: dict[str, int | None] = {}
        for question in golden:
            hits = retrieve(question.question, embedder, store, depth, mode).hits
            ranks[question.id] = first_relevant_rank(hits, question.expected)
        results.append(ModeResult(mode, ranks))
    return results


def save_run(path: Path | str, results: list[ModeResult], model: str) -> None:
    """Write each mode's recall@1/3/5, MRR and per-question ranks to a JSON file."""
    run = {
        "model": model,
        "modes": {
            result.mode: {
                "recall@1": result.recall(1),
                "recall@3": result.recall(3),
                "recall@5": result.recall(5),
                "mrr": result.mrr(),
                "ranks": result.ranks,
            }
            for result in results
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")


def _recall_for_kind(result: ModeResult, golden: list[GoldenQuestion], kind: str, k: int) -> float:
    return recall_at_k([result.ranks[q.id] for q in golden if q.kind == kind], k)


def format_table(results: list[ModeResult], golden: list[GoldenQuestion]) -> str:
    """One row per mode, then how many questions of each kind there are."""
    header = ["mode", "recall@1", "recall@3", "recall@5", "MRR", "r@3 keyword", "r@3 paraphrase"]
    lines = [" | ".join(f"{name:>14}" for name in header)]
    for result in results:
        cells = [
            result.mode,
            f"{result.recall(1):.2f}",
            f"{result.recall(3):.2f}",
            f"{result.recall(5):.2f}",
            f"{result.mrr():.2f}",
            f"{_recall_for_kind(result, golden, 'keyword', 3):.2f}",
            f"{_recall_for_kind(result, golden, 'paraphrase', 3):.2f}",
        ]
        lines.append(" | ".join(f"{cell:>14}" for cell in cells))

    kinds = Counter(q.kind for q in golden)
    lines.append(f"questions: {kinds['keyword']} keyword, {kinds['paraphrase']} paraphrase")
    return "\n".join(lines)


def misses(results: list[ModeResult], golden: list[GoldenQuestion], k: int = 3) -> list[str]:
    """One line per question that at least one mode did not get into its top k."""
    lines = []
    for question in golden:
        ranks = {result.mode: result.ranks[question.id] for result in results}
        if all(rank is not None and rank <= k for rank in ranks.values()):
            continue
        wanted = ", ".join(f"{e.source} > {e.section}" for e in question.expected)
        found = ", ".join(
            f"{mode}: {'miss' if rank is None else rank}" for mode, rank in ranks.items()
        )
        lines.append(f"{question.id} [{question.kind}] {question.question} -> {wanted} | {found}")
    return lines
