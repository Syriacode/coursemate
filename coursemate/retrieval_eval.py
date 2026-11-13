from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


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
