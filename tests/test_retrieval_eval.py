import json
from pathlib import Path

import pytest

from coursemate.chunking import chunk_document
from coursemate.retrieval_eval import GoldenSetError, load_golden

ROOT = Path(__file__).parents[1]
SAMPLE_NOTES = ROOT / "notes" / "sample"
GOLDEN = ROOT / "evals" / "retrieval_golden.jsonl"


def row(id, reviewed=True):
    return {
        "id": id,
        "question": "When is Lab 4 due?",
        "expected": [{"source": "lecture-02-hash-tables.md", "section": "Lab 4"}],
        "kind": "keyword",
        "reviewed": reviewed,
    }


def write_rows(tmp_path, rows):
    path = tmp_path / "golden.jsonl"
    # a blank line between rows checks that blank lines are skipped
    path.write_text("\n\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_reviewed_rows_load_and_blank_lines_are_skipped(tmp_path):
    path = write_rows(tmp_path, [row("q01"), row("q02")])

    assert [q.id for q in load_golden(path)] == ["q01", "q02"]


def test_unreviewed_rows_are_refused(tmp_path):
    path = write_rows(tmp_path, [row("q01"), row("q02", reviewed=False)])

    with pytest.raises(GoldenSetError, match="q02") as info:
        load_golden(path)

    assert "q01" not in str(info.value)


def test_duplicate_ids_are_refused(tmp_path):
    path = write_rows(tmp_path, [row("q01"), row("q01")])

    with pytest.raises(GoldenSetError, match="duplicate id 'q01'"):
        load_golden(path)


def test_every_golden_question_points_at_a_real_section():
    # (source file, last heading) of every chunk in the sample notes
    real = set()
    for file in SAMPLE_NOTES.glob("*.md"):
        for chunk in chunk_document(file.name, file.read_text(encoding="utf-8")):
            real.add((file.name, chunk.heading.split(" > ")[-1]))

    # read the rows directly: they are not reviewed yet, so load_golden would refuse them
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line]
    missing = [
        (row["id"], e["source"], e["section"])
        for row in rows
        for e in row["expected"]
        if (e["source"], e["section"]) not in real
    ]

    assert len(rows) == 25
    assert missing == []
