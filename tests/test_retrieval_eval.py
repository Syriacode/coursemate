import json
from pathlib import Path

import pytest

from coursemate.chunking import chunk_document
from coursemate.embeddings import FakeEmbedder
from coursemate.retrieval import MODES
from coursemate.retrieval_eval import (
    Expected,
    GoldenSetError,
    ModeResult,
    build_index,
    evaluate,
    first_relevant_rank,
    is_relevant,
    load_golden,
    mean_reciprocal_rank,
    recall_at_k,
    save_run,
)
from coursemate.store import Hit

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


def hit(source, heading):
    return Hit(source=source, heading=heading, text="...", score=0.5)


LAB4 = [Expected(source="lecture-02-hash-tables.md", section="Lab 4")]


def test_is_relevant_matches_exact_heading_and_heading_path():
    assert is_relevant(hit("lecture-02-hash-tables.md", "Lab 4"), LAB4)
    assert is_relevant(hit("lecture-02-hash-tables.md", "Lecture 2: Hash Tables > Lab 4"), LAB4)


def test_is_relevant_rejects_wrong_source_or_similar_heading():
    assert not is_relevant(hit("syllabus.md", "Lecture 2: Hash Tables > Lab 4"), LAB4)
    assert not is_relevant(
        hit("lecture-02-hash-tables.md", "Lecture 2: Hash Tables > Big Lab 4"), LAB4
    )


def test_first_relevant_rank_is_one_based_or_none():
    hits = [
        hit("syllabus.md", "CS 201 Data Structures: Syllabus > Grading"),
        hit("lecture-02-hash-tables.md", "Lecture 2: Hash Tables > Lab 4"),
        hit("lecture-02-hash-tables.md", "Lecture 2: Hash Tables > Lab 4"),
    ]

    assert first_relevant_rank(hits, LAB4) == 2
    assert first_relevant_rank(hits[:1], LAB4) is None
    assert first_relevant_rank([], LAB4) is None


@pytest.mark.parametrize("k, expected", [(1, 0.25), (2, 0.5), (3, 0.75)])
def test_recall_at_k(k, expected):
    assert recall_at_k([1, 3, None, 2], k) == pytest.approx(expected)


def test_recall_of_an_empty_list_is_zero():
    assert recall_at_k([], 3) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([1, 2, None, 4]) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert mean_reciprocal_rank([]) == 0.0


def test_evaluate_on_sample_notes_with_fake_embedder():
    golden = load_golden(GOLDEN)
    embedder = FakeEmbedder()

    results = evaluate(golden, embedder, build_index(SAMPLE_NOTES, embedder))

    assert [r.mode for r in results] == list(MODES)
    for result in results:
        assert set(result.ranks) == {q.id for q in golden}
        assert 0.0 <= result.recall(1) <= result.recall(3) <= result.recall(5) <= 1.0


def test_save_run_writes_metrics_and_ranks(tmp_path):
    results = [
        ModeResult("vector", {"q01": 1, "q02": None}),
        ModeResult("keyword", {"q01": 3, "q02": 2}),
    ]
    path = tmp_path / "runs" / "toy.json"  # the runs folder doesn't exist yet

    save_run(path, results, "toy-model")

    run = json.loads(path.read_text(encoding="utf-8"))
    assert run["model"] == "toy-model"
    assert list(run["modes"]) == ["vector", "keyword"]
    assert run["modes"]["vector"]["recall@1"] == 0.5
    assert run["modes"]["vector"]["mrr"] == pytest.approx(0.5)
    assert run["modes"]["vector"]["ranks"] == {"q01": 1, "q02": None}
    assert run["modes"]["keyword"]["recall@3"] == 1.0
