from pathlib import Path

from coursemate.embeddings import FakeEmbedder
from coursemate.ingest import find_files, ingest
from coursemate.store import MemoryStore

SAMPLE_NOTES = Path(__file__).parents[1] / "notes" / "sample"


def test_finds_markdown_and_text_files_only(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nText.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("Text.")
    (tmp_path / "slides.pdf").write_bytes(b"%PDF")

    assert [p.name for p in find_files(tmp_path)] == ["a.md", "b.txt"]


def test_ingest_sample_notes_twice_gives_the_same_index():
    store = MemoryStore()
    embedder = FakeEmbedder()

    first = ingest(SAMPLE_NOTES, embedder, store)
    counts_after_first = store.counts()
    second = ingest(SAMPLE_NOTES, embedder, store)

    assert first.files == 4
    assert first.chunks >= 4
    assert second == first
    assert store.counts() == counts_after_first
