import uuid

import pytest
from psycopg import sql

from coursemate.chunking import Chunk
from coursemate.store import MemoryStore, PgVectorStore, StoreError, to_pgvector
from tests.conftest import TEST_DB, require_db


def chunk(source: str, text: str, index: int = 0) -> Chunk:
    return Chunk(source, "", index, text)


@pytest.fixture(params=["memory", pytest.param("pg", marks=pytest.mark.db)])
def store(request):
    if request.param == "memory":
        yield MemoryStore()
        return

    require_db()
    table = f"test_{uuid.uuid4().hex[:8]}"
    pg = PgVectorStore(TEST_DB, table=table)
    try:
        yield pg
    finally:
        pg.conn.execute(sql.SQL("DROP TABLE IF EXISTS {t}").format(t=sql.Identifier(table)))
        pg.conn.execute("DELETE FROM index_meta WHERE table_name = %s", (table,))
        pg.close()


def test_search_returns_closest_first_with_cosine_scores(store):
    store.prepare("toy", 2)
    store.replace_source(
        "toy.md",
        [chunk("toy.md", "east", 0), chunk("toy.md", "north", 1), chunk("toy.md", "north-east", 2)],
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
    )

    hits = store.search([1.0, 0.1], k=2)

    assert [h.text for h in hits] == ["east", "north-east"]
    assert hits[0].score == pytest.approx(0.995, abs=1e-3)


def test_reingesting_a_source_replaces_its_chunks(store):
    store.prepare("toy", 2)
    store.replace_source("a.md", [chunk("a.md", "old")], [[1.0, 0.0]])
    store.replace_source("b.md", [chunk("b.md", "other")], [[0.0, 1.0]])

    store.replace_source(
        "a.md",
        [chunk("a.md", "new one", 0), chunk("a.md", "new two", 1)],
        [[1.0, 0.0], [0.5, 0.5]],
    )

    assert store.counts() == {"a.md": 2, "b.md": 1}
    texts = [h.text for h in store.search([1.0, 0.0], k=10)]
    assert "old" not in texts
    assert len(texts) == 3


def test_all_chunks_are_ordered_by_source_then_index(store):
    store.prepare("toy", 2)
    a0 = Chunk("a.md", "A > One", 0, "first")
    a1 = Chunk("a.md", "A > Two", 1, "second")
    b0 = Chunk("b.md", "B > One", 0, "third")

    # insert out of order: b.md first, and a.md's chunks in reverse index order
    store.replace_source("b.md", [b0], [[0.0, 1.0]])
    store.replace_source("a.md", [a1, a0], [[0.5, 0.5], [1.0, 0.0]])

    assert store.all_chunks() == [a0, a1, b0]


def test_switching_embedding_model_requires_reset(store):
    store.prepare("nomic-embed-text", 768)

    with pytest.raises(StoreError, match="--reset"):
        store.prepare("fake", 256)

    store.prepare("fake", 256, reset=True)
    assert store.counts() == {}


def test_vectors_are_sent_as_pgvector_text():
    assert to_pgvector([1, 0.5]) == "[1.0,0.5]"
