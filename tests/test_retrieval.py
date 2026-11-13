import pytest

from coursemate.chunking import Chunk
from coursemate.embeddings import FakeEmbedder
from coursemate.retrieval import MODES, retrieve
from coursemate.store import MemoryStore

TEXTS = {
    "late.md": "Late projects lose ten percent every day",
    "sorting.md": "Merge sort splits the array and merges sorted halves",
    "graphs.md": "Breadth first search uses a queue to explore a graph",
    "hash.md": "Hash tables use chaining to handle collisions",
}


@pytest.fixture
def store_and_embedder():
    embedder = FakeEmbedder()
    store = MemoryStore()
    store.prepare(embedder.model, embedder.dim)
    for source, text in TEXTS.items():
        chunks = [Chunk(source, "Notes", 0, text)]
        store.replace_source(source, chunks, embedder.embed_documents([text]))
    return store, embedder


def test_vector_mode_returns_k_hits_and_the_top_cosine_score(store_and_embedder):
    store, embedder = store_and_embedder

    result = retrieve("how does merge sort work", embedder, store, k=2)

    assert len(result.hits) == 2
    assert result.hits[0].source == "sorting.md"
    assert result.best_vector_score == result.hits[0].score


def test_keyword_mode_finds_the_exact_word(store_and_embedder):
    store, embedder = store_and_embedder

    result = retrieve("chaining", embedder, store, k=3, mode="keyword")

    assert [h.source for h in result.hits] == ["hash.md"]  # chunks without the word are dropped


def test_hybrid_mode_fuses_and_keeps_the_vector_score(store_and_embedder):
    store, embedder = store_and_embedder

    hybrid = retrieve("merge sort", embedder, store, k=2, mode="hybrid")
    vector = retrieve("merge sort", embedder, store, k=2, mode="vector")

    assert len(hybrid.hits) <= 2
    assert hybrid.hits[0].source == "sorting.md"
    assert hybrid.best_vector_score == vector.best_vector_score


@pytest.mark.parametrize("mode", MODES)
def test_an_empty_store_gives_no_hits(mode):
    embedder = FakeEmbedder()
    store = MemoryStore()
    store.prepare(embedder.model, embedder.dim)

    result = retrieve("anything", embedder, store, k=3, mode=mode)

    assert result.hits == []
    assert result.best_vector_score == 0.0


def test_an_unknown_mode_is_an_error(store_and_embedder):
    store, embedder = store_and_embedder

    with pytest.raises(ValueError, match="unknown mode"):
        retrieve("x", embedder, store, k=1, mode="bm25")  # type: ignore[arg-type]
