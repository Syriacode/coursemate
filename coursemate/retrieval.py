"""One entry point for the three ways of searching: vector, keyword and hybrid."""

from dataclasses import dataclass
from typing import Literal

from coursemate.embeddings import Embedder
from coursemate.hybrid import rrf
from coursemate.keyword import KeywordIndex
from coursemate.store import Hit, VectorStore

Mode = Literal["vector", "keyword", "hybrid"]
MODES: tuple[Mode, ...] = ("vector", "keyword", "hybrid")


@dataclass(frozen=True)
class Retrieval:
    hits: list[Hit]
    # Top cosine score of the vector search. BM25 and RRF scores are not on the cosine scale,
    # so the "I don't know" gate must always use this number.
    best_vector_score: float


def retrieve(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    k: int,
    mode: Mode = "vector",
    candidates: int = 20,
) -> Retrieval:
    """Return the top k hits for the query, searching the way `mode` says."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose one of {', '.join(MODES)}")

    # Keyword and hybrid look at more candidates, so a chunk that is 6th in one list
    # and 2nd in the other can still be fused.
    depth = k if mode == "vector" else max(k, candidates)

    # Vector search always runs: we need its top cosine score in every mode.
    vector_hits = store.search(embedder.embed_query(query), depth)
    best_vector_score = vector_hits[0].score if vector_hits else 0.0

    if mode == "vector":
        return Retrieval(vector_hits, best_vector_score)

    keyword_hits = KeywordIndex(store.all_chunks()).search(query, depth)
    if mode == "keyword":
        return Retrieval(keyword_hits[:k], best_vector_score)

    return Retrieval(rrf([vector_hits, keyword_hits])[:k], best_vector_score)
