"""Hybrid search: fuse ranked lists with Reciprocal Rank Fusion (RRF)."""

from dataclasses import replace

from coursemate.store import Hit


def rrf(rankings: list[list[Hit]], c: int = 60) -> list[Hit]:
    """Each list gives a chunk 1 / (c + rank); chunks high in several lists win."""
    scores: dict[tuple[str, str, str], float] = {}
    first_seen: dict[tuple[str, str, str], Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.source, hit.heading, hit.text)
            scores[key] = scores.get(key, 0.0) + 1 / (c + rank)
            first_seen.setdefault(key, hit)
    fused = [replace(first_seen[key], score=score) for key, score in scores.items()]
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused
