"""BM25 keyword search over chunks: rank by shared words, rare words counting more."""

import math
import re
from collections import Counter

from coursemate.chunking import Chunk
from coursemate.store import Hit

STOPWORDS = frozenset(
    {"when", "is", "are", "the", "what", "how", "do", "i", "my", "a", "an"}
    | {"to", "of", "for", "in", "on", "and", "can"}
)


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, without stopwords."""
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


def bm25_term_score(
    tf: int, df: int, n_docs: int, doc_len: int, avg_len: float, k1: float = 1.5, b: float = 0.75
) -> float:
    """How much one query word adds to one chunk's score."""
    if tf == 0:
        return 0.0
    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    length_norm = 1 - b + b * doc_len / avg_len
    return idf * tf * (k1 + 1) / (tf + k1 * length_norm)


class KeywordIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        # word counts per chunk, using the heading and the text, like the embeddings do
        self.counts = [Counter(tokenize(f"{c.heading} {c.text}")) for c in chunks]
        self.lengths = [sum(counts.values()) for counts in self.counts]
        self.avg_len = sum(self.lengths) / len(chunks) if chunks else 0.0
        # document frequency: in how many chunks does each word appear?
        self.df: Counter[str] = Counter()
        for counts in self.counts:
            self.df.update(counts.keys())

    def search(self, query: str, k: int) -> list[Hit]:
        words = list(dict.fromkeys(tokenize(query)))  # distinct words, in order
        scored: list[tuple[float, Chunk]] = []
        for chunk, counts, length in zip(self.chunks, self.counts, self.lengths, strict=True):
            score = sum(
                bm25_term_score(counts[word], self.df[word], len(self.chunks), length, self.avg_len)
                for word in words
                if counts[word] > 0
            )
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [Hit(c.source, c.heading, c.text, score) for score, c in scored[:k]]
