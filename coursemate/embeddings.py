import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

import httpx

from coursemate.llm import LLMError

Vector = list[float]

# nomic-embed-text was trained with a task prefix on every input
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

STOPWORDS = {"the", "and", "for", "what", "how", "are", "does", "with", "this", "that", "you"}


class Embedder(Protocol):
    model: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        dim: int = 768,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        doc_prefix: str = DOC_PREFIX,
        query_prefix: str = QUERY_PREFIX,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.dim = dim
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout)

    def _embed(self, inputs: list[str]) -> list[Vector]:
        try:
            response = self._http.post("/api/embed", json={"model": self.model, "input": inputs})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama embed failed: {exc}") from exc

        vectors = response.json()["embeddings"]
        if len(vectors) != len(inputs):
            raise LLMError(f"{self.model} returned {len(vectors)} vectors for {len(inputs)} inputs")
        for vector in vectors:
            if len(vector) != self.dim:
                raise LLMError(
                    f"{self.model} returned {len(vector)} dimensions, expected {self.dim}"
                )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return self._embed([self.doc_prefix + text for text in texts])

    def embed_query(self, text: str) -> Vector:
        return self._embed([self.query_prefix + text])[0]


class FakeEmbedder:
    """Deterministic bag-of-words vectors; texts that share words get similar vectors.

    No model, no network.
    """

    def __init__(self, dim: int = 256) -> None:
        self.model = "fake"
        self.dim = dim

    def _vector(self, text: str) -> Vector:
        vector = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) >= 3 and word not in STOPWORDS:
                vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return self._vector(text)


def cosine(a: Vector, b: Vector) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
