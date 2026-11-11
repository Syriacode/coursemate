import json

import httpx
import pytest

from coursemate.embeddings import FakeEmbedder, OllamaEmbedder, cosine
from coursemate.llm import LLMError


def embedder_with(handler, dim=3):
    http = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaEmbedder(dim=dim, http=http)


def test_documents_and_queries_get_different_prefixes():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]] * len(body["input"])})

    embedder = embedder_with(handler)
    embedder.embed_documents(["a", "b"])
    embedder.embed_query("c")

    assert bodies[0] == {
        "model": "nomic-embed-text",
        "input": ["search_document: a", "search_document: b"],
    }
    assert bodies[1]["input"] == ["search_query: c"]


def test_wrong_dimension_is_an_error():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    with pytest.raises(LLMError, match="2 dimensions"):
        embedder_with(handler).embed_query("c")


def test_connection_error_becomes_llm_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMError):
        embedder_with(handler).embed_query("c")


def test_fake_embedder_puts_related_text_closer():
    embedder = FakeEmbedder()
    query = embedder.embed_query("late penalty for projects")
    related, unrelated = embedder.embed_documents(
        [
            "projects lose 10% per day late: the late penalty",
            "binary search trees keep keys in sorted order",
        ]
    )

    assert cosine(query, related) > cosine(query, unrelated)


def test_cosine():
    assert cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
