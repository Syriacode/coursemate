import json

import pytest

from coursemate.embeddings import FakeEmbedder
from coursemate.ingest import ingest
from coursemate.llm import FakeClient
from coursemate.rag import IDK, answer
from coursemate.store import MemoryStore

NOTES = """# Syllabus

## Late work

Projects lose 10% per day late, up to three days. Late work after three days is not accepted.

## Office hours

Office hours are Tuesdays 2 to 4 pm in office 312.
"""

QUESTION = "What is the late penalty for projects?"


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "syllabus.md"
    path.write_text(NOTES, encoding="utf-8")
    embedder = FakeEmbedder()
    store = MemoryStore()
    ingest(path, embedder, store)
    return embedder, store


def reply(found=True, text="Projects lose 10% per day late.", citations=(1,)):
    return json.dumps({"found": found, "answer": text, "citations": list(citations)})


def test_grounded_answer_comes_back_with_its_sources(index):
    embedder, store = index
    client = FakeClient([reply()])

    result = answer(QUESTION, embedder, store, client, k=2, min_score=0.1)

    assert result.found
    assert result.answer == "Projects lose 10% per day late."
    assert [h.heading for h in result.sources] == ["Syllabus > Late work"]
    assert (result.input_tokens, result.output_tokens) == (100, 20)
    user_message = client.calls[0][1]["content"]
    assert "[1] syllabus.md > Syllabus > Late work" in user_message
    assert user_message.endswith(QUESTION)


def test_low_retrieval_score_refuses_without_calling_the_model(index):
    embedder, store = index
    client = FakeClient([])

    result = answer("Who won the 2022 World Cup?", embedder, store, client, k=2, min_score=0.5)

    assert not result.found
    assert result.answer == IDK
    assert client.calls == []
    assert "best" in result.reason


def test_model_saying_not_found_means_i_dont_know(index):
    embedder, store = index
    client = FakeClient([reply(found=False, text="", citations=())])

    result = answer(QUESTION, embedder, store, client, k=2, min_score=0.1)

    assert not result.found
    assert result.answer == IDK
    assert len(client.calls) == 1


def test_answer_without_citations_is_not_trusted(index):
    embedder, store = index
    client = FakeClient([reply(citations=())])

    result = answer(QUESTION, embedder, store, client, k=2, min_score=0.1)

    assert not result.found
    assert result.answer == IDK


def test_citing_a_source_that_was_not_retrieved_fails_closed(index):
    embedder, store = index
    client = FakeClient([reply(citations=(1, 7))])

    result = answer(QUESTION, embedder, store, client, k=2, min_score=0.1)

    assert not result.found
    assert "7" in result.reason


def test_invalid_json_fails_closed(index):
    embedder, store = index
    client = FakeClient(["not json"])

    result = answer(QUESTION, embedder, store, client, k=2, min_score=0.1)

    assert not result.found
    assert result.answer == IDK
