import json

import pytest
from fastapi.testclient import TestClient

from coursemate.api import create_app
from coursemate.embeddings import FakeEmbedder
from coursemate.ingest import ingest
from coursemate.llm import LLMError
from coursemate.rag import IDK
from coursemate.store import MemoryStore
from coursemate.streaming import FakeStreamer

NOTES = """# Syllabus

## Late work

Projects lose 10% per day late, up to three days. Late work after three days is not accepted.

## Office hours

Office hours are Tuesdays 2 to 4 pm in office 312.
"""

QUESTION = "What is the late penalty for projects?"
GOOD = "Projects lose 10% per day late [1]."


@pytest.fixture
def make_client(tmp_path):
    path = tmp_path / "syllabus.md"
    path.write_text(NOTES, encoding="utf-8")
    embedder = FakeEmbedder()
    store = MemoryStore()
    ingest(path, embedder, store)

    def make(replies):
        fake = FakeStreamer(replies)
        app = create_app(embedder, store, fake, k=2, min_score=0.1)
        return TestClient(app), fake

    return make


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split an SSE body into (event name, data) pairs."""
    events = []
    for block in body.strip().split("\n\n"):
        name_line, data_line = block.split("\n")
        events.append(
            (name_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: ")))
        )
    return events


def test_events_come_in_order_with_the_right_headers(make_client):
    client, _ = make_client([GOOD])

    response = client.post("/api/chat", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "session"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"token"} and len(names) > 3
    assert "".join(data["text"] for name, data in events if name == "token") == GOOD

    done = events[-1][1]
    assert done["found"] is True
    assert done["answer"] == GOOD
    assert done["sources"][0]["n"] == 1
    assert done["sources"][0]["source"] == "syllabus.md"
    assert done["sources"][0]["heading"] == "Syllabus > Late work"
    assert isinstance(done["sources"][0]["score"], float)
    assert (done["provider"], done["model"]) == ("fake", "fake")
    assert (done["input_tokens"], done["output_tokens"]) == (100, 20)
    assert (done["first_token_ms"], done["total_ms"]) == (1.0, 2.0)


def test_a_second_question_in_the_same_session_sees_the_history(make_client):
    client, fake = make_client([GOOD, GOOD])
    first = parse_sse(client.post("/api/chat", json={"question": QUESTION}).text)
    session_id = first[0][1]["session_id"]

    second = client.post(
        "/api/chat", json={"question": "And after three days?", "session_id": session_id}
    )

    assert parse_sse(second.text)[0][1]["session_id"] == session_id
    assert [m["role"] for m in fake.calls[1]] == ["system", "user", "assistant", "user"]
    assert fake.calls[1][1]["content"] == QUESTION
    turns = client.get(f"/api/sessions/{session_id}").json()
    assert turns == [
        {"question": QUESTION, "answer": GOOD},
        {"question": "And after three days?", "answer": GOOD},
    ]


def test_a_retracted_reply_ends_with_found_false(make_client):
    client, _ = make_client(["Projects lose 10% per day late."])  # no citation

    events = parse_sse(client.post("/api/chat", json={"question": QUESTION}).text)

    assert any(name == "token" for name, _ in events)  # the words were streamed...
    name, done = events[-1]
    assert name == "done"
    assert done["found"] is False  # ...and the page is told to retract them
    assert done["answer"] == IDK
    assert done["sources"] == []


def test_an_llm_error_becomes_an_error_event(make_client):
    client, _ = make_client([LLMError("boom")])

    response = client.post("/api/chat", json={"question": QUESTION})

    assert response.status_code == 200  # the stream had already started
    assert parse_sse(response.text)[1:] == [("error", {"message": "boom"})]


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_bad_questions_get_422_without_calling_the_model(make_client, question):
    client, fake = make_client([GOOD])

    response = client.post("/api/chat", json={"question": question})

    assert response.status_code == 422
    assert fake.calls == []


def test_unknown_session_is_404(make_client):
    client, fake = make_client([GOOD])

    chat = client.post("/api/chat", json={"question": QUESTION, "session_id": "nope"})
    lookup = client.get("/api/sessions/nope")

    assert chat.status_code == 404
    assert lookup.status_code == 404
    assert fake.calls == []


def test_health_sessions_and_the_placeholder_page(make_client):
    client, _ = make_client([])

    assert client.get("/api/health").json() == {"status": "ok", "provider": "fake", "model": "fake"}
    session_id = client.post("/api/sessions").json()["session_id"]
    assert client.get(f"/api/sessions/{session_id}").json() == []
    page = client.get("/")
    assert page.status_code == 200
    assert "<title>CourseMate</title>" in page.text
