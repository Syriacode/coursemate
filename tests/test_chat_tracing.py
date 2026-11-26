import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from opentelemetry.trace import StatusCode

from coursemate.api import ChatRequest, create_app
from coursemate.cache import CachedStreamer, ResponseCache
from coursemate.chat import Final, Token, stream_answer
from coursemate.embeddings import FakeEmbedder
from coursemate.ingest import ingest
from coursemate.llm import LLMError
from coursemate.pricing import PRICES, Price
from coursemate.store import MemoryStore
from coursemate.streaming import FakeStreamer
from coursemate.tracing import iso_time

NOTES = """# Syllabus

## Late work

Projects lose 10% per day late, up to three days. Late work after three days is not accepted.

## Office hours

Office hours are Tuesdays 2 to 4 pm in office 312.
"""

QUESTION = "What is the late penalty for projects?"
GOOD = "Projects lose 10% per day late [1]."


class HostedStreamer(FakeStreamer):
    local = False  # priced like a hosted API, but still entirely fake


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "syllabus.md"
    path.write_text(NOTES, encoding="utf-8")
    embedder = FakeEmbedder()
    store = MemoryStore()
    ingest(path, embedder, store)
    return embedder, store


def answer(index, llm, question=QUESTION, **kwargs):
    embedder, store = index
    kwargs.setdefault("k", 2)
    kwargs.setdefault("min_score", 0.1)
    return stream_answer(question, [], embedder, store, llm, **kwargs)


async def final_of(events) -> Final:
    return [event async for event in events][-1]


def by_name(spans):
    finished = spans.get_finished_spans()
    named = {s.name: s for s in finished}
    assert len(named) == len(finished)  # no stage was traced twice
    return named


@pytest.mark.anyio
async def test_one_answer_is_one_trace_of_four_spans(index, spans):
    await final_of(answer(index, FakeStreamer([GOOD]), session_id="s1"))

    named = by_name(spans)
    assert set(named) == {"chat", "retrieve", "llm", "check"}
    chat = named["chat"]
    assert chat.parent is None
    for name in ("retrieve", "llm", "check"):
        assert named[name].context.trace_id == chat.context.trace_id
        assert named[name].parent.span_id == chat.context.span_id

    root = chat.attributes
    assert root["langfuse.trace.name"] == "chat"
    assert root["langfuse.session.id"] == "s1"
    assert root["langfuse.observation.input"] == QUESTION
    assert root["langfuse.observation.output"] == GOOD
    assert (root["coursemate.found"], root["coursemate.cached"]) == (True, False)
    assert (root["coursemate.input_tokens"], root["coursemate.output_tokens"]) == (100, 20)
    assert root["coursemate.first_token_ms"] == 1.0
    assert root["coursemate.cost_usd"] == 0.0

    retrieve = named["retrieve"].attributes
    assert retrieve["langfuse.observation.type"] == "retriever"
    assert retrieve["langfuse.observation.input"] == QUESTION
    hits = json.loads(retrieve["langfuse.observation.output"])
    assert len(hits) == 2 and set(hits[0]) == {"source", "heading", "score"}
    assert (retrieve["coursemate.k"], retrieve["coursemate.mode"]) == (2, "vector")
    assert retrieve["coursemate.best_score"] == hits[0]["score"]

    llm = named["llm"].attributes
    assert llm["langfuse.observation.type"] == "generation"
    messages = json.loads(llm["langfuse.observation.input"])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert llm["langfuse.observation.output"] == GOOD
    assert llm["gen_ai.request.model"] == "fake"
    assert (llm["gen_ai.usage.input_tokens"], llm["gen_ai.usage.output_tokens"]) == (100, 20)
    assert json.loads(llm["langfuse.observation.usage_details"]) == {"input": 100, "output": 20}
    assert json.loads(llm["langfuse.observation.cost_details"]) == {"total": 0.0}
    assert llm["coursemate.cached"] is False
    # FakeStreamer says the first word came after 1 ms
    first_word = datetime.fromisoformat(llm["langfuse.observation.completion_start_time"])
    started = datetime.fromisoformat(iso_time(named["llm"].start_time))
    assert first_word - started == timedelta(milliseconds=1)

    check = named["check"].attributes
    assert check["langfuse.observation.type"] == "guardrail"
    assert check["langfuse.observation.output"] == "passed"
    assert "langfuse.observation.level" not in check


@pytest.mark.anyio
async def test_gate_one_refusal_has_no_llm_span(index, spans):
    fake = FakeStreamer([])

    await final_of(answer(index, fake, "Who won the 2022 World Cup?", min_score=0.5))

    named = by_name(spans)
    assert set(named) == {"chat", "retrieve"}
    assert named["chat"].attributes["coursemate.found"] is False
    assert "langfuse.session.id" not in named["chat"].attributes  # none was given
    assert fake.calls == []


@pytest.mark.anyio
async def test_a_failed_citation_check_is_a_warning(index, spans):
    final = await final_of(answer(index, FakeStreamer(["Projects lose 10% per day late."])))

    named = by_name(spans)
    check = named["check"].attributes
    assert check["langfuse.observation.level"] == "WARNING"
    assert check["langfuse.observation.output"] == final.reason
    assert named["chat"].attributes["coursemate.found"] is False


@pytest.mark.anyio
async def test_a_model_error_ends_every_started_span(index, spans):
    with pytest.raises(LLMError):
        await final_of(answer(index, FakeStreamer([LLMError("model is down")])))

    named = by_name(spans)
    assert set(named) == {"chat", "retrieve", "llm"}
    assert named["llm"].status.status_code is StatusCode.ERROR
    assert named["chat"].status.status_code is StatusCode.ERROR
    assert named["retrieve"].status.status_code is StatusCode.UNSET


@pytest.mark.anyio
async def test_closing_after_the_first_token_ends_every_started_span(index, spans):
    events = answer(index, FakeStreamer([GOOD]))
    assert isinstance(await anext(events), Token)
    assert [s.name for s in spans.get_finished_spans()] == ["retrieve"]  # the rest are open

    await events.aclose()  # what happens when the student closes the page

    named = by_name(spans)
    assert set(named) == {"chat", "retrieve", "llm"}
    assert all(s.status.status_code is StatusCode.UNSET for s in named.values())


@pytest.mark.anyio
async def test_closing_the_api_stream_ends_every_started_span(index, spans):
    embedder, store = index
    app = create_app(embedder, store, FakeStreamer([GOOD]), k=2, min_score=0.1)
    [chat_route] = [r for r in app.routes if getattr(r, "path", "") == "/api/chat"]
    body = (await chat_route.endpoint(ChatRequest(question=QUESTION))).body_iterator
    assert (await anext(body)).startswith("event: session")
    assert (await anext(body)).startswith("event: token")

    await body.aclose()  # closes only the API's generator; it must close stream_answer too

    assert set(by_name(spans)) == {"chat", "retrieve", "llm"}


@pytest.mark.anyio
async def test_an_unknown_price_gives_no_cost(index, spans):
    final = await final_of(answer(index, HostedStreamer([GOOD])))

    assert final.cost_usd is None
    named = by_name(spans)
    assert "langfuse.observation.cost_details" not in named["llm"].attributes
    assert "coursemate.cost_usd" not in named["chat"].attributes


@pytest.mark.anyio
async def test_a_known_price_is_costed(index, spans, monkeypatch):
    monkeypatch.setitem(PRICES, "fake", Price(1.0, 2.0))

    final = await final_of(answer(index, HostedStreamer([GOOD])))

    expected = pytest.approx(0.00014)  # (100 × $1 + 20 × $2) per million tokens
    assert final.cost_usd == expected
    llm = by_name(spans)["llm"].attributes
    assert json.loads(llm["langfuse.observation.cost_details"])["total"] == expected


@pytest.mark.anyio
async def test_the_same_question_twice_hits_the_cache(index, spans):
    fake = FakeStreamer([GOOD])  # one reply: the second answer must not call the model
    llm = CachedStreamer(fake, ResponseCache())

    first = await final_of(answer(index, llm))
    second = await final_of(answer(index, llm))

    assert [first.cached, second.cached] == [False, True]
    assert second.answer == first.answer and second.found
    assert (second.input_tokens, second.output_tokens) == (0, 0)
    roots = [s for s in spans.get_finished_spans() if s.name == "chat"]
    assert [s.attributes["coursemate.cached"] for s in roots] == [False, True]


def test_the_api_session_id_is_on_the_root_span(index, spans):
    embedder, store = index
    app = create_app(embedder, store, FakeStreamer([GOOD]), k=2, min_score=0.1)

    response = TestClient(app).post("/api/chat", json={"question": QUESTION})

    events = [block.split("\n") for block in response.text.strip().split("\n\n")]
    data = {name: json.loads(line.removeprefix("data: ")) for name, line in events}
    session_id = data["event: session"]["session_id"]
    assert by_name(spans)["chat"].attributes["langfuse.session.id"] == session_id
    assert data["event: done"]["cached"] is False
