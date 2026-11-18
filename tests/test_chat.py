import pytest

from coursemate.chat import (
    Final,
    Token,
    Turn,
    extract_citations,
    stream_answer,
)
from coursemate.embeddings import FakeEmbedder
from coursemate.ingest import ingest
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


class RecordingEmbedder(FakeEmbedder):
    """A FakeEmbedder that remembers what it was asked to search for."""

    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    def embed_query(self, text):
        self.queries.append(text)
        return super().embed_query(text)


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "syllabus.md"
    path.write_text(NOTES, encoding="utf-8")
    embedder = RecordingEmbedder()
    store = MemoryStore()
    ingest(path, embedder, store)
    embedder.queries.clear()  # forget the ingest-time calls
    return embedder, store


async def collect(question, history, embedder, store, llm, **kwargs):
    kwargs.setdefault("k", 2)
    kwargs.setdefault("min_score", 0.1)
    events = [e async for e in stream_answer(question, history, embedder, store, llm, **kwargs)]
    tokens = [e for e in events if isinstance(e, Token)]
    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1 and events[-1] is finals[0]  # exactly one Final, and it comes last
    return tokens, finals[0]


@pytest.mark.anyio
async def test_tokens_then_a_found_final(index):
    embedder, store = index
    fake = FakeStreamer([GOOD])

    tokens, final = await collect(QUESTION, [], embedder, store, fake)

    assert len(tokens) > 1
    assert "".join(t.text for t in tokens) == GOOD
    assert final.found
    assert final.answer == GOOD
    assert final.citations == [1]
    assert [h.heading for h in final.sources] == ["Syllabus > Late work"]
    assert (final.model, final.input_tokens, final.output_tokens) == ("fake", 100, 20)
    assert (final.first_token_ms, final.total_ms) == (1.0, 2.0)


@pytest.mark.anyio
async def test_low_retrieval_score_refuses_without_calling_the_model(index):
    embedder, store = index
    fake = FakeStreamer([])

    tokens, final = await collect(
        "Who won the 2022 World Cup?", [], embedder, store, fake, min_score=0.5
    )

    assert tokens == []
    assert not final.found
    assert final.answer == IDK
    assert "best" in final.reason
    assert fake.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "bad_reply",
    [
        "Projects lose 10% per day late.",  # no citation
        "Projects lose 10% per day late [7].",  # source 7 was never retrieved
        "I don't know.",  # straight apostrophe
        "I don’t know.",  # curly apostrophe
    ],
)
async def test_bad_reply_streams_its_words_but_ends_not_found(index, bad_reply):
    embedder, store = index
    fake = FakeStreamer([bad_reply])

    tokens, final = await collect(QUESTION, [], embedder, store, fake)

    assert "".join(t.text for t in tokens) == bad_reply  # the page showed these words...
    assert not final.found  # ...and now has to retract them
    assert final.answer == IDK
    assert final.sources == [] and final.citations == []
    assert final.reason


@pytest.mark.anyio
async def test_history_reaches_the_model(index):
    embedder, store = index
    fake = FakeStreamer([GOOD])
    history = [Turn(QUESTION, "Projects lose 10% per day [1].")]

    await collect("And after three days?", history, embedder, store, fake)

    messages = fake.calls[0]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == QUESTION
    assert messages[2]["content"] == "Projects lose 10% per day [1]."
    assert messages[3]["content"].endswith("Question:\nAnd after three days?")
    assert "[1] syllabus.md > Syllabus > Late work" in messages[3]["content"]


@pytest.mark.anyio
async def test_follow_up_searches_with_the_previous_question(index):
    embedder, store = index
    fake = FakeStreamer([GOOD, GOOD])

    await collect(QUESTION, [], embedder, store, fake)
    assert embedder.queries == [QUESTION]

    embedder.queries.clear()
    history = [Turn(QUESTION, GOOD)]
    await collect("And after three days?", history, embedder, store, fake)
    assert embedder.queries == [f"{QUESTION} And after three days?"]


def test_extract_citations_keeps_first_use_order_without_duplicates():
    assert extract_citations("a [2] b [1][2] c [3]") == [2, 1, 3]
    assert extract_citations("no citations here") == []
