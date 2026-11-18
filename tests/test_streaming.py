import json

import httpx
import pytest

from coursemate.llm import LLMError
from coursemate.streaming import (
    FakeStreamer,
    OllamaStreamer,
    StreamEnd,
    TextDelta,
    make_streamer,
)

MESSAGES = [{"role": "user", "content": "What is the late penalty?"}]

# recorded-style Ollama NDJSON: one JSON object per line, the last one has done: true
NDJSON = "\n".join(
    json.dumps(line)
    for line in [
        {"model": "qwen2.5:3b", "message": {"role": "assistant", "content": "The"}, "done": False},
        {
            "model": "qwen2.5:3b",
            "message": {"role": "assistant", "content": " penalty"},
            "done": False,
        },
        {
            "model": "qwen2.5:3b",
            "message": {"role": "assistant", "content": " is 10%."},
            "done": False,
        },
        {
            "model": "qwen2.5:3b",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "prompt_eval_count": 812,
            "eval_count": 9,
        },
    ]
)


def streamer_with(handler):
    http = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaStreamer(model="qwen2.5:3b", http=http)


async def collect(streamer, **kwargs):
    return [event async for event in streamer.stream(MESSAGES, **kwargs)]


@pytest.mark.anyio
async def test_pieces_join_into_the_right_text():
    streamer = streamer_with(lambda request: httpx.Response(200, content=NDJSON))
    events = await collect(streamer)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "The penalty is 10%."
    assert isinstance(events[-1], StreamEnd)
    assert sum(isinstance(e, StreamEnd) for e in events) == 1


@pytest.mark.anyio
async def test_token_counts_come_from_the_last_line():
    streamer = streamer_with(lambda request: httpx.Response(200, content=NDJSON))
    end = (await collect(streamer))[-1]
    assert (end.model, end.input_tokens, end.output_tokens) == ("qwen2.5:3b", 812, 9)
    assert 0 <= end.first_token_ms <= end.total_ms


@pytest.mark.anyio
async def test_request_asks_for_a_stream_with_the_right_options():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=NDJSON)

    await collect(streamer_with(handler), temperature=0.2, max_tokens=256)
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/api/chat"
    assert body["stream"] is True
    assert body["messages"] == MESSAGES
    assert body["options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 256}


@pytest.mark.anyio
async def test_http_error_becomes_llm_error():
    streamer = streamer_with(lambda request: httpx.Response(500))
    with pytest.raises(LLMError):
        await collect(streamer)


@pytest.mark.anyio
async def test_a_stream_that_ends_without_a_final_line_is_an_error():
    first_line = NDJSON.split("\n")[0]
    streamer = streamer_with(lambda request: httpx.Response(200, content=first_line))
    with pytest.raises(LLMError):
        await collect(streamer)


@pytest.mark.anyio
async def test_fake_streamer_streams_word_by_word_and_records_calls():
    fake = FakeStreamer()
    events = await collect(fake)
    pieces = [e.text for e in events if isinstance(e, TextDelta)]
    assert len(pieces) > 1
    assert "".join(pieces) == "Fake provider: no model was called [1]."
    assert events[-1] == StreamEnd("fake", 100, 20, 1.0, 2.0)
    assert fake.calls == [MESSAGES]


@pytest.mark.anyio
async def test_fake_streamer_raises_an_exception_reply():
    fake = FakeStreamer([LLMError("boom")])
    with pytest.raises(LLMError, match="boom"):
        await collect(fake)


def test_make_streamer_picks_the_provider():
    assert isinstance(make_streamer("fake", "x", "http://localhost:11434"), FakeStreamer)
    ollama = make_streamer("ollama", "qwen2.5:3b", "http://localhost:11434")
    assert isinstance(ollama, OllamaStreamer)
    assert ollama.model == "qwen2.5:3b"
    with pytest.raises(ValueError):
        make_streamer("nope", "x", "http://localhost:11434")
