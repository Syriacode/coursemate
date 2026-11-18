import json

import httpx
import pytest

from coursemate.llm import LLMError
from coursemate.streaming import (
    AnthropicStreamer,
    FakeStreamer,
    OllamaStreamer,
    OpenAIStreamer,
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


def test_make_streamer_knows_the_hosted_formats(monkeypatch):
    # point both at local Ollama through the environment, so no key is needed
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    openai = make_streamer("openai", "qwen2.5:3b", "http://localhost:11434")
    anthropic = make_streamer("anthropic", "qwen2.5:3b", "http://localhost:11434")

    assert isinstance(openai, OpenAIStreamer)
    assert isinstance(anthropic, AnthropicStreamer)
    assert (openai.provider, openai.model, openai.local) == ("openai", "qwen2.5:3b", True)
    assert (anthropic.provider, anthropic.model, anthropic.local) == (
        "anthropic",
        "qwen2.5:3b",
        True,
    )


# ---- OpenAI and Anthropic formats: recorded-style SSE bodies, no network ----------------------

OPENAI_BASE = "https://api.openai.com/v1"
ANTHROPIC_BASE = "https://api.anthropic.com"

CHAT = [
    {"role": "system", "content": "Answer only from the sources."},
    {"role": "user", "content": "What is the late penalty?"},
]


def sse_body(*events):
    """Build an SSE body from (event name or None, data) pairs; a dict becomes a JSON line."""
    parts = []
    for name, data in events:
        payload = data if isinstance(data, str) else json.dumps(data)
        parts.append((f"event: {name}\n" if name else "") + f"data: {payload}\n\n")
    return "".join(parts)


def sse_response(body, status=200):
    return httpx.Response(status, content=body, headers={"content-type": "text/event-stream"})


def recording(body):
    """A MockTransport handler that remembers every request it was sent."""
    requests = []

    def handler(request):
        requests.append(request)
        return sse_response(body)

    return handler, requests


def chat_chunk(content=None, finish=None):
    # what OpenAI sends for each piece: text in choices[0].delta.content, usage is null
    delta = {} if content is None else {"content": content}
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "gpt-test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        "usage": None,
    }


OPENAI_SSE = sse_body(
    (None, chat_chunk("The")),
    (None, chat_chunk(" penalty")),
    (None, chat_chunk(" is 10%.")),
    (None, chat_chunk(finish="stop")),
    # the usage chunk has no choices, and only comes because we ask for include_usage
    (
        None,
        {
            "id": "chatcmpl-1",
            "model": "gpt-test",
            "choices": [],
            "usage": {"prompt_tokens": 812, "completion_tokens": 9, "total_tokens": 821},
        },
    ),
    (None, "[DONE]"),
)


def text_delta(text):
    return {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": text},
    }


ANTHROPIC_START = (
    "message_start",
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [],
            "usage": {"input_tokens": 812, "output_tokens": 1},
        },
    },
)

ANTHROPIC_SSE = sse_body(
    ANTHROPIC_START,
    (
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ),
    ("ping", {"type": "ping"}),
    ("content_block_delta", text_delta("The")),
    ("content_block_delta", text_delta(" penalty")),
    ("content_block_delta", text_delta(" is 10%.")),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 9},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
)


def openai_with(handler, api_key="test-key", base_url=OPENAI_BASE):
    http = httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))
    return OpenAIStreamer(model="gpt-test", base_url=base_url, api_key=api_key, http=http)


def anthropic_with(handler, api_key="test-key", base_url=ANTHROPIC_BASE):
    http = httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))
    return AnthropicStreamer(model="claude-test", base_url=base_url, api_key=api_key, http=http)


async def collect_chat(streamer, **kwargs):
    return [event async for event in streamer.stream(CHAT, **kwargs)]


def pieces_of(events):
    return [e.text for e in events if isinstance(e, TextDelta)]


@pytest.mark.anyio
async def test_openai_pieces_join_and_counts_come_from_the_usage_chunk():
    events = await collect_chat(openai_with(lambda request: sse_response(OPENAI_SSE)))

    assert pieces_of(events) == ["The", " penalty", " is 10%."]
    assert sum(isinstance(e, StreamEnd) for e in events) == 1
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert (end.model, end.input_tokens, end.output_tokens) == ("gpt-test", 812, 9)
    assert 0 <= end.first_token_ms <= end.total_ms


@pytest.mark.anyio
async def test_openai_sends_the_exact_url_headers_and_body():
    handler, requests = recording(OPENAI_SSE)

    await collect_chat(openai_with(handler), temperature=0.2, max_tokens=256)

    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert "x-api-key" not in request.headers
    # the system prompt stays inside messages, as the first message
    assert json.loads(request.content) == {
        "model": "gpt-test",
        "messages": CHAT,
        "temperature": 0.2,
        "max_completion_tokens": 256,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


@pytest.mark.anyio
async def test_openai_401_becomes_llm_error_with_the_status():
    body = {"error": {"message": "Incorrect API key provided"}}
    streamer = openai_with(lambda request: httpx.Response(401, json=body))

    with pytest.raises(LLMError, match="401"):
        await collect_chat(streamer)


@pytest.mark.anyio
async def test_openai_stream_without_done_is_an_error():
    only_one_chunk = sse_body((None, chat_chunk("The")))

    with pytest.raises(LLMError, match="DONE"):
        await collect_chat(openai_with(lambda request: sse_response(only_one_chunk)))


@pytest.mark.anyio
async def test_anthropic_pieces_join_and_counts_come_from_start_and_delta():
    events = await collect_chat(anthropic_with(lambda request: sse_response(ANTHROPIC_SSE)))

    assert pieces_of(events) == ["The", " penalty", " is 10%."]
    assert sum(isinstance(e, StreamEnd) for e in events) == 1
    end = events[-1]
    assert isinstance(end, StreamEnd)
    # input tokens come from message_start, output tokens from message_delta (not the 1 in start)
    assert (end.model, end.input_tokens, end.output_tokens) == ("claude-test", 812, 9)
    assert 0 <= end.first_token_ms <= end.total_ms


@pytest.mark.anyio
async def test_anthropic_sends_the_exact_url_headers_and_body_with_system_moved_out():
    handler, requests = recording(ANTHROPIC_SSE)

    await collect_chat(anthropic_with(handler), temperature=0.2, max_tokens=256)

    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in request.headers
    # the system prompt is a top-level field, and is gone from messages
    assert json.loads(request.content) == {
        "model": "claude-test",
        "max_tokens": 256,
        "temperature": 0.2,
        "stream": True,
        "system": "Answer only from the sources.",
        "messages": [{"role": "user", "content": "What is the late penalty?"}],
    }


@pytest.mark.anyio
async def test_anthropic_joins_several_system_messages_and_omits_system_when_there_is_none():
    handler, requests = recording(ANTHROPIC_SSE)
    streamer = anthropic_with(handler)
    two_systems = [
        {"role": "system", "content": "Rule one."},
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "Rule two."},
    ]

    _ = [event async for event in streamer.stream(two_systems)]
    _ = [event async for event in streamer.stream(MESSAGES)]

    first, second = (json.loads(r.content) for r in requests)
    assert first["system"] == "Rule one.\n\nRule two."
    assert first["messages"] == [{"role": "user", "content": "Hi"}]
    assert "system" not in second
    assert first["max_tokens"] == 512  # required by Anthropic, so always sent


@pytest.mark.anyio
async def test_anthropic_401_becomes_llm_error_with_the_status():
    body = {
        "type": "error",
        "error": {"type": "authentication_error", "message": "invalid x-api-key"},
    }
    streamer = anthropic_with(lambda request: httpx.Response(401, json=body))

    with pytest.raises(LLMError, match="401"):
        await collect_chat(streamer)


@pytest.mark.anyio
async def test_anthropic_error_event_becomes_llm_error():
    overloaded = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    body = sse_body(ANTHROPIC_START, ("error", overloaded))

    with pytest.raises(LLMError, match="Overloaded"):
        await collect_chat(anthropic_with(lambda request: sse_response(body)))


@pytest.mark.anyio
async def test_anthropic_stream_without_message_stop_is_an_error():
    body = sse_body(ANTHROPIC_START, ("content_block_delta", text_delta("The")))

    with pytest.raises(LLMError, match="message_stop"):
        await collect_chat(anthropic_with(lambda request: sse_response(body)))


def test_hosted_openai_url_without_a_key_is_refused(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LLMError, match="costs money") as info:
        OpenAIStreamer(model="gpt-test", base_url=OPENAI_BASE)

    assert "OPENAI_API_KEY" in str(info.value)
    assert "OPENAI_BASE_URL=http://localhost:11434/v1" in str(info.value)  # how to go free


def test_hosted_anthropic_url_without_a_key_is_refused(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LLMError, match="costs money") as info:
        AnthropicStreamer(model="claude-test", base_url=ANTHROPIC_BASE)

    assert "ANTHROPIC_API_KEY" in str(info.value)
    assert "ANTHROPIC_BASE_URL=http://localhost:11434" in str(info.value)


def test_the_default_url_is_hosted_so_no_key_is_refused_too(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(LLMError, match="costs money"):
        OpenAIStreamer(model="gpt-test")
    with pytest.raises(LLMError, match="costs money"):
        AnthropicStreamer(model="claude-test")


@pytest.mark.anyio
async def test_a_key_from_the_environment_is_sent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    handler, requests = recording(OPENAI_SSE)
    http = httpx.AsyncClient(base_url=OPENAI_BASE, transport=httpx.MockTransport(handler))

    await collect_chat(OpenAIStreamer(model="gpt-test", base_url=OPENAI_BASE, http=http))

    assert requests[0].headers["authorization"] == "Bearer env-key"


@pytest.mark.anyio
async def test_openai_on_localhost_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    handler, requests = recording(OPENAI_SSE)
    streamer = openai_with(handler, api_key=None, base_url="http://localhost:11434/v1")

    await collect_chat(streamer)

    assert streamer.local is True
    assert str(requests[0].url) == "http://localhost:11434/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer ollama"


@pytest.mark.anyio
async def test_anthropic_on_localhost_needs_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    handler, requests = recording(ANTHROPIC_SSE)
    streamer = anthropic_with(handler, api_key=None, base_url="http://localhost:11434")

    await collect_chat(streamer)

    assert streamer.local is True
    assert str(requests[0].url) == "http://localhost:11434/v1/messages"
    assert requests[0].headers["x-api-key"] == "ollama"


def test_hosted_streamers_are_not_local():
    assert openai_with(lambda request: sse_response("")).local is False
    assert anthropic_with(lambda request: sse_response("")).local is False
