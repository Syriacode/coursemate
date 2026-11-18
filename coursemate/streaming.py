import json
import os
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from coursemate.llm import LLMError, Message
from coursemate.sse import sse_data

DEFAULT_STREAM_REPLY = "Fake provider: no model was called [1]."
OPENAI_URL = "https://api.openai.com/v1"
ANTHROPIC_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class TextDelta:
    """One piece of the answer text."""

    text: str


@dataclass(frozen=True)
class StreamEnd:
    """The last event of a stream: which model answered, and what it cost in tokens and time."""

    model: str
    input_tokens: int
    output_tokens: int
    first_token_ms: float
    total_ms: float


class StreamingLLM(Protocol):
    provider: str
    model: str
    local: bool  # True when nothing leaves this computer
    base_url: str  # where the requests go ("" when nothing is sent anywhere)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[TextDelta | StreamEnd]:
        """Yield TextDelta pieces, then exactly one StreamEnd."""
        ...


class OllamaStreamer:
    provider = "ollama"
    local = True

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        num_ctx: int = 8192,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.num_ctx = num_ctx
        self._http = http or httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[TextDelta | StreamEnd]:
        body = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }

        start = time.perf_counter()
        first_token_ms: float | None = None
        last: dict | None = None
        try:
            # the request stays open while we read the reply line by line
            async with self._http.stream("POST", "/api/chat", json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise LLMError(f"ollama stream failed: {data['error']}")
                    text = data.get("message", {}).get("content", "")
                    if text:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - start) * 1000
                        yield TextDelta(text)
                    if data.get("done"):
                        last = data
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc

        if last is None:
            raise LLMError("ollama stream ended before the final line")
        total_ms = (time.perf_counter() - start) * 1000
        yield StreamEnd(
            model=last.get("model", self.model),
            input_tokens=last.get("prompt_eval_count", 0),
            output_tokens=last.get("eval_count", 0),
            first_token_ms=total_ms if first_token_ms is None else first_token_ms,
            total_ms=total_ms,
        )


def _is_local(base_url: str) -> bool:
    """True when the URL points at this computer (for example Ollama), so nothing is billed."""
    return httpx.URL(base_url).host in LOCAL_HOSTS


def _resolve_key(
    api_key: str | None, key_env: str, base_url: str, base_url_env: str, local_example: str
) -> str:
    """Pick the API key. A hosted URL without one is refused, so a typo can't cost money."""
    key = api_key or os.environ.get(key_env)
    if key:
        return key
    if _is_local(base_url):
        return "ollama"  # Ollama ignores the key, but the header must not be empty
    raise LLMError(
        f"{key_env} is not set, and {base_url} is a hosted API that costs money for every token. "
        f"To use your local Ollama model for free instead, set {base_url_env}={local_example}."
    )


async def _raise_for_status(response: httpx.Response, provider: str) -> None:
    """Turn an HTTP error status into an LLMError with the status code and the start of the body."""
    if response.status_code >= 400:
        await response.aread()  # a streamed response has no body until we read it
        raise LLMError(
            f"{provider} request failed: HTTP {response.status_code}: {response.text[:200]}"
        )


def _parse_event(data: str, provider: str) -> dict[str, Any]:
    try:
        event = json.loads(data)
    except json.JSONDecodeError as exc:
        raise LLMError(f"{provider} sent something that is not JSON: {data[:80]!r}") from exc
    if not isinstance(event, dict):
        raise LLMError(f"{provider} sent an unexpected event: {data[:80]!r}")
    return event


def _error_text(error: object) -> str:
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(error)


def _stream_end(
    model: str, input_tokens: int, output_tokens: int, first_token_ms: float | None, start: float
) -> StreamEnd:
    total_ms = (time.perf_counter() - start) * 1000
    return StreamEnd(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        first_token_ms=total_ms if first_token_ms is None else first_token_ms,
        total_ms=total_ms,
    )


class OpenAIStreamer:
    """OpenAI Chat Completions format: SSE `data:` lines that end with `data: [DONE]`."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_URL
        self.base_url = base_url
        self.local = _is_local(base_url)
        # resolved here, not on the first request, so a missing key fails before anything is sent
        self._api_key = _resolve_key(
            api_key, "OPENAI_API_KEY", base_url, "OPENAI_BASE_URL", "http://localhost:11434/v1"
        )
        self._http = http or httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[TextDelta | StreamEnd]:
        body = {
            "model": self.model,
            "messages": list(messages),  # the system prompt stays in the list, as a system message
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},  # without this there are no token counts
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        start = time.perf_counter()
        first_token_ms: float | None = None
        model = self.model
        input_tokens = output_tokens = 0
        done = False
        try:
            async with self._http.stream(
                "POST", "/chat/completions", json=body, headers=headers
            ) as response:
                await _raise_for_status(response, self.provider)
                async for data in sse_data(response.aiter_lines()):
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = _parse_event(data, self.provider)
                    if "error" in chunk:
                        raise LLMError(f"openai stream failed: {_error_text(chunk['error'])}")
                    model = chunk.get("model") or model
                    for choice in chunk.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            if first_token_ms is None:
                                first_token_ms = (time.perf_counter() - start) * 1000
                            yield TextDelta(text)
                    usage = chunk.get("usage")  # null on every chunk except the last one
                    if usage:
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)
        except httpx.HTTPError as exc:
            raise LLMError(f"openai request failed: {exc}") from exc

        if not done:
            raise LLMError("openai stream ended before [DONE]")
        yield _stream_end(model, input_tokens, output_tokens, first_token_ms, start)


class AnthropicStreamer:
    """Anthropic Messages format: SSE events with a type, ending with `message_stop`."""

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL") or ANTHROPIC_URL
        self.base_url = base_url
        self.local = _is_local(base_url)
        self._api_key = _resolve_key(
            api_key, "ANTHROPIC_API_KEY", base_url, "ANTHROPIC_BASE_URL", "http://localhost:11434"
        )
        self._http = http or httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[TextDelta | StreamEnd]:
        # Anthropic has no system *message*: the system prompt is a separate top-level field
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,  # required by this API
            "temperature": temperature,
            "stream": True,
            "messages": [m for m in messages if m["role"] != "system"],
        }
        if system:
            body["system"] = system
        headers = {"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION}

        start = time.perf_counter()
        first_token_ms: float | None = None
        model = self.model
        input_tokens = output_tokens = 0
        done = False
        try:
            async with self._http.stream(
                "POST", "/v1/messages", json=body, headers=headers
            ) as response:
                await _raise_for_status(response, self.provider)
                async for data in sse_data(response.aiter_lines()):
                    event = _parse_event(data, self.provider)
                    kind = event.get("type")
                    if kind == "message_start":
                        message = event.get("message") or {}
                        model = message.get("model") or model
                        input_tokens = (message.get("usage") or {}).get("input_tokens", 0)
                    elif kind == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            if first_token_ms is None:
                                first_token_ms = (time.perf_counter() - start) * 1000
                            yield TextDelta(delta["text"])
                    elif kind == "message_delta":
                        usage = event.get("usage") or {}
                        output_tokens = usage.get("output_tokens", output_tokens)
                    elif kind == "error":
                        raise LLMError(
                            f"anthropic stream failed: {_error_text(event.get('error'))}"
                        )
                    elif kind == "message_stop":
                        done = True
                        break
                    # ping, content_block_start, content_block_stop: nothing to do
        except httpx.HTTPError as exc:
            raise LLMError(f"anthropic request failed: {exc}") from exc

        if not done:
            raise LLMError("anthropic stream ended before message_stop")
        yield _stream_end(model, input_tokens, output_tokens, first_token_ms, start)


class FakeStreamer:
    provider = "fake"
    local = True

    def __init__(self, replies: Sequence[str | Exception] | None = None) -> None:
        self.model = "fake"
        self.base_url = ""  # nothing is sent anywhere
        self.calls: list[list[Message]] = []
        self._replies = None if replies is None else list(replies)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[TextDelta | StreamEnd]:
        self.calls.append(list(messages))
        if self._replies is None:
            reply: str | Exception = DEFAULT_STREAM_REPLY
        elif not self._replies:
            raise AssertionError("FakeStreamer ran out of replies")
        else:
            reply = self._replies.pop(0)

        if isinstance(reply, Exception):
            raise reply
        for i, word in enumerate(reply.split()):
            yield TextDelta(word if i == 0 else " " + word)
        yield StreamEnd("fake", 100, 20, 1.0, 2.0)


def make_streamer(provider: str, model: str, ollama_url: str) -> StreamingLLM:
    if provider == "ollama":
        return OllamaStreamer(model=model, base_url=ollama_url)
    if provider == "openai":
        return OpenAIStreamer(model=model)  # URL and key come from the environment
    if provider == "anthropic":
        return AnthropicStreamer(model=model)
    if provider == "fake":
        return FakeStreamer()
    raise ValueError(f"unknown provider {provider!r} (known: ollama, openai, anthropic, fake)")
