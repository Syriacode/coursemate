import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from coursemate.llm import LLMError, Message

DEFAULT_STREAM_REPLY = "Fake provider: no model was called [1]."


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


class FakeStreamer:
    provider = "fake"
    local = True

    def __init__(self, replies: Sequence[str | Exception] | None = None) -> None:
        self.model = "fake"
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
    if provider == "fake":
        return FakeStreamer()
    raise ValueError(f"unknown provider {provider!r} (known: ollama, fake)")
