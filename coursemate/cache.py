"""A response cache in front of the model: the same request gets the stored reply."""

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass

from coursemate.llm import Message
from coursemate.streaming import StreamEnd, StreamingLLM, TextDelta


def cache_key(
    provider: str, model: str, messages: Sequence[Message], temperature: float, max_tokens: int
) -> str:
    """Everything that can change the reply goes into the key, and nothing else."""
    payload = {
        "provider": provider,
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # sort_keys: the same dict written in a different order must give the same key
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedReply:
    """A finished reply, with what it cost the first time."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int


class ResponseCache:
    """Replies by key, in memory. Entries expire (TTL), and when full the one used
    longest ago is dropped (LRU).

    No lock: get() and put() never await, so on one event loop they can't interleave.
    """

    def __init__(
        self,
        ttl_s: float = 3600,
        max_entries: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self._clock = clock
        # key -> (time stored, reply). Front: least recently used. End: most recently used.
        self._entries: OrderedDict[str, tuple[float, CachedReply]] = OrderedDict()

    def get(self, key: str) -> CachedReply | None:
        if key not in self:
            self._entries.pop(key, None)  # remove it if it was there but expired
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return self._entries[key][1]

    def put(self, key: str, reply: CachedReply) -> None:
        self._entries[key] = (self._clock(), reply)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        """True for an entry that hasn't expired. Doesn't count as a use."""
        entry = self._entries.get(key)
        return entry is not None and self._clock() - entry[0] <= self.ttl_s


class CachedStreamer:
    """A StreamingLLM that answers repeated requests from the cache."""

    def __init__(self, inner: StreamingLLM, cache: ResponseCache) -> None:
        self.inner = inner
        self.cache = cache
        self.provider = inner.provider
        self.model = inner.model
        self.local = inner.local
        self.base_url = inner.base_url

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[TextDelta | StreamEnd]:
        # above temperature 0 the reply is supposed to differ each time, so never cache it
        key = None
        if temperature == 0:
            start = time.perf_counter()
            key = cache_key(self.provider, self.model, messages, temperature, max_tokens)
            saved = self.cache.get(key)
            if saved is not None:
                if saved.text:
                    yield TextDelta(saved.text)
                elapsed_ms = (time.perf_counter() - start) * 1000
                # 0 tokens: no model was called
                yield StreamEnd(saved.model, 0, 0, elapsed_ms, elapsed_ms, cached=True)
                return

        parts: list[str] = []
        request = self.inner.stream(messages, temperature=temperature, max_tokens=max_tokens)
        async with aclosing(request) as events:
            async for event in events:
                if isinstance(event, TextDelta):
                    parts.append(event.text)
                elif key is not None:
                    # only a stream that reached its end is stored; a failed one raised earlier
                    reply = CachedReply(
                        "".join(parts), event.model, event.input_tokens, event.output_tokens
                    )
                    self.cache.put(key, reply)
                yield event
