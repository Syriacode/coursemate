import re

import pytest

from coursemate.cache import CachedReply, CachedStreamer, ResponseCache, cache_key
from coursemate.llm import LLMError
from coursemate.streaming import FakeStreamer, TextDelta

SYSTEM = {"role": "system", "content": "You are CourseMate."}


def asking(question):
    return [SYSTEM, {"role": "user", "content": question}]


# the request in the first line of the guide's table (section 2d)
REQUEST = {
    "provider": "ollama",
    "model": "qwen2.5:3b",
    "messages": asking("When is Lab 4 due?"),
    "temperature": 0.0,
    "max_tokens": 512,
}


def key(**changes):
    return cache_key(**{**REQUEST, **changes})


def test_the_same_request_gives_the_same_64_hex_key():
    first = key()
    again = key(messages=asking("When is Lab 4 due?"))  # built afresh, an hour later

    assert first == again
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_key_order_inside_a_message_does_not_matter():
    reordered = [{"content": m["content"], "role": m["role"]} for m in REQUEST["messages"]]

    assert key(messages=reordered) == key()


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"model": "llama3.2:3b"}, id="model"),
        pytest.param({"provider": "openai"}, id="provider"),
        pytest.param({"max_tokens": 50}, id="max_tokens"),
        pytest.param({"temperature": 0.7}, id="temperature"),
        pytest.param({"messages": asking("When is Lab 5 due?")}, id="question"),
        pytest.param({"messages": asking("When is Lab 4 due? ")}, id="extra-space"),
    ],
)
def test_anything_that_can_change_the_reply_changes_the_key(change):
    assert key(**change) != key()


def reply(text):
    return CachedReply(text, "fake", 100, 20)


def test_the_paper_timeline():
    now = 0.0
    cache = ResponseCache(ttl_s=60, max_entries=2, clock=lambda: now)

    cache.put("A", reply("a"))
    now = 10
    cache.put("B", reply("b"))
    now = 20
    assert cache.get("A") == reply("a")  # hit: A is now the most recently used
    now = 30
    cache.put("C", reply("c"))  # full: B was used longest ago
    assert len(cache) == 2
    assert "A" in cache and "B" not in cache and "C" in cache
    assert cache.get("B") is None
    now = 61
    assert cache.get("A") is None  # 61 - 0 > 60: expired, even though it was used at t=20
    assert len(cache) == 1  # and removed
    assert cache.get("C") == reply("c")  # 61 - 30 <= 60
    assert (cache.hits, cache.misses) == (2, 2)


async def events_of(llm, **kwargs):
    return [event async for event in llm.stream(REQUEST["messages"], **kwargs)]


def text_of(events):
    return "".join(e.text for e in events if isinstance(e, TextDelta))


@pytest.mark.anyio
async def test_a_miss_calls_the_model_and_a_hit_does_not():
    fake = FakeStreamer(["Friday [1]."])  # one reply: a second model call would fail
    cached = CachedStreamer(fake, ResponseCache())

    miss = await events_of(cached)
    hit = await events_of(cached)

    assert len(fake.calls) == 1
    assert text_of(miss) == text_of(hit) == "Friday [1]."
    assert (miss[-1].cached, miss[-1].input_tokens, miss[-1].output_tokens) == (False, 100, 20)
    assert (hit[-1].cached, hit[-1].input_tokens, hit[-1].output_tokens) == (True, 0, 0)
    assert hit[-1].model == "fake"
    assert len(hit) == 2  # the whole text at once, then the end


@pytest.mark.anyio
async def test_temperature_above_zero_is_never_cached():
    fake = FakeStreamer(["Friday [1].", "On Friday [1]."])
    cache = ResponseCache()
    cached = CachedStreamer(fake, cache)

    first = await events_of(cached, temperature=0.7)
    second = await events_of(cached, temperature=0.7)

    assert len(fake.calls) == 2
    assert text_of(first) != text_of(second)
    assert not second[-1].cached
    assert (len(cache), cache.hits, cache.misses) == (0, 0, 0)


class BreaksMidway(FakeStreamer):
    """Sends one word, then the connection drops."""

    async def stream(self, messages, *, temperature=0.0, max_tokens=512):
        self.calls.append(list(messages))
        yield TextDelta("Friday")
        raise LLMError("connection lost")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failing",
    [FakeStreamer([LLMError("model is down")]), BreaksMidway()],
    ids=["before-any-word", "after-a-word"],
)
async def test_a_failed_stream_is_not_cached(failing):
    cache = ResponseCache()

    with pytest.raises(LLMError):
        await events_of(CachedStreamer(failing, cache))

    assert len(cache) == 0


def test_the_wrapper_looks_like_the_model_inside_it():
    inner = FakeStreamer()
    inner.provider, inner.model = "openai", "some-model"
    inner.local, inner.base_url = False, "http://example.invalid/v1"

    wrapper = CachedStreamer(inner, ResponseCache())

    for name in ("provider", "model", "local", "base_url"):
        assert getattr(wrapper, name) == getattr(inner, name)
