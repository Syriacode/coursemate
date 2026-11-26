import httpx
import pytest

from coursemate.cache import CachedStreamer, ResponseCache
from coursemate.streaming import OllamaStreamer, TextDelta

pytestmark = pytest.mark.live

# the address, not "localhost": on Windows a new connection by name can cost about 2 s,
# which would slow the first call and make the speed check pass too easily
OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5:3b"


@pytest.fixture(scope="module", autouse=True)
def ollama_running():
    try:
        httpx.get(f"{OLLAMA}/api/version", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("Ollama is not running (start it, then: ollama pull qwen2.5:3b)")


@pytest.mark.anyio
async def test_the_second_ask_comes_from_the_cache():
    llm = CachedStreamer(OllamaStreamer(model=MODEL, base_url=OLLAMA), ResponseCache())
    messages = [{"role": "user", "content": "What color is a clear daytime sky? One word."}]

    async def ask():
        events = [event async for event in llm.stream(messages, max_tokens=20)]
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        return text, events[-1]

    first_text, first = await ask()
    second_text, second = await ask()

    assert not first.cached
    assert first.input_tokens > 0
    assert second.cached
    assert (second.input_tokens, second.output_tokens) == (0, 0)
    assert first_text and second_text == first_text
    assert second.total_ms < first.total_ms / 10
