import httpx
import pytest

from coursemate.streaming import (
    AnthropicStreamer,
    OllamaStreamer,
    OpenAIStreamer,
    StreamEnd,
    TextDelta,
)

pytestmark = pytest.mark.live

OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5:3b"

# the same local model, reached through three different packagings; none of them costs money
STREAMERS = {
    "ollama": lambda: OllamaStreamer(model=MODEL, base_url=OLLAMA),
    "openai": lambda: OpenAIStreamer(model=MODEL, base_url=OLLAMA + "/v1"),
    "anthropic": lambda: AnthropicStreamer(model=MODEL, base_url=OLLAMA),
}


@pytest.fixture(scope="module", autouse=True)
def ollama_running():
    try:
        httpx.get(f"{OLLAMA}/api/version", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("Ollama is not running (start it, then: ollama pull qwen2.5:3b)")


@pytest.mark.anyio
@pytest.mark.parametrize("name", list(STREAMERS))
async def test_the_same_question_streams_through_each_format(name):
    streamer = STREAMERS[name]()
    messages = [{"role": "user", "content": "What color is a clear daytime sky?"}]

    events = [event async for event in streamer.stream(messages)]

    pieces = [e.text for e in events if isinstance(e, TextDelta)]
    assert len(pieces) > 1  # it streamed: more than one piece
    assert "blue" in "".join(pieces).lower()
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.input_tokens > 0
    # output tokens are not asserted: some Ollama versions omit them from Anthropic's message_delta
