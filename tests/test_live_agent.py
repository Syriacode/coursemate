from pathlib import Path

import httpx
import pytest

from coursemate.agent import run_agent
from coursemate.agent_tools import CourseTools, Outbox
from coursemate.embeddings import OllamaEmbedder
from coursemate.llm import OllamaClient
from coursemate.retrieval_eval import build_index

pytestmark = pytest.mark.live

SAMPLE = Path(__file__).parents[1] / "notes" / "sample"

TASK = "Write an announcement reminding students when Lab 4 is due, and send it."


@pytest.fixture(scope="module")
def index():
    try:
        httpx.get("http://localhost:11434/api/version", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("Ollama is not running (start it, then: ollama pull qwen2.5:3b)")

    embedder = OllamaEmbedder()
    return embedder, build_index(SAMPLE, embedder), OllamaClient(model="qwen2.5:3b")


@pytest.fixture
def course(index, tmp_path):
    embedder, store, client = index
    return CourseTools(embedder, store, client, Outbox(tmp_path / "outbox"))


def test_office_hours_answer_comes_from_a_search(index, course):
    _, _, client = index

    result = run_agent(client, "When are office hours?", course.registry())

    assert result.stopped == "answer"
    assert "search_notes" in result.tool_calls
    assert "tuesday" in result.answer.lower()


def test_a_no_means_nothing_is_sent_by_the_real_model(index, course):
    _, _, client = index

    result = run_agent(client, TASK, course.registry(), approve=lambda call: False, max_steps=8)

    assert course.outbox.sent_ids() == []
    assert result.steps <= 8
