import json
from pathlib import Path

import pytest
from mcp import Client

from coursemate.agent_tools import CourseTools, Outbox
from coursemate.embeddings import FakeEmbedder
from coursemate.llm import FakeClient
from coursemate.mcp_server import build_server
from coursemate.retrieval_eval import build_index

SAMPLE = Path(__file__).parents[1] / "notes" / "sample"


@pytest.fixture(scope="module")
def index():
    embedder = FakeEmbedder()
    return embedder, build_index(SAMPLE, embedder)


@pytest.fixture
def course(index, tmp_path):
    embedder, store = index
    return CourseTools(embedder, store, FakeClient([]), Outbox(tmp_path / "outbox"), min_score=0.1)


@pytest.fixture
def server(course):
    return build_server(course, SAMPLE)


@pytest.mark.anyio
async def test_three_tools_with_their_hints(server):
    async with Client(server) as client:
        listed = await client.list_tools()

    tools = {tool.name: tool for tool in listed.tools}
    assert set(tools) == {"search_notes", "make_quiz", "draft_announcement"}
    for name in ("search_notes", "make_quiz"):
        assert tools[name].annotations.read_only_hint is True
        assert tools[name].annotations.open_world_hint is False
    assert tools["draft_announcement"].annotations.read_only_hint is False
    assert tools["draft_announcement"].annotations.destructive_hint is False
    assert tools["search_notes"].input_schema["required"] == ["query"]


@pytest.mark.anyio
async def test_search_finds_lab_4_first(server):
    async with Client(server) as client:
        result = await client.call_tool("search_notes", {"query": "When is Lab 4 due?"})

    assert not result.is_error
    found = json.loads(result.content[0].text)
    assert found["results"][0]["source"] == "lecture-02-hash-tables.md"


@pytest.mark.anyio
async def test_bad_arguments_come_back_as_errors(server, course):
    async with Client(server) as client:
        blank = await client.call_tool("draft_announcement", {"title": " ", "body": "Due Friday."})
        quiz = await client.call_tool("make_quiz", {"topic": "hash tables", "questions": 9})

    assert blank.is_error
    assert "must not be empty" in blank.content[0].text
    assert quiz.is_error
    assert "between 1 and 5" in quiz.content[0].text
    assert course.outbox.draft_ids() == []
