import json
from pathlib import Path

import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from coursemate.agent_tools import CourseTools, Outbox
from coursemate.embeddings import FakeEmbedder
from coursemate.llm import FakeClient
from coursemate.mcp_server import build_server, send_if_approved
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
async def test_four_tools_with_their_hints(server):
    async with Client(server) as client:
        listed = await client.list_tools()

    tools = {tool.name: tool for tool in listed.tools}
    assert set(tools) == {"search_notes", "make_quiz", "draft_announcement", "send_announcement"}
    for name in ("search_notes", "make_quiz"):
        assert tools[name].annotations.read_only_hint is True
        assert tools[name].annotations.open_world_hint is False
    assert tools["draft_announcement"].annotations.read_only_hint is False
    assert tools["draft_announcement"].annotations.destructive_hint is False
    assert tools["send_announcement"].annotations.destructive_hint is True
    assert tools["send_announcement"].annotations.open_world_hint is True
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


LAB_4 = ("Lab 4", "Lab 4 is due Friday of week 5.")
PARTY = ("Party", "Class is cancelled tomorrow!")


def test_send_if_approved_follows_the_paper_trace(tmp_path):
    outbox = Outbox(tmp_path / "outbox")
    outbox.save_draft(*LAB_4)
    outbox.save_draft(*PARTY)
    outbox.approve(1)

    assert send_if_approved(outbox, 1) == {"draft_id": 1, "status": "sent"}
    with pytest.raises(ToolError, match="not approved"):
        send_if_approved(outbox, 2)

    assert outbox.draft_ids() == [2]
    assert outbox.sent_ids() == [1]


def test_a_draft_edited_after_approval_is_refused(tmp_path):
    outbox = Outbox(tmp_path / "outbox")
    outbox.save_draft(*LAB_4)
    outbox.approve(1)
    (outbox.drafts / "001.md").write_text("# Lab 4\n\nLab 4 is cancelled!\n", encoding="utf-8")

    with pytest.raises(ToolError, match="not approved"):
        send_if_approved(outbox, 1)

    assert outbox.sent_ids() == []


@pytest.mark.anyio
async def test_send_without_approval_is_refused_through_the_client(server, course):
    async with Client(server) as client:
        await client.call_tool("draft_announcement", {"title": LAB_4[0], "body": LAB_4[1]})
        result = await client.call_tool("send_announcement", {"draft_id": 1})

    assert result.is_error
    assert "py -m coursemate approve 1" in result.content[0].text
    assert course.outbox.sent_ids() == []


@pytest.mark.anyio
async def test_send_after_approval_works_through_the_client(server, course):
    async with Client(server) as client:
        await client.call_tool("draft_announcement", {"title": LAB_4[0], "body": LAB_4[1]})
        course.outbox.approve(1)
        result = await client.call_tool("send_announcement", {"draft_id": 1})

    assert not result.is_error
    assert json.loads(result.content[0].text) == {"draft_id": 1, "status": "sent"}
    assert course.outbox.sent_ids() == [1]
