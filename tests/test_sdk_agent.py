from pathlib import Path

import pytest

pytest.importorskip("claude_agent_sdk")  # an optional extra: CI does not install it

from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny  # noqa: E402

from coursemate.agent_tools import CourseTools, Outbox  # noqa: E402
from coursemate.embeddings import FakeEmbedder  # noqa: E402
from coursemate.llm import FakeClient  # noqa: E402
from coursemate.retrieval_eval import build_index  # noqa: E402
from coursemate.sdk_agent import (  # noqa: E402
    build_server,
    make_options,
    make_permission_check,
    wrap,
)

SAMPLE = Path(__file__).parents[1] / "notes" / "sample"


@pytest.fixture(scope="module")
def index():
    embedder = FakeEmbedder()
    return embedder, build_index(SAMPLE, embedder)


@pytest.fixture
def course(index, tmp_path):
    embedder, store = index
    return CourseTools(embedder, store, FakeClient([]), Outbox(tmp_path / "outbox"), min_score=0.1)


def fake_ask(answer):
    """An `ask` that gives a fixed answer and remembers the prompts it was shown."""
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return answer

    return ask, prompts


@pytest.mark.anyio
async def test_wrap_returns_the_result_as_json_text(course):
    result = await wrap(course.search_notes)({"query": "Lab 4 due date"})

    assert "is_error" not in result
    assert result["content"][0]["type"] == "text"
    assert "Friday of week 5" in result["content"][0]["text"]


@pytest.mark.anyio
async def test_wrap_turns_an_exception_into_an_error_result(course):
    result = await wrap(course.send_announcement)({"draft_id": 9})

    assert result["is_error"] is True
    assert result["content"][0]["text"] == "error: LookupError: no draft 9; drafts: none"


def test_build_server_is_an_sdk_server_named_coursemate(course):
    server = build_server(course)

    assert server["type"] == "sdk"
    assert server["name"] == "coursemate"


@pytest.mark.anyio
async def test_a_no_denies_the_send_and_says_nothing_was_sent(course):
    course.draft_announcement("Lab 4", "Lab 4 is due Friday.")
    ask, prompts = fake_ask("n")
    can_use = make_permission_check(course.outbox, ask)

    decision = await can_use("mcp__coursemate__send_announcement", {"draft_id": 1}, None)

    assert isinstance(decision, PermissionResultDeny)
    assert "nothing was sent" in decision.message
    assert prompts == ["Send it? [y/N] "]


@pytest.mark.anyio
async def test_a_yes_allows_the_send(course):
    course.draft_announcement("Lab 4", "Lab 4 is due Friday.")
    ask, _ = fake_ask("y")
    can_use = make_permission_check(course.outbox, ask)

    decision = await can_use("mcp__coursemate__send_announcement", {"draft_id": 1}, None)

    assert isinstance(decision, PermissionResultAllow)


@pytest.mark.anyio
async def test_any_other_tool_is_denied_without_asking(course):
    ask, prompts = fake_ask("y")
    can_use = make_permission_check(course.outbox, ask)

    decision = await can_use("Bash", {"command": "echo hi"}, None)

    assert isinstance(decision, PermissionResultDeny)
    assert prompts == []


def test_the_options_are_locked_down_and_allow_only_the_read_tools(course):
    options = make_options(course, "haiku", 8)

    assert options.tools == []
    assert options.setting_sources == []
    assert sorted(options.allowed_tools) == [
        "mcp__coursemate__draft_announcement",
        "mcp__coursemate__make_quiz",
        "mcp__coursemate__search_notes",
    ]
