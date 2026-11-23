import json
from pathlib import Path

import pytest

from coursemate.agent import AGENT_SYSTEM, deny_all, run_agent, run_tool, show_message
from coursemate.agent_tools import TOOL_SCHEMAS, CourseTools, Outbox
from coursemate.embeddings import FakeEmbedder
from coursemate.llm import FakeClient, ToolCall
from coursemate.retrieval_eval import build_index

SAMPLE = Path(__file__).parents[1] / "notes" / "sample"

TASK = "Write an announcement reminding students when Lab 4 is due, and send it."
FINAL = (
    "I drafted the Lab 4 reminder (draft 1) but it was not sent. Source: lecture-02-hash-tables.md."
)

SEARCH = ToolCall("search_notes", {"query": "Lab 4 due date"})
DRAFT = ToolCall("draft_announcement", {"title": "Lab 4", "body": "Lab 4 is due Friday."})
SEND = ToolCall("send_announcement", {"draft_id": 1})


def paper_trace():
    """The guide's paper trace: search, draft, send, then an answer."""
    return FakeClient([[SEARCH], [DRAFT], [SEND], FINAL])


def approver(answer):
    """An approver that gives a fixed answer and remembers every call it was asked about."""
    seen = []

    def approve(call):
        seen.append(call)
        return answer

    return approve, seen


@pytest.fixture(scope="module")
def index():
    embedder = FakeEmbedder()
    return embedder, build_index(SAMPLE, embedder)


@pytest.fixture
def course(index, tmp_path):
    embedder, store = index
    return CourseTools(embedder, store, FakeClient([]), Outbox(tmp_path / "outbox"), min_score=0.1)


# ---- run_tool ----


def test_a_search_runs_without_asking_the_approver(course):
    approve, seen = approver(True)

    result = run_tool(SEARCH, course.registry(), approve)

    assert "Friday of week 5" in result
    assert seen == []


def test_send_declined_is_an_error_and_nothing_is_sent(course):
    course.draft_announcement("Lab 4", "Lab 4 is due Friday.")
    approve, seen = approver(False)

    result = run_tool(SEND, course.registry(), approve)

    assert result.startswith("error: not approved")
    assert "nothing was sent" in result
    assert seen == [SEND]
    assert course.outbox.draft_ids() == [1]
    assert course.outbox.sent_ids() == []


def test_send_approved_moves_the_draft(course):
    course.draft_announcement("Lab 4", "Lab 4 is due Friday.")
    approve, seen = approver(True)

    result = run_tool(SEND, course.registry(), approve)

    assert json.loads(result) == {"draft_id": 1, "status": "sent"}
    assert seen == [SEND]
    assert course.outbox.sent_ids() == [1]


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (ToolCall("delete_everything", {}), "unknown tool"),
        (ToolCall("send_announcement", {"draft_id": 9}), "LookupError"),
        (ToolCall("search_notes", {"q": "Lab 4"}), "TypeError"),
    ],
)
def test_errors_come_back_as_text(course, call, expected):
    approve, _ = approver(True)

    result = run_tool(call, course.registry(), approve)

    assert result.startswith("error:")
    assert expected in result


def test_deny_all_says_no():
    assert deny_all(SEND) is False


# ---- run_agent ----


def test_paper_trace_with_the_default_approver(course):
    fake = paper_trace()

    result = run_agent(fake, TASK, course.registry())

    assert result.stopped == "answer"
    assert result.answer == FINAL
    assert result.steps == 4
    assert result.input_tokens == 400
    assert result.tool_calls == ["search_notes", "draft_announcement", "send_announcement"]
    assert [m["role"] for m in result.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.messages[7]["content"].startswith("error: not approved")
    assert [len(messages) for messages in fake.calls] == [2, 4, 6, 8]
    assert course.outbox.draft_ids() == [1]
    assert course.outbox.sent_ids() == []


def test_paper_trace_with_a_yes_sends_the_draft(course):
    approve, seen = approver(True)

    result = run_agent(paper_trace(), TASK, course.registry(), approve=approve)

    assert json.loads(result.messages[7]["content"]) == {"draft_id": 1, "status": "sent"}
    assert seen == [SEND]
    assert course.outbox.sent_ids() == [1]


def test_every_call_gets_the_system_prompt_first_and_the_tool_schemas(course):
    fake = paper_trace()

    run_agent(fake, TASK, course.registry())

    assert all(
        messages[0] == {"role": "system", "content": AGENT_SYSTEM} for messages in fake.calls
    )
    assert fake.tools_seen == [TOOL_SCHEMAS] * 4


def test_an_empty_reply_stops_with_empty(course):
    result = run_agent(FakeClient(["   "]), TASK, course.registry())

    assert result.stopped == "empty"
    assert result.steps == 1
    assert result.tool_calls == []


def test_max_steps_stops_and_says_so(course):
    result = run_agent(paper_trace(), TASK, course.registry(), max_steps=2)

    assert result.stopped == "max_steps"
    assert result.steps == 2
    assert "(2)" in result.answer


def test_show_message_cuts_tool_content_to_200_characters(capsys):
    show_message({"role": "tool", "tool_name": "search_notes", "content": "x" * 300})

    out = capsys.readouterr().out
    assert "x" * 200 in out
    assert "x" * 201 not in out
