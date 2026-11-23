import json
from pathlib import Path

import pytest

from coursemate.agent_tools import NEEDS_APPROVAL, TOOL_SCHEMAS, CourseTools, Outbox
from coursemate.embeddings import FakeEmbedder
from coursemate.llm import FakeClient
from coursemate.retrieval_eval import build_index

SAMPLE = Path(__file__).parents[1] / "notes" / "sample"

NOTHING = {"results": [], "note": "Nothing in the course notes matches. Say so."}

QUIZ_REPLY = json.dumps(
    {
        "questions": [
            {
                "question": "What does a hash table use to find a slot?",
                "choices": ["A hash function", "A sort", "A stack", "A heap"],
                "answer": 0,
                "source": 1,
            }
        ]
    }
)


@pytest.fixture(scope="module")
def index():
    embedder = FakeEmbedder()
    return embedder, build_index(SAMPLE, embedder)


@pytest.fixture
def make_tools(index, tmp_path):
    embedder, store = index

    def make(client=None, min_score=0.1):
        outbox = Outbox(tmp_path / "outbox")
        return CourseTools(embedder, store, client or FakeClient([]), outbox, min_score=min_score)

    return make


def test_search_finds_lab_4_first(make_tools):
    result = make_tools().search_notes("When is Lab 4 due?")

    first = result["results"][0]
    assert first["source"] == "lecture-02-hash-tables.md"
    assert "Friday of week 5" in first["text"]


def test_search_below_min_score_returns_the_note(make_tools):
    assert make_tools(min_score=0.99).search_notes("When is Lab 4 due?") == NOTHING


def test_quiz_sends_numbered_sources_and_returns_the_choices(make_tools):
    client = FakeClient([QUIZ_REPLY])

    result = make_tools(client).make_quiz("hash tables", questions=1)

    assert "[1] lecture-02-hash-tables.md" in client.calls[0][1]["content"]
    assert result["questions"][0]["choices"] == ["A hash function", "A sort", "A stack", "A heap"]
    assert result["sources"]
    assert all(" > " in source or source.endswith(".md") for source in result["sources"])


def test_quiz_with_bad_json_is_a_value_error(make_tools):
    with pytest.raises(ValueError):
        make_tools(FakeClient(["not json"])).make_quiz("hash tables")


def test_quiz_with_three_choices_is_a_value_error(make_tools):
    bad = json.loads(QUIZ_REPLY)
    bad["questions"][0]["choices"] = ["A", "B", "C"]

    with pytest.raises(ValueError):
        make_tools(FakeClient([json.dumps(bad)])).make_quiz("hash tables")


def test_quiz_with_40_questions_is_a_value_error_and_no_model_call(make_tools):
    client = FakeClient([QUIZ_REPLY])

    with pytest.raises(ValueError):
        make_tools(client).make_quiz("hash tables", questions=40)

    assert client.calls == []


def test_quiz_with_no_matching_notes_is_a_lookup_error(make_tools):
    client = FakeClient([QUIZ_REPLY])

    with pytest.raises(LookupError):
        make_tools(client, min_score=0.99).make_quiz("hash tables")

    assert client.calls == []


def test_drafting_writes_a_file_and_sends_nothing(tmp_path, make_tools):
    tools = make_tools()

    result = tools.draft_announcement("Lab 4", "Due Friday.")

    assert result == {"draft_id": 1, "status": "draft (not sent)"}
    assert (tmp_path / "outbox" / "drafts" / "001.md").read_text() == "# Lab 4\n\nDue Friday.\n"
    assert tools.outbox.sent_ids() == []


@pytest.mark.parametrize(
    ("title", "body"), [("", "Due Friday."), ("Lab 4", ""), ("Lab 4", "x" * 2001)]
)
def test_bad_drafts_are_value_errors(make_tools, title, body):
    tools = make_tools()

    with pytest.raises(ValueError):
        tools.draft_announcement(title, body)

    assert tools.outbox.draft_ids() == []


def test_send_moves_the_draft_and_the_next_id_is_2(make_tools):
    tools = make_tools()
    tools.draft_announcement("Lab 4", "Due Friday.")

    result = tools.send_announcement(1)

    assert result == {"draft_id": 1, "status": "sent"}
    assert tools.outbox.draft_ids() == []
    assert tools.outbox.sent_ids() == [1]
    assert tools.draft_announcement("Quiz", "Monday.")["draft_id"] == 2
    assert tools.outbox.texts() == ["# Quiz\n\nMonday.\n", "# Lab 4\n\nDue Friday.\n"]


def test_sending_a_missing_draft_is_a_lookup_error(make_tools):
    tools = make_tools()
    tools.draft_announcement("Lab 4", "Due Friday.")

    with pytest.raises(LookupError, match="no draft 7; drafts: "):
        tools.send_announcement(7)

    assert tools.outbox.draft_ids() == [1]
    assert tools.outbox.sent_ids() == []


def test_only_send_needs_approval():
    assert NEEDS_APPROVAL == {"send_announcement"}


def test_schema_names_match_the_registry(make_tools):
    names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]

    assert names == list(make_tools().registry())
    assert all(schema["type"] == "function" for schema in TOOL_SCHEMAS)


def test_a_draft_id_passed_as_a_string_is_converted(make_tools):
    tools = make_tools()
    tools.draft_announcement("Lab 4", "Due Friday.")

    result = tools.send_announcement("1")

    assert result == {"draft_id": 1, "status": "sent"}
    assert tools.outbox.sent_ids() == [1]


def test_a_draft_id_that_is_not_a_number_is_a_lookup_error(make_tools):
    with pytest.raises(LookupError, match="no draft 'abc'"):
        make_tools().send_announcement("abc")
