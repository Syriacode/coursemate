import json
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from coursemate.agent import run_agent
from coursemate.agent_tools import CourseTools, Outbox
from coursemate.embeddings import FakeEmbedder
from coursemate.llm import FakeClient
from coursemate.retrieval_eval import build_index
from coursemate.tracing import use_provider
from tests.test_agent import FINAL, SAMPLE, TASK, paper_trace

TOOLS = ["search_notes", "draft_announcement", "send_announcement"]


class StartOrder(SpanProcessor):
    """Remember span names as they start. Start times can tie on a coarse clock."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def on_start(self, span, parent_context=None) -> None:
        self.names.append(span.name)


@pytest.fixture
def traced() -> Iterator[tuple[StartOrder, InMemorySpanExporter]]:
    """Like the spans fixture, but it also records the order in which spans start."""
    started, exporter = StartOrder(), InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(started)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = use_provider(provider)
    yield started, exporter
    use_provider(previous)
    provider.shutdown()


@pytest.fixture(scope="module")
def index():
    embedder = FakeEmbedder()
    return embedder, build_index(SAMPLE, embedder)


@pytest.fixture
def course(index, tmp_path):
    embedder, store = index
    return CourseTools(embedder, store, FakeClient([]), Outbox(tmp_path / "outbox"), min_score=0.1)


def named(exporter, name):
    """The finished spans with this name, in the order they ended."""
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_spans_start_in_the_order_of_the_paper_run(course, traced):
    started, exporter = traced

    run_agent(paper_trace(), TASK, course.registry())

    assert started.names == [
        "agent",
        "llm",
        "search_notes",
        "llm",
        "draft_announcement",
        "llm",
        "send_announcement",
        "llm",
    ]
    assert len(exporter.get_finished_spans()) == 8  # every span ended


def test_every_step_is_a_child_of_the_agent_span(course, traced):
    _, exporter = traced

    run_agent(paper_trace(), TASK, course.registry())

    [root] = named(exporter, "agent")
    assert root.parent is None
    children = [s for s in exporter.get_finished_spans() if s is not root]
    assert len(children) == 7
    for child in children:
        assert child.context.trace_id == root.context.trace_id
        assert child.parent.span_id == root.context.span_id


def test_the_agent_span_says_how_the_run_ended(course, traced):
    _, exporter = traced

    run_agent(paper_trace(), TASK, course.registry())

    [root] = named(exporter, "agent")
    assert root.attributes["langfuse.trace.name"] == "agent"
    assert root.attributes["langfuse.observation.type"] == "agent"
    assert root.attributes["langfuse.observation.input"] == TASK
    assert root.attributes["langfuse.observation.output"] == FINAL
    assert root.attributes["coursemate.stopped"] == "answer"
    assert root.attributes["coursemate.steps"] == 4
    assert json.loads(root.attributes["coursemate.tool_calls"]) == TOOLS


def test_each_model_call_records_what_the_model_saw_and_said(course, traced):
    _, exporter = traced

    run_agent(paper_trace(), TASK, course.registry())

    calls = [s.attributes for s in named(exporter, "llm")]
    assert [c["coursemate.step"] for c in calls] == [1, 2, 3, 4]
    assert [len(json.loads(c["langfuse.observation.input"])) for c in calls] == [2, 4, 6, 8]
    for c in calls:
        assert c["langfuse.observation.type"] == "generation"
        assert c["gen_ai.request.model"] == "fake"
        assert (c["gen_ai.usage.input_tokens"], c["gen_ai.usage.output_tokens"]) == (100, 20)
    first = json.loads(calls[0]["langfuse.observation.output"])
    assert first["tool_calls"][0]["function"]["name"] == "search_notes"
    assert json.loads(calls[3]["langfuse.observation.output"])["content"] == FINAL


def test_a_declined_send_is_a_denied_warning(course, traced):
    _, exporter = traced

    run_agent(paper_trace(), TASK, course.registry())  # the default approver says no

    [send] = named(exporter, "send_announcement")
    assert send.attributes["langfuse.observation.type"] == "tool"
    assert json.loads(send.attributes["langfuse.observation.input"]) == {"draft_id": 1}
    assert send.attributes["langfuse.observation.output"].startswith("error: not approved")
    assert send.attributes["coursemate.needs_approval"] is True
    assert send.attributes["coursemate.denied"] is True
    assert send.attributes["langfuse.observation.level"] == "WARNING"
    assert course.outbox.sent_ids() == []  # the gate still holds
    for name in ("search_notes", "draft_announcement"):
        [tool] = named(exporter, name)
        assert tool.attributes["coursemate.needs_approval"] is False
        assert tool.attributes["coursemate.denied"] is False
        assert "langfuse.observation.level" not in tool.attributes


def test_an_approved_send_has_no_warning(course, traced):
    _, exporter = traced

    run_agent(paper_trace(), TASK, course.registry(), approve=lambda call: True)

    [send] = named(exporter, "send_announcement")
    assert send.attributes["coursemate.needs_approval"] is True
    assert send.attributes["coursemate.denied"] is False
    assert "langfuse.observation.level" not in send.attributes
    assert course.outbox.sent_ids() == [1]
