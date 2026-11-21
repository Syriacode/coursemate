"""The CourseMate agent: ask the model, run the tools it asks for, repeat.

Side effects (sending an announcement) need a human's yes, checked in run_tool.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from coursemate.agent_tools import NEEDS_APPROVAL, TOOL_SCHEMAS  # noqa: F401
from coursemate.llm import LLMClient, Message, ToolCall, assistant_message

# NEEDS_APPROVAL is not used yet. run_tool needs it (you type that body), so the noqa
# keeps ruff from removing the import before you do.

AGENT_SYSTEM = (
    "You are CourseMate, an assistant for the instructor of a university course. "
    "Use search_notes to find facts in the course notes before you answer, write a quiz "
    "or write an announcement; never invent dates, policies or numbers. "
    "Use make_quiz for quizzes and draft_announcement to write announcements. "
    "Call send_announcement only when the instructor asked you to send something. "
    "When you are done, reply in two or three sentences and name the note files you used."
)

# Asks a human "may this tool call run?". The default says no, so forgetting to wire up
# a human means nothing gets sent (fail closed).
Approver = Callable[[ToolCall], bool]

SHOW_LIMIT = 200


def deny_all(call: ToolCall) -> bool:
    return False


@dataclass(frozen=True)
class AgentResult:
    answer: str
    stopped: Literal["answer", "empty", "max_steps"]
    steps: int
    input_tokens: int
    tool_calls: list[str]
    messages: list[Message]


def show_message(message: Message) -> None:
    """Print one message as one short line, indented two spaces (one line per tool call)."""
    if message["role"] == "tool":
        content = message["content"]
        if len(content) > SHOW_LIMIT:
            content = content[:SHOW_LIMIT] + "…"
        print(f"  [tool → {message.get('tool_name', '?')}] {content}")
        return

    tool_calls = message.get("tool_calls", [])
    for call in tool_calls:
        function = call["function"]
        print(f"  [assistant wants] {function['name']}({json.dumps(function['arguments'])})")
    if not tool_calls:
        print(f"  [{message['role']}] {message['content']}")


def run_tool(call: ToolCall, tools: dict[str, Callable[..., Any]], approve: Approver) -> str:
    """Run one tool call and return the result as text. Side effects need a human's yes."""
    if call.name not in tools:
        return f"error: unknown tool {call.name!r}. Available tools: {', '.join(tools)}"
    if call.name in NEEDS_APPROVAL and not approve(call):
        return (
            f"error: not approved. The instructor did not approve {call.name}, so nothing "
            "was sent. Do not try again; tell the instructor it was not sent."
        )
    try:
        result = tools[call.name](**call.arguments)
    except Exception as exc:  # bad arguments, unknown draft, a failed lookup...
        return f"error: {type(exc).__name__}: {exc}"
    return json.dumps(result)


def run_agent(
    client: LLMClient,
    task: str,
    tools: dict[str, Callable[..., Any]],
    *,
    schemas: list[dict[str, Any]] = TOOL_SCHEMAS,
    approve: Approver = deny_all,
    max_steps: int = 6,
    verbose: bool = False,
) -> AgentResult:
    """Ask the model, run the tools it asks for, send the results back, repeat until it answers."""
    messages: list[Message] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": task},
    ]
    input_tokens = 0
    tool_calls: list[str] = []
    for step in range(1, max_steps + 1):
        response = client.chat(messages, tools=schemas)
        input_tokens += response.input_tokens
        messages.append(assistant_message(response))
        if verbose:
            print(f"step {step} ({response.input_tokens} input tokens)")
            show_message(messages[-1])

        if not response.tool_calls:
            # stop 1: the model answered in words. stop 2: it said nothing at all.
            if not response.text.strip():
                empty = "The model returned an empty reply."
                return AgentResult(empty, "empty", step, input_tokens, tool_calls, messages)
            return AgentResult(response.text, "answer", step, input_tokens, tool_calls, messages)

        for call in response.tool_calls:
            tool_calls.append(call.name)
            messages.append(
                {"role": "tool", "tool_name": call.name, "content": run_tool(call, tools, approve)}
            )
            if verbose:
                show_message(messages[-1])

    # stop 3: the safety limit
    return AgentResult(
        f"I hit the step limit ({max_steps}) before finishing.",
        "max_steps",
        max_steps,
        input_tokens,
        tool_calls,
        messages,
    )
