"""The CourseMate agent on the Claude Agent SDK (optional extra: pip install -e ".[sdk]").

Same four tools and the same human "yes" before a send as coursemate/agent.py, but
Claude Code's engine runs the loop: query() replaces run_agent(), and can_use_tool
replaces the approve() function.
"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

from coursemate.agent import AGENT_SYSTEM
from coursemate.agent_tools import NEEDS_APPROVAL, TOOL_SCHEMAS, CourseTools, Outbox
from coursemate.cli import DEFAULT_DB
from coursemate.embeddings import OllamaEmbedder
from coursemate.llm import OllamaClient
from coursemate.store import PgVectorStore

SERVER = "coursemate"
SHOW_LIMIT = 200
DENY_MESSAGE = (
    "The instructor did not approve this, so nothing was sent. "
    "Do not try again; tell the instructor it was not sent."
)


def full_name(name: str) -> str:
    """The SDK names a tool from an MCP server mcp__<server>__<tool>."""
    return f"mcp__{SERVER}__{name}"


def wrap(function: Callable[..., Any]) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Turn one of our plain functions into the async handler the SDK expects."""

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = function(**args)
        except Exception as exc:  # same as run_tool: the model reads the error and can react
            text = f"error: {type(exc).__name__}: {exc}"
            return {"content": [{"type": "text", "text": text}], "is_error": True}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    return handler


def build_server(tools: CourseTools) -> Any:
    """Serve our tools to Claude in-process, from the same JSON Schemas as TOOL_SCHEMAS."""
    registry = tools.registry()
    sdk_tools = []
    for schema in TOOL_SCHEMAS:
        function = schema["function"]
        name = function["name"]
        sdk_tools.append(
            tool(name, function["description"], function["parameters"])(wrap(registry[name]))
        )
    return create_sdk_mcp_server(name=SERVER, tools=sdk_tools)


def make_permission_check(
    outbox: Outbox, ask: Callable[[str], str] = input
) -> Callable[[str, dict[str, Any], ToolPermissionContext], Awaitable[Any]]:
    """can_use_tool: the SDK calls this for every tool that is not in allowed_tools."""
    needs_approval = {full_name(name) for name in NEEDS_APPROVAL}

    async def can_use_tool(
        name: str, tool_input: dict[str, Any], context: ToolPermissionContext
    ) -> Any:
        if name not in needs_approval:
            return PermissionResultDeny(message=f"{name} is not allowed in this agent.")
        try:
            draft = outbox.read_draft(tool_input.get("draft_id", ""))
        except LookupError:
            return PermissionResultAllow()  # nothing to show; the tool reports "no draft"
        print("\nClaude wants to SEND this announcement:\n")
        print(draft)
        # input() blocks, so run it in a thread and keep the event loop free.
        answer = await asyncio.to_thread(ask, "Send it? [y/N] ")
        if answer.strip().lower() == "y":
            return PermissionResultAllow()
        return PermissionResultDeny(message=DENY_MESSAGE)

    return can_use_tool


def make_options(tools: CourseTools, model: str, max_turns: int) -> ClaudeAgentOptions:
    """Locked down: no built-in tools, no settings files, read tools allowed, send asks."""
    return ClaudeAgentOptions(
        system_prompt=AGENT_SYSTEM,
        mcp_servers={SERVER: build_server(tools)},
        tools=[],  # Claude Code's own Bash, Read, Edit... are off
        allowed_tools=[full_name(name) for name in tools.registry() if name not in NEEDS_APPROVAL],
        can_use_tool=make_permission_check(tools.outbox),
        setting_sources=[],  # do not load ~/.claude/settings.json or CLAUDE.md
        model=model,
        max_turns=max_turns,
    )


def _shorten(text: str) -> str:
    return text[:SHOW_LIMIT] + "..." if len(text) > SHOW_LIMIT else text


def _result_text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(str(part.get("text", part)) for part in content)
    return "" if content is None else str(content)


def show(message: Any) -> None:
    """Print one SDK message the way agent.py prints its steps."""
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                print(f"  [assistant wants] {block.name}({json.dumps(block.input)})")
            elif isinstance(block, TextBlock):
                print(f"  [assistant] {block.text}")
    elif isinstance(message, UserMessage) and isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                label = "tool result (error)" if block.is_error else "tool result"
                print(f"  [{label}] {_shorten(_result_text(block.content))}")
    elif isinstance(message, ResultMessage):
        print(f"\n{message.result}")
        seconds = message.duration_ms / 1000
        cost = message.total_cost_usd or 0.0
        print(
            f"\n(turns: {message.num_turns}, {seconds:.1f} s, API-price equivalent: ${cost:.4f})",
            file=sys.stderr,
        )


async def _prompt(task: str) -> AsyncIterator[dict[str, Any]]:
    # can_use_tool only works when the prompt is streamed, not passed as a plain string.
    yield {"type": "user", "message": {"role": "user", "content": task}}


async def run(task: str, options: ClaudeAgentOptions) -> None:
    async for message in query(prompt=_prompt(task), options=options):
        show(message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coursemate.sdk_agent")
    parser.add_argument("task")
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--outbox", type=Path, default=Path("outbox"))
    parser.add_argument("--min-score", type=float, default=0.55)
    args = parser.parse_args(argv)

    if os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "error: ANTHROPIC_API_KEY is set, so this would bill the API instead of your plan. "
            "Unset it and run again.",
            file=sys.stderr,
        )
        return 1

    store = PgVectorStore(os.environ.get("DATABASE_URL") or DEFAULT_DB)
    try:
        embedder = OllamaEmbedder(model="nomic-embed-text", dim=768)
        store.prepare(embedder.model, embedder.dim)
        outbox = Outbox(args.outbox)
        # make_quiz still uses the local model; only the agent loop runs on Claude.
        tools = CourseTools(
            embedder, store, OllamaClient("qwen2.5:3b"), outbox, min_score=args.min_score
        )
        asyncio.run(run(args.task, make_options(tools, args.model, args.max_turns)))
    finally:
        store.close()

    print(f"(outbox: drafts {outbox.draft_ids()}, sent {outbox.sent_ids()})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
