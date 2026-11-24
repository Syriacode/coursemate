"""The CourseMate MCP server: the agent's tools, served over MCP on stdio.

stdout is the protocol here, so nothing in this file may print to it.
"""

import argparse
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resources import FileResource
from mcp.types import ToolAnnotations

from coursemate.agent_tools import TOOL_SCHEMAS, CourseTools, Outbox
from coursemate.cli import DEFAULT_DB
from coursemate.embeddings import FakeEmbedder, OllamaEmbedder
from coursemate.llm import FakeClient, LLMError, OllamaClient
from coursemate.rag import DEFAULT_MIN_SCORE
from coursemate.retrieval_eval import build_index
from coursemate.store import PgVectorStore, StoreError

REPO = Path(__file__).resolve().parents[1]  # so the defaults work from any working directory
FAKE_MIN_SCORE = 0.1

# Hints for the client: this tool only reads, and never reaches outside the notes.
READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
# A draft writes a file, but never overwrites or deletes anything, and sends nothing.
DRAFT = ToolAnnotations(read_only_hint=False, destructive_hint=False)
# A send cannot be undone, and it reaches people outside this program.
SEND = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=True)
SEND_DESCRIPTION = (
    "Send a saved draft to all students. This cannot be undone. The instructor approves "
    "drafts in their own terminal, never through this tool; a draft that is not approved "
    "is refused and nothing is sent."
)

ANNOUNCEMENT_PROMPT = (
    "Use search_notes to find what the course notes say about {topic}. Then use "
    "draft_announcement to write a short announcement for students. Use only facts from "
    "those search results, and name the note file each fact comes from. Do not send it. "
    "Show me the draft id and the full text, so I can read it and approve it."
)

# Reuse Friday's descriptions so the model reads the same text as in the agent.
DESCRIPTIONS = {
    schema["function"]["name"]: schema["function"]["description"] for schema in TOOL_SCHEMAS
}


@contextmanager
def tool_errors() -> Iterator[None]:
    """Turn the errors a tool is expected to raise into ToolError, which the SDK sends to the
    client as is_error with the message. Anything else is a bug and is not caught."""
    try:
        yield
    except (LookupError, ValueError, LLMError, StoreError) as exc:
        raise ToolError(str(exc)) from exc


def send_if_approved(outbox: Outbox, draft_id: int) -> dict[str, Any]:
    """The server's own gate: send only a draft the instructor approved, word for word."""
    if not outbox.is_approved(draft_id):
        raise ToolError(
            f"draft {draft_id} is not approved, so nothing was sent. Tell the instructor to "
            f"read it and run this in their own terminal: py -m coursemate approve {draft_id}"
        )
    outbox.send(draft_id)
    return {"draft_id": draft_id, "status": "sent"}


def build_server(tools: CourseTools, notes_dir: Path) -> MCPServer:
    """Serve the tools, the notes in notes_dir as resources, and the announcement prompt."""
    server = MCPServer("coursemate")

    @server.tool(description=DESCRIPTIONS["search_notes"], annotations=READ)
    def search_notes(query: str) -> dict[str, Any]:
        with tool_errors():
            return tools.search_notes(query)

    @server.tool(description=DESCRIPTIONS["make_quiz"], annotations=READ)
    def make_quiz(topic: str, questions: int = 3) -> dict[str, Any]:
        with tool_errors():
            return tools.make_quiz(topic, questions)

    @server.tool(description=DESCRIPTIONS["draft_announcement"], annotations=DRAFT)
    def draft_announcement(title: str, body: str) -> dict[str, Any]:
        with tool_errors():
            return tools.draft_announcement(title, body)

    @server.tool(description=SEND_DESCRIPTION, annotations=SEND)
    def send_announcement(draft_id: int) -> dict[str, Any]:
        with tool_errors():
            return send_if_approved(tools.outbox, draft_id)

    # An allow-list: only the files found now (not subfolders) can ever be read.
    for path in sorted(notes_dir.glob("*.md")):
        server.add_resource(
            FileResource(
                uri=f"notes://sample/{path.name}",
                name=path.stem,
                mime_type="text/markdown",
                path=path.resolve(),
            )
        )

    @server.prompt(description="Draft (never send) an announcement about a topic from the notes.")
    def announcement(topic: str) -> str:
        return ANNOUNCEMENT_PROMPT.format(topic=topic)

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coursemate.mcp_server")
    parser.add_argument("--fake", action="store_true", help="no database, no model: for tests")
    parser.add_argument("--outbox", type=Path, default=REPO / "outbox")
    parser.add_argument("--notes", type=Path, default=REPO / "notes" / "sample")
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args(argv)

    outbox = Outbox(args.outbox)
    if args.fake:
        embedder = FakeEmbedder()
        store = build_index(args.notes, embedder)
        tools = CourseTools(
            embedder,
            store,
            FakeClient(),
            outbox,
            min_score=FAKE_MIN_SCORE if args.min_score is None else args.min_score,
        )
        build_server(tools, args.notes).run("stdio")
        return 0

    store = PgVectorStore(os.environ.get("DATABASE_URL") or DEFAULT_DB)
    try:
        embedder = OllamaEmbedder(model="nomic-embed-text", dim=768)
        store.prepare(embedder.model, embedder.dim)
        tools = CourseTools(
            embedder,
            store,
            OllamaClient("qwen2.5:3b"),
            outbox,
            min_score=DEFAULT_MIN_SCORE if args.min_score is None else args.min_score,
        )
        build_server(tools, args.notes).run("stdio")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
