# coursemate

This repo is coursemate: CourseMate: answers questions about my course notes with citations, using local models and Postgres + pgvector

Language: python

## How to work with me

- I'm learning, so explain each change and the reasoning behind it.
- Always run the tests after changing code, and show me the output:

```bash
docker compose up -d
py -m pip install -e ".[dev]"
py -m pytest
py -m ruff check .
py -m ruff format --check .
```

- Activate the venv before starting Claude Code: `source .venv/Scripts/activate`.
- Start Postgres with `docker compose up -d` before running tests.
- Tests never call a real model or the network: use `FakeStreamer`, `FakeClient`, `FakeEmbedder`, and `httpx.MockTransport` for HTTP.
- Live tests are marked `@pytest.mark.live`.
- Never call a hosted API (`api.openai.com`, `api.anthropic.com`): it costs money.
- Async tests use `@pytest.mark.anyio`.
- Never read, print or commit anything in `notes/private/`.
- Run `py -m ruff format .` after editing Python files.
- The agent may never send anything without a human's yes: send_announcement goes through the approval gate in run_tool, and tests never remove it.
- Nothing in the MCP server may print to stdout: stdout is the protocol.
- Over MCP, send_announcement sends only drafts the instructor approved with py -m coursemate approve; tests never remove that gate, and Claude never runs that command.
- Tests never send spans anywhere: they use the `spans` fixture (an in-memory exporter).
- Never read .env: it holds the Langfuse keys.

## Plan

- `GUIDE-retrieval.md` (Nov 12): better retrieval.
- `GUIDE-api.md` (Nov 16): API and streaming.
- `GUIDE-agent.md` (Nov 20): the CourseMate agent: tools, human approval, agent evals, Claude Agent SDK.
- `GUIDE-mcp.md` (Nov 22): the CourseMate MCP server.
- `GUIDE-observability.md` (Nov 24): tracing with OpenTelemetry and Langfuse, latency and cost, a response cache.

## Environment

- My machine runs Windows with Git Bash. Shell commands are POSIX bash.
