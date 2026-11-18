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

## Plan

- `GUIDE.md` holds the step-by-step plan once it's added.

## Environment

- My machine runs Windows with Git Bash. Shell commands are POSIX bash.
