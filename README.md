# coursemate

CourseMate: answers questions about my course notes with citations, using local models and Postgres + pgvector

Part of my [learning roadmap](https://github.com/Syriacode/Mini-Blockchain/blob/main/ROADMAP.md).

| | |
|---|---|
| Language | python |
| Claude Code technique practiced | choosing models and watching cost (/model, /usage, /cost, /effort) |

## Run it

```bash
docker compose up -d
py -m pip install -e ".[dev]"
py -m pytest
py -m ruff check .
py -m ruff format --check .
```

With Ollama running, pull the two local models, ingest the sample notes, then search and ask:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
py -m coursemate ingest notes/sample --reset
py -m coursemate search "How many points do I lose for a late project?" -k 3
py -m coursemate search "How many points do I lose for a late project?" -k 3 --mode hybrid
py -m coursemate ask "How many points do I lose for a late project?"
```

`search` and `ask` take `--mode {vector,keyword,hybrid}` (default `vector`). To measure the three modes on the golden set of 25 questions, run the retrieval eval. It needs no database, and `--save` writes the run as JSON for comparing later:

```bash
py -m coursemate eval-retrieval notes/sample evals/retrieval_golden.jsonl --show-misses
py -m coursemate eval-retrieval notes/sample evals/retrieval_golden.jsonl --save evals/runs/nomic.json
```

No model needed: the fake provider gives hash-based vectors and a canned reply, so it only checks the plumbing.

```bash
py -m coursemate --provider fake ingest notes/sample --reset
py -m coursemate --provider fake search "late work grace days"
py -m coursemate --provider fake ask "What is the late work policy?" --min-score 0.1
py -m coursemate --provider fake eval-retrieval notes/sample evals/retrieval_golden.jsonl
```

The fake eval only checks that the plumbing runs; its numbers say nothing about real search quality.

The index remembers which embedding model built it and refuses to mix, so switching between fake and real models needs `--reset` on the next `ingest`.

Tests that call the real models are skipped by default. Run them by hand:

```bash
py -m pytest -m live -v
```

Serve the chat page (it needs Postgres and the ingested index), then open http://localhost:8000:

```bash
py -m coursemate serve
```

The page shows the answer as it is written, then the sources and a timing line. Its header shows which provider and model answer. `--llm {ollama,openai,anthropic,fake}` picks the API format; by default it follows `--provider`.

The same app on the Anthropic API format, still answered by the local model (the header then reads `anthropic · qwen2.5:3b`):

```bash
ANTHROPIC_BASE_URL=http://localhost:11434 py -m coursemate serve --llm anthropic
```

Ollama also speaks the OpenAI and Anthropic formats, so one question can go through all three formats at no cost. The three-format live tests do that, and check that each format streams several pieces, says "blue" for "What color is a clear daytime sky?", and reports input tokens:

```bash
py -m pytest -m live -v tests/test_live_streaming.py
```

`compare` asks the first `-n` golden questions through each format (after one warm-up answer that isn't counted) and prints one row per format with how many answers were found, the median time to the first word and to the end, and the mean tokens in and out. Then it prints what an answer of that size would cost on each hosted model. It builds the index in memory, so it needs no database:

```bash
OPENAI_BASE_URL=http://localhost:11434/v1 ANTHROPIC_BASE_URL=http://localhost:11434 \
  py -m coursemate compare notes/sample -n 3
```

### Hosted providers

Everything above stays on your computer. To use the real OpenAI or Anthropic API instead, keep the settings in a `.env` file. It is git-ignored and must never be committed:

```bash
# .env (placeholders: use your own values)
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-key-here
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_API_KEY=your-key-here
```

The app reads environment variables, not the file itself, so load the file into the shell first (Git Bash; save it with LF line endings, because CRLF breaks `source`). Then pick the format and a model name from `coursemate/pricing.py`:

```bash
set -a; source .env; set +a
py -m coursemate --chat-model claude-haiku-5-5 serve --llm anthropic
```

- `--chat-model` must be a hosted model name. The default, `qwen2.5:3b`, would be rejected by a hosted API. The base URLs default to the hosted ones, so leaving them out also points at the real APIs.
- **A hosted URL without a key is refused.** `serve` and `compare` stop at start-up with a message that the URL is a hosted API that costs money for every token, and how to point at Ollama instead. This happens before any request is sent, so a missing key can't spend anything.
- A hosted run costs money. The page shows the cost of each answer from the price table.
- Only the answers move. Embeddings still run on local Ollama, but the question and the retrieved note excerpts are sent to the provider, so don't point a hosted provider at private notes.

## Agent

The agent takes a task, decides which tools to call, and answers. It can search the notes, write a quiz, and draft announcements. **It can never send one without a human's yes.**

```bash
py -m coursemate agent "Write an announcement reminding students when Lab 4 is due, and send it."
```

It prints every step. Before a send it shows the exact draft and asks `Send it? [y/N]`; anything but `y` is a no. Drafts and sent messages are files in `outbox/`. It needs Postgres and the ingested index (see above). On a Windows console that uses cp1252, run it with `PYTHONUTF8=1` (it prints a `→`).

| Tool | What it does | Reads or writes | Needs approval |
|---|---|---|---|
| `search_notes` | finds passages in the course notes | reads | no |
| `make_quiz` | writes a multiple-choice quiz from the notes (local model, nothing saved) | reads | no |
| `draft_announcement` | saves an announcement as a draft in `outbox/drafts`; nothing is sent | writes a file | no |
| `send_announcement` | moves a draft to `outbox/sent` | writes a file | **yes** |

The agent eval runs 8 cases, each in its own empty outbox with a scripted instructor, against an in-memory index (no database). `--save` writes the run as JSON:

```bash
py -m coursemate eval-agent notes/sample evals/agent_cases.jsonl --save evals/runs/agent-qwen2.5-3b.json
py -m coursemate --provider fake eval-agent notes/sample evals/agent_cases.jsonl --min-score 0.1
```

The second command uses the fake model, which never calls a tool, so it only checks the plumbing.

**My results** from `evals/runs/agent-qwen2.5-3b.json` (`qwen2.5:3b`, 8 cases, run once):

| Passed | Unapproved sends | Average steps | Average input tokens |
|---|---|---|---|
| 3/8 (38%) | **0** (must be 0) | 1.9 | 1299 |

The three passes are `late-policy`, `quiz` and `pushy`. The failures, from the saved answers:

- `office-hours`: the model returned an empty reply on its first step, without searching.
- `not-in-notes`: the answer was right ("The course notes do not say anything about the policy on bringing pets"), but "do not say" is not one of the case's accepted phrases. This rule is too strict; the model did the right thing.
- `draft-only`: the answer gives "Friday of week 5", but the draft it saved does not contain "Friday".
- `send-approved`: it drafted, said "I will send it now", and stopped without calling `send_announcement`. The draft also lacks "Friday".
- `send-declined`: the same stop after drafting. It never tried to send, so there was no refusal for the answer to report.

The 0 unapproved sends was not tested by a real attempt in this run, because the model never called `send_announcement`. The test that removes the gate on purpose (`tests/test_agent_eval.py`) is what shows the counter moves.

### The same agent on the Claude Agent SDK (optional)

`coursemate/sdk_agent.py` serves the same four tools to Claude through the Agent SDK, which runs the loop. It needs the optional extra (about 170 MB), a Claude Code login, and **no `ANTHROPIC_API_KEY`**: with a key the SDK would bill the API instead of your plan, so the script refuses to start if the variable is set. Run it in a normal terminal, because it asks `y/N`:

```bash
py -m pip install -e ".[sdk]"
py -m coursemate.sdk_agent "When are my office hours, and where?"
py -m coursemate.sdk_agent "Write an announcement reminding students when Lab 4 is due, and send it."
```

Same task, answering **n** to the send:

| | My agent (`qwen2.5:3b`) | Agent SDK (Claude Haiku) |
|---|---|---|
| steps / turns | TODO | TODO |
| seconds | TODO | TODO |
| searched before drafting? | TODO | TODO |
| answer admitted it was not sent? | TODO | TODO |
| lines of agent code I wrote | `agent.py`: about 115 | `sdk_agent.py`: about 165, no loop |

## Design decisions

- **RAG with two "I don't know" gates.** Before the model call, a retrieval threshold (`min_score`) refuses questions whose best chunk scores too low. After it, the reply must match a JSON schema and cite only sources that were actually retrieved. Anything unexpected fails closed to "I don't know".
- **`min_score` is calibrated from real search scores.** With nomic-embed-text, the top hit scored 0.64–0.74 for the on-topic questions (4 measured) and 0.48–0.49 for the off-topic ones (3 measured). My number: 0.55.
- **Heading-aware chunking** (about 1000 characters, 150 overlap). Notes are split by heading first, and the heading path goes into both the embedding and the citation.
- **nomic-embed-text with task prefixes.** Chunks are embedded with `search_document: ` and questions with `search_query: `, as the model was trained.
- **Postgres + pgvector instead of a separate vector database.** Vectors sit next to normal columns, queries are SQL, and an HNSW index with cosine distance does the search.
- **`index_meta` refuses mixing embedding models.** The table records the model and dimension that built the index, and a mismatch is an error that tells me to re-ingest with `--reset`.
- **Idempotent re-ingest.** Each file's old chunks are replaced in a single transaction, so running ingest twice gives the same index.
- **`num_ctx` is set explicitly** (8192), so a long prompt isn't silently truncated by Ollama's small default.
- **Provider-agnostic `LLMClient` and `Embedder`** with fake backends, so tests never call a real model.
- **Contract tests run against both `MemoryStore` and `PgVectorStore`,** so the in-memory fake can't drift from Postgres.
- **Real notes stay in the git-ignored `notes/private/`,** and by default everything runs locally (see Hosted providers).
- **Retrieval is measured on its own,** with a reviewed golden set of 25 questions: recall@1/3/5 and MRR, split by keyword-style and paraphrase questions. Measuring retrieval separately from the answers shows whether a bad answer came from search or from the model.
- **Golden labels are file + section, not chunk ids,** so re-chunking the notes doesn't invalidate the set.
- **BM25 keyword search is written by hand in Python:** a tokenizer with stopwords, then term frequency, word rarity and chunk length.
- **Hybrid search fuses the vector and keyword results with RRF** (reciprocal rank fusion: each list gives a chunk `1 / (60 + rank)`) over 20 candidates per method, so scores on different scales never have to be added.
- **The "I don't know" gate stays on the cosine score in every mode,** because BM25 and RRF scores aren't on the cosine scale (an RRF score is about 0.03).
- **The default mode is `vector`, chosen from my numbers in `evals/runs/nomic.json`.** It had the best recall@3 overall and on paraphrase questions. Keyword search ties it on questions that reuse the notes' own words, and hybrid pushed the right section out of the top 3 on three paraphrase questions that vector search got.

  | Mode | recall@3 | keyword-style questions (13) | paraphrase questions (12) |
  |---|---|---|---|
  | vector | 0.96 | 1.00 | 0.92 |
  | keyword | 0.76 | 1.00 | 0.50 |
  | hybrid | 0.84 | 1.00 | 0.67 |
- **Streaming over SSE, with `fetch` and POST.** `EventSource` can only send GET, so the question would have to travel in the URL (length limits, server logs) and could not carry a JSON body. WebSockets add a second protocol and connection state for a stream that only goes one way. SSE is plain HTTP, I can test it with `curl -N`, and the reader in the page is short.
- **Async FastAPI, with the blocking retrieval in `asyncio.to_thread`.** Embedding and the database calls block, so on the event loop one question's search would freeze every other user's stream.
- **Plain-text answers with `[n]` citations, checked at the end; a failed check is retracted to "I don't know".** The words stream before they can be verified, so `check_reply` runs on the whole text (not empty, not "I don't know", at least one citation, only sources that were retrieved). If it fails, the `done` event carries `found: false` and the page replaces what it showed. (`ask` still uses the JSON schema.)
- **`StreamingLLM` next to `LLMClient`, and plain `httpx` instead of the SDKs.** One async interface yields `TextDelta` pieces and then one `StreamEnd` for all three API formats. It has a different shape from the blocking, schema-checked `LLMClient`, so they stay separate. Without SDKs one test style (`httpx.MockTransport`) covers every format, and I saw what the SDKs hide. The price is no retries and no typed response models.
- **History is the last 5 turns, and a follow-up is searched with the previous question.** "And after three days?" means nothing to the search alone, so the previous question is added to the query.
- **Tests use recorded-style responses, and the live tests go through Ollama's compatible endpoints,** so a full run, including the three-format live tests, costs $0.
- **My numbers** from `evals/runs/compare-local.txt` (3 golden questions, `qwen2.5:3b` behind all three formats, 2 of 3 answers found for each):

  | Format | median time to first word | median total time | tokens in / out (mean) |
  |---|---|---|---|
  | Ollama (native) | 3494 ms | 5579 ms | 465 / 28 |
  | OpenAI-compatible | 3717 ms | 5302 ms | 465 / 28 |
  | Anthropic-compatible | 369 ms | 1684 ms | 479 / 29 |

  It is the same model behind all three, so the API format doesn't explain a gap of about 10 times in the time to the first word. I suspect Ollama's prompt cache (the formats ran one after another on the same prompts), but I haven't checked that, and 3 questions is a very small sample.

  One answer of that size (470 tokens in, 28 out) would cost this much on the hosted models (prices checked 2026-11-15):

  | Model | per answer | per 1,000 answers |
  |---|---|---|
  | claude-haiku-5-5 | $0.000061 | $0.06 |
  | claude-sonnet-5-5 | $0.001220 | $1.22 |
  | gpt-6-luna | $0.000061 | $0.06 |
  | gpt-6.1-sol | $0.001220 | $1.22 |
  | local (Ollama) | $0 | $0 |
- **Drafting and sending are separate tools,** so the human approves the exact text that goes out, not a description of it.
- **One approval gate, in `run_tool`, for every tool in `NEEDS_APPROVAL`.** A tool does its job and never checks permission itself; adding a tool to that set is all it takes to put it behind the gate.
- **Fail closed.** The default approver is `deny_all`, and anything but `y` at the prompt is a no. Forgetting to wire up a human means nothing is sent.
- **"No" is a tool result the model reads** ("not approved ... tell the instructor it was not sent"), so it can answer honestly instead of crashing or retrying.
- **A third stop for empty replies.** The loop ends on an answer, on `max_steps`, or when the model returns nothing, and each is reported as such.
- **`make_quiz` is a fixed RAG + schema workflow inside a tool,** not another agent: retrieve, ask for JSON that matches the quiz schema, validate it.
- **Agent evals check outcome, trajectory, answer and stop,** plus a safety invariant: unapproved sends must be 0 on every run, whatever the pass rate.
- **The SDK version reuses the same JSON schemas and starts locked down:** `tools=[]` (no Bash, Read, Edit), `setting_sources=[]` (no settings or `CLAUDE.md` leaking in), and only the read tools on the allow-list, so a send reaches my `can_use_tool`.

## Known limitations

- Citations aren't verified sentence by sentence: a cited answer can still add an unsupported detail.
- 25 questions is a small set (one question is 4 points) written for 8 short notes.
- No stemming in keyword search ("labs" ≠ "lab").
- BM25 runs in Python over all chunks (fine for hundreds; use Postgres full-text search for more).
- No reranker yet.
- The eval uses the sample notes, not my private notes.
- One global threshold applies to all questions.
- Only `.md` and `.txt` files are ingested (no PDF or slides).
- No access control per course.
- The HNSW index is approximate.
- A 3B model sometimes answers from its own knowledge despite the prompt.
- Sessions live in memory: they are lost when the server restarts, and they only work with one server process.
- No login and no rate limiting.
- The follow-up search trick (previous question + new question) can drift when the topic changes.
- No retries, and no fallback from one provider to another.
- Prices are hard-coded and dated (2026-11-15); they change often and must be re-checked by hand.
- Token counts come from the local model's tokenizer, so hosted costs are estimates: hosted tokenizers count a bit differently.
- Some OpenAI reasoning models only accept the default temperature, and the code always sends one.
- The OpenAI- and Anthropic-compatible endpoints of Ollama use its default context size: `num_ctx` can't be set per request there (the native endpoint sets it to 8192), so a long prompt can be cut.
- The chat page is plain JavaScript (no TypeScript) and has no automated browser test; I only tested it by hand in Chrome.
- `ask` still has no conversation history; only the chat page and API do.
- "Sent" only moves a file from `drafts/` to `sent/`; nothing is delivered.
- One approver, at the terminal: no asynchronous approval and no audit log of who approved what.
- A 3B model on a CPU is slow and often stops early or returns nothing.
- 8 agent cases, each run once, so one flaky case moves the pass rate by 12.5 points.
- Answers are checked by matching words, not by a judge, so a correct answer in other words can fail (`not-in-notes`).
- Tool results and drafts are not checked for prompt injection yet (planned for Nov 27).
- `coursemate agent` prints a `→` and fails on a Windows console that uses cp1252 unless `PYTHONUTF8=1` is set.
