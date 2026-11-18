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
