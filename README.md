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
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

With Ollama running, pull the two local models, ingest the sample notes, then search and ask:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
python -m coursemate ingest notes/sample --reset
python -m coursemate search "How many points do I lose for a late project?" -k 3
python -m coursemate ask "How many points do I lose for a late project?"
```

No model needed: the fake provider gives hash-based vectors and a canned reply, so it only checks the plumbing.

```bash
python -m coursemate --provider fake ingest notes/sample --reset
python -m coursemate --provider fake search "late work grace days"
python -m coursemate --provider fake ask "What is the late work policy?" --min-score 0.1
```

The index remembers which embedding model built it and refuses to mix, so switching between fake and real models needs `--reset` on the next `ingest`.

Tests that call the real models are skipped by default. Run them by hand:

```bash
python -m pytest -m live -v
```

## Design decisions

- **RAG with two "I don't know" gates.** Before the model call, a retrieval threshold (`min_score`) refuses questions whose best chunk scores too low. After it, the reply must match a JSON schema and cite only sources that were actually retrieved. Anything unexpected fails closed to "I don't know".
- **`min_score` is calibrated from real search scores.** With nomic-embed-text, the top hit scored 0.64–0.74 for the on-topic questions (4 measured) and 0.48–0.49 for the off-topic ones (3 measured). My number: TODO.
- **Heading-aware chunking** (about 1000 characters, 150 overlap). Notes are split by heading first, and the heading path goes into both the embedding and the citation.
- **nomic-embed-text with task prefixes.** Chunks are embedded with `search_document: ` and questions with `search_query: `, as the model was trained.
- **Postgres + pgvector instead of a separate vector database.** Vectors sit next to normal columns, queries are SQL, and an HNSW index with cosine distance does the search.
- **`index_meta` refuses mixing embedding models.** The table records the model and dimension that built the index, and a mismatch is an error that tells me to re-ingest with `--reset`.
- **Idempotent re-ingest.** Each file's old chunks are replaced in a single transaction, so running ingest twice gives the same index.
- **`num_ctx` is set explicitly** (8192), so a long prompt isn't silently truncated by Ollama's small default.
- **Provider-agnostic `LLMClient` and `Embedder`** with fake backends, so tests never call a real model.
- **Contract tests run against both `MemoryStore` and `PgVectorStore`,** so the in-memory fake can't drift from Postgres.
- **Real notes stay in the git-ignored `notes/private/`,** and everything runs locally.

## Known limitations

- Citations aren't verified sentence by sentence: a cited answer can still add an unsupported detail.
- Retrieval quality isn't measured yet (next: recall@k).
- Retrieval is vector-only: no keyword or hybrid search, no reranking.
- One global threshold applies to all questions.
- Only `.md` and `.txt` files are ingested (no PDF or slides).
- No conversation history.
- No access control per course.
- The HNSW index is approximate.
- A 3B model sometimes answers from its own knowledge despite the prompt.

## How I used Claude Code

### What I delegated

### What I reviewed

### A bug I caught

### Something I changed by hand

### What I'd do differently
