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

## Known limitations

- Citations aren't verified sentence by sentence: a cited answer can still add an unsupported detail.
- 25 questions is a small set (one question is 4 points) written for 8 short notes.
- No stemming in keyword search ("labs" ≠ "lab").
- BM25 runs in Python over all chunks (fine for hundreds; use Postgres full-text search for more).
- No reranker yet.
- The eval uses the sample notes, not my private notes.
- One global threshold applies to all questions.
- Only `.md` and `.txt` files are ingested (no PDF or slides).
- No conversation history.
- No access control per course.
- The HNSW index is approximate.
- A 3B model sometimes answers from its own knowledge despite the prompt.
