# Week 2 Thu — Better retrieval: keyword, vector and hybrid search, measured (step-by-step guide)

> Read the **Project overview** first, on its own, before you open a terminal. It is longer than usual on purpose:
> it teaches the AI ideas with small examples you can work out on paper. The steps start at Step 0.
>
> New pace: one project every other day. Do the overview and Steps 0–4 today if you like, and the rest next time,
> or spread it however you want. The **Check your understanding** questions at the end are for the day in between.

---

## Project overview

### 1. First, what you built yesterday (in plain words)

`coursemate` answers questions about your course notes and says where the answer came from. It works in two halves,
and this split is the most important idea in RAG:

```text
                      ┌─────────────── RETRIEVAL (find) ───────────────┐   ┌──── GENERATION (write) ────┐
 "What is the late   │ 1. turn the question into numbers (embedding)  │   │ 4. give the model ONLY     │
  work policy?"  ──► │ 2. compare with the numbers of every chunk     │──►│    those chunks + question │──► answer + [1] syllabus.md
                     │ 3. keep the k closest chunks (k = 4)           │   │ 5. check its JSON reply     │
                      └────────────────────────────────────────────────┘   └────────────────────────────┘
       ingest (once): notes/*.md ──► chunks (one per heading) ──► embeddings ──► Postgres + pgvector
```

Where that lives in your code:

| Part | File and function |
|---|---|
| Split notes into chunks by heading | `coursemate/chunking.py`, `chunk_document()` |
| Turn text into numbers | `coursemate/embeddings.py`, `OllamaEmbedder.embed_query()` / `embed_documents()` |
| Store chunks and find the closest | `coursemate/store.py`, `PgVectorStore.search()` (and `MemoryStore.search()` for tests) |
| Build the prompt, call the model, check the reply | `coursemate/rag.py`, `answer()` |
| Commands (`ingest`, `search`, `ask`) | `coursemate/cli.py` |

### 2. What this project is about

**If retrieval doesn't find the right chunk, the answer is wrong no matter how good the model is.** The model can
only use the chunks you hand it. So today you work only on the *find* half, and you do two things:

1. **Measure it.** Right now you have no idea how often `search` puts the right section in the top 4. Your README
   even says *"Retrieval quality isn't measured yet"*. You'll write 25 questions where you know the right section,
   and count how often each search method finds it.
2. **Improve it** by adding a second, older kind of search (**keyword search**) and combining both (**hybrid
   search**). Then you let the numbers, not your gut, decide which one CourseMate uses.

No chat model is called today. Only the small embedding model runs, so everything is fast.

### 3. The AI basics you need today (work these on paper)

#### 3a. Embeddings and cosine similarity: "search by meaning"

An **embedding model** turns a text into a list of numbers, called a **vector**. Your model, `nomic-embed-text`,
always gives 768 numbers. Nobody decides what each number means; the model learned them from millions of texts, so
that **texts with similar meaning get vectors that point in a similar direction**.

Pretend vectors had only 2 numbers, so you can draw them:

```text
 a = "late penalty"                [0.90, 0.10]
 b = "extra time without penalty"  [0.80, 0.30]
 c = "merge sort"                  [0.10, 0.95]
```

**Cosine similarity** measures the angle between two vectors: 1.0 = same direction, 0 = unrelated.

```text
 cosine(a, b) = (a·b) / (|a| × |b|)
 a·b = 0.90×0.80 + 0.10×0.30 = 0.75       |a| = √(0.90² + 0.10²) = 0.906     |b| = √(0.80² + 0.30²) = 0.854
 cosine(a, b) = 0.75 / (0.906 × 0.854) = 0.97      ← very similar
 cosine(a, c) = 0.21                               ← unrelated
```

That's all **vector search** is: compute the cosine of the question against every chunk and keep the top k.
pgvector does exactly this, just with 768 numbers and a fast index. You'll see real 768-number vectors in Step 1.

**Strength:** it matches meaning even when no word is shared ("extra time" ≈ "late work, grace days").
**Weakness:** it is fuzzy about exact things: "Lab 4" and "Lab 5", or "Project 2" and "Project 1", have almost
the same meaning to it, so it can return the wrong one.

#### 3b. Keyword search and BM25: "search by shared words"

Keyword search is how search engines worked for decades: score each chunk by **the words it shares with the
question**. Two ideas make it good:

- **Rare words count more.** "due" is in many chunks, so it says little. "lab" or "4" are in few chunks, so they
  say a lot. This is **IDF** (inverse document frequency).
- **Repeating a word helps, but less and less,** and a **long chunk** gets a small penalty, because a long text
  contains many words by chance.

**BM25** is the standard formula that does both. For each question word that a chunk contains:

```text
 idf        = ln(1 + (N − df + 0.5) / (df + 0.5))        N = number of chunks, df = chunks containing the word
 word score = idf × tf × (k1 + 1) / (tf + k1 × (1 − b + b × len / avg_len))
              tf = times the word is in this chunk, len = chunk length in words,
              k1 = 1.5 (how fast repeats stop helping), b = 0.75 (how much length matters)
 chunk score = sum of the word scores
```

Worked example. Three tiny chunks and the question *"When is Lab 4 due?"*. First we drop **stopwords** (words like
"when", "is" that carry no meaning), so the question becomes `lab, 4, due`:

```text
 C1 "Lab 4 is due Friday"        → lab, 4, due, friday        (4 words)
 C2 "Labs are worth 30%"         → labs, worth, 30            (3 words)
 C3 "Project 2 is due week 13"   → project, 2, due, week, 13  (5 words)      avg_len = 12 / 3 = 4

 idf(lab) = idf(4) = ln(1 + 2.5/1.5) = 0.98      (in 1 of 3 chunks: rare, worth a lot)
 idf(due)          = ln(1 + 1.5/2.5) = 0.47      (in 2 of 3 chunks: common, worth less)

 C1: len = avg_len, tf = 1, so each word scores exactly its idf:  0.98 + 0.98 + 0.47 = 2.43
 C3: only "due", and C3 is a bit longer than average:              0.47 × 2.5 / (1 + 1.5 × 1.1875) = 0.42
 C2: shares no word ("labs" is not "lab"!)                         0
```

Ranking: C1, C3. Notice two lessons: exact words like "4" are a strength, and "labs" ≠ "lab" is a weakness
(real systems add **stemming**, cutting words to their root; we won't today).

**Strength:** exact terms, numbers, names, codes ("Lab 4", "O(V + E)", "GitHub Classroom").
**Weakness:** no shared words, no match. "Can I get extra time?" shares nothing with "grace days".

#### 3c. Hybrid search and Reciprocal Rank Fusion (RRF)

The two methods fail on *different* questions, so run both and combine the lists. Problem: their scores are on
different scales (cosine 0 to 1, BM25 0 to maybe 15), so you can't just add them. **RRF** ignores the scores and
uses only the **rank** (1st, 2nd, 3rd...):

```text
 RRF score of a chunk = sum over the lists of  1 / (60 + rank in that list)

 vector list:   1. Late work     2. Office hours   3. Regrade requests
 keyword list:  1. Grading       2. Late work      3. Lab 4

 Late work     = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325   ← in BOTH lists: wins
 Grading       = 1/(60+1)                         = 0.0164
 Office hours  = 1/(60+2)                         = 0.0161
 Lab 4, Regrade = 1/(60+3)                        = 0.0159
```

A chunk that both methods like rises to the top. The 60 is a damping constant from the original paper: it keeps
one list's first place from dominating. It's a default, not something to tune today.

#### 3d. Measuring retrieval: recall@k and MRR

You'll write a **golden set**: questions where *you* decided which section answers them. Then for each question,
you look at the ranked hits and note the **rank** of the first correct one (or "missed").

- **recall@k** = share of questions whose correct section is somewhere in the top k.
- **MRR** (mean reciprocal rank) = average of 1/rank (0 when missed). It rewards putting the right one *first*.

```text
 4 questions, rank of the right section:  1, 3, missed, 2
 recall@1 = 1/4 = 0.25      recall@3 = 3/4 = 0.75
 MRR = (1/1 + 1/3 + 0 + 1/2) / 4 = 0.46
```

Why recall@k matters for RAG: `ask` hands the model k chunks. If recall@4 is 0.80, then for 1 question in 5 the
model **never sees** the answer, and the best it can do is say "I don't know".

(Strictly, *recall* means "how many of all the relevant chunks did we find". Each of our questions has one right
section, so it becomes "did we find it in the top k", sometimes called **hit rate**. Interviewers use both names.)

#### 3e. Reranking (what it is; not built today)

A **reranker** is a second, slower model that reads the question and each candidate chunk *together* and scores how
well the chunk answers it. You retrieve 20–50 candidates cheaply, then rerank them and keep the best 4. It usually
beats plain hybrid search, but it's another model to run on a CPU. It's moved to the optional list in the roadmap;
today's eval is exactly what you'd use later to prove a reranker helps.

### 4. What we'll build (the parts and how they fit)

```text
 evals/retrieval_golden.jsonl  25 questions + the section that answers each (Claude drafts, YOU check)
          │
          ▼   py -m coursemate eval-retrieval notes/sample evals/retrieval_golden.jsonl
 coursemate/retrieval_eval.py
   build_index(): chunk + embed the notes into a MemoryStore (no Postgres needed)
   for each mode in vector, keyword, hybrid:
       for each question:  retrieve() ──► top 5 hits ──► rank of the right section
   recall@1, recall@3, recall@5, MRR, split by question kind (keyword-style vs paraphrase)
          │
 coursemate/retrieval.py  retrieve(question, mode)
   vector  ──► store.search()                       (what you have today)
   keyword ──► coursemate/keyword.py  KeywordIndex  (BM25, new)
   hybrid  ──► both, fused by coursemate/hybrid.py  rrf()  (new)
          │
          ▼
 notebooks/retrieval.ipynb   look at real vectors, try one question three ways, chart the results, read the misses
          │
          ▼
 coursemate search / ask --mode hybrid   (CourseMate uses the winner)
```

| New or changed file | What it is |
|---|---|
| `notes/sample/*.md` | 4 new sample notes, so search has something to get wrong (34 chunks instead of 17) |
| `evals/retrieval_golden.jsonl` | the golden set |
| `coursemate/keyword.py` | BM25 keyword search (you type `bm25_term_score` yourself) |
| `coursemate/hybrid.py` | RRF (you type `rrf` yourself) |
| `coursemate/retrieval.py` | `retrieve()`: one function for the three modes |
| `coursemate/retrieval_eval.py` | golden set loader, recall@k (you type `recall_at_k`), MRR, the eval, saved runs |
| `coursemate/store.py` | new `all_chunks()` on both stores (keyword search needs the text of every chunk) |
| `coursemate/rag.py`, `cli.py` | `--mode` for `search` and `ask`; new `eval-retrieval` command |
| `notebooks/retrieval.ipynb` | the lab notebook |
| `evals/runs/*.json` | saved eval runs, to compare versions |

**Same repo, on a branch.** This improves CourseMate itself, so you work on a branch `retrieval` in `coursemate`
and merge it with a pull request, like Tuesday's evals.

### 5. How we'll test it

1. **Unit tests, no model.** The three functions you type get tests with the **numbers from section 3** (2.43, the
   RRF example, recall 0.75). If your code gives the same numbers you got on paper, it's right.
2. **The eval, with the real embedding model.** `eval-retrieval` runs the 25 questions through all three modes.
   This tests *search quality*, not code.
3. **The fake embedder as a contrast.** `--provider fake` makes vectors from word overlap only, with no meaning.
   You'll run the eval with both and compare: that comparison shows you, in numbers, what "meaning" adds.
4. **CI** runs the unit tests, the eval and the whole notebook with the fake embedder, so nothing silently breaks.

### 6. New words (short definitions)

- **Retrieval:** finding the chunks to give the model. **Generation:** the model writing the answer.
- **Embedding / vector:** a list of numbers for a text; similar meaning → similar direction.
- **Cosine similarity:** how close two vectors' directions are (1 = same, 0 = unrelated).
- **Vector search (dense, semantic search):** rank chunks by cosine to the question.
- **Keyword search (lexical, sparse search):** rank chunks by shared words. **BM25** is the standard formula.
- **TF / IDF:** term frequency (how often a word is in this chunk) / inverse document frequency (how rare the word
  is across chunks).
- **Stopwords:** very common words ("the", "is", "when") removed before keyword scoring.
- **Stemming:** cutting words to their root ("labs" → "lab"). Not used today.
- **Hybrid search:** run both and combine. **RRF:** combine by rank, `1 / (60 + rank)`.
- **Reranker (cross-encoder):** a slower model that scores (question, chunk) pairs to reorder candidates.
- **Golden set:** questions with the answer *you* decided. **recall@k / hit rate:** share of questions with the
  right section in the top k. **MRR:** average of 1/rank of the first right hit.
- **Jupyter notebook (`.ipynb`):** a file of **cells**, each either text (markdown) or code. You run cells one by
  one and the output is saved under each cell. Great for exploring and showing results; bad for code you rely on,
  which belongs in `.py` files with tests.
- **Kernel:** the Python process behind a notebook that runs the cells. It must be your project's venv.

### 7. Today's Claude Code technique: working with notebooks

Claude Code can **read a notebook including its outputs** (tables, printed numbers) and **edit individual cells**
(it has a dedicated notebook-edit tool: add a cell, replace a cell, delete a cell). It can't click "Run" in your
browser, but it can run the whole notebook from the terminal with `py -m jupyter nbconvert --execute`, then read the
new outputs. So the loop is:

```text
 you run cells in the browser and look  ──►  you ask Claude about @notebooks/retrieval.ipynb
        ▲                                         (it sees code AND outputs, explains, adds cells)
        └──────────────── you re-run and look again ◄──┘
```

You'll use that to compare two runs side by side and have Claude explain the differences question by question.

### 8. Time and cost

About **2 hours** of hands-on time, plus reading. Split it over two sittings if you like (a natural break is after
Step 4). **Cost: $0.** Only `nomic-embed-text` runs locally; no chat model is needed.

---

## How to use this guide with Claude Code

Every step has:

- **Learn:** the idea. Read it first.
- **Predict:** write down what you expect *before* running anything. Being wrong is how you learn.
- **Prompt:** what to type into Claude Code in **plan mode** (`Shift+Tab` until you see `⏸ plan mode on`). The
  prompts are short on purpose: they say *what* and *why*. Read Claude's plan and ask "why?" about anything
  you don't understand before you approve. That conversation is part of the lesson.
- **You check:** what you verify yourself, with the file and function to open.
- **Done when:** a command whose output proves the step works.
- **Commit:** the message.

**Python is `py` on your machine.** Every command in this guide uses `py -m ...`, and inside Claude Code you type
`!py -m pytest`. CI on GitHub runs Linux, where it stays `python`.

---

## Step 0 — Get up to date, branch, install the notebook tools (10 min)

In Git Bash:

```bash
cd coursemate                 # wherever you cloned it
source .venv/Scripts/activate
git status
git pull
git switch -c retrieval
docker compose up -d
py -m pip install notebook matplotlib
py -m pytest -q
```

**What you should see:**
- `git status`: `nothing to commit, working tree clean` (if not, commit or stash first).
- `git pull` brings in this guide, `GUIDE-retrieval.md`.
- `git switch -c retrieval`: `Switched to a new branch 'retrieval'`.
- pip ends with `Successfully installed ...` (Jupyter is a big install, 1–2 minutes).
- pytest: `32 passed, 2 deselected` (with Docker running).

Check Ollama has the embedding model (start the Ollama app first if this fails):

```bash
ollama list
```

You should see `nomic-embed-text`. Then start Claude Code:

```bash
claude
```

**Prompt (plan mode):**
> On my machine Python runs as `py`, not `python` (in Git Bash and in your Bash tool). Update CLAUDE.md and the
> "Run it" section of README.md so every command uses `py -m ...`. Leave .github/workflows/ci.yml alone: CI runs
> on Linux, where it's `python`. Also add `notebook` and `matplotlib` as a new optional dependency group called
> `notebook` in pyproject.toml, next to `dev`. Then run `py -m pytest -q`.

Then let Claude run tests without asking each time: type `/permissions`, choose **Allow**, and add these two rules:
`Bash(py -m pytest:*)` and `Bash(py -m ruff:*)`.

**You check:** `CLAUDE.md` no longer says `python -m`. `pyproject.toml` has `notebook = [...]` under
`[project.optional-dependencies]`.

**Done when:** `!py -m pytest -q` shows `32 passed, 2 deselected`.

**Commit:** `Use py in docs; add notebook extras`

---

## Step 1 — Look inside an embedding, in a notebook (15 min)

**Learn.** Before measuring search, *see* what it works with. You'll embed four sentences and look at the numbers
and at the cosine between every pair. Read section 3a again if "cosine" feels fuzzy.

**Predict.** Here are the four sentences. Which pair will have the highest cosine (other than a sentence with
itself)? Which pair the lowest? Write it down.

```text
 s0 "Late projects lose 10% per day."
 s1 "How much is the penalty for handing in work after the deadline?"
 s2 "Merge sort runs in O(n log n) time."
 s3 "Office hours are on Tuesday afternoons."
```

**Prompt (plan mode):**
> Create notebooks/retrieval.ipynb, a Jupyter notebook for exploring retrieval. Only these cells for now:
> 1. Markdown: title "Retrieval lab: how CourseMate finds the right notes" and one line saying to run cells top to
>    bottom with Shift+Enter.
> 2. Code (setup): find the repo root (the parent folder if the notebook runs inside notebooks/), read the
>    COURSEMATE_PROVIDER environment variable (default "ollama"), and create FakeEmbedder() for "fake" or
>    OllamaEmbedder() otherwise. Print the model name and dimensions.
> 3. Markdown: "Part A. What an embedding looks like", one sentence of explanation.
> 4. Code: embed these 4 sentences with embed_documents: [the four sentences from section 3a's Predict list, typed
>    exactly]. Print the length of one vector and its first 5 numbers rounded to 3 decimals.
> 5. Code: print a 4×4 table of the cosine similarity between every pair, using cosine() from
>    coursemate.embeddings, with s0..s3 as row and column labels.
>
> Keep the code simple and commented for a beginner. Don't run it; I'll run it in Jupyter.

*(Copy the four sentences from the Predict box above into the prompt.)*

Now open the notebook. In a **second** Git Bash window (keep Claude Code open in the first):

```bash
cd coursemate
source .venv/Scripts/activate
py -m notebook notebooks/retrieval.ipynb
```

Your browser opens Jupyter. Top right should say **Python 3 (ipykernel)**. Click the first code cell and press
**Shift+Enter** on each cell in turn.

**What you should see:**
- `embedding model: nomic-embed-text | dimensions: 768`
- `first 5 numbers ...`: five small numbers, positive and negative, like `0.031, -0.042, ...`. None of them means
  anything on its own; only the direction of all 768 together does.
- A 4×4 table with `1.00` on the diagonal.

**You check:** compare the table with your prediction. Is s0–s1 (late penalty vs. handing in after the deadline,
almost no shared words) the highest pair? Is s2 (merge sort) low against everything? Notice that unrelated pairs
are **not** near 0 with real models; they often sit around 0.4–0.5. Only the **gap** between related and
unrelated pairs means something. That's why yesterday's `min_score` had to be measured.

Now try the technique. Back in Claude Code:

> Read @notebooks/retrieval.ipynb including its outputs. Explain the cosine table to me in plain words: which
> pairs are close, and why s0 and s1 can be close even though they share almost no words.

**Then try the contrast.** Stop Jupyter (`Ctrl+C` twice in its window), and start it with the fake embedder:

```bash
COURSEMATE_PROVIDER=fake py -m notebook notebooks/retrieval.ipynb
```

Run the cells again. Now s0–s1 is **0.00**: the fake embedder only matches identical words, and those two share
none that it counts. That is the difference between "search by words" and "search by meaning", in one table.
Stop Jupyter and start it again without `COURSEMATE_PROVIDER=fake`, run all cells (menu **Run → Run All Cells**),
and **save** (`Ctrl+S`), so the real outputs are stored in the file.

**Done when:** the notebook shows real `nomic-embed-text` outputs and is saved.

**Commit:** `Add retrieval notebook: look inside embeddings`

---

## Step 2 — More notes, and a golden set you check yourself (20 min)

**Learn.** The sample notes have 17 chunks. With k = 3, almost any method finds the right one among so few, so a
test would prove nothing. Four more notes (34 chunks) give search a chance to be wrong, especially with
**near-misses**: two "due" dates, two projects, "stable" in two sections.

The golden set needs **two kinds** of question, because the two methods have opposite strengths (section 3):

- **keyword:** uses the same exact words as the notes ("When is **Lab 4** due?").
- **paraphrase:** asks in a student's own words, sharing few words with the notes ("Can I get extra time without
  a penalty?" for the grace days rule).

Each row names the right section by **file and last heading**, not by chunk number, because chunk numbers change
whenever chunking changes and headings don't:

```json
{"id": "q12", "question": "When is Lab 4 due?", "expected": [{"source": "lecture-02-hash-tables.md", "section": "Lab 4"}], "kind": "keyword", "reviewed": false}
```

Same rule as Tuesday: Claude drafts, **you** check every label, and the code refuses unchecked rows.

**Prompt 1 (plan mode) — the notes:**
> Add 4 short sample notes to notes/sample/, in the same style as the existing lectures (a "# Lecture N: ..."
> title, 3 to 5 "## " sections, 2 to 4 sentences each, CS 201 Data Structures level):
> lecture-04-sorting.md (why sorting matters, insertion sort, merge sort, quicksort, stability),
> lecture-05-graphs.md (representing a graph, BFS, DFS, running time),
> lecture-06-recursion.md (recursion and base cases, memoization, dynamic programming),
> projects.md ("# CS 201 Projects": Project 1 text statistics with a hash map due week 6; Project 2 campus route
> planner with BFS due week 13, pairs allowed; a "Submitting work" section about GitHub Classroom and what to do
> if the autograder fails).
> Then update the two tests that count the sample files (tests/test_ingest.py and tests/test_cli.py) from 4 to 8,
> because the specification changed: there are now 8 sample files. Don't change any other test.

**Prompt 2 (plan mode) — the loader and its tests:**
> Create coursemate/retrieval_eval.py with Pydantic models `Expected` (source: str, section: str, the last heading)
> and `GoldenQuestion` (id, question, expected: non-empty list of Expected, kind: "keyword" or "paraphrase",
> reviewed: bool; extra fields forbidden), a `GoldenSetError` exception, and `load_golden(path)` that reads JSONL,
> skips blank lines, and raises GoldenSetError with the line number on a bad row, with the ids when any row is not
> reviewed, and on duplicate ids. Add tests in tests/test_retrieval_eval.py for: unreviewed rows refused (the
> message names the id) and duplicate ids refused. Run the tests.

**Prompt 3 (plan mode) — the draft data:**
> Read every file in @notes/sample/. Write evals/retrieval_golden.jsonl: 25 questions a CS 201 student might ask,
> each answered by exactly one section of these notes. About half "keyword" (uses the notes' exact terms, names or
> numbers) and half "paraphrase" (a student's own words, sharing as few words as possible with the section). Cover
> all 8 files, at least 2 questions each. Ids q01 to q25. Set "reviewed": false on every row. `section` is the
> text of the "## " heading only, exactly as written.
> Then add a test test_every_golden_question_points_at_a_real_section: chunk every sample note with
> chunk_document and check that each expected (source, section) exists. Don't load_golden in that test (rows aren't
> reviewed yet); read the JSON lines directly. Run the tests.

**Commit now, before you review:** `Add 4 sample notes and the retrieval golden set draft (unreviewed)`

**You check (in your editor, not with Claude).** Open `evals/retrieval_golden.jsonl`. For each row:

1. **Cover the answer.** Read only `question` and decide which file and section answers it. Then compare.
2. Is the `kind` honest? A "paraphrase" that repeats three words from the section is really a keyword question:
   reword the question or change the kind.
3. Could **two** sections answer it? Then reword the question so only one can.
4. Set `"reviewed": true`.

**Done when:**

```bash
git diff --stat
py -m pytest -q
```

`git diff --stat` shows what you changed during review (write the number down: "I changed N of 25", it's an
interview line), and pytest shows `0 failed`.

**Commit:** `Review retrieval golden set`

---

## Step 3 — Measure today's vector search: recall@k (20 min)

**Learn.** Re-read section 3d. You'll type `recall_at_k` yourself: it's the one function where today's measuring
idea lives. The rest (running each question, finding the rank of the right section) Claude writes.

To keep the eval simple and repeatable, it doesn't use Postgres: it chunks and embeds `notes/sample` into a
`MemoryStore` each run (34 chunks, a few seconds). Your contract tests from yesterday already prove `MemoryStore`
searches exactly like `PgVectorStore`, so the numbers are the same.

**Predict.** Out of 25 questions, how many do you think vector search gets in the top 3? Which kind will it miss
more, keyword or paraphrase?

**Prompt 1 (plan mode):**
> In coursemate/retrieval_eval.py add:
> - `is_relevant(hit, expected)`: true when hit.source equals an expected source and hit.heading equals the
>   section or ends with " > " + section.
> - `first_relevant_rank(hits, expected)`: 1-based rank of the first relevant hit, or None.
> - `recall_at_k(ranks: list[int | None], k: int) -> float`: leave the body as `raise NotImplementedError`. I'll
>   write it.
> - `mean_reciprocal_rank(ranks)`.
> Write tests first, using the example from my guide: ranks [1, 3, None, 2] give recall@1 = 0.25, recall@2 = 0.5,
> recall@3 = 0.75, and an empty list gives 0.0; MRR of [1, 2, None, 4] is (1 + 0.5 + 0 + 0.25) / 4. Run the
> tests and show me the recall tests failing.

**Now you type** the body of `recall_at_k` in `coursemate/retrieval_eval.py`, in your editor:

```python
def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Share of questions whose right section is somewhere in the top k."""
    if not ranks:
        return 0.0
    found = sum(1 for rank in ranks if rank is not None and rank <= k)
    return found / len(ranks)
```

Read it line by line: `rank is not None` skips the misses, `rank <= k` keeps only the top k.

```bash
py -m pytest -q tests/test_retrieval_eval.py
```

**Prompt 2 (plan mode):**
> Add an eval for vector search only, for now:
> - `build_index(notes, embedder) -> MemoryStore` using ingest().
> - `evaluate(golden, embedder, store)`: for every question, search the top 5 and record the rank of the first
>   relevant hit. Return a small dataclass `ModeResult(mode, ranks: dict[question id, rank or None])` with methods
>   recall(k) and mrr().
> - `format_table(results, golden)`: one row per mode with recall@1, recall@3, recall@5, MRR, and recall@3 for
>   each kind separately; then a line with the number of questions of each kind.
> - `misses(results, golden, k=3)`: one line per question that some mode missed in the top 3.
> - A CLI command `eval-retrieval NOTES GOLDEN [--show-misses]` in coursemate/cli.py that needs no database, and
>   prints "embedding model: <name>" and the table.
> Add a test that runs evaluate() on the real sample notes and golden set with FakeEmbedder (recall@1 ≤ recall@3 ≤
> recall@5), and a CLI test with --provider fake. Run the tests.

**You check:** `evaluate()` in `coursemate/retrieval_eval.py`: it searches **5** deep, then the table reads off
recall@1, @3 and @5 from the same ranks. Ask Claude: *"Why search 5 deep once instead of searching 1, 3 and 5
separately?"* (Same results, a third of the work: the top 3 is just the first 3 of the top 5.)

**Done when:** your first real number:

```bash
py -m coursemate eval-retrieval notes/sample evals/retrieval_golden.jsonl --show-misses
```

You see `embedding model: nomic-embed-text`, one row `vector | ...`, and the missed questions. Compare with your
prediction. Read every miss: what did the question say, and what does the right section say?

**Commit:** `Measure vector retrieval with recall@k`

---

## Step 4 — Keyword search with BM25 (20 min)

**Learn.** Re-read section 3b and its worked example. Keyword search needs the **text** of every chunk, not just a
way to find the nearest vector, so both stores get a new method `all_chunks()`. The `KeywordIndex` then counts
words in Python. That's fine for a few hundred chunks; with millions you'd let Postgres do it (`tsvector`).

**Predict.** Which of your golden questions will keyword search get right that vector search missed?

**Prompt 1 (plan mode):**
> Add `all_chunks() -> list[Chunk]` to the VectorStore protocol, MemoryStore and PgVectorStore in
> coursemate/store.py, ordered by source then chunk index. Add one contract test in tests/test_store.py that runs on
> both stores, like the existing ones.

**Prompt 2 (plan mode):**
> Create coursemate/keyword.py:
> - `tokenize(text)`: lowercase words and numbers (regex [a-z0-9]+), minus a small English stopword set (include
>   when, is, the, what, how, do, i, my, a, an, to, of, for, in, on, and, can).
> - `bm25_term_score(tf, df, n_docs, doc_len, avg_len, k1=1.5, b=0.75) -> float`: leave the body as
>   `raise NotImplementedError`. I'll write it.
> - `class KeywordIndex`: built from a list of Chunks (heading + text), keeps word counts per chunk, chunk lengths,
>   average length and document frequencies; `search(query, k) -> list[Hit]` sums bm25_term_score over the query's
>   distinct words, skips chunks scoring 0, sorts by score.
> Write tests first in tests/test_keyword.py, using my guide's worked example: tokenize("When is Lab 4 due?") ==
> ["lab", "4", "due"]; bm25_term_score(tf=2, df=1, n_docs=3, doc_len=4, avg_len=4) equals
> ln(1 + 2.5/1.5) × 2 × 2.5 / (2 + 1.5) ≈ 1.40; rare words beat common words; repeating a word helps but less than
> proportionally; a longer chunk scores lower for the same count; and the 3-chunk example ("Lab 4 is due Friday",
> "Labs are worth 30%", "Project 2 is due week 13") ranks the first chunk with a score of about 2.43, the third
> about 0.42, and doesn't return the second. Run the tests and show me them failing.

**Now you type** the body of `bm25_term_score` in `coursemate/keyword.py`. It's the formula from section 3b,
nothing more:

```python
def bm25_term_score(tf: int, df: int, n_docs: int, doc_len: int, avg_len: float,
                    k1: float = 1.5, b: float = 0.75) -> float:
    """How much one query word adds to one chunk's score."""
    if tf == 0:
        return 0.0
    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    length_norm = 1 - b + b * doc_len / avg_len
    return idf * tf * (k1 + 1) / (tf + k1 * length_norm)
```

Match each line to the formula: `idf` is "rare words count more"; `length_norm` is 1 for an average-length chunk,
more for a long one; the last line is "repeats help, but less and less".

```bash
py -m pytest -q tests/test_keyword.py
```

When they pass, your code computes the same 2.43 you computed on paper.

**Done when:** `!py -m pytest -q` shows `0 failed` (Claude may have formatted your function differently; that's
fine). Run `!py -m ruff format .` if the format check complains.

**Commit:** `Add BM25 keyword search`

> Good place to stop for today if you're splitting the project. Next time, start at Step 5.

---

## Step 5 — Hybrid search with RRF, and the three-way comparison (20 min)

**Learn.** Re-read section 3c. One function, `retrieve(question, mode)`, will hide the three ways of searching, so
the eval, `search` and `ask` all call the same code.

For hybrid, each method first finds **20 candidates**, then RRF fuses the two lists and we keep the top k. Why 20
and not k? A chunk that's 6th in vector search and 2nd in keyword search should be able to win, but it can only be
fused if it's in both lists.

**Predict.** Will hybrid beat both methods on recall@3? Or land in between?

**Prompt 1 (plan mode):**
> Create coursemate/hybrid.py with `rrf(rankings: list[list[Hit]], c: int = 60) -> list[Hit]`. Leave the body as
> `raise NotImplementedError`; I'll write it. Write tests first in tests/test_hybrid.py with my guide's example:
> vector [A, B] and keyword [B, C] fuse to [B, A, C] with scores 1/62 + 1/61, 1/61 and 1/62; two lists that rank
> the same two chunks in opposite order give them equal scores even if the original scores are very different
> (0.51 vs 900); and empty lists give []. A chunk is identified by (source, heading, text). Show me the tests
> failing.

**Now you type** the body of `rrf` in `coursemate/hybrid.py` (add `from dataclasses import replace` at the top):

```python
def rrf(rankings: list[list[Hit]], c: int = 60) -> list[Hit]:
    """Each list gives a chunk 1 / (c + rank); chunks high in several lists win."""
    scores: dict[tuple[str, str, str], float] = {}
    first_seen: dict[tuple[str, str, str], Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.source, hit.heading, hit.text)
            scores[key] = scores.get(key, 0.0) + 1 / (c + rank)
            first_seen.setdefault(key, hit)
    fused = [replace(first_seen[key], score=score) for key, score in scores.items()]
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused
```

`enumerate(..., start=1)` gives ranks 1, 2, 3; `scores.get(key, 0.0) + ...` adds up a chunk's share from every
list; `replace(...)` makes a copy of the `Hit` with the new score (`Hit` is frozen, so it can't be changed).

```bash
py -m pytest -q tests/test_hybrid.py
```

**Prompt 2 (plan mode):**
> Create coursemate/retrieval.py with `Mode = Literal["vector", "keyword", "hybrid"]`, `MODES`, a dataclass
> `Retrieval(hits, best_vector_score)` and `retrieve(query, embedder, store, k, mode="vector", candidates=20)`:
> always run vector search (k deep for vector mode, max(k, candidates) for the others) and keep the top cosine
> score as best_vector_score; keyword mode uses KeywordIndex(store.all_chunks()); hybrid fuses the vector and
> keyword lists with rrf. Then make evaluate() in retrieval_eval.py run all three modes, and update its tests.
> Also add `save_run(path, results, model)` that writes recall@1/3/5, MRR and the per-question ranks of each mode
> to JSON, and a `--save PATH` option on eval-retrieval. Run the tests.

**You check:** `retrieve()` in `coursemate/retrieval.py`. Ask Claude: *"Why does keyword mode still run vector
search?"* (To get `best_vector_score` for the "I don't know" gate in Step 7. BM25 and RRF scores aren't on the
cosine scale, so yesterday's `min_score` only makes sense for the cosine.)

**Done when:** you save two runs, the real model and the fake one:

```bash
py -m coursemate eval-retrieval notes/sample evals/retrieval_golden.jsonl --show-misses --save evals/runs/nomic.json
py -m coursemate --provider fake eval-retrieval notes/sample evals/retrieval_golden.jsonl --save evals/runs/fake.json
```

Each prints a table with three rows: `vector`, `keyword`, `hybrid`. When I tested this guide with the fake
embedder it looked like this (your nomic numbers will be different, that's the point):

```text
          mode |       recall@1 |       recall@3 |       recall@5 |            MRR |    r@3 keyword | r@3 paraphrase
        vector |           0.64 |           0.76 |           0.84 |           0.72 |           1.00 |           0.54
       keyword |           0.76 |           1.00 |           1.00 |           0.88 |           1.00 |           1.00
        hybrid |           0.80 |           0.92 |           1.00 |           0.87 |           1.00 |           0.85
```

With the fake embedder, "vector" is really just a cruder word match, so keyword wins: there's no meaning to add.
With nomic, look at the **paraphrase** column: that's where meaning should show up.

**Commit:** `Add hybrid search (RRF) and compare three retrieval modes`

---

## Step 6 — Compare in the notebook, and read the misses with Claude (20 min)

**Learn.** A table tells you *how much*; only reading the misses tells you *why*. This is where the notebook
technique pays off: Claude adds the cells, you run them, then Claude reads the outputs with you.

**Prompt (plan mode):**
> Add these cells to the end of @notebooks/retrieval.ipynb (keep the existing ones):
> - Markdown "Part B. One question, three ways to search", then code: build_index on notes/sample, print the number
>   of chunks, and for the question "Can I get extra time on an assignment without a penalty?" print the top 3 of
>   each mode as "score  source > last heading".
> - Markdown "Part C. Measure all three on the golden set", then code: load the golden set, evaluate, print
>   format_table; then a matplotlib line chart of recall@k for k = 1, 3, 5, one line per mode, y from 0 to 1.05.
> - Markdown "Part D. Read the misses", then code: print misses(results, golden, k=3).
> - Markdown "Part E. Compare saved runs", then code: load every evals/runs/*.json and print one row per run and
>   mode: run name, embedding model, mode, recall@1, @3, @5, MRR.
> - Markdown "What I learned" with a placeholder line for me to fill in.

In Jupyter: **Run → Run All Cells**, look at every output, save.

Then, in Claude Code:

> Read @notebooks/retrieval.ipynb with its outputs, and @evals/runs/nomic.json and @evals/runs/fake.json. For each
> question that vector and keyword search disagree on (one finds it in the top 3, the other doesn't), show me the
> question, its kind, and in one sentence why one method found it and the other didn't. Then: did hybrid fix those
> cases, and where did it make things worse?

**You check:**
- Part B: for the "extra time" question, which mode put `Late work` first? Does that match section 3?
- Part E: is nomic's **paraphrase** recall higher than fake's? If not, your "paraphrase" questions may share too
  many words with the notes. That's a golden-set finding, not a failure.
- If keyword search wins on almost everything with nomic too, ask yourself whether your questions are realistic:
  students rarely use the exact words of your notes. Is the golden set fair to both methods?

Write 3–5 sentences under **What I learned** in your own words: which mode won, on which kind of question, and why.
Save the notebook.

**Done when:** every cell has output, "What I learned" is filled in, and this runs without errors (it's what CI
will run, with the fake embedder):

```bash
COURSEMATE_PROVIDER=fake MPLBACKEND=Agg py -m jupyter nbconvert --to notebook --execute notebooks/retrieval.ipynb --output-dir /tmp
```

You should see `[NbConvertApp] Writing ... bytes to ...`. (This runs a copy, so your saved real outputs stay.)

**Commit:** `Compare retrieval modes in the notebook`

---

## Step 7 — Let CourseMate use the winner (15 min)

**Learn.** Now `search` and `ask` get a `--mode` option, and **your numbers** pick the default. One trap: the
"I don't know" gate in `answer()` compares the top score with `min_score`, which was calibrated on **cosine**
scores. An RRF score is around 0.03 and a BM25 score can be 12, so the gate would break. That's why `retrieve()`
returns `best_vector_score`: the gate keeps using the cosine, whatever mode found the chunks.

**Prompt (plan mode):**
> Add `--mode {vector,keyword,hybrid}` to the `search` and `ask` commands in coursemate/cli.py, using retrieve().
> Add a `mode` parameter to answer() in coursemate/rag.py: it gets its hits from retrieve(), and the min_score gate
> must use best_vector_score, not hits[0].score. Add tests: in hybrid mode a far-off question is still refused
> without calling the model; in hybrid mode a good question is answered; and a CLI test (needs the database) that
> "When is Lab 4 due?" puts the Lab 4 section first in all three modes with the fake provider. Run all tests.

**You check:** `answer()` in `coursemate/rag.py`. The gate line reads `best_vector_score`, not `hits[0].score`.
Ask Claude: *"Show me what would happen to the gate in hybrid mode if it used hits[0].score."*

Now decide the default. Look at **recall@3 and MRR** for nomic in `evals/runs/nomic.json`. If hybrid is best or tied,
change the `--mode` default in `coursemate/cli.py` to `"hybrid"` (one word, by hand). If vector is still best on
your data, keep `"vector"` and write down why in the README in Step 8. Either is a correct, data-driven decision.

**Done when:** re-ingest the bigger notes into Postgres and try it:

```bash
py -m coursemate ingest notes/sample --reset
py -m coursemate search "Can I get extra time on an assignment without a penalty?" -k 3 --mode vector
py -m coursemate search "Can I get extra time on an assignment without a penalty?" -k 3 --mode hybrid
py -m pytest -q
```

The two searches print 3 hits each; compare their order. In hybrid mode the score column shows RRF scores (around
0.03), not cosines, so don't compare it with the vector scores. pytest shows `0 failed`.

**Commit:** `Add --mode to search and ask; default chosen from the retrieval eval`

---

## Step 8 — CI, README, interview questions, pull request (15 min)

**Prompt 1 (plan mode):**
> In .github/workflows/ci.yml: install with `pip install -e ".[dev,notebook]"`, and after the existing steps add
> (1) "Retrieval eval runs (fake embedder, plumbing only)": `python -m coursemate --provider fake eval-retrieval
> notes/sample evals/retrieval_golden.jsonl`, and (2) "Notebook runs top to bottom (fake embedder)": with env
> COURSEMATE_PROVIDER=fake and MPLBACKEND=Agg, `python -m jupyter nbconvert --to notebook --execute
> notebooks/retrieval.ipynb --output-dir /tmp`. Linux CI uses `python`, not `py`.

**Prompt 2 (plan mode):**
> Update README.md: in "Run it" add the eval-retrieval command and `--mode`. In "Design decisions" add bullets:
> retrieval measured on its own with a reviewed golden set (recall@1/3/5, MRR, split by keyword vs paraphrase
> questions); golden labels are file + section, not chunk ids; BM25 keyword search written by hand in Python; hybrid
> with RRF over 20 candidates per method; the "I don't know" gate stays on the cosine score in every mode; the
> default mode, chosen from my numbers in evals/runs/nomic.json (put my real recall@3 numbers in a small table).
> In "Known limitations" replace the two lines about retrieval not being measured and being vector-only with: 25
> questions is a small set (one question is 4 points) written for 8 short notes; no stemming in keyword search
> ("labs" ≠ "lab"); BM25 runs in Python over all chunks (fine for hundreds, use Postgres full-text search for more);
> no reranker yet; the eval uses the sample notes, not my private notes. Remove the empty "How I used Claude Code"
> section; the roadmap no longer asks for it.

**Prompt 3 (plan mode):**
> Add 5 questions with model answers (5–8 sentences each, plain language) to @INTERVIEW.md under
> "## Retrieval (Week 2 Thu)", based on this code and my results in evals/runs/nomic.json:
> 1. What's the difference between keyword (BM25) and vector search? Give one question from my golden set where
>    each one wins, and why.
> 2. How does hybrid search with Reciprocal Rank Fusion work, and why combine by rank instead of adding scores?
> 3. How did you measure retrieval? Explain recall@k and MRR, and why you measure retrieval separately from the
>    final answers.
> 4. What is a reranker, when would you add one, and how would you prove it helps?
> 5. LLM system design: users of a RAG system over 10,000 documents complain that answers are "often wrong". How do
>    you find out whether retrieval or generation is the problem, and what would you change in each case?

**You check:** fill your own numbers into the README table if Claude left placeholders. Read every interview
answer, and rewrite Q1 in your own words with your own example.

**Done when:**

```bash
py -m ruff check . && py -m ruff format --check . && py -m pytest -q
```

all pass. **Commit:** `Document retrieval in README and interview Q&A`

```text
!git push -u origin retrieval
!gh pr create --fill
!gh pr checks --watch
!gh pr merge --merge --delete-branch
!git switch main
!git pull
```

All CI steps should be green, including the two new ones. If the notebook step fails, open the failed step in
GitHub, read the error at the bottom, and ask Claude to fix it.

---

## Check your understanding (for the day in between)

Answer in your own words first, then open the answer.

1. A student asks "When is Lab 5 due?" and vector search returns the Lab 4 section first. Why is that likely, and
   which method would do better?
   <details><summary>Answer</summary>Vector search compares meaning, and "Lab 4 due date" and "Lab 5 due date" mean
   almost the same thing to an embedding model, so it ranks Lab 4 high. Keyword search scores the exact token "5"
   (or "4"), so it can tell them apart. Hybrid keeps both signals.</details>
2. In BM25, why does the word "due" add less than the word "lab" in the worked example?
   <details><summary>Answer</summary>"due" appears in 2 of the 3 chunks, so its IDF is low (0.47); "lab" appears
   in 1 of 3, so its IDF is high (0.98). Rare words tell you more about which chunk is the right one.</details>
3. Why can't you just add the cosine score and the BM25 score to combine them?
   <details><summary>Answer</summary>They're on different scales: cosine is 0 to 1, BM25 can be 0 to 15 or more,
   so BM25 would drown out the cosine. RRF uses ranks, which are on the same scale for every method.</details>
4. recall@3 is 0.80 and `ask` uses k = 4. What's the most that the answers can be right, roughly, and why?
   <details><summary>Answer</summary>About 80–84%: for at least 1 question in 5, the right section isn't among the
   chunks given to the model (recall@4 is at least recall@3), so the model can't answer from the notes. Retrieval
   puts a ceiling on answer quality.</details>
5. Why are golden labels "file + section" instead of chunk numbers?
   <details><summary>Answer</summary>Chunk numbers change whenever you change chunking (size, overlap, more notes);
   headings don't. Labels that survive a chunking change let you compare two chunking versions with the same
   golden set.</details>
6. Why does the "I don't know" gate use the vector score even in hybrid mode?
   <details><summary>Answer</summary>`min_score` was calibrated on cosine scores. RRF scores are around 0.01–0.03
   and BM25 scores are unbounded, so comparing them with a cosine threshold would refuse everything or nothing.
   </details>
7. With the fake embedder, keyword search beat "vector" search. Does that mean keyword search is better?
   <details><summary>Answer</summary>No. The fake embedder only hashes words, so it's a worse keyword search, with no
   meaning. The comparison that matters is with a real embedding model, on realistic questions.</details>

---

## When you're done

Tell the roadmap chat **"done"**. It will review the repo, run the tests and tick this row.

## Level up

**Explore in a notebook, keep in a module, and let Claude read the outputs.** A notebook is the fastest way to *see*
what a model does (vectors, scores, misses), and Claude Code reads its outputs and edits single cells, so you can
ask "why did this question miss?" about real numbers instead of guesses. But code you rely on belongs in `.py`
files with tests; the notebook should only *call* that code, as yours does. Save every run you might compare
(`--save evals/runs/NAME.json`) and have Claude diff runs question by question. And run the notebook in CI
(`nbconvert --execute`), because a notebook nobody re-runs quietly stops working.
