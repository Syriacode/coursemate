# Phase 3 Tue — Observability for CourseMate: traces, latency and cost, and a cache (step-by-step guide)

> Read the **Project overview** first, on its own, before you open a terminal. It teaches today's ideas with small
> examples you can work out on paper. The steps start at Step 0.
>
> **Two days.** **Part A** (day 1, Steps 0–6, about 2–2.5 hours) ends at a working, tested, committed checkpoint:
> every answer and every agent run is a trace you can open in Langfuse. **Part B** (day 2, Steps 7–11, about
> 2–2.5 hours) adds the cache, the latency and cost report with budgets, a small dashboard, a real run on
> `qwen2.5:3b`, the README, the interview questions and the pull request. On day 2, re-read this overview and answer
> the **Check your understanding** questions at the end **before** you start Part B.

---

## Project overview

### 1. What you've built so far (in plain words)

- **CourseMate RAG (Nov 11) and better retrieval (Nov 12):** your course notes are cut into **chunks**; each chunk
  becomes a vector (a list of 768 numbers) with the local model `nomic-embed-text`, and the vectors live in
  **Postgres + pgvector** (in Docker, port 5433). A question is turned into a vector the same way, the closest chunks
  come back (`retrieve()` in `coursemate/retrieval.py`), and `qwen2.5:3b` answers **only from those chunks**, citing
  them as `[1]`, `[2]`. If the best chunk scores below `min_score` (0.55), CourseMate says "I don't know" **without**
  calling the model.
- **CourseMate API (Nov 16):** the same answers behind a **FastAPI** server (`coursemate/api.py`). The answer is
  **streamed**: words reach the page as they're written, over **SSE** (server-sent events: `token` events, then one
  `done` event with the sources, tokens, timings and `cost_usd`). The code that does one answer is
  `stream_answer()` in `coursemate/chat.py`, an **async generator**: a function that `yield`s events one by one and
  pauses between them. Three API formats (Ollama, OpenAI, Anthropic) sit behind one `StreamingLLM` interface
  (`coursemate/streaming.py`), and `coursemate/pricing.py` turns tokens into dollars.
- **CourseMate agent (Nov 20):** `run_agent()` in `coursemate/agent.py` is a loop: ask the model, run the tools it
  asks for (`search_notes`, `make_quiz`, `draft_announcement`, `send_announcement`), send the results back, repeat.
  `run_tool()` holds the **approval gate**: nothing is sent without your "yes".
- **CourseMate MCP server (Nov 22):** the same four tools offered to Claude Code over MCP.

So CourseMate works. But when an answer takes 25 seconds, or says "I don't know" when it shouldn't, or the agent
does something odd, today you can only guess **where** the time went and **what** the model saw. Production AI teams
answer those questions with **observability**: every request leaves a record of each step (what went in, what came
out, how long it took, what it cost). Today you add that, then use it to make CourseMate cheaper and faster with a
**cache**, and you prove the improvement with numbers.

| Part | Question | What you'll see |
|---|---|---|
| A | Where does the time go in one answer? | one **trace** per answer in **Langfuse**: `chat` → `retrieve`, `llm`, `check`, with the prompt, the chunks, tokens, cost and timings |
| A | What did the agent actually do? | one trace per agent run: every model call and every tool call, the declined send marked as a warning |
| B | What do we pay, and how slow are we, over many answers? | `coursemate stats`: p50/p95 latency, time to first word, tokens, cost per answer, **budgets** that pass or fail, and a one-page dashboard |
| B | Can we skip the model for repeated questions? | a **cache**: the second time a question is asked, the answer comes back in milliseconds, with 0 tokens |

### 2. The ideas you need today (work these on paper)

#### 2a. Traces and spans: a stopwatch with a family tree

A **span** is one timed step: it has a **name**, a **start** time, an **end** time, and **attributes** (labels like
`model = qwen2.5:3b` or `input_tokens = 1200`). A span can have a **parent**: the step it is part of. All the spans
of one request share a **trace id**; together they are a **trace**. The span without a parent is the **root**.

One CourseMate answer, drawn as a trace (numbers made up but realistic for your CPU):

```text
 trace 4bf9…  "When is Lab 4 due?"
 time (ms) →   0        2,000      4,000      6,000      8,000      9,500
 chat          |==========================================================|  9,500 ms   (root, no parent)
  retrieve     |=|                                                           120 ms    parent: chat
  llm            |========================================================|  9,300 ms   parent: chat
                 ↑ first word at 4,000 ms after the llm span started
  check                                                                   |  1 ms      parent: chat
```

Read it like this: of 9,500 ms, retrieval took 120 ms and the model took 9,300 ms, so **making retrieval faster is
pointless**; the model is 98% of the wait. The student saw the first word after about 120 + 4,000 = 4,120 ms. The
gap that's left (9,500 − 120 − 9,300 − 1 = **79 ms**) is our own Python code. That's the whole point of a trace: it
turns "it's slow" into "**this** step is slow".

Three rules that every span follows, and that you'll type into code in Step 3:

1. **A child points at its parent.** `retrieve`, `llm` and `check` carry the span id of `chat`, and all four carry the
   same trace id.
2. **A child lies inside its parent in time.** `chat` starts first and ends last.
3. **A span always ends**, even when the step fails. A failed step ends with status **ERROR** and the error message
   attached. A span that never ends is never sent, so the one request you most need to see (the failed one) would
   be missing.

#### 2b. Cost and latency per answer

You already compute cost in `coursemate/pricing.py` (Nov 16):

```text
 cost = input_tokens × input price per token + output_tokens × output price per token
 one answer: 1,200 tokens in (system prompt + 4 chunks + question), 150 tokens out

 claude-haiku-5-5   ($0.10 / $0.50 per million):  1,200 × 0.10/1M + 150 × 0.50/1M = 0.000120 + 0.000075 = $0.000195
 claude-sonnet-5-5  ($2.00 / $10.00 per million): 1,200 × 2.00/1M + 150 × 10.00/1M = 0.002400 + 0.001500 = $0.003900
 qwen2.5:3b on your PC:                           $0 (but you pay in time: your CPU is slow)
```

Per 1,000 answers that's about **$0.20** on Haiku and **$3.90** on Sonnet. Notice that input tokens are most of the
bill on Haiku (0.000120 of 0.000195): **the retrieved chunks cost more than the answer**. A **budget** is a number
you promise not to cross, written down so a program can check it: "cost per answer ≤ $0.001" or "95% of answers
within 15 seconds". A trace records tokens and cost on every model call, so the budget can be checked on real
traffic instead of a guess.

Latency has two numbers in a streaming app (Nov 16):

- **Time to first token (TTFT):** how long the student stares at nothing. This is what *feels* slow.
- **Total time:** when the answer is complete (and checked: CourseMate only knows the citations are valid at the end).

#### 2c. Percentiles: why p95 and not the average

Ten answer times in ms, as they came in:

```text
 1200, 900, 15000, 1100, 1000, 950, 1300, 1250, 980, 20000
 sorted:  900, 950, 980, 1000, 1100, 1200, 1250, 1300, 15000, 20000
 rank:     1    2    3     4     5     6     7     8      9     10
```

The **nearest-rank percentile** p: sort, then take the value at rank **⌈p/100 × n⌉** (⌈ ⌉ means "round up").

- **p50** (the median): ⌈0.50 × 10⌉ = 5 → **1,100 ms**. Half the answers are this fast or faster.
- **p90**: ⌈0.90 × 10⌉ = 9 → **15,000 ms**.
- **p95**: ⌈0.95 × 10⌉ = ⌈9.5⌉ = 10 → **20,000 ms**. 95% of answers are this fast or faster.
- **Mean**: 43,680 / 10 = **4,368 ms**. That's slower than 8 of the 10 answers and faster than the other 2; it
  describes **nobody's** experience.

That's why latency budgets use **p95** (or p99): they're about the slow answers real students get. One more trap you
will meet in Step 10: if half the answers come from a **cache** in 50 ms, p50 looks wonderful even when every real
model answer is still 15 seconds. So the report shows **cached and not-cached answers separately**.

#### 2d. Caching: skip the model when you already know the answer

A **cache** keeps results you already computed, so you don't compute them again. For an LLM app the expensive
result is the model's reply. Two questions decide whether a cache is safe:

**1. What goes into the key?** The key must change whenever the reply could change, and stay the same otherwise.
Today's cache sits **right in front of the model**, so its key is built from exactly what the model receives:

```text
 key = sha256( provider, model, messages, temperature, max_tokens )      messages = system prompt + chunks + question
```

Because the retrieved **chunks are inside the messages**, a change to your notes changes the chunks, which changes
the key: an old answer can't come back for new notes. A new system prompt changes the key too. Trace these on paper:

| Request | Same key as "When is Lab 4 due?" on qwen2.5:3b? | Why |
|---|---|---|
| the same question, same notes, an hour later | **yes** → cache hit | nothing the model sees changed |
| `When is Lab 4 due? ` (an extra space) | no | the text differs by one character; an **exact** cache only hits exact repeats |
| the same question to `llama3.2:3b` | no | another model may answer differently |
| the same question, `max_tokens` 50 | no | a shorter limit can cut the reply |
| a follow-up "And Lab 5?" after it | no | the history is in the messages |
| the same dict written in a different key order | **yes** | we write the JSON with `sort_keys=True`, so the order of keys can't matter |

**sha256** turns any text into a 64-character hex fingerprint; the same text always gives the same fingerprint, and
a one-character change gives a completely different one. We store the fingerprint, not the whole prompt.

**Only at temperature 0.** At temperature 0 the model picks the most likely word every time (lab 1, Nov 14), so the
same messages give (almost) the same reply, and replaying it is honest. At temperature 0.7 the reply is *supposed* to
differ each time, so the cache never stores it.

**2. When does an entry go away?** Two rules:

- **TTL (time to live):** an entry older than `ttl` seconds is treated as missing. (What could change the reply
  without changing the key? You run `ollama pull qwen2.5:3b` and get new weights under the same name. TTL limits
  how long you'd keep serving the old model's answers.)
- **LRU (least recently used):** the cache holds at most `max_entries`; when it's full, the entry used longest ago is
  dropped.

Trace this timeline with `ttl = 60` seconds and `max_entries = 2` (it's the test in Step 8):

```text
 t=0   put A                       cache: A(0)
 t=10  put B                       cache: A(0) B(10)
 t=20  get A  → HIT                cache: B(10) A(0)        A is now the most recently used
 t=30  put C  → full: drop B       cache: A(0) C(30)        B was used longest ago
 t=30  get B  → MISS               (it was dropped)
 t=61  get A  → MISS               61 − 0 = 61 > 60: expired, removed
 t=61  get C  → HIT                61 − 30 = 31 ≤ 60
 hits 2, misses 2
```

**A cache hit costs 0 tokens and almost 0 time**, but retrieval still runs (we need the chunks to build the key), so
a hit takes about as long as embedding one question: tens of milliseconds instead of 10 seconds.

**Why not a "semantic" cache?** Some systems match *similar* questions by embedding them: if a new question's vector
is close enough to a cached one (say cosine ≥ 0.95), return the cached answer. On paper: "When is Lab **4** due?" and
"When is Lab **5** due?" differ by one character; their embeddings are almost identical (they share every word but
one), so a semantic cache would happily give the Lab 4 date to the Lab 5 question. For course facts, being wrong is
worse than being slow, so today's cache is **exact**. (You'll try the semantic idea as a thought experiment in a
forked Claude Code session in Step 7.)

#### 2e. OpenTelemetry and Langfuse: who does what

```text
 your code ──spans──▶ OpenTelemetry SDK ──batch──▶ exporter ──OTLP over HTTP──▶ Langfuse (Docker, localhost:3000)
                       (in your process)              │
                                                      └──▶ JSONL exporter ──▶ logs/spans.jsonl ──▶ coursemate stats
```

- **OpenTelemetry (OTel)** is the open standard for traces: a Python library to create spans, plus a wire format
  (**OTLP**) to send them. It doesn't care where they go.
- An **exporter** sends finished spans somewhere. A **BatchSpanProcessor** collects them and sends them every few
  seconds in a background thread, so an answer never waits for the network. (That's why a trace shows up in Langfuse
  a few seconds **after** the answer, and why a program must **flush** the batch before it exits.)
- **Langfuse** is an open-source LLM observability app: it receives spans, stores them, and shows them as traces,
  sessions and dashboards, with LLM-specific views (prompt, reply, tokens, cost). It understands certain **attribute
  names**: `langfuse.observation.type = generation` makes a span show as a model call; `gen_ai.usage.input_tokens` are
  its input tokens. You'll use the table in section 3.
- **Why OTel and not the Langfuse Python SDK?** The Langfuse SDK is built on OTel and is nicer to use (`@observe`),
  but it hides what's happening. Plain OTel shows you the moving parts, and your code isn't tied to one vendor: the
  same spans could go to Jaeger, Grafana Tempo, Datadog or Arize Phoenix by changing the exporter. (Same rule as the
  roadmap's "plain SDK calls first".)

### 3. What we'll build (the parts and how they fit)

```text
 py -m coursemate serve            py -m coursemate agent "…"          py -m coursemate bench notes/sample -n 5 --repeat 2
        │ (api.py: session_id)            │                                    │ (asks golden questions twice, in memory)
        ▼                                 ▼                                    ▼
 chat.py stream_answer()           agent.py run_agent()                 chat.py stream_answer()
   span "chat" (root)                span "agent" (root)
    ├─ span "retrieve"               ├─ span "llm"   (one per step)
    ├─ span "llm"  ◀── cache.py      ├─ span "search_notes" / "draft_announcement" / "send_announcement"
    └─ span "check"   CachedStreamer └─ …
        │                 ▲ key = cache_key(…) ← YOU type (Step 7)
        ▼
 tracing.py   span() ← YOU type (Step 3);  attrs();  setup_tracing() → exporters
        │                                         │
        ▼                                         ▼
 Langfuse (Docker, :3000)                  logs/spans.jsonl ──▶ stats.py: percentile() ← YOU type (Step 9)
 traces, sessions, dashboards                                   report, budgets, logs/dashboard.html
```

| New or changed file | What it is |
|---|---|
| `coursemate/tracing.py` | the tracer, **your** `span()`, `attrs()`, the Langfuse exporter, the JSONL exporter, `setup_tracing()` |
| `coursemate/chat.py` (changed) | spans around each stage of an answer; `Final` gets `cached` and `cost_usd` |
| `coursemate/agent.py` (changed) | spans for the agent run, each model call and each tool call |
| `coursemate/cache.py` | **your** `cache_key()`, `ResponseCache` (TTL + LRU), `CachedStreamer` |
| `coursemate/stats.py` | **your** `percentile()`, reading the span log, the report, budgets, the HTML dashboard |
| `coursemate/cli.py` (changed) | `--span-log`, tracing for `serve` and `agent`, `serve --cache-ttl`, new `bench` and `stats` commands |
| `coursemate/api.py`, `static/index.html` (changed) | the session id goes into the trace; the page shows "cached" |
| `.env.example`, `.claude/settings.json` | the Langfuse settings (keys stay in `.env`, which Claude Code may not read) |
| `tests/test_tracing.py`, `test_chat_tracing.py`, `test_agent_tracing.py`, `test_cache.py`, `test_stats.py`, `test_live_cache.py` | tests with the paper numbers |

The attributes we send (Langfuse reads the `langfuse.*` and `gen_ai.*` names; `coursemate.*` are ours):

| Span | Attributes |
|---|---|
| `chat` (root) | `langfuse.trace.name` = chat, `langfuse.session.id`, `langfuse.observation.input` (question), `langfuse.observation.output` (answer), `coursemate.found`, `coursemate.cached`, `coursemate.input_tokens`, `coursemate.output_tokens`, `coursemate.first_token_ms`, `coursemate.cost_usd` |
| `retrieve` | `langfuse.observation.type` = retriever, input = the search query, output = the hits (source, heading, score), `coursemate.k`, `coursemate.mode`, `coursemate.best_score` |
| `llm` | type = generation, input = the messages, output = the reply text, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `langfuse.observation.usage_details` (`{"input": …, "output": …}`), `langfuse.observation.cost_details` (`{"total": …}`), `langfuse.observation.completion_start_time` (when the first word came), `coursemate.cached` |
| `check` | type = guardrail, output = "passed" or the reason it failed, `langfuse.observation.level` = WARNING when it failed |
| `agent` (root) | type = agent, input = the task, output = the answer, `coursemate.stopped`, `coursemate.steps`, `coursemate.tool_calls` |
| a tool, e.g. `send_announcement` | type = tool, input = the arguments, output = the result, `coursemate.needs_approval`, `coursemate.denied`, level WARNING on an error |

Same `coursemate` repo, on a branch called `observability`, merged with a pull request at the end.

### 4. How we'll test it

1. **Your `span()`** is tested with the paper tree from 2a: children share the root's trace id and point at it, a
   child lies inside its parent in time, and an error ends the span with status ERROR (rule 3).
2. **The chat trace** is tested with `FakeStreamer` and an **in-memory exporter** (a fake backend that keeps spans in
   a Python list): four spans, the right types and attributes, no `llm` span when gate 1 refuses, a WARNING when the
   citation check fails, and every span ended even when the model fails or the browser goes away mid-answer.
3. **The agent trace** reuses the agent guide's paper run (search, draft, send declined, answer): spans in the order
   `agent, llm, search_notes, llm, draft_announcement, llm, send_announcement, llm`, history sizes `[2, 4, 6, 8]`.
4. **Your `cache_key()`** is tested with the table in 2d; `ResponseCache` with the t=0…61 timeline.
5. **Your `percentile()`** is tested with the ten times from 2c: p50 1,100, p90 15,000, p95 20,000.
6. **Sending to Langfuse** is tested against a tiny pretend Langfuse on `127.0.0.1` inside the test: the right
   path, the Basic auth header and a real OTLP message. No test needs Langfuse, Ollama or the network.
7. **By hand:** you open the traces in the real Langfuse (Step 5) and run the real model (Step 10).
8. **CI** runs a smoke test with the fake model: `bench` (cache hits in round 2) and `stats` with budgets.

### 5. New words (short definitions)

- **Observability:** being able to tell what your system did and why, from the records it leaves.
  **LLMOps:** the day-to-day work of running LLM apps (tracing, evals, cost, latency, prompt versions).
- **Trace / span / root span / parent / attribute:** see 2a. **Trace id / span id:** random ids (32 and 16 hex
  characters) that link spans together.
- **Status:** OK, ERROR or UNSET (nothing went wrong, nothing was said). **Event:** a timestamped note inside a span;
  an exception is recorded as an event named `exception`.
- **OpenTelemetry (OTel):** the open standard and libraries for traces (and metrics and logs). **OTLP:** its wire
  format. **Tracer provider:** the object that creates tracers and owns the exporters. **Exporter:** sends finished
  spans somewhere. **Batch span processor:** collects spans and exports them in the background. **Flush / shutdown:**
  send what's waiting now (do it before the program exits).
- **No-op tracer:** a tracer whose spans do nothing; what you get when tracing is off, so the code needs no `if`s.
- **GenAI semantic conventions:** OTel's agreed attribute names for model calls (`gen_ai.request.model`,
  `gen_ai.usage.input_tokens`, ...).
- **Langfuse:** open-source LLM observability (traces, sessions, cost, dashboards, evals). **Observation:**
  Langfuse's word for a span; its **type** (generation, retriever, tool, guardrail, agent, span) sets its icon and
  view. **Session:** traces grouped by a session id (one chat conversation).
- **Self-hosted:** you run the software yourself (here in Docker) instead of using the vendor's cloud. Langfuse
  needs six containers: `langfuse-web` (the app and API, port 3000), `langfuse-worker` (background processing),
  **Postgres** (users, projects, keys), **ClickHouse** (a fast database for analytics: the traces themselves),
  **Redis** (a queue) and **MinIO** (an S3-compatible file store for raw events).
- **Headless initialization:** creating Langfuse's first user, project and API keys from settings, without clicking.
- **Latency / TTFT / p50 / p95 / budget:** see 2b and 2c.
- **Cache / hit / miss / hit rate / key / TTL / LRU / exact vs. semantic cache:** see 2d. **Hit rate** = hits ÷
  (hits + misses).
- **sha256:** a hash function: any text in, a fixed 64-hex-character fingerprint out.
- **`aclosing`:** a Python helper that closes an async generator when a block ends; you'll see why it matters in Step 4.
- **Session (Claude Code):** a saved Claude Code conversation you can resume. See section 6.

### 6. Today's Claude Code technique: sessions (`-n`, `--continue`, `--resume`, `/resume`, `/branch`)

Every Claude Code conversation is saved on your disk as a **session** (one transcript file per conversation, under
`~/.claude/projects/`). You can come back to it later, with everything Claude read, planned and decided still there.
That fits today perfectly: Part A on day 1, Part B on day 2, **in the same conversation**.

| What | Command |
|---|---|
| start a session with a name | `claude -n observability` |
| name the session you're in | `/rename observability` |
| resume the most recent session in this folder | `claude --continue` (or `claude -c`) |
| resume a session by name | `claude --resume observability` |
| pick from a list (search, preview) | `claude --resume` with no name, or `/resume` inside Claude Code |
| try something without changing the main conversation | `/branch try-semantic-cache` inside, or `claude --continue --fork-session` from the terminal |

What a resumed session **does** remember: the whole conversation, the files Claude read (as they were then), its
plans, your corrections, and its permission mode. What it **doesn't** know: anything that happened outside it since
(your own edits, a `git pull`, a teammate's commit). So the first prompt after a resume is always "check `git status`
and the last commit, then continue". And a long resumed session carries all its old tokens; if `/context` shows it
nearly full, `/compact` it or start fresh.

You'll use it in four places: `claude -n observability` in Step 0, a check at the end of Part A, `claude --resume
observability` to start Part B (Step 7), and a `/branch` for a thought experiment (Step 7).

### 7. Talking to hosted APIs (interviewers will ask)

- **Tracing hosted calls:** the same spans work for the OpenAI and Anthropic formats you added on Nov 16: the `llm`
  span records the model, tokens and cost, whatever the provider. Hosted SDKs also have ready-made OTel
  instrumentation (for example OpenLLMetry or OpenInference packages) that creates these spans for you; you wrote
  them by hand today so you know what such packages do.
- **Your cache vs. provider prompt caching:** your cache skips the **whole** call when the request is identical.
  Anthropic and OpenAI also offer **prompt caching** inside their API: when many requests start with the **same long
  prefix** (system prompt, tool definitions, a big document), the provider reuses its work on that prefix and charges
  much less for those input tokens (check each provider's pricing page for the current discount and how long the
  cache lives). That helps even when every question is different. A good production setup uses both: put the stable
  parts first in the prompt (for the provider's cache), and keep an exact response cache for true repeats.
- **Cost from the provider vs. your table:** hosted APIs report tokens in each response (you read them from the
  stream on Nov 16), and your `pricing.py` turns them into dollars. Langfuse can also compute cost itself from its
  own model price list; sending `cost_details` yourself keeps one source of truth (your table) for the report and
  the dashboard.
- **Privacy:** a trace contains the prompt and the answer. With your own course notes on your own PC that's fine. With
  student messages or a hosted observability service, you'd redact personal data before export, limit retention,
  and maybe record only metadata (tokens, times) for some routes.

### 8. Time and cost

About **4–5 hours** of hands-on time over two days. **Cost: $0.** Langfuse, Postgres and the models all run on your
PC; nothing calls a hosted API. **Memory:** Langfuse is six containers. On your 16 GB PC with Docker Desktop (which
by default may use up to half your RAM), Ollama running `qwen2.5:3b` and CourseMate's Postgres, expect it to fit,
but watch `docker stats` in Step 1. Stop Langfuse with `docker compose down` (in its folder) at the end of each
session; its containers are set to `restart: always`, so otherwise they start with Docker every time. Your traces
stay (they live in Docker volumes).

> **Note on the expected outputs.** I built and tested a reference solution for this guide: every test, command and
> file name below works, with the fake model, a mocked Ollama, a real Postgres + pgvector, and a pretend Langfuse
> that checks the exact OTLP request. I couldn't run the real Langfuse containers or `qwen2.5:3b` while writing it,
> so the Langfuse screens (Steps 1, 5, 6, 10) and the real-model numbers (Step 10) are **what to expect**, not
> copies. Langfuse's UI also changes often, so a menu name may differ a little; look for the nearest match.

---

## How to use this guide with Claude Code

Every step has:

- **Learn:** the idea. Read it first.
- **Predict:** write down what you expect *before* running anything. Being wrong is how you learn.
- **Prompt:** what to type into Claude Code in **plan mode** (`Shift+Tab` until you see `⏸ plan mode on`). Read
  Claude's plan and ask "why?" about anything you don't understand. The prompts are short and point Claude at the
  sections of this guide (it's in the repo), so the details come from one place.
- **You check:** what you verify yourself, with the file and function to open.
- **Done when:** a command whose output proves the step works.
- **Commit:** the message.

**Python is `py` on your machine.** Commands use `py -m ...`; inside Claude Code type `!py -m pytest`. CI on GitHub
runs Linux, where it stays `python`.

**Three windows.** Today you'll have Claude Code in one Git Bash window, and sometimes a second Git Bash window for
`py -m coursemate serve` (it keeps running) and a third for Langfuse's folder. Each new window needs
`source .venv/Scripts/activate` (in `coursemate`) before `py -m coursemate`.

---

# Part A (day 1): every answer and every agent run becomes a trace

## Step 0 — Get ready: merge the MCP branch, start a named session (15 min)

**Learn.** Today's work goes on a new branch, `observability`, made from `main`. One loose end first: Sunday's MCP
work is still on the branch `mcp`, and `GUIDE-mcp.md` may not be committed yet. Merge that first, so today starts
from everything you've built.

In Git Bash, in your `coursemate` folder:

```bash
git switch mcp
git status
```

**What you should see:** `On branch mcp`. If `GUIDE-mcp.md` is listed under "Untracked files", commit it:

```bash
git add GUIDE-mcp.md
git commit -m "Add the MCP guide"
```

Then push and open (or find) the pull request:

```bash
git push -u origin mcp
gh pr list
```

If `gh pr list` shows no PR from `mcp`, create one with `gh pr create --base main --fill`. Then:

```bash
gh pr checks --watch
```

**What you should see:** the checks end with `pass`. (If the notebook step fails, that's the known CI issue from
Nov 22; it's not part of today. Merge anyway only if the **Test** step itself passed.) Merge it on GitHub (or with
`gh pr merge --merge`), then come back to `main` and start today's branch:

```bash
git switch main
git pull
git switch -c observability
docker compose up -d
source .venv/Scripts/activate
py -m pip install -e ".[dev]"
py -m pytest -q
```

**What you should see:** `Switched to a new branch 'observability'`, and the tests ending with `0 failed`. On the
merged code I counted `223 passed, 1 skipped, 7 deselected` (the skipped one is the Agent SDK test, the 7 deselected
are live tests); yours may differ by a few.

Save **this file** as `GUIDE-observability.md` in the `coursemate` folder.

**Claude Code technique, part 1: a named session.** Start Claude Code with a name, so you can find this exact
conversation tomorrow:

```bash
claude -n observability
```

**What you should see:** Claude Code starts, and the name `observability` shows on the prompt bar (if you forget the
`-n`, type `/rename observability` now).

**Prompt (plan mode):**
> Read @GUIDE-observability.md (Project overview only). Then: in CLAUDE.md, add GUIDE-observability.md (Nov 24:
> tracing with OpenTelemetry and Langfuse, latency and cost, a response cache) to the Plan section, and add to "How to
> work with me": "Tests never send spans anywhere: they use the `spans` fixture (an in-memory exporter)." and "Never
> read .env: it holds the Langfuse keys." Add `Read(./.env)` and `Bash(cat .env:*)` to the deny list in
> .claude/settings.json. Create .env.example with LANGFUSE_BASE_URL=http://localhost:3000,
> LANGFUSE_PUBLIC_KEY=pk-lf-coursemate-local, LANGFUSE_SECRET_KEY=sk-lf-CHANGE-ME and
> COURSEMATE_SPAN_LOG=logs/spans.jsonl, with a comment saying to copy it to .env (git-ignored) and load it with
> `set -a; source .env; set +a`. Check .gitignore already has `.env` and `logs/`. Change nothing else.

**You check:** open `.claude/settings.json`: the two new lines are in `"deny"`. Open `.gitignore`: `.env` and `logs/`
are there (both were added in earlier projects). Why deny `.env` when the keys only work on your own PC? Because the
habit matters more than these keys: on Dec 3 the same file will hold AWS settings.

**Done when:** `!git status` shows `CLAUDE.md` and `.claude/settings.json` modified, and `GUIDE-observability.md` and
`.env.example` untracked.

**Commit:** `Add the observability guide` (include all four files).

---

## Step 1 — Run Langfuse in Docker, with your own keys (20 min)

**Learn.** Langfuse publishes a `docker-compose.yml` that starts its six containers (section 5). Every password in it
has a default marked `# CHANGEME`, and each default can be replaced from a `.env` file next to it, because the file
writes them as `${SALT:-mysalt}` ("use `SALT` from the environment or `.env`; if it's missing, use `mysalt`"). The same
`.env` can also create your first user, project and API keys at startup (**headless initialization**), so you never
copy keys out of a web page.

You'll keep Langfuse in its **own folder**, outside `coursemate`: it's a separate system you run, not part of your
repo. Do this step in a **second Git Bash window**, not in Claude Code: these are real passwords, and Claude doesn't
need to see them.

**Predict.** The compose file publishes ports 3000 (the app), 9090 (MinIO), 5432 (Langfuse's own Postgres) and a few
more. CourseMate's Postgres uses 5433. Will they clash?

In the second window:

```bash
mkdir -p ~/langfuse-local
cd ~/langfuse-local
curl -fsSLO https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml
ls
```

**What you should see:** `docker-compose.yml`.

Now write its `.env` with random secrets (`openssl` comes with Git Bash). Copy the whole block; the `EOF` lines
matter:

```bash
MINIO_SECRET=$(openssl rand -hex 16)
cat > .env <<EOF
TELEMETRY_ENABLED=false
NEXTAUTH_SECRET=$(openssl rand -hex 32)
SALT=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(openssl rand -hex 32)
MINIO_ROOT_PASSWORD=$MINIO_SECRET
LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=$MINIO_SECRET
LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=$MINIO_SECRET
LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY=$MINIO_SECRET
LANGFUSE_INIT_ORG_ID=coursemate-org
LANGFUSE_INIT_ORG_NAME=CourseMate
LANGFUSE_INIT_PROJECT_ID=coursemate
LANGFUSE_INIT_PROJECT_NAME=CourseMate
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-coursemate-local
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-$(openssl rand -hex 16)
LANGFUSE_INIT_USER_EMAIL=instructor@coursemate.local
LANGFUSE_INIT_USER_NAME=Instructor
LANGFUSE_INIT_USER_PASSWORD=$(openssl rand -hex 12)
EOF
grep -c = .env
```

**What you should see:** `17` (seventeen settings). What they do: `TELEMETRY_ENABLED=false` stops Langfuse from
sending anonymous usage statistics to its makers; the next three are the app's own secrets (session signing, password
hashing, encrypting stored keys); the four `MINIO`/`S3` lines give the file store one random password that the app
also uses; the `LANGFUSE_INIT_*` lines create your user, organization, project and API keys on first start. (Don't put
quotes around any value: Docker Compose would keep them as part of the value.) The databases keep their default
passwords; the compose file only lets programs on your own PC reach them (`127.0.0.1`).

Start it. `langfuse-web` and `langfuse-worker` pull in the four databases they depend on; naming them skips the
optional AI Gateway container, which you don't need:

```bash
docker compose up -d langfuse-web langfuse-worker
docker compose ps
```

**What you should see:** the first time, Docker downloads several images (a few minutes; about 2–3 GB). Then
`docker compose ps` lists six containers (`langfuse-web`, `langfuse-worker`, `postgres`, `clickhouse`, `redis`,
`minio`), the databases `healthy`. The web app needs another 1–3 minutes to run its database migrations. Wait for
it:

```bash
docker compose logs -f langfuse-web
```

When you see a line with **Ready**, press `Ctrl+C` (that stops *watching* the logs, not Langfuse). Check it answers:

```bash
curl -s http://localhost:3000/api/public/health
```

**What you should see:** a short JSON reply with `"status":"OK"` and a version number starting with `4`. (Your
prediction: no clash; 5432 and 5433 are different ports. If Docker says `port is already allocated`, something else
on your PC uses that port, often a Postgres you installed on Windows; stop it, or ask me how to remap the port.)

Log in: open **http://localhost:3000** and sign in with the email and password from your `.env`:

```bash
grep -E "USER_EMAIL|USER_PASSWORD" .env
```

**What you should see:** after signing in, the organization **CourseMate** with the project **CourseMate**, and an
empty tracing page. Under the project's settings → API keys, one key starting with `pk-lf-coursemate-local`.

**Memory check:**

```bash
docker stats --no-stream
```

**What you should see:** one line per container with a MEM USAGE column. Add them up roughly; I'd expect about 2–4 GB
for Langfuse when idle, ClickHouse and the web app being the biggest. If your PC starts swapping (everything gets
slow), close browser tabs first; if that's not enough, tell me the numbers and we'll set memory limits.

**Now give CourseMate the keys.** Back in the `coursemate` folder (still in this second window, not in Claude Code),
create its `.env` from the template, filling in the secret key from Langfuse's `.env`:

```bash
cd ~/path/to/coursemate     # use the real path of your coursemate folder
SK=$(grep '^LANGFUSE_INIT_PROJECT_SECRET_KEY=' ~/langfuse-local/.env | cut -d= -f2)
sed "s/^LANGFUSE_SECRET_KEY=.*/LANGFUSE_SECRET_KEY=$SK/" .env.example > .env
git status --short
```

**What you should see:** `git status --short` does **not** list `.env` (it's ignored). Now test the keys on
Langfuse's OTLP endpoint, the exact address your code will send to, with an empty message:

```bash
set -a; source .env; set +a
curl -s -o /dev/null -w "%{http_code}\n" -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  -H "Content-Type: application/json" -d '{"resourceSpans":[]}' \
  "$LANGFUSE_BASE_URL/api/public/otel/v1/traces"
```

**What you should see:** `200`. Try a wrong key once to see the difference (`-u "$LANGFUSE_PUBLIC_KEY:wrong"`):
`401` (not authorized). `set -a; source .env; set +a` means "export every variable the file sets": `set -a` turns on
auto-export, `source` runs the file's lines, `set +a` turns it off again. You'll run it in every window that starts
`coursemate serve`, `agent` or `bench`.

**Done when:** Langfuse is up, you're logged in, `curl` printed `200`, and `.env` is not in `git status`.

**Commit:** nothing (Langfuse's folder isn't a repo, and `.env` must never be committed).

---

## Step 2 — The tracing module: a tracer, exporters, and a test fixture (15 min)

**Learn.** Re-read 2e. Three pieces are plain plumbing, and Claude writes them:

- **`tracer()`**: the object that starts spans. When tracing is off it returns OpenTelemetry's **no-op tracer**, so
  `with span(...)` costs almost nothing and the rest of the code never needs `if tracing_enabled:`.
- **Exporters:** `langfuse_exporter()` builds the OTLP/HTTP exporter. Its URL is
  `LANGFUSE_BASE_URL + /api/public/otel/v1/traces`; its `Authorization` header is `Basic` + base64 of
  `public_key:secret_key` (the same thing `curl -u` did in Step 1); and `x-langfuse-ingestion-version: 4` asks
  Langfuse to show traces right away instead of minutes later. `JsonlSpanExporter` writes each finished span as one
  JSON line to `logs/spans.jsonl`, for your own report in Part B.
- **`setup_tracing()`** creates a **tracer provider**, gives it one `BatchSpanProcessor` per exporter, and returns a
  list of where spans will go (empty: tracing stays off).

One design choice: OpenTelemetry normally keeps **one global** provider per process, and it can be set only once.
That makes tests awkward (each test wants its own fake backend), so `tracing.py` keeps its **own** provider in a
module variable, with `use_provider()` to swap it. Tests get a fixture, `spans`, that installs a provider with an
**in-memory exporter** and a `SimpleSpanProcessor` (it exports each span the moment it ends, so a test never waits for
a batch).

The one function where today's idea lives, `span()`, is yours: you type it in Step 3. For now it's a stub.

**Predict.** With no `LANGFUSE_*` settings and no span log, `setup_tracing()` returns `[]`. What does
`tracer().start_span("x").is_recording()` return then, and why is that good for CI?

**Prompt (plan mode):**
> Add OpenTelemetry tracing as described in @GUIDE-observability.md sections 2e and 3 and Step 2. Add
> `opentelemetry-sdk>=1.40` and `opentelemetry-exporter-otlp-proto-http>=1.40` to the dependencies. Create
> coursemate/tracing.py with: a module-level provider (not OpenTelemetry's global one); `tracer()` (a NoOpTracer when
> off); `use_provider()`; `attrs(**values)` (a double underscore in a name becomes a dot, None values are dropped,
> str/bool/int/float stay, anything else becomes JSON); `iso_time(epoch_ns)`; `JsonlSpanExporter(path)` (one JSON
> line per span: name, trace_id, span_id, parent_id, start, duration_ms, status, attributes); `langfuse_exporter(environ)`
> (None unless LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are all set; import the OTLP exporter
> inside the function); `setup_tracing(span_log, environ=None)` returning where spans go; `shutdown_tracing()`.
> Leave `span(name, parent=None, **attributes)` as a @contextmanager stub with its docstring and
> `raise NotImplementedError`: I'll type it. Add a `spans` fixture to tests/conftest.py. In tests/test_tracing.py,
> test attrs, setup with nothing configured, the span log, the three Langfuse settings, and a pretend Langfuse
> (http.server on 127.0.0.1 port 0 in a thread) that receives the OTLP request: check the path, the Basic auth
> header, the ingestion-version header, and the span name (parse the body as ExportTraceServiceRequest). Also write
> the four span() tests from section 4.1 now. Install, run the tests, ruff fix and format.

**You check:** open `coursemate/tracing.py`.

1. `tracer()`: find the `NoOpTracer`. (Your prediction: `is_recording()` is `False`. CI and every existing test run
   with tracing off and pay nothing.)
2. `langfuse_exporter()`: find `base64.b64encode(f"{public_key}:{secret_key}"...)`. Compare with `curl -u` in
   Step 1: same header, built by hand.
3. `setup_tracing()`: find `BatchSpanProcessor`. Then open `tests/conftest.py` and find `SimpleSpanProcessor` in the
   `spans` fixture. Say in one sentence why the app uses one and the tests the other.
4. `tests/test_tracing.py`: find the pretend Langfuse (`class _Collector`). It's a real HTTP server on your own PC,
   started inside the test, so the test proves the **real** exporter sends the **real** bytes without needing
   Docker.

**Done when:**

```bash
py -m pytest -q tests/test_tracing.py
```

shows some passed and some failed, and **every failure** ends in `NotImplementedError` (they all use `span()`).
`py -m pytest -q --ignore=tests/test_tracing.py` (everything else) shows `0 failed`.

**Commit:** not yet: the tests are red on purpose. Step 3 makes them green.

---

## Step 3 — You type `span()`: parent, timing, errors, always end (15 min)

**Learn.** Re-read the three rules in 2a. Here they are as code. Type this into `span()` in `coursemate/tracing.py`,
replacing the `raise NotImplementedError` (keep the decorator, the signature and the docstring):

```python
    context = trace.set_span_in_context(parent) if parent is not None else None
    current = tracer().start_span(name, context=context, attributes=attrs(**attributes))
    try:
        yield current
    except Exception as exc:
        current.record_exception(exc)
        current.set_status(Status(StatusCode.ERROR, str(exc)))
        raise
    finally:
        current.end()
```

Line by line:

- **Line 1, rule 1 (a child points at its parent).** `set_span_in_context(parent)` wraps the parent in a "context"
  object; a span started with that context becomes the parent's child, with the same trace id. With no parent,
  `context=None` starts a **new trace**: a root.
- **Line 2.** `start_span` notes the start time now. The attributes go through your `attrs()` so dicts become JSON.
- **`yield current`, rule 2 (a child lies inside its parent).** This is a **context manager** (the `@contextmanager`
  above it): the code inside `with span(...) as s:` runs at the `yield`. So the span starts before the block and ends
  after it. Nest one `with` inside another and the inner span lies inside the outer one in time.
- **`except Exception`, the ERROR status.** If the block raises, we attach the exception (as an event named
  `exception`), set the status to ERROR with the message, and **re-raise**: tracing records the error, it never
  swallows it.
- **`finally: current.end()`, rule 3 (a span always ends).** `finally` runs whether the block finished, raised, or
  was stopped from outside. Only an ended span is exported.

Two more questions you might ask:

- *Why `except Exception` and not every error?* When a browser closes the page, Python stops the answer generator
  with `GeneratorExit`, which is **not** an `Exception` (it's a `BaseException`). The student leaving isn't a failed
  step, so we don't mark it ERROR; `finally` still ends the span.
- *Why not OpenTelemetry's usual `start_as_current_span`?* That one also makes the span "current", stored in a Python
  context variable, so that code deeper down can find it. But `stream_answer` is an **async generator**: it pauses at
  every `yield` while the web server runs other code, and resumes later. A "current" span set inside it can be reset
  from the wrong place, and OpenTelemetry then complains ("Failed to detach context"). Passing the parent **explicitly**
  avoids all of that, and it's easier to read: you can see who the parent is.

**Predict.** The test `test_an_error_marks_the_span_and_still_ends_it` raises `ValueError` inside `span("llm")`. If
you delete the two `finally` lines, will that test still find the span in the exporter? Which other tests will fail?

**Prompt:** none. You type it.

**You check:** run the tests:

```bash
py -m pytest -q tests/test_tracing.py
```

**What you should see:** all pass (`10 passed` in my version; yours may have a different count). Now **break it on
purpose**: put a `#` in front of the two lines `finally:` and `current.end()`, and run again.

**What you should see:** many failures, including the error test: a span that never ends is never exported, so the
in-memory exporter is **empty** and the test can't even find it. (Your prediction.) That's rule 3 in action: without
`end()`, the failed request you most need to see is the one that disappears. Remove the two `#` and run again: all
pass.

**Done when:**

```bash
py -m pytest -q
py -m ruff check . && py -m ruff format --check .
```

show `0 failed` and no ruff errors.

**Commit:** `Tracing: span(), a no-op default, exporters for Langfuse and a JSONL log`

---

## Step 4 — Trace one answer: `chat`, `retrieve`, `llm`, `check` (20 min)

**Learn.** Now put spans around the stages of `stream_answer()` in `coursemate/chat.py`, the function that does one
streamed answer (Nov 16). A quick reminder of what it does, stage by stage:

1. **Retrieve** (in a helper thread, because embedding and the database are blocking calls), then **gate 1**: if no
   chunk scores above `min_score`, yield "I don't know" and stop, **without** calling the model.
2. **Stream** the model's words: each piece is yielded as a `Token`, which the API turns into an SSE `token` event.
3. **Check** the whole reply (citations exist and point at real sources), then yield one `Final`.

The trace mirrors that: a root span `chat`, and children `retrieve`, `llm` and `check`, with the attributes from the
table in section 3. The session id (from the API) goes on the root, so Langfuse can group a conversation.

One trap, and it's why the tests include a strange case. When a student closes the page mid-answer, the web server
**closes** the `stream_answer` generator (Python raises `GeneratorExit` at the paused `yield`). If the stages live in
a second, inner generator (Claude may split the code this way, so the root span wraps everything), closing the outer
one does **not** close the inner one right away: Python leaves an abandoned async generator for the garbage collector,
which may close it much later, or never before the program exits. Until then the inner spans (`llm`) stay open and are
never exported. The fix is one line: `contextlib.aclosing(...)`, which closes the inner generator when the block ends.

**Predict.** A browser closes the page right after the first word. Which of `chat`, `retrieve`, `llm`, `check` should
end up exported? (Think: which stages had **started**?)

**Prompt (plan mode):**
> Trace each answer in coursemate/chat.py as described in @GUIDE-observability.md sections 2a and 3 and Step 4.
> `stream_answer()` gets `session_id: str = ""` and opens the root span "chat"; the stages become children
> "retrieve", "llm" and "check", made with `span()` and the parent passed explicitly, with the attributes from the
> table in section 3 (completion_start_time = the llm span's start + first_token_ms). `Final` gets
> `cached: bool = False` and `cost_usd: float | None = 0.0` (from pricing.price_for and cost_usd; None when the price
> is unknown); in api.py, `done_payload` uses `final.cost_usd` and adds `cached`, and the chat route passes the
> session id. `StreamEnd` in streaming.py gets `cached: bool = False` (Part B uses it). If the client goes away after
> the first token, every span that started must still end. Add tests/test_chat_tracing.py with the cases in section
> 4.2 (the `spans` fixture, FakeStreamer, a MemoryStore), including one that reads the first token and then calls
> `aclose()`, and an API test that the root span's langfuse.session.id equals the id in the `session` event. Don't
> change anything the user sees. Run all tests, ruff fix and format.

**You check:** open `coursemate/chat.py`.

1. Find `stream_answer` and the root `span("chat", ...)`. Inside it, find where the stages run and look for
   `aclosing`. If Claude didn't split the code into two generators, there's no inner generator to close and no
   `aclosing`; that's fine too, as long as the `aclose()` test passes.
2. Find `span("llm", root, ...)`. The `async for event in llm.stream(...)` loop is **inside** the `with`, so the `llm`
   span's time is the model's time, and the `yield Token(...)` happens inside it. That's why the span must survive
   pauses: your explicit-parent `span()` from Step 3 is what makes this safe.
3. Find the `check` span and `langfuse.observation.level`: a failed citation check becomes a **WARNING**, so in
   Langfuse you can filter for answers the guardrail retracted.
4. Open `tests/test_chat_tracing.py` and find the test that calls `aclose()`.

**Break it on purpose** (only if Claude used `aclosing`). First commit, so you can undo with git:

```bash
git add -A && git commit -m "Trace each answer: chat, retrieve, llm, check"
```

In `stream_answer`, replace the `async with aclosing(...) as events:` line so it loops over the inner generator
directly (or ask Claude: *"Temporarily remove the aclosing wrapper in stream_answer and run
tests/test_chat_tracing.py. Don't fix anything."*). Run the tests:

```bash
py -m pytest -q tests/test_chat_tracing.py
```

**What you should see:** the `aclose()` test fails: only `chat` and `retrieve` were exported; `llm` stayed open.
(Your prediction: `chat`, `retrieve` and `llm` had started, so all three should end; `check` never started.) Undo:

```bash
git restore coursemate/chat.py
py -m pytest -q tests/test_chat_tracing.py
```

**Done when:**

```bash
py -m pytest -q
py -m ruff check . && py -m ruff format --check .
```

show `0 failed` and no ruff errors.

**Commit:** `Trace each answer: chat, retrieve, llm, check` (if you didn't already commit it above).

---

## Step 5 — See your traces in Langfuse (20 min)

**Learn.** The spans exist in code, but nothing turns tracing **on** yet. The commands that run for real (`serve` and
`agent`) need to call `setup_tracing()` when they start and `shutdown_tracing()` when they stop (to flush the last
batch). We also add one global option, `--span-log PATH`, which defaults to `COURSEMATE_SPAN_LOG` from your `.env`.

**Prompt (plan mode):**
> In coursemate/cli.py: add a global `--span-log` option (type Path, default from the environment variable
> COURSEMATE_SPAN_LOG, else None); a helper `start_tracing(args, stderr)` that calls setup_tracing(args.span_log) and
> prints "tracing: " plus where spans go, or "tracing: off"; call it at the start of `run_serve` and
> `run_agent_command`, and call shutdown_tracing() in their `finally` blocks. Run all tests, ruff fix and format.

**Done when** the tests pass. **Commit:** `Turn tracing on for serve and agent`

Now run it for real. In your **second** Git Bash window, in the `coursemate` folder:

```bash
source .venv/Scripts/activate
set -a; source .env; set +a
py -m coursemate serve
```

**What you should see:** first `tracing: langfuse at http://localhost:3000, span log logs/spans.jsonl`, then
Uvicorn's lines ending with `Uvicorn running on http://127.0.0.1:8000`. (If it says `tracing: off`, the `.env` wasn't
loaded in this window.)

Open **http://localhost:8000** and ask, in this order, in one conversation:

1. `When is Lab 4 due?`
2. `And how is it graded?` (a follow-up)
3. `Who won the 2022 World Cup?` (it should say "I don't know")

**Predict.** How many traces will Langfuse show? Which one has no `llm` observation? How many sessions?

Wait 5–10 seconds (the batch), then open **http://localhost:3000** → your project → **Tracing**.

**You check (in Langfuse):**

1. **Three traces named `chat`.** (If the list shows every observation instead of traces, filter to root
   observations, or to the name `chat`.) Open the first one.
2. **The tree:** `chat` with `retrieve`, `llm` and `check` under it, each with a duration bar. Write down the three
   durations and the total. Which stage dominates? (Section 2a predicted the model; check it on your CPU.)
3. **The `llm` observation** (a generation): the model `qwen2.5:3b`, input and output tokens, a cost of `$0`, and the
   **input**: the system prompt, the numbered sources and the question, exactly what the model read. This is the view
   that answers "why did it say that?". Find the **time to first token** (Langfuse computes it from
   `completion_start_time`).
4. **The `retrieve` observation:** the hits with their scores. Is `lecture-02-hash-tables.md > ... > Lab 4` first?
5. **The third trace** (World Cup): only `chat` and `retrieve`, and `coursemate.found = false` in the root's metadata.
   Gate 1 refused, so no model call: 0 tokens, a few hundred ms.
6. **Sessions:** the first two traces share one session id (the second was a follow-up in the same conversation); the
   third is in the same session too if you asked it in the same chat. Open the session view: one row per question.

Now the same data, without Langfuse. The batch writes to the span log every few seconds too, so you can read it while
the server runs. Open a third Git Bash window in the `coursemate` folder (or stop the server with `Ctrl+C` first):

```bash
tail -n 1 logs/spans.jsonl | py -m json.tool
```

**What you should see:** the last span, as JSON: probably the `chat` root of the World Cup question, with
`"parent_id": null`, a `duration_ms`, and its attributes. That file is what `coursemate stats` reads in Part B.

**Done when:** you've opened all three traces and written down the stage timings of the first one.

---

## Step 6 — Trace the agent: every model call and every tool call (15 min)

**Learn.** The agent (Nov 20) is a loop, so its trace is a list of steps under one root: `agent` → `llm`, a tool,
`llm`, a tool, ... Each tool span shows **the arguments the model chose** and **the result it got back**, which is
exactly what you needed on Nov 20 when `qwen2.5:3b` "drafted, said it would send, and stopped". A declined send gets
`coursemate.denied = true` and level WARNING, so you can find every refused action across all runs with one filter.

Re-read the agent guide's paper trace: search, draft, send (declined), answer. That's 4 model calls and 3 tool runs.

**Predict.** List the span names of that run in the order they **start**. How many messages does the model see in its
4 calls?

**Prompt (plan mode):**
> Trace the agent in coursemate/agent.py as described in @GUIDE-observability.md section 3 (rows "agent" and "a
> tool"): `run_agent` opens a root span "agent" (type agent, input = the task, output = the answer, plus
> coursemate.stopped, coursemate.steps and coursemate.tool_calls); every model call is a child span "llm" (type
> generation, input = the messages so far, output = the assistant message, tokens, coursemate.step); every tool call
> is a child span named after the tool (type tool, input = the arguments, output = the result,
> coursemate.needs_approval, coursemate.denied, level WARNING when the result starts with "error"). Keep run_tool and
> the approval gate exactly as they are. Add tests/test_agent_tracing.py with the paper run from the agent guide
> (FakeClient: search, draft, send, then an answer; the default approver): span names in start order, all children of
> agent, the send span denied with WARNING, steps 1–4 with history sizes [2, 4, 6, 8]. Run all tests, ruff fix and
> format.

**You check:** open `coursemate/agent.py` and find where the tool span wraps `run_tool(call, tools, approve)`. The
gate is untouched: tracing **observes** the decision, it doesn't make it. In `tests/test_agent_tracing.py`, find the
expected order: `agent, llm, search_notes, llm, draft_announcement, llm, send_announcement, llm` (your prediction),
and `[2, 4, 6, 8]` (the model sees 2, 4, 6, then 8 messages).

**Done when** `py -m pytest -q` and ruff pass. **Commit:** `Trace the agent: model calls and tool calls`

Now a real run. In the second window (with `.env` loaded):

```bash
py -m coursemate agent "When are office hours? Answer in one sentence."
```

**What you should see:** `tracing: langfuse at ...`, the agent's steps, and an answer about Tuesdays. In Langfuse, a
new trace named `agent`: open it and read the tool span `search_notes`: the query `qwen2.5:3b` chose, and the chunks
it got back. Optional: run the Lab 4 "write and send" task from the agent guide and answer **n** when asked. If the
model calls `send_announcement`, its span shows a WARNING and `coursemate.denied = true`. (A 3B model doesn't always
try to send, and that's information too: the trace shows you exactly where it stopped.)

---

> **Stop here. Part A is done when:** `py -m pytest -q` shows `0 failed`, ruff is clean, Steps 3–6 are committed on
> the `observability` branch, and you've seen a `chat` trace and an `agent` trace in Langfuse.
>
> **Before you close Claude Code** (sessions, part 2), ask it:
>
> > Write a 5-line note on where we are: what's committed, what Part B (Steps 7–11 in GUIDE-observability.md) will
> > add, and anything you noticed that I should check tomorrow. Don't change any files.
>
> Read it, then exit (`/exit`). Run `claude --resume` once to see the session picker: `observability` should be at
> the top. Press `Esc` to leave it. Then stop Langfuse in its folder, so it doesn't start with Docker tomorrow
> morning (your traces stay in Docker volumes):
>
> ```bash
> cd ~/langfuse-local && docker compose down
> ```
>
> Before Part B: re-read the Project overview and answer the **Check your understanding** questions.

---

# Part B (day 2): a cache, latency and cost numbers, and budgets

## Step 7 — Resume yesterday's session, then you type `cache_key()` (20 min)

**Learn (sessions, part 3).** Start Langfuse again (`cd ~/langfuse-local && docker compose up -d langfuse-web
langfuse-worker`, about a minute until it's Ready), and Postgres (`docker compose up -d` in `coursemate`). Then, in a
fresh Git Bash window in `coursemate` where you have **not** loaded `.env` (Claude's commands inherit the window's
environment variables, and the keys should stay out of its reach):

```bash
source .venv/Scripts/activate
claude --resume observability
```

**What you should see:** yesterday's conversation, scrolled to the end, with your 5-line note. Claude remembers what
it read and decided, but **not** anything that happened outside the session since. So always start with a check:

**Prompt (plan mode):**
> We're resuming. Run `git status` and `git log --oneline -6`, compare them with your note from yesterday, and tell
> me if anything changed that you didn't do. Then read Step 7 of @GUIDE-observability.md. Don't change any files.

**What you should see:** a clean tree and your Part A commits (`Trace the agent...` on top). If Claude says something
changed, read what it found before going on.

**A thought experiment in a branch.** Section 2d argued against a semantic cache. Test that with real numbers, in a
**branch** of this conversation, so the experiment doesn't clutter the main thread:

```text
/branch try-semantic-cache
```

**Prompt (in the branch, plan mode off):**
> Thought experiment, don't edit any files: with Ollama's nomic-embed-text (use OllamaEmbedder.embed_query from
> coursemate/embeddings.py in a one-off `py -c` command), compute the cosine similarity between "When is Lab 4 due?"
> and "When is Lab 5 due?", and between "When is Lab 4 due?" and "when's lab 4 due". If a semantic cache returned a
> cached answer whenever the cosine is at least 0.95, what would each pair get? What would you need to make a
> semantic cache safe for course facts?

**What you should see:** both pairs score high, probably above 0.9 (write down your numbers). The Lab 4 / Lab 5 pair
is the dangerous one: a high score, a different right answer. Now go back to the main conversation:

```text
/resume observability
```

**What you should see:** the main thread, without the experiment in it. (`claude --resume` shows the branch grouped
under its root, if you ever want it back.)

**Learn (the cache key).** Re-read 2d. The key must contain everything that can change the reply, and nothing else.
`CachedStreamer` (Step 8) will sit in front of the model, so "everything" is exactly the five things a model call
receives. Type this into `cache_key()` in `coursemate/cache.py` once Claude has made the stub:

```python
    payload = {
        "provider": provider,
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # sort_keys: the same dict written in a different order must give the same key
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

- `list(messages)`: the messages may arrive as any sequence; JSON needs a list.
- `json.dumps(..., sort_keys=True)`: the dict becomes one string, with keys in alphabetical order at every level, so
  `{"role": ..., "content": ...}` and `{"content": ..., "role": ...}` give the **same** string.
  `ensure_ascii=False` keeps non-English text as it is (both choices work, as long as you never change it: a different
  choice would make different keys for old entries).
- `sha256(...).hexdigest()`: the 64-character fingerprint from 2d.

**Predict.** Using the table in 2d: which rows give the same key? If you delete the `"model": model,` line, which test
fails?

**Prompt (plan mode):**
> Create coursemate/cache.py with a stub `cache_key(provider, model, messages, temperature, max_tokens) -> str`
> (docstring "Everything that can change the reply goes into the key, and nothing else." and
> `raise NotImplementedError`): I'll type it. Add tests/test_cache.py with the table in @GUIDE-observability.md
> section 2d: the same request gives the same 64-hex key; a different key order inside a message gives the same key;
> each of model, provider, max_tokens, temperature, a different question, and one extra space changes the key. Only
> the cache_key tests for now. Run them; they should fail with NotImplementedError.

Type the function, then:

```bash
py -m pytest -q tests/test_cache.py
```

**What you should see:** all pass (`8 passed` in my version). **Break it:** put a `#` before `"model": model,` and run
again: one case fails (the `llama3.2:3b` row). Without the model in the key, switching models would serve the old
model's answers. Remove the `#`.

**Done when** `py -m pytest -q` and ruff pass. **Commit:** `cache_key: everything that can change the reply`

---

## Step 8 — The cache: TTL, LRU, and a wrapper around the model (20 min)

**Learn.** Two classes, both written by Claude:

- **`ResponseCache`** stores `key → (time stored, reply)` in an `OrderedDict`, a dict that remembers order and can
  move a key to the end (`move_to_end`) or drop the first one (`popitem(last=False)`). The **end** is "most recently
  used", the **front** "least recently used". `get()` treats an entry older than `ttl_s` as missing; `put()` drops
  from the front while there are more than `max_entries`. It takes a `clock` argument (a function that returns the
  time), so the test can **set** the time to 0, 10, 20, 30, 61 like the paper timeline in 2d, instead of sleeping.
- **`CachedStreamer`** wraps any `StreamingLLM` and looks like one (same `provider`, `model`, `local`, `base_url`), so
  `stream_answer`, the API and the tracing don't change at all. On a **hit** it yields the whole saved text as one
  `TextDelta`, then a `StreamEnd` with `cached=True` and **0 tokens** (no model was called). On a **miss** it passes
  every event through and saves the reply when the `StreamEnd` arrives: a stream that fails raises before that, so a
  failed reply is never cached. At temperature above 0 it never caches.

This is the **wrapper** (or decorator) pattern: add a feature around an object without changing it. It's also why
`cache_key` uses exactly the arguments of `stream()`.

**Predict.** A student asks "When is Lab 4 due?" twice. The second time, which spans does the trace have, how many
tokens does the `llm` span show, and what does `coursemate.cached` say on the root?

**Prompt (plan mode):**
> In coursemate/cache.py add `CachedReply` (text, model, input_tokens, output_tokens), `ResponseCache(ttl_s=3600,
> max_entries=1000, clock=time.monotonic)` with get/put, hits/misses counters, `__len__` and `__contains__`, and
> `CachedStreamer(inner, cache)`, all as described in @GUIDE-observability.md Step 8 and section 2d. Wire it in:
> `serve` gets `--cache-ttl` (seconds, default 3600, 0 = no cache), and the chat page's timing line adds " · cached (no
> model call)" when `cached` is true. Tests: the t=0…61 timeline from section 2d (hits 2, misses 2); a miss calls the
> model and a hit doesn't (a FakeStreamer with only one reply); temperature 0.7 is never cached; a failed stream isn't
> cached; the wrapper has the inner model's provider, model, local and base_url; and in tests/test_chat_tracing.py,
> the same question twice through stream_answer gives cached False then True, 0 input tokens the second time, and
> root spans with coursemate.cached [False, True]. Run all tests, ruff fix and format.

**You check:** open `coursemate/cache.py`.

1. In `ResponseCache.get`, find `move_to_end`, and in `put`, find `popitem(last=False)`. Trace the paper timeline
   through them: at t=20, `get("A")` moves A to the end, so at t=30 the front is B, and B is dropped.
2. In `CachedStreamer.stream`, find where it calls `self.cache.put(...)`: inside the branch for the `StreamEnd`, so
   only a finished stream is stored.
3. In `coursemate/cli.py`, find where `serve` wraps the streamer (`CachedStreamer(llm, ResponseCache(ttl_s=...))`).

**Done when** `py -m pytest -q` and ruff pass. **Commit:** `A response cache in front of the model`

Try it. Second window (`.env` loaded):

```bash
py -m coursemate serve
```

Ask `When is Lab 4 due?` and wait for the answer. Then ask exactly `When is Lab 4 due?` again **in a new chat**
(reload the page). (A follow-up in the same chat would put the first answer into the history, which changes the
messages and the key.)

**What you should see:** the first answer as before (seconds, tokens in and out). The second appears all at once, and
its timing line ends with `cached (no model call)`, tokens `0 in / 0 out`, a total of a few ms (the model part only;
the request also spent tens of ms on retrieval). In Langfuse, the second trace's `llm` observation shows 0 tokens and
`coursemate.cached: true`, and the whole trace is a small fraction of the first one. (Your prediction: the same four
spans; retrieval still runs, because the chunks are part of the key.) Now ask `When is Lab 4 due? ` with a space at
the end: a miss, the model runs again. That's an **exact** cache.

---

## Step 9 — The numbers: you type `percentile()`, then `bench` and `stats` (25 min)

**Learn.** Langfuse is great for looking at **one** trace, and its dashboards show trends. You'll also build your own
small report from `logs/spans.jsonl`, for three reasons: you'll understand every number in it, it runs in CI without
Langfuse, and it can **fail** (exit code 1) when a budget is broken, which a dashboard can't.

- **`coursemate bench notes/sample -n 5 --repeat 2`** asks the first 5 golden questions (from
  `evals/retrieval_golden.jsonl`), twice, through the real `stream_answer`, with tracing and the cache on, against an
  in-memory index (no database). Round 2 should be all cache hits. All answers share one Langfuse session.
- **`coursemate stats LOG`** reads the root `chat` spans from a span log and prints: answers, found, cached and the
  hit rate; p50/p95/max latency for all answers, not-cached, cached, and first token; mean tokens per model call; what
  those answers would cost on each hosted model in `pricing.py`; and the **budgets** you pass (`--max-p95-ms`,
  `--max-cost-usd` per answer), each `OK` or `OVER`. Any `OVER` → exit code 1. `--html PATH` also writes a one-page
  dashboard: one bar per answer, green for cached, blue for model answers, red for "I don't know".

The idea that matters is the **percentile**. Type this into `percentile()` in `coursemate/stats.py` once Claude has
made the stub:

```python
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))  # 1-based: the rank-th smallest value
    return ordered[max(rank, 1) - 1]
```

- `sorted(values)` makes a new sorted list and leaves the caller's list alone.
- `math.ceil(p / 100 * n)` is the ⌈ ⌉ from 2c: p95 of 10 values → ⌈9.5⌉ = 10.
- `max(rank, 1) - 1`: ranks start at 1, Python lists at 0; `max(..., 1)` makes p0 the smallest value instead of
  index −1 (which in Python would be the **largest**).

**Predict.** For the ten times in 2c: p50, p90, p95? And for the report: if 5 answers are cached at about 50 ms and 5
come from the model at about 12,000 ms, what are p50 and p95 of **all** answers? (Rank ⌈0.5 × 10⌉ = 5: which group is
the 5th smallest in?)

**Prompt (plan mode):**
> Add coursemate/stats.py as described in @GUIDE-observability.md Step 9: a stub `percentile(values, p)` (nearest
> rank; docstring and `raise NotImplementedError`, I'll type it); `load_answers(path)` (root "chat" spans from the
> JSONL span log, oldest first; first_token_ms only for answers that called the model); `summarize()` (rows for all
> answers, not cached, cached and first token, each with n, p50, p95, max; mean tokens per model call; total cost,
> None if any is unknown; the number of answers that called a model); `format_stats(summary, max_p95_ms=None,
> max_cost_usd=None)` returning (text, ok), including the hosted-price lines (cost of one model call at the mean
> tokens × the share of answers that called a model); `write_html(path, ...)`. In cli.py add the `bench` and `stats`
> commands as described there (bench: notes path, --golden, -n 5, --repeat 2, --cache-ttl 3600, --min-score; one event
> loop for the whole run; one line per answer with round, question id, wall-clock ms, "cached"/"model"/"-", and
> found or I don't know; stats: log path, --max-p95-ms, --max-cost-usd, --html; exit 1 when a budget is over). Tests
> in tests/test_stats.py: the paper numbers from section 2c (p50 1100, p90 15000, p95 20000, mean 4368), edge cases,
> a small hand-written log (one model answer, the same one cached, one refused), budgets OK and OVER, the hosted price
> with 1 of 3 answers calling the model ($0.000065 per answer on claude-haiku-5-5 for 1,200 in and 150 out), and an
> unknown cost. CLI tests: fake bench + stats (round 2 all cached, "50% hit rate"), bench with --cache-ttl 0 never
> hits, stats over budget exits 1. Add a CI step after the MCP smoke test that runs the fake bench and
> `stats --max-p95-ms 5000 --max-cost-usd 0`. Run all tests, ruff fix and format.

Type `percentile()`, then run the tests:

```bash
py -m pytest -q tests/test_stats.py
```

**What you should see:** all pass. (Your first prediction: p50 = 1,100, p90 = 15,000, p95 = 20,000. Your second: the
5th smallest of 10 is the slowest cached answer, so **p50 ≈ 50 ms** while **p95 ≈ 12,000 ms**: the cache makes the
median look wonderful and hides nothing in the tail. That's why the report has separate rows.)

**You check:** in `coursemate/stats.py`, find `load_answers` and the line that skips spans whose `parent_id` isn't
`None`: only roots are answers. In `format_stats`, find `per_answer = each * s.model_calls / s.n`: cached and refused
answers cost nothing, so a 50% hit rate halves the bill.

Now try both commands with the fake model, inside Claude Code:

```text
!py -m coursemate --provider fake --span-log logs/fake.jsonl bench notes/sample -n 3 --repeat 2 --min-score 0.1
```

**What you should see** (the session id and the ms differ):

```text
tracing: span log logs/fake.jsonl
session bench-c30da54a: 3 questions x 2
round 1  q01          3 ms  model   found
round 1  q02          2 ms  model   found
round 1  q03          2 ms  model   found
round 2  q01          2 ms  cached  found
round 2  q02          2 ms  cached  found
round 2  q03          2 ms  cached  found
spans written to logs/fake.jsonl
```

```text
!py -m coursemate stats logs/fake.jsonl --max-p95-ms 1000 --max-cost-usd 0.001
```

**What you should see:**

```text
answers: 6   found: 6   cached: 3 (50% hit rate)

                 n    p50 ms    p95 ms    max ms
all answers      6         2         3         3
not cached       3         2         3         3
cached           3         2         2         2
first token      3         1         1         1

tokens per model call: 100 in, 20 out
if hosted (prices checked 2026-11-15; cached answers cost nothing):
  claude-haiku-5-5    $0.000010 per answer   $0.01 per 1,000
  claude-sonnet-5-5   $0.000200 per answer   $0.20 per 1,000
  gpt-6-luna          $0.000010 per answer   $0.01 per 1,000
  gpt-6.1-sol         $0.000200 per answer   $0.20 per 1,000
cost of these answers: $0.000000

budgets:
  p95 latency 3 ms <= 1,000 ms  OK
  cost per answer $0.000000 <= $0.001000  OK
```

(The fake model "answers" in 2 ms and always reports 100 tokens in and 20 out, so these numbers only prove the
plumbing. Check one by hand: Haiku, 100 × 0.10/1M + 20 × 0.50/1M = $0.00002 per model call, × 3 model calls / 6
answers = **$0.000010** per answer.)

**Done when** `py -m pytest -q` and ruff pass. **Commit:** `bench and stats: latency, cost, hit rate and budgets`

---

## Step 10 — The real run: measure CourseMate on `qwen2.5:3b` (20 min)

**Learn.** Now real numbers. Use a **separate** span log for this run, so yesterday's manual questions don't mix in.
Budgets for today: **p95 latency ≤ 15,000 ms** and **cost per answer ≤ $0.001**. Whether you pass the latency budget
on a CPU is an honest open question; either result is useful, as long as you write it down.

**Predict.** Before you run it, write down: your guess for an uncached answer's time on your PC (yesterday's trace
tells you), a cached answer's time, the hit rate after 2 rounds of 5 questions, and p50 and p95 of all 10 answers.

In the second window (`.env` loaded, Ollama running):

```bash
py -m coursemate --span-log logs/bench.jsonl bench notes/sample -n 5 --repeat 2
```

**What you should see:** `tracing: langfuse at ..., span log logs/bench.jsonl` (the `--span-log` flag replaces the
`.env` one for this run), a pause while the notes are embedded into memory (tens of seconds), then 10 lines. Round 1:
`model`, each taking many seconds (my guess for your CPU: 8–25 s). Round 2: `cached`, each tens to a few hundred
milliseconds (retrieval only: embedding the question). Some questions may say `I don't know` if they fall below `min_score`; that's the gate doing its job,
and those show `-` instead of `model` (no model call, so nothing to cache: they're fast in both rounds).

Now the report, saved as your run record, and the dashboard:

```bash
py -m coursemate stats logs/bench.jsonl --max-p95-ms 15000 --max-cost-usd 0.001 --html logs/dashboard.html | tee evals/runs/observability-qwen2.5-3b.txt
start logs/dashboard.html
```

**What you should see:** the report from Step 9 with real numbers: about 50% hit rate (fewer if some answers were
refused), `not cached` p50 in the seconds, `cached` p50 in tens to hundreds of ms, `first token` a few seconds, about 1,000–1,500
tokens in and 30–150 out per model call, `cost of these answers: $0.000000`, and the hosted lines (on Haiku, a small
fraction of a cent per answer). The budgets: cost `OK`; latency `OK` or `OVER` (the p95 of 10 answers is the slowest
one). `start` opens the dashboard in your browser: 10 bars, the green cached ones barely visible next to the blue ones.

**You check (in Langfuse):**

1. **Sessions:** one session named `bench-...` with 10 traces. The 5 cached ones are short; open one: `llm` shows 0
   tokens and `coursemate.cached: true`.
2. **A chart:** open the observations table, filter to the name `llm`, switch the view from **Table** to **Chart**,
   pick the metric **Latency** with **p95** (and try **Tokens**). Use **Add to dashboard** to save it on a new
   dashboard called `CourseMate`. Add a second widget: **Count** of `chat` traces, or **Cost**. (Langfuse also has a
   Home dashboard with latency percentiles and model usage built in; look at it too.)
3. Do Langfuse's numbers and your `stats` agree? They read the same spans, so the `llm` latencies should match.
   Differences come from what each one measures: your `stats` uses the **root** `chat` span (the whole answer).

Write in your notes: the not-cached p50 and p95, the first-token p50, the cached p50, the hit rate, and whether you
passed the latency budget. If you didn't, which stage would you attack first, looking at a trace? (Section 2a: the
model; and the trace's tokens tell you how much of that is reading the prompt.)

**Done when:** `evals/runs/observability-qwen2.5-3b.txt` exists and you've written down your numbers.

**Commit:** `Observability run on qwen2.5:3b` (only the `.txt`; `logs/` is ignored).

---

## Step 11 — Live test, README, interview questions, pull request (20 min)

**Prompt 1 (plan mode):**
> Add tests/test_live_cache.py (pytestmark live; skip if Ollama isn't reachable): ask qwen2.5:3b "What color is a
> clear daytime sky? One word." twice through CachedStreamer(OllamaStreamer(...), ResponseCache()) with
> max_tokens=20: the first isn't cached and has input tokens, the second is cached with 0 tokens and the same text,
> and its total_ms is less than a tenth of the first.

```bash
py -m pytest -m live -v tests/test_live_cache.py
```

**What you should see:** `1 passed`.

**Prompt 2 (plan mode):**
> Update README.md: add a section "Observability" with: how to start Langfuse (a short version of Step 1 in
> GUIDE-observability.md, pointing to it), the .env settings (from .env.example) and `set -a; source .env; set +a`;
> what each answer's trace contains (the span table from section 3); the cache (`serve --cache-ttl`, what's in the key,
> exact not semantic, temperature 0 only); `bench` and `stats` with budgets and `--html`; and my results from
> evals/runs/observability-qwen2.5-3b.txt as a small table (I'll add one line on what I'd improve first). Design
> decisions: plain OpenTelemetry instead of the Langfuse SDK (vendor-neutral, shows what SDKs hide); our own provider
> instead of the global one (testable); explicit parents instead of "current span" (safe in async generators);
> aclosing so a closed stream ends its spans; tracing off = no-op, so tests and CI never need Langfuse; a batch
> exporter so answers never wait for the network; the cache keyed on exactly what the model receives, so new notes or
> prompts can't return stale answers; a JSONL span log so the report and budgets run anywhere, including CI. Known
> limitations: an in-memory cache per process (lost on restart, not shared between servers; production would use
> Redis); no redaction of prompts in traces (fine for my notes, not for student data); no sampling (every request is
> traced); Langfuse runs locally without backups; the report's p95 of 10 answers is just the slowest answer; budgets
> are checked after the fact, not enforced per request.

**Prompt 3 (plan mode):**
> Add a section "## Observability, latency, cost and caching" to @INTERVIEW.md with 5 questions and model answers
> (5–8 sentences each, plain language), using this repo and my numbers in evals/runs/observability-qwen2.5-3b.txt:
> 1. What are traces and spans? How did you instrument CourseMate, and why OpenTelemetry rather than a vendor SDK?
> 2. How do you measure latency in a streaming LLM app? Why p95 and not the average, and what did you measure?
> 3. How does your cache work? What's in the key, when is caching wrong, and how does it compare with a semantic cache
>    and with the providers' prompt caching?
> 4. How do you know what an answer costs, and how would you keep a monthly budget in production?
> 5. LLM system design: design observability for an AI teaching assistant serving 5,000 students on a hosted model.
>    Cover what to trace on each request, sampling, personal data in prompts (redaction, retention, who can see
>    traces), alerts (p95 latency, error rate, refusal rate, cost per day), dashboards, linking traces to evals and
>    to student feedback, and what the observability itself costs.

**You check:** every number in the README and in answer 2 is one you actually saw. Rewrite answer 3 in your own words,
using your Lab 4 / Lab 5 cosine from Step 7.

**Done when:**

```bash
py -m ruff check . && py -m ruff format --check . && py -m pytest -q
```

all pass (I had `272 passed, 1 skipped, 8 deselected`; yours will differ a little). **Commit:** `Document
observability: README, interview questions, live cache test`

Push the branch and open a pull request:

```text
!git push -u origin observability
!gh pr create --base main --title "Observability: traces in Langfuse, latency and cost, a response cache" --body "Every answer and agent run is an OpenTelemetry trace (Langfuse, self-hosted, plus a JSONL span log), an exact-match response cache in front of the model, bench and stats commands with p95 and cost budgets, a small HTML dashboard, and a real run on qwen2.5:3b."
!gh pr checks --watch
```

**What you should see:** a PR link, then the checks finishing with `pass` (CI runs the new tests and the fake bench +
stats smoke test; it never needs Langfuse). If a step fails, open the PR on GitHub, click the failed check, read the
error at the bottom, and ask Claude to fix it. Leave the PR open: when you tell the roadmap chat "done", it reviews
the PR before you merge it.

Finally, stop Langfuse: `cd ~/langfuse-local && docker compose down`.

---

## Check your understanding (for the reading day)

Answer in your own words first, then open the answer.

1. A trace shows `chat` 21,000 ms, `retrieve` 90 ms, `llm` 20,800 ms with the first word at 9,000 ms, `check` 1 ms.
   Where does the time go, what did the student experience, and what would you **not** bother optimizing?
   <details><summary>Answer</summary>The model is 20,800 of 21,000 ms (about 99%); retrieval (90 ms) and our own code
   (21,000 − 90 − 20,800 − 1 = 109 ms) don't matter. The student waited about 90 + 9,000 ≈ 9.1 s for the first word
   and 21 s for the full, checked answer. Don't optimize retrieval or the database. Attack the model time: a shorter
   prompt (fewer or smaller chunks: 9 s before the first word means reading the prompt is slow on a CPU), fewer
   output tokens, a smaller or faster model, a GPU or a hosted model, and the cache for repeats.</details>
2. Why does `span()` end the span in `finally`, and why does it catch `Exception` but not `GeneratorExit`?
   <details><summary>Answer</summary>Only an ended span is exported, so a span that doesn't end on an error would make
   the failed request (the one you most need) disappear; finally runs whether the block finished, raised or was
   stopped. The except block marks real failures as ERROR and re-raises. GeneratorExit is what Python raises when a
   generator is closed from outside (the browser went away); that's not a failed step, so it isn't marked ERROR, but
   finally still ends the span.</details>
3. What goes wrong without `aclosing` when a student closes the page after the first word?
   <details><summary>Answer</summary>The web server closes the outer generator, but the inner one (which holds the open
   llm span) is just abandoned; Python closes abandoned async generators later, through the garbage collector, maybe
   only at exit. Until then the llm span stays open and isn't exported, so the trace has a root and a retrieve span
   but no model call, exactly when you'd want to see how far it got. aclosing closes the inner generator when the
   outer block ends, so its finally blocks run and the spans end right away.</details>
4. For each request, hit or miss, if "When is Lab 4 due?" on qwen2.5:3b at temperature 0 is cached: (a) the same
   question after you edit the Lab 4 section of your notes; (b) the same question at temperature 0.7; (c) the same
   question after `ollama pull qwen2.5:3b` downloaded new weights; (d) "when is lab 4 due?".
   <details><summary>Answer</summary>(a) Miss: the retrieved chunk text changed, and it's inside the messages, so the
   key changed. (b) Miss, and nothing is stored: CachedStreamer never caches above temperature 0. (c) Hit, and that's
   the danger: the model name is the same, so the key is the same, and you'd get the old model's answer until the TTL
   expires (a reason for a TTL, or to put a model version or digest in the key). (d) Miss: different capital letters
   are a different string; an exact cache only hits exact repeats.</details>
5. Ten answers: five cached at 40–60 ms, five from the model at 9,000–16,000 ms. Give p50 and p95 of all ten. Why
   does the report split cached and not-cached rows?
   <details><summary>Answer</summary>Sorted, the five cached times come first, so p50 (rank 5) is the slowest cached
   time, about 60 ms, and p95 (rank 10) is the slowest model answer, 16,000 ms. The overall p50 suggests CourseMate
   answers in 60 ms, which is true only for repeats; every new question still takes 9–16 s. Split rows show both
   truths, and the not-cached row is the one to budget and improve.</details>
6. Your report says `cost per answer $0.000000` and Langfuse says `$0`. Why is that true and also misleading? Which
   line of the report answers "what would this cost in production"?
   <details><summary>Answer</summary>True: qwen2.5:3b runs on your PC, so price_for returns FREE for a local model.
   Misleading: you pay in time (seconds per answer on a CPU) and in electricity and hardware, and production would
   likely use a hosted model. The "if hosted" lines answer it: mean tokens per model call × each hosted price × the
   share of answers that reached the model (cached and refused answers cost nothing).</details>
7. You resume yesterday's session with `claude --resume observability`. What does Claude still know, what doesn't it
   know, and what do you check first? When would you use `/branch` instead of just asking in the same session?
   <details><summary>Answer</summary>It knows the whole conversation: what it read (as it was then), its plans, your
   corrections and its permission mode. It doesn't know what happened outside the session since: your own edits, a
   pull, a merge. So first check git status and the last commits and compare them with its note. Use /branch for an
   experiment or a side question you may throw away (like the semantic cache): the main conversation stays focused,
   and the branch is still in the picker if you want it later.</details>

---

## When you're done

Tell the roadmap chat **"done"**. It will review the PR, run the tests and tick this row.

## Level up

**Name a session per task, stop at a commit, and resume instead of re-explaining.** Start with `claude -n <task>` (or
`/rename`), end each day at a committed checkpoint with a short "where we are" note from Claude, and come back with
`claude --resume <task>` (or `claude --continue` for the latest one in this folder). The first prompt after a resume
is always "check git status and the last commits against your note", because the session can't see what changed
outside it. Put experiments and side questions in a `/branch` (or `claude --continue --fork-session`) so the main
thread stays clean, and start a fresh session (or `/compact`) when `/context` shows a resumed conversation is mostly
old detail. And the same habit for your own app: **measure before you optimize**. A trace tells you which step is
slow, p95 tells you what your slowest users feel, and a budget in CI tells you the day it gets worse.
