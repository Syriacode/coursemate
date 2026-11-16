# Phase 2 Mon — CourseMate API + UI: streaming answers, a chat page, and three LLM providers (step-by-step guide)

> Read the **Project overview** first, on its own, before you open a terminal. It teaches today's ideas with small
> examples you can work out on paper. The steps start at Step 0.
>
> Pace: about 2 hours of hands-on time. There's a natural place to stop after **Step 5** (the API works from the
> terminal). The **Check your understanding** questions at the end are for the reading day.

---

## Project overview

### 1. First, what you have so far (in plain words)

`coursemate` answers questions about your course notes and says where each answer came from. You built it in two
projects:

- **Nov 11 (RAG):** notes are split into chunks by heading, each chunk becomes a vector (768 numbers) with
  `nomic-embed-text`, and the vectors live in Postgres + pgvector. `ask` finds the 4 closest chunks, gives them to
  `qwen2.5:3b` with the question, and checks the JSON reply. If anything looks wrong, it says "I don't know".
- **Nov 12 (retrieval):** you measured search with a golden set of 25 questions (recall@k, MRR), compared vector,
  keyword (BM25) and hybrid search, and kept **vector** because it won on your numbers.

```text
 ingest (once):  notes/*.md ──► chunks ──► embeddings ──► Postgres + pgvector

 ask:  question ──► RETRIEVE the 4 closest chunks ──► "I don't know" gate 1 (score too low?)
                ──► GENERATE: model reads ONLY those chunks ──► gate 2 (valid JSON? real citations?)
                ──► answer + [1] syllabus.md > Late work
```

Where that lives in your code (you'll reuse most of it today):

| Part | File and function |
|---|---|
| Find the closest chunks (vector, keyword or hybrid) | `coursemate/retrieval.py`, `retrieve()` |
| Number the sources for the prompt | `coursemate/rag.py`, `format_sources()` |
| The `ask` pipeline with both gates | `coursemate/rag.py`, `answer()` |
| The "I don't know" text and threshold | `coursemate/rag.py`, `IDK` and `DEFAULT_MIN_SCORE` |
| Talk to the model (one reply, all at once) | `coursemate/llm.py`, `LLMClient`, `OllamaClient.chat()`, `FakeClient` |
| Build an index in memory, without Postgres | `coursemate/retrieval_eval.py`, `build_index()` |
| Commands | `coursemate/cli.py`, `build_parser()` and `main()` |

### 2. What this project is about

Today CourseMate stops being a command-line tool and becomes something a student can use: a **web page** where
you type a question and watch the answer appear word by word, and can ask a follow-up ("and for labs?").

That needs four new ideas, and each one is something AI engineer job posts ask for by name:

1. **Streaming:** send the answer while the model is still writing it, using **server-sent events (SSE)**.
2. **Async Python:** a web server that can wait on many slow model calls at once without freezing.
3. **Sessions (conversation history):** the model remembers nothing, so *you* keep the conversation and resend it.
4. **Hosted LLM APIs:** the OpenAI and Anthropic API formats, next to Ollama, behind one small interface, so
   switching provider is a setting, not a rewrite. Then you **compare latency and cost**.

**The zero-cost rule still holds: nothing today costs money and you need no API key.** Here's how:

- The OpenAI and Anthropic code is tested with **recorded-style responses** (no network), like your other tests.
- Ollama itself speaks both formats: it has an **OpenAI-compatible endpoint** (`http://localhost:11434/v1`) and an
  **Anthropic-compatible endpoint** (`http://localhost:11434/v1/messages`). So your OpenAI and Anthropic code runs
  live, for free, against your own local model. Pointing the same code at `api.openai.com` or `api.anthropic.com`
  would only need a URL and a key (which we won't use).
- Costs for hosted models are **calculated** from your real token counts and the providers' published prices.

### 3. The ideas you need today (work these on paper)

#### 3a. Why stream? Time to first token

A model writes one token at a time. On your laptop's CPU, a CourseMate answer goes roughly like this (made-up but
realistic numbers; you'll measure yours in Step 9):

```text
 reading the prompt (≈ 700 tokens)       4 s     ← nothing can come out yet
 writing the answer (60 tokens, 10/s)    6 s
 total                                  10 s
```

- **Without streaming:** the page shows nothing for **10 s**, then the whole answer.
- **With streaming:** the first word shows at **4 s**, then words keep coming until 10 s.

The total time is the same. What changes is the **time to first token (TTFT)**: 4 s instead of 10 s. People judge
speed by when something starts happening, so TTFT is the number chat apps care about most. You'll measure both
TTFT and total time for every answer.

#### 3b. Server-sent events (SSE): the format of a stream

**SSE** is a simple text format for a server to push a series of messages over one ordinary HTTP response. Each
message (an **event**) is a few lines, and **a blank line ends it**:

```text
 event: token
 data: {"text": "Projects "}
                                    ← blank line: event finished, the browser can use it now
 event: token
 data: {"text": "lose 10% per day [1]."}

 event: done
 data: {"found": true, "answer": "Projects lose 10% per day [1].", "sources": [...]}

```

- `event:` is the name (we use `session`, `token`, `done`, `error`).
- `data:` is the content. We always put **JSON** there, so a newline inside the answer can't break the format:
  `json.dumps` writes a newline as the two characters `\n`.
- The response has the content type `text/event-stream` and is never "finished" until the server closes it.

You'll hand-type the function that writes one event. Work it out on paper first:

```text
 format_sse("token", {"text": "Hi"})
   = "event: token" + "\n" + "data: " + '{"text": "Hi"}' + "\n" + "\n"
   = 'event: token\ndata: {"text": "Hi"}\n\n'
```

**Ollama's own API streams differently:** one JSON object per line, called **NDJSON** (newline-delimited JSON),
with no `event:`/`data:` lines. OpenAI and Anthropic both use SSE. Same idea, different packaging, which is
exactly why we hide it behind one interface.

#### 3c. Async Python: one waiter, many tables

A web server spends almost all its time **waiting**: for the database, for the embedding model, and above all for
the chat model. Think of a waiter in a restaurant:

- **Synchronous (blocking):** the waiter takes table 1's order, walks to the kitchen and **stands there** until the
  food is ready, then serves it. Only then does table 2 get a waiter.
- **Asynchronous:** the waiter takes table 1's order, hands it to the kitchen, and **while the kitchen cooks**, takes
  table 2's order. Whenever a dish is ready, the waiter brings it.

Numbers: three students ask at the same moment, and each answer is 2 s of waiting on a hosted API.

```text
 one blocking worker:   student 1 done at 2 s, student 2 at 4 s, student 3 at 6 s
 async (one thread):    all three done at about 2 s: the waits overlap
```

In Python, `async def` makes a function that can pause, and `await` is where it pauses ("I'm waiting; serve someone
else"). The **event loop** is the waiter: one thread that runs whichever task is ready.

**The one rule you must not break:** never call slow, *blocking* code directly inside an `async def`. It stops the
waiter for everyone. Your embedder (`httpx.Client`) and your Postgres store (`psycopg`) are blocking, so we run
them with `await asyncio.to_thread(...)`: "a helper thread does this, the waiter keeps serving". The model call
uses `httpx.AsyncClient`, which is truly async.

(Honest note: Ollama on your CPU answers one question at a time anyway, so locally the model is the bottleneck. With
a hosted API, which handles many requests in parallel, async is what lets one server handle hundreds of students.)

#### 3d. Conversation history: the model remembers nothing

Every call to a model starts from zero. A "conversation" is just **you resending the earlier turns** each time:

```text
 Turn 1 sends:  [system] [user: "What is the late penalty for projects?" + sources]
 Turn 2 sends:  [system] [user: "What is the late penalty for projects?"] [assistant: "10% per day [1]."]
                [user: "And after three days?" + sources]
```

Two consequences:

- **Prompts grow.** Say the system prompt plus sources is 700 tokens and each earlier turn adds 80. Turn 6 sends
  700 + 5 × 80 = **1,100** input tokens. So we keep only the **last 5 turns**.
- **Retrieval needs the history too.** "And after three days?" means nothing on its own, so its search would find the
  wrong notes. A simple fix: search with **the previous question + the new one**
  ("What is the late penalty for projects? And after three days?"). Real systems often ask the model to rewrite the
  question instead (**query rewriting**); that costs one more model call.

The server keeps each conversation under a **session id** (a random string). The browser sends the id with every
question. We keep sessions in memory today, so they disappear when the server restarts (a known limitation).

#### 3e. Streaming vs. checking: you show words before you can check them

Your `ask` command checks the whole JSON reply before printing anything. With streaming, the student sees words
**as they arrive**, but you can only check citations **at the end**. So the streaming answer works like this:

1. Gate 1 (retrieval score) runs **before** any streaming, as before.
2. Instead of JSON, the model writes **plain text with citations like `[1]`** (JSON is awkward to show while it's
   half-written).
3. When the stream ends, check: are there citations? Do they point at sources that exist? Did it say "I don't know"?
4. If the check fails, the final `done` event says `found: false`, and **the page replaces the streamed words with
   "I don't know"**. This is called **retracting**.

Example with 2 retrieved sources:

```text
 streamed text                               check                         page shows at the end
 "Projects lose 10% per day late [1]."       cites [1], exists             the answer + source [1]
 "Projects lose 10% per day late."           no citation                   "I don't know..."
 "Projects lose 10% per day late [7]."       [7] doesn't exist (only 2)    "I don't know..."
```

This trade-off (fast feedback vs. checked output) is a classic interview topic.

#### 3f. Three APIs, one idea

All three providers take the same thing (a list of messages) and give back the same thing (text, then token counts).
Only the packaging differs:

| | Ollama (native) | OpenAI Chat Completions | Anthropic Messages |
|---|---|---|---|
| Endpoint | `POST /api/chat` | `POST /v1/chat/completions` | `POST /v1/messages` |
| Key goes in | (none) | `Authorization: Bearer <key>` | `x-api-key: <key>` + `anthropic-version` header |
| System prompt | a `system` message | a `system` message | a separate `system` field |
| Max length | optional (`num_predict`) | optional (`max_completion_tokens`) | **required** (`max_tokens`) |
| Stream format | NDJSON lines | SSE `data:` lines, ends with `data: [DONE]` | SSE with typed events |
| Text pieces in | `message.content` | `choices[0].delta.content` | `content_block_delta` → `delta.text` |
| Token counts in | the last line (`prompt_eval_count`, `eval_count`) | a last chunk, **only if you ask** (`stream_options.include_usage`) | `message_start` (input) and `message_delta` (output) |

Your code hides all of this behind one method, `stream(messages)`, that yields **`TextDelta`** pieces and then one
**`StreamEnd`** with the model name, token counts, time to first token and total time. Everything above it (the chat
logic, the API, the page) never knows which provider it's talking to.

**Why not the official SDKs** (`pip install openai anthropic`)? They're what you'd use at work, and they do exactly
what you'll write today (plus retries and types). Writing it once with plain `httpx` shows you what they hide, and
lets one test style (`httpx.MockTransport`) cover all three. Interviewers like hearing that you know both.

#### 3g. What an answer costs

Hosted models charge **per million tokens (MTok)**, with output tokens dearer than input tokens. Prices on the
providers' pricing pages on Nov 15, 2026 (they change often, so check again before you quote them):

| Model | Input $/MTok | Output $/MTok |
|---|---|---|
| Claude Haiku 5.5 (`claude-haiku-5-5`, prompts up to 100k tokens) | 0.10 | 0.50 |
| Claude Sonnet 5.5 (`claude-sonnet-5-5`) | 2.00 | 10.00 |
| GPT-6 Luna (`gpt-6-luna`) | 0.10 | 0.50 |
| GPT-6.1 Sol (`gpt-6.1-sol`, up to 272k input tokens) | 2.00 | 10.00 |

The formula, which you'll hand-type:

```text
 cost = (input_tokens × input_price + output_tokens × output_price) / 1,000,000

 one CourseMate answer: 700 tokens in, 60 out
 Sonnet 5.5:  (700 × 2.00 + 60 × 10.00) / 1,000,000 = (1,400 + 600) / 1,000,000 = $0.002
 Haiku 5.5:   (700 × 0.10 + 60 × 0.50)  / 1,000,000 = (70 + 30)     / 1,000,000 = $0.0001

 5,000 questions a day × 30 days = 150,000 answers a month
 Sonnet 5.5: 150,000 × $0.002  = $300 a month
 Haiku 5.5:  150,000 × $0.0001 = $15 a month
 Local Ollama: $0 per token, but your hardware, slower answers, and one question at a time
```

Notice that the **input** is most of the cost here: 700 tokens of sources and system prompt. That's why `k` (how
many chunks you send) and the 5-turn history limit matter for cost, not just quality.

### 4. What we'll build (the parts and how they fit)

```text
 Browser: coursemate/static/index.html
   │  POST /api/chat {"question": "...", "session_id": "..."}
   │  ◄── SSE: session, token, token, token, ..., done (answer, sources, tokens, timings, cost)
   ▼
 coursemate/api.py            FastAPI app (async): /, /api/health, /api/sessions, /api/chat
   │   sessions.py            SessionStore: session id → last 5 turns (in memory)
   │   sse.py                 format_sse()  (you type it)  and sse_data() (reads SSE)
   ▼
 coursemate/chat.py           stream_answer(): retrieve (in a thread) → gate 1 → stream the model
   │                          → check citations at the end → retract if needed
   ▼
 coursemate/streaming.py      StreamingLLM.stream(messages) → TextDelta ... StreamEnd
        ├── OllamaStreamer     /api/chat, NDJSON                    ┐
        ├── OpenAIStreamer     /v1/chat/completions, SSE            ├─ all three run on your local Ollama today
        ├── AnthropicStreamer  /v1/messages, SSE                    ┘
        └── FakeStreamer       no model, for tests
 coursemate/pricing.py        price list + cost_usd()  (you type it)
 coursemate/compare.py        same questions through several providers: latency, tokens, cost
 cli.py                       new commands: serve, compare
```

| New or changed file | What it is |
|---|---|
| `coursemate/sse.py` | write one SSE event (`format_sse`, you type it) and read a stream of them (`sse_data`) |
| `coursemate/streaming.py` | the `StreamingLLM` interface and its four backends |
| `coursemate/chat.py` | `stream_answer()`, citation check, history in the prompt, follow-up search |
| `coursemate/sessions.py` | `SessionStore` (conversation history in memory) |
| `coursemate/api.py` | the FastAPI app, `create_app()` |
| `coursemate/static/index.html` | the chat page (HTML + plain JavaScript, no build step) |
| `coursemate/pricing.py` | prices + `cost_usd` (you type it) |
| `coursemate/compare.py` | the latency and cost comparison |
| `coursemate/cli.py` | `serve` and `compare` commands |
| `tests/test_*.py` | one test file per new module, plus `tests/test_live_streaming.py` (marked `live`) |

**Same repo, on a branch** `api`, merged with a pull request, like Nov 12.

**Why plain JavaScript and not TypeScript?** The roadmap row says HTML/TS. TypeScript needs Node and a build step for
one small page, so the page is plain JavaScript in one file today. Converting it to TypeScript is a good small task
for a later day (it shows up in forward-deployed job posts).

**Why a new interface next to `LLMClient`?** Your `LLMClient.chat()` is synchronous and returns the whole reply as
JSON. Streaming needs an async method that yields pieces, so `StreamingLLM` is a sibling interface in the same
spirit: provider-agnostic, with a fake for tests. `ask` keeps using `LLMClient`; the web app uses `StreamingLLM`.

### 5. How we'll test it

1. **Unit tests, no model, no network** (the default `py -m pytest`):
   - your two hand-typed functions with the **paper numbers** (`'event: token\ndata: {"text": "Hi"}\n\n'`, $0.002);
   - each provider's wire format, with recorded-style responses fed through `httpx.MockTransport` (the same trick as
     `tests/test_llm.py`), including the headers and body we send;
   - `stream_answer()` with `FakeStreamer` and `MemoryStore`: tokens, then a checked final answer, and every way a
     reply gets retracted;
   - the API with FastAPI's `TestClient`: the SSE events in order, sessions, bad input.
2. **Live tests, marked `live`,** run by hand: the three wire formats against your real local model (free).
3. **By hand in a terminal:** `curl -N` shows the raw SSE stream arriving piece by piece.
4. **In a real browser, with Claude:** today's Claude Code technique (section 7).
5. **CI** runs the unit tests and starts the server with the fake provider to check one real SSE answer.

### 6. New words (short definitions)

- **Streaming:** sending a response in pieces while it's being produced.
- **TTFT (time to first token):** time from sending the request to the first piece of the answer.
- **SSE (server-sent events):** a text format for a server to push events over one HTTP response:
  `event:` + `data:` lines, blank line between events, content type `text/event-stream`. One-way: server → browser.
- **NDJSON:** one JSON object per line. Ollama's native streaming format.
- **WebSocket:** a two-way connection between browser and server. More powerful than SSE, more complex; a chat that
  only needs server → browser streaming doesn't need it.
- **Async / `async def` / `await`:** functions that can pause while waiting, so one thread can serve many requests.
  **Event loop:** the thing that runs whichever paused function is ready. **Blocking call:** code that makes the whole
  thread wait (like `time.sleep` or a normal `httpx.Client` request).
- **`asyncio.to_thread`:** run a blocking function in a helper thread and `await` its result.
- **FastAPI:** a Python web framework built for async; it reads type hints and Pydantic models to validate requests.
- **Uvicorn:** the server program that runs a FastAPI app (an **ASGI** server: the standard way async Python web apps
  talk to servers).
- **Endpoint / route:** one URL + method the API answers, like `POST /api/chat`.
- **Session:** one conversation, identified by a session id; **history:** its earlier turns.
- **Retracting:** replacing words you already streamed, because the final check failed.
- **Provider-agnostic:** code that works with any model provider through one interface.
- **OpenAI-compatible / Anthropic-compatible endpoint:** a server that speaks the same request and response format
  as OpenAI's or Anthropic's API. Ollama has both, which is how we test today for free.
- **MTok:** a million tokens, the unit for API prices.
- **TestClient:** FastAPI's way to call your app in tests without starting a real server.
- **Claude in Chrome:** a browser extension that lets Claude see and use web pages in your Chrome; Claude Code can
  connect to it with `--chrome`.

### 7. Today's Claude Code technique: testing a web page with Claude in a browser

Your unit tests prove the **server** sends the right events. They don't prove the **page** works: that words appear
one by one, the button disables while answering, a retracted answer turns into "I don't know", nothing errors in the
browser console, and a question containing HTML shows up as text instead of being run.

Claude Code can **drive your real Chrome** through the **Claude in Chrome** extension: open a page, type, click,
read what's on screen, read the **console errors**, and report back, then fix the code in the same session.

```text
 Claude Code (terminal) ──► Chrome extension ──► http://localhost:8000
      ▲  reads the code                              types a question, watches the answer stream,
      └── fixes bugs ◄── reports what it saw ◄────── reads console errors, tries the unhappy paths
```

What you need (you'll check these in Step 7):

- Google Chrome (or Microsoft Edge) with the **Claude in Chrome** extension, version 1.0.36 or later, signed in to
  the same Claude account as Claude Code.
- Claude Code signed in with `/login` on a paid plan (Pro, Max, Team or Enterprise). It doesn't work when Claude
  Code uses an API key.
- Start Claude Code with `claude --chrome`, or type `/chrome` inside a running session.

It runs in a visible Chrome window, so you can watch every click. It uses your real browser, so keep it to
`localhost` today. If it doesn't connect on your machine, Step 7 has a manual checklist that tests the same things.

### 8. Time and cost

About **2 hours** of hands-on time, plus reading. Stop after **Step 5** if you want to split it. **Cost: $0.** No
API keys, no hosted calls: everything runs on your local Ollama.

---

## How to use this guide with Claude Code

Every step has:

- **Learn:** the idea. Read it first.
- **Predict:** write down what you expect *before* running anything.
- **Prompt:** what to type into Claude Code in **plan mode** (`Shift+Tab` until you see `⏸ plan mode on`). Prompts
  are short on purpose: they say *what* and *why*. Read Claude's plan, ask "why?" about anything unclear, then approve.
- **You check:** what you verify yourself, with the file and function to open.
- **Done when:** a command whose output proves the step works.
- **Commit:** the message.

**Python is `py` on your machine.** Inside Claude Code, run commands with `!`, for example `!py -m pytest -q`.

You'll often need **two Git Bash windows**: one running Claude Code, one for running the server and `curl`. In every
new window, first run `cd coursemate` and `source .venv/Scripts/activate`.

---

## Step 0 — Get ready: branch, install, check Ollama speaks all three formats (15 min)

**Learn.** Before writing code, check the tools. Two new libraries: **FastAPI** (the web framework) and **uvicorn**
(the server that runs it). And one check that matters today: your Ollama must be new enough to have the
**Anthropic-compatible** endpoint, or Step 8 can't run live.

In Git Bash:

```bash
cd coursemate
source .venv/Scripts/activate
git status
git switch main
git pull
git switch -c api
docker compose up -d
py -m pip install fastapi uvicorn
py -m pytest -q
```

**What you should see:**
- `git status`: `nothing to commit, working tree clean` (commit or stash first if not).
- `git pull` brings in this guide, `GUIDE-api.md`.
- `Switched to a new branch 'api'`.
- pip ends with `Successfully installed ... fastapi-... uvicorn-...` (plus a few helpers like `starlette`).
- pytest: `69 passed, 2 deselected` (with Docker running).

Now check Ollama. Start the Ollama app if it isn't running, then:

```bash
ollama --version
ollama list
```

You should see `qwen2.5:3b` and `nomic-embed-text`. **Predict:** the next two commands send "Say hi" to your local
model in the **OpenAI** format and in the **Anthropic** format. Will the two replies look the same?

```bash
curl -s http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "Say hi"}]}'
echo
curl -s http://localhost:11434/v1/messages -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:3b", "max_tokens": 30, "messages": [{"role": "user", "content": "Say hi"}]}'
echo
```

**What you should see:** two JSON replies from the same model, packaged differently. The first has
`"choices": [{"message": {"content": "Hi! ..."}}]` and `"usage": {"prompt_tokens": ..., "completion_tokens": ...}`.
The second has `"content": [{"type": "text", "text": "Hi! ..."}]` and
`"usage": {"input_tokens": ..., "output_tokens": ...}`. Same model, two API formats: that's today's idea in one
picture.

**If the second command says `404 page not found`,** your Ollama is too old for the Anthropic format: download the
latest from ollama.com, install it over the old one, and run the command again.

Now start Claude Code:

```bash
claude
```

**Prompt (plan mode):**
> Today I add a streaming web API and chat page to CourseMate (see GUIDE-api.md). First, small setup only:
> 1. Add `fastapi>=0.115` and `uvicorn>=0.30` to the dependencies in pyproject.toml.
> 2. In CLAUDE.md, add these rules: tests never call a real model or the network (use FakeStreamer, FakeClient,
>    FakeEmbedder, and httpx.MockTransport for HTTP); live tests are marked `@pytest.mark.live`; never call a
>    hosted API (api.openai.com, api.anthropic.com), because it costs money; async tests use
>    `@pytest.mark.anyio`.
> 3. In tests/conftest.py add an `anyio_backend` fixture that returns "asyncio", so async tests run with the anyio
>    pytest plugin that comes with FastAPI.
> Then reinstall with `py -m pip install -e ".[dev]"` and run the tests.

**You check (by hand, one number).** Open `coursemate/rag.py` and find `DEFAULT_MIN_SCORE = 0.0` with a TODO from
Nov 11. Zero means gate 1 is **off**, so off-topic questions always reach the model. Your README says on-topic
questions scored 0.64–0.74 and off-topic ones 0.48–0.49. Pick a number in the gap, for example `0.55`, type it in,
and delete the TODO line. Then open `README.md` and replace `My number: TODO.` with `My number: 0.55.` (or yours).

**Done when:** `!py -m pytest -q` shows `69 passed, 2 deselected`.

**Commit:** `Add FastAPI and uvicorn; set the retrieval threshold`

---

## Step 1 — Write one SSE event by hand (10 min)

**Learn.** Re-read section 3b. The whole streaming API rests on one tiny function. You'll type it yourself.

**Predict.** What exact string does `format_sse("token", {"text": "a\nb"})` return? Careful with the newline
inside the text.

**Prompt (plan mode):**
> Create coursemate/sse.py with:
> 1. A stub `def format_sse(event: str, data: dict[str, Any]) -> str:` with a docstring ("One server-sent event: a
>    name line, a JSON data line, then a blank line.") and `raise NotImplementedError`. I'll write the body myself.
> 2. `async def sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]`: reads lines of an SSE stream and
>    yields the data of each event (an event ends at a blank line; join several `data:` lines with "\n"; skip
>    `event:`, `id:`, `retry:` and `:` comment lines). OpenAI and Anthropic streams will be read with it.
> Then create tests/test_sse.py with the tests below, plus tests for sse_data: it reads back what format_sse
> writes, it skips a `: keep-alive` comment, and it joins two data lines. Don't run them yet.

Then paste these two tests into the prompt as well (they use the paper numbers):

```python
def test_format_sse_matches_the_paper_example():
    assert format_sse("token", {"text": "Hi"}) == 'event: token\ndata: {"text": "Hi"}\n\n'


def test_format_sse_keeps_a_newline_inside_the_text_on_one_data_line():
    # json.dumps writes a newline as the two characters \n, so the event can't break early
    assert format_sse("token", {"text": "a\nb"}) == 'event: token\ndata: {"text": "a\\nb"}\n\n'
```

**Now type the body yourself** in `coursemate/sse.py`, replacing `raise NotImplementedError`. One line: an
f-string with `event`, then `json.dumps(data)`, then two newlines (remember `import json` at the top).

**You check:** in `sse_data()`, find the line that handles a blank line: that's the "event finished" moment from
section 3b. Ask Claude: *"Why does sse_data need to handle a stream that ends without a final blank line?"*

**Done when:** `!py -m pytest -q tests/test_sse.py` shows all passed. Then break it on purpose: change your `\n\n`
to a single `\n`, run the tests, watch the paper-example test fail, and change it back.

**Commit:** `Add format_sse (by hand) and an SSE reader`

---

## Step 2 — One streaming interface: Ollama and a fake (15 min)

**Learn.** Re-read section 3f. Before code, *see* Ollama's native stream. In your second Git Bash window:

```bash
curl -N http://localhost:11434/api/chat -d '{"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "Count from 1 to 5"}]}'
```

(`-N` tells curl not to buffer, so you see lines as they arrive.)

**Predict:** how many lines will you get, and what's different about the last one?

**What you should see:** lines appear one by one, each a JSON object like
`{"model":"qwen2.5:3b",...,"message":{"role":"assistant","content":"1"},"done":false}`. Each holds one small piece of
text (a token or two). The **last** line has `"done":true` and the counts: `prompt_eval_count` (input tokens) and
`eval_count` (output tokens), plus timings in nanoseconds. That's NDJSON.

**Prompt (plan mode):**
> Create coursemate/streaming.py: a provider-agnostic streaming interface (see GUIDE-api.md section 3f).
> - Two frozen dataclasses: `TextDelta(text)` and `StreamEnd(model, input_tokens, output_tokens, first_token_ms,
>   total_ms)`.
> - A Protocol `StreamingLLM` with attributes `provider`, `model`, `local` (True when nothing leaves this computer)
>   and `def stream(messages, *, temperature=0.0, max_tokens=512) -> AsyncIterator[TextDelta | StreamEnd]`: it
>   yields TextDelta pieces, then exactly one StreamEnd.
> - `OllamaStreamer`: like OllamaClient in llm.py but with `httpx.AsyncClient` and `"stream": true` on /api/chat,
>   reading NDJSON lines; options temperature, num_ctx (8192) and num_predict (= max_tokens); time to first token
>   and total time measured with time.perf_counter; HTTP errors become LLMError. Accept an optional `http`
>   AsyncClient for tests.
> - `FakeStreamer`: no network; takes a list of replies (or exceptions) like FakeClient, records the messages it
>   got in `.calls`, streams each reply word by word, then StreamEnd("fake", 100, 20, 1.0, 2.0). Default reply:
>   "Fake provider: no model was called [1]."
> - `make_streamer(provider, model, ollama_url)`.
> Tests in tests/test_streaming.py with httpx.MockTransport and a recorded-style NDJSON body (no network): the
> pieces join into the right text, token counts come from the last line, and the request has stream true and the
> right options. Plus a FakeStreamer test.

**You check:** open `coursemate/streaming.py`, `OllamaStreamer.stream()`. Find (1) the `async with ... stream(...)`
line: the request stays open while you read it; (2) `aiter_lines()`: lines arrive as Ollama sends them; (3) where
`first_token_ms` is recorded: only on the **first** piece that has text. Ask Claude: *"Why is stream() an async
generator (async def with yield) instead of returning a list?"*

**Done when:** `!py -m pytest -q` shows `0 failed`.

**Commit:** `Add a streaming LLM interface with Ollama and fake backends`

---

## Step 3 — Streaming answers that are still checked, with history (20 min)

**Learn.** Re-read sections 3d and 3e. This is the heart of today: the RAG pipeline from `rag.py`, but streaming,
with history, and with the citation check moved to the end.

**Predict.** With 2 retrieved sources, which of these will the page finally show as an answer, and which as
"I don't know"? Write it down.

```text
 a) "Office hours are Tuesdays [2]."
 b) "Office hours are Tuesdays."
 c) "I don't know."
 d) "Office hours are Tuesdays [3]."
```

**Prompt (plan mode):**
> Create coursemate/chat.py: streaming RAG answers with conversation history (GUIDE-api.md sections 3d and 3e).
> - A frozen dataclass `Turn(question, answer)`.
> - `CHAT_SYSTEM_PROMPT`: like SYSTEM_PROMPT in rag.py, but the model replies in plain text, cites sources like
>   [1] or [2][3] after each fact, replies exactly "I don't know." when the sources don't contain the answer, keeps
>   to 5 sentences, and uses earlier messages only to understand the question.
> - `build_chat_messages(question, hits, history)`: system, then each earlier turn as a user and an assistant
>   message, then the sources (format_sources from rag.py) and the question.
> - `retrieval_query(question, history)`: the question alone, or the previous question + " " + the question.
> - `extract_citations(text)`: numbers in [n], first-use order, no duplicates.
> - `async def stream_answer(question, history, embedder, store, llm, *, k=4, min_score=DEFAULT_MIN_SCORE,
>   mode="vector", max_tokens=512)`: run `retrieve` with `await asyncio.to_thread(...)` (it's blocking code); if gate
>   1 fails, yield one `Final` with found False and IDK, without calling the model. Otherwise yield a `Token` for
>   every TextDelta, and when the stream ends, check the whole text and yield one `Final` (found, answer, sources,
>   citations = the [n] number of each source, reason, model, token counts, timings). Fail closed to IDK when the
>   text is empty or starts with "I don't know" (straight or curly apostrophe), has no citations, or cites a number
>   that wasn't retrieved.
> Tests in tests/test_chat.py with FakeStreamer, FakeEmbedder and MemoryStore (copy the NOTES fixture idea from
> tests/test_rag.py): tokens then a found Final; gate 1 refuses without a model call; each bad reply (no citation,
> [7], "I don't know.", "I don’t know.") streams its words but ends with found False; history reaches the model;
> follow-ups search with the previous question.

**You check:**
- `coursemate/chat.py`, `stream_answer()`: find the `asyncio.to_thread` line. That's the "helper thread" from 3c.
- Find where the `Final` is built after the loop: the citation check runs only **after** every token was already
  yielded. That's why the page has to retract.
- Check your prediction: a) answer, b) c) d) "I don't know".
- Ask Claude: *"What would happen to other users of the server if retrieve() were called directly, without
  asyncio.to_thread?"*

**Done when:** `!py -m pytest -q` shows `0 failed`.

**Commit:** `Stream checked RAG answers with conversation history`

---

## Step 4 — Sessions: where the history lives (10 min)

**Learn.** `stream_answer()` takes a history, but someone has to keep it between questions. A **session store** maps
a session id to its last turns.

**Prompt (plan mode):**
> Create coursemate/sessions.py with a `SessionStore` kept in memory: `create()` returns a new random id
> (uuid4 hex); `exists(id)`; `history(id)` returns a copy of the turns or raises `UnknownSession` (a KeyError
> subclass); `add(id, turn)` appends and keeps only the last `max_turns` (default 5). Also cap the number of
> sessions (`max_sessions`, default 1000): when full, drop the oldest. Tests in tests/test_sessions.py.

**You check:** `SessionStore.add()`: find the line that trims to the last 5 turns. Re-read the 1,100-token example in
3d. Ask Claude: *"What are two ways this in-memory store breaks when CourseMate runs as 3 server processes behind
a load balancer, and what would you use instead?"*

**Done when:** `!py -m pytest -q` shows `0 failed`.

**Commit:** `Add an in-memory session store`

---

## Step 5 — The async API, tried with curl (20 min)

**Learn.** Now wire it into **FastAPI**. Two FastAPI ideas:

- A **route** is a function with a decorator: `@app.post("/api/chat")` means "POST requests to /api/chat run this".
  If the function takes a Pydantic model (`ChatRequest`), FastAPI parses and validates the JSON body for you, and
  answers **422** for bad input before your code runs.
- **`StreamingResponse`** takes an async generator and sends each string it yields as soon as it's yielded. With
  `media_type="text/event-stream"`, that's SSE.

The app is built by a function, `create_app(embedder, store, llm)`, instead of at import time. That's **dependency
injection**: tests pass `FakeEmbedder`, `MemoryStore` and `FakeStreamer`; `serve` passes the real ones.

**Prompt (plan mode):**
> Create coursemate/api.py with `create_app(embedder, store, llm, *, sessions=None, k=4,
> min_score=DEFAULT_MIN_SCORE, mode="vector") -> FastAPI`:
> - `GET /` serves coursemate/static/index.html (create a placeholder page with title CourseMate for now).
> - `GET /api/health` → {"status": "ok", "provider", "model"}.
> - `POST /api/sessions` → {"session_id"}; `GET /api/sessions/{id}` → the turns, or 404.
> - `POST /api/chat` with a Pydantic body: `question` (whitespace stripped, 1–2000 characters) and optional
>   `session_id` (unknown id → 404; none → create one). It returns a StreamingResponse, media type
>   text/event-stream, header Cache-Control: no-cache, with SSE events made by format_sse: first `session`
>   {session_id}, then `token` {text} for each Token, then `done` with found, answer, sources (n = the citation
>   number, source, heading, score), reason, provider, model, token counts, first_token_ms, total_ms. Save the turn
>   (question + final answer) in the session when done. An LLMError or StoreError becomes an `error` {message}
>   event.
> Add a `serve` command to cli.py: `--llm` (ollama, openai, anthropic, fake; default = --provider), --host
> 127.0.0.1, --port 8000, -k, --mode, --min-score; it uses make_streamer, PgVectorStore (call prepare first, so a
> wrong embedding model fails with the usual message) and uvicorn.run. Add `[tool.setuptools.package-data]
> coursemate = ["static/*"]` to pyproject.toml.
> Tests in tests/test_api.py with FastAPI's TestClient and fakes: the events come in order (session, tokens, done)
> with the right content type; a second question in the same session sees the history; a retracted reply ends with
> found false; an LLMError becomes an error event; empty, blank and 2001-character questions get 422 without a
> model call; unknown session is 404.

If pytest prints a warning saying `Using httpx with starlette.testclient is deprecated`, it's harmless: newer
Starlette prefers a package called `httpx2` for TestClient. Ask Claude to add a `filterwarnings` entry for that one
message in `[tool.pytest.ini_options]`.

**You check:** in `coursemate/api.py`, the `chat()` route: find the inner `async def events()` generator and the
`yield format_sse(...)` lines. Your Step 1 function is now the whole wire format of the API.

**Done when:** first the tests: `!py -m pytest -q` shows `0 failed`. Then the real thing. In your **second** Git Bash
window (Claude Code stays in the first):

```bash
py -m coursemate ingest notes/sample --reset
py -m coursemate serve
```

You should see `Uvicorn running on http://127.0.0.1:8000`. (If port 8000 is taken, add `--port 8001` and use that
below.) Open a **third** Git Bash window and **predict** what you'll see, then:

```bash
curl -s http://localhost:8000/api/health
echo
curl -N -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question": "How many points do I lose for a late project?"}'
```

**What you should see:** the health check prints `{"status":"ok","provider":"ollama","model":"qwen2.5:3b"}`. The chat
prints `event: session` at once, then after a pause (the model reading the prompt), `event: token` blocks arrive a
few at a time, then one `event: done` with `"found": true`, a source from `syllabus.md`, the token counts,
`first_token_ms` and `total_ms`. Write down your `first_token_ms` and `total_ms`: that's 3a with
your own numbers.

Try an off-topic question to see gate 1 (no tokens at all, straight to `done` with `found: false`):

```bash
curl -N -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question": "Who won the 2022 FIFA World Cup?"}'
```

Leave the server running for Step 6, or stop it with `Ctrl+C`.

**Commit:** `Add the streaming chat API and serve command`

> **Good place to stop.** Everything after this builds on a working API.

---

## Step 6 — The chat page (10 min)

**Learn.** The browser has a built-in reader for SSE called `EventSource`, but it only does `GET` requests and it
**reconnects automatically** when the stream ends, which would ask the question again. So the page uses `fetch()`
with `POST` and reads the stream itself: it reads bytes, splits them at blank lines, and parses each `data:` line as
JSON. The same job as your Python `sse_data()`, in JavaScript.

One security rule: **model output is untrusted text.** If the page put it into the page as HTML (`innerHTML`), a note
or a student question containing `<img src=x onerror=...>` could run code in the browser. The page only ever sets
`textContent`.

**Prompt (plan mode):**
> Replace coursemate/static/index.html with a minimal chat page: one self-contained HTML file with plain JavaScript
> (no build step, no libraries, no external files). It shows provider and model from /api/health; a message list;
> a textarea (Enter sends, Shift+Enter is a new line) and a Send button that is disabled while an answer streams;
> a "New conversation" button that forgets the session id. On send: POST /api/chat with the question and the session
> id (kept in a JavaScript variable only), read the SSE stream with fetch and a ReadableStream reader, show tokens
> as they arrive, and on `done`: if found, show the answer and the sources as "[n] source > heading"; if not found,
> replace the streamed text with the answer ("I don't know...") in a different style. Below every answer show
> first-word time, total time and tokens in/out, plus the cost if the done event has a `cost_usd` field (Step 9
> adds it): "$0 (local)" when it's 0, "cost unknown" when it's null. Show `error` events and
> HTTP errors in red; on 404, forget the session id. Use textContent only, never innerHTML, for anything from the
> server or the user. Add `<link rel="icon" href="data:,">` so the browser doesn't request a missing favicon.
> Keep it short and commented for a beginner.

**You check:** open `coursemate/static/index.html`. Find (1) the function that reads SSE from `fetch` and splits at
`"\n\n"`; (2) the `done` handler that replaces the text when `found` is false; (3) that there's no `innerHTML`
anywhere (ask Claude to search for it).

**Done when:** restart the server (`Ctrl+C`, then `py -m coursemate serve`), open **http://localhost:8000** in your
browser, ask *"How many points do I lose for a late project?"* and watch the words appear. Then ask a follow-up in
the same conversation: *"And after three days?"*

**Commit:** `Add the chat page`

---

## Step 7 — Test the page with Claude in Chrome (15 min)

**Learn.** Re-read section 7. You'll ask Claude to be a careful tester: use the page like a student, try the
unhappy paths, read the console, and report. Then you fix what it finds.

**Set up (one time).**
1. In Chrome, install the **Claude** extension ("Claude in Chrome") from the Chrome Web Store and sign in with the
   same Claude account you use for Claude Code. Check its version on `chrome://extensions`: 1.0.36 or later.
2. In Claude Code, run `/status` and check you're signed in with your Claude account (not an API key). If not, run
   `/login`.
3. Exit Claude Code (`/exit`) and start it again with the browser connected:

```bash
claude --chrome
```

The first time, Claude Code shows a short dialog about the integration and site permissions: read it and continue.
Type `/chrome` to see the connection status. (If browser tools stop working later, `/chrome` → **Reconnect
extension**.)

Make sure the server is running in your second window (`py -m coursemate serve`).

**Predict.** Which of the checks in the prompt below do you expect to fail, if any?

**Prompt (plan mode is fine; it will ask before acting):**
> Test the CourseMate chat page in Chrome at http://localhost:8000 like a careful QA tester. Don't change any code
> yet. Check and report each item as pass or fail with what you saw:
> 1. The header shows the provider and model.
> 2. Ask "How many points do I lose for a late project?": text appears in several steps (not all at once), Send is
>    disabled while it streams, and the answer ends with a [n] source line and a timing line.
> 3. Follow-up in the same conversation: "And after three days?" gets an answer about late work.
> 4. "Who won the 2022 FIFA World Cup?" ends with the "I don't know" style.
> 5. Ask exactly: <b>What is the late policy?</b> — it must appear as literal text, not bold.
> 6. Pressing Enter on an empty box sends nothing.
> 7. "New conversation" clears the messages.
> 8. Read the browser console: report any errors or warnings.
> Then give me a short list of bugs with the file and line you think causes each one.

Watch the Chrome window while it works: you'll see it type and click.

**You check:** read the report. For each real bug, ask Claude to fix it (in plan mode), then to **write a test that
would have caught it** if it's a server bug (in `tests/test_api.py`), and re-run the browser check for that item.
If everything passed, try breaking it on purpose: ask Claude to change one `textContent` that shows the answer to
`innerHTML`, re-run check 5, see it fail, and change it back with `Esc Esc` (rewind).

**If Claude in Chrome won't connect** (it's still a beta feature), don't lose time: do the 8 checks yourself in your
browser, with the console open (`F12` → **Console**), and note pass/fail. Same lesson, by hand.

**Done when:** all 8 checks pass and `!py -m pytest -q` shows `0 failed`.

**Commit:** `Fix issues found in the browser check` (or skip the commit if nothing changed)

---

## Step 8 — Two hosted API formats, tested for free (20 min)

**Learn.** Re-read section 3f. First, look at both streams raw, from your local model. Second Git Bash window (stop
the server first with `Ctrl+C`, or use a third window):

```bash
curl -N http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:3b", "stream": true, "stream_options": {"include_usage": true}, "messages": [{"role": "user", "content": "Count from 1 to 5"}]}'
```

**Predict:** where will the token counts be? What's the very last line?

**What you should see:** SSE lines `data: {"id":...,"choices":[{"index":0,"delta":{"content":"1"},...}]}` with a blank
line between them; near the end, one chunk with `"choices":[]` and a `"usage"` object; the last line is
`data: [DONE]`. Now the Anthropic format:

```bash
curl -N http://localhost:11434/v1/messages -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:3b", "max_tokens": 50, "stream": true, "messages": [{"role": "user", "content": "Count from 1 to 5"}]}'
```

**What you should see:** pairs of `event: ...` and `data: {...}` lines: `message_start` (with `usage.input_tokens`),
`content_block_start`, many `content_block_delta` with `"delta":{"type":"text_delta","text":"..."}`, then
`content_block_stop`, `message_delta` (with the output token count, if your Ollama version reports it) and
`message_stop`. Same text, third packaging.

**Prompt (plan mode):**
> Add OpenAIStreamer and AnthropicStreamer to coursemate/streaming.py, using plain httpx (no SDKs) and sse_data
> from sse.py (GUIDE-api.md section 3f):
> - OpenAI: POST chat/completions under base_url (default from OPENAI_BASE_URL, else https://api.openai.com/v1),
>   header Authorization: Bearer <key>, body with model, messages, temperature, max_completion_tokens, stream true
>   and stream_options include_usage; text from choices[].delta.content, tokens from the usage chunk, stop at
>   [DONE].
> - Anthropic: POST /v1/messages under base_url (default from ANTHROPIC_BASE_URL, else https://api.anthropic.com),
>   headers x-api-key and anthropic-version 2023-06-01, system messages moved to the `system` field, max_tokens
>   required; text from content_block_delta text_delta, input tokens from message_start, output tokens from
>   message_delta, an `error` event raises LLMError, stop at message_stop.
> - Keys come from OPENAI_API_KEY / ANTHROPIC_API_KEY. If the base URL is localhost (Ollama), no key is needed
>   ("ollama" is sent) and `local` is True. If it's a hosted URL and there's no key, raise LLMError in the
>   constructor with a clear message that a hosted API costs money and how to point it at Ollama instead.
> - HTTP error statuses become LLMError with the status code. Register both in make_streamer.
> Tests in tests/test_streaming.py with httpx.MockTransport and recorded-style SSE bodies like the ones I saw with
> curl (no network): text, token counts, model name, the exact headers and body sent (system prompt placement!),
> 401 → LLMError, Anthropic error event → LLMError, hosted URL without key → LLMError, localhost needs no key.
> Also add tests/test_live_streaming.py, marked live and skipped when Ollama isn't running: the same question
> through OllamaStreamer, OpenAIStreamer (http://localhost:11434/v1) and AnthropicStreamer
> (http://localhost:11434) streams several pieces, contains "blue" for "What color is a clear daytime sky?", and
> reports input tokens.

**You check:**
- `AnthropicStreamer.stream()` in `coursemate/streaming.py`: find where the system message is moved out of
  `messages`. Compare it with `OpenAIStreamer.stream()`, where it stays in.
- Find the constructor check that refuses a hosted URL without a key. That's your **seatbelt** against spending
  money by accident: run `!py -m pytest -q -k key` and read which tests cover it.
- Ask Claude: *"Which three things would change if I pointed AnthropicStreamer at api.anthropic.com with a real
  key? Which would not change?"* (Answer to look for: the URL, the key, and the model name change; the code, the
  tests and everything above `stream()` don't.)

**Done when:**

```bash
py -m pytest -q
py -m pytest -m live -v tests/test_live_streaming.py
```

The first shows `0 failed`. The second shows `3 passed`, one per format, all from your own local model. Then run the
**whole app** on the Anthropic format (still your local model, still free):

```bash
ANTHROPIC_BASE_URL=http://localhost:11434 py -m coursemate serve --llm anthropic
```

Open http://localhost:8000: the header now says `anthropic · qwen2.5:3b`, and the answers work as before. That's the
roadmap's promise: **swapping provider is a config change, not a rewrite.** Stop the server with `Ctrl+C`.

**Commit:** `Add OpenAI and Anthropic API formats behind StreamingLLM`

---

## Step 9 — Compare latency and cost (15 min)

**Learn.** Re-read section 3g. You'll hand-type the cost formula, then measure the same questions through all three
formats and price them on hosted models.

**Predict.** Your local model runs behind all three formats. Will the three rows have very different times? Will
the hosted prices per answer be closer to $0.0001 or to $0.002?

**Prompt 1 (plan mode):**
> Create coursemate/pricing.py with a frozen dataclass `Price(input_per_mtok, output_per_mtok)`, `FREE`, a
> `PRICES` dict for claude-haiku-5-5 (0.10/0.50), claude-sonnet-5-5 (2.00/10.00), gpt-6-luna (0.10/0.50) and
> gpt-6.1-sol (2.00/10.00) with `PRICES_CHECKED = "2026-11-15"` and a comment to re-check them, a stub
> `def cost_usd(input_tokens: int, output_tokens: int, price: Price) -> float` that raises NotImplementedError
> (I'll write it), and `price_for(model, local)`: FREE when local, the table price for a known hosted model, else
> None. Add a `cost_usd` field to the `done` event in api.py using price_for and the llm's `local`. Write
> tests/test_pricing.py with the tests below plus price_for tests. Don't run them yet.

Paste this test into the prompt too (the paper numbers):

```python
def test_cost_matches_the_paper_example():
    # 700 tokens in, 60 out
    assert cost_usd(700, 60, Price(2.00, 10.00)) == pytest.approx(0.002)
    assert cost_usd(700, 60, Price(0.10, 0.50)) == pytest.approx(0.0001)
```

**Now type the body of `cost_usd` yourself** in `coursemate/pricing.py`: the formula from 3g, one `return` line.
(`pytest.approx` is there because money in floats is never exact: 0.1 × 700 isn't exactly 70 in binary. Real
billing systems count in integer micro-dollars; see the Go ledger project later.)

**Prompt 2 (plan mode):**
> Add coursemate/compare.py and a `compare` command in cli.py: `py -m coursemate compare NOTES [--llm ollama openai
> anthropic] [-n 3] [--golden evals/retrieval_golden.jsonl]`. Take the first n golden questions, build the index in
> memory with build_index (no Postgres), and for each provider: one warm-up answer that isn't counted (it loads the
> model), then the n questions through stream_answer. Print one row per provider: provider, model, endpoint host,
> found/n, median time to first token, median total time, mean tokens in/out. Then print a table of what one answer
> of that size would cost on each model in PRICES (per answer and per 1,000 answers), with the date the prices were
> checked, and a note that hosted tokenizers count a bit differently. Run it inside asyncio.run. Tests in
> tests/test_compare.py with FakeStreamer (the warm-up isn't counted; the cost table shows $0.002000 and $2.00 for
> 700 in / 60 out on claude-sonnet-5-5).

**You check:** `coursemate/compare.py`: find the warm-up call. Ask Claude: *"Why is the median a better summary than
the mean for latency?"*

**Done when:** tests pass (`!py -m pytest -q` shows `0 failed`), then in your second window:

```bash
OPENAI_BASE_URL=http://localhost:11434/v1 ANTHROPIC_BASE_URL=http://localhost:11434 \
  py -m coursemate compare notes/sample -n 3
```

It takes a minute or two (4 answers per row on your CPU). **What you should see:** three rows, all
`qwen2.5:3b` on `localhost:11434`, with similar times (it's the same model; the format adds almost nothing), and a
cost table with `$0` for local, about `$0.0001` per answer for Haiku 5.5 and GPT-6 Luna, and about `$0.0015`–`$0.002`
for Sonnet 5.5 and GPT-6.1 Sol (the exact numbers depend on your token counts). Copy the output into a file:

```bash
OPENAI_BASE_URL=http://localhost:11434/v1 ANTHROPIC_BASE_URL=http://localhost:11434 \
  py -m coursemate compare notes/sample -n 3 > evals/runs/compare-local.txt
```

Check your prediction. Then try the seatbelt: run `py -m coursemate compare notes/sample --llm openai` **without**
the URL: it must stop with `OPENAI_API_KEY is not set. A hosted API costs money...` and spend nothing. (Only if you
have never set `OPENAI_API_KEY` on this computer. Check first with `echo $OPENAI_API_KEY`: it should print an empty
line.)

**Commit:** `Add cost_usd (by hand) and the provider comparison`

---

## Step 10 — CI, README, interview questions, pull request (15 min)

**Prompt 1 (plan mode):**
> In .github/workflows/ci.yml, after the existing smoke test, add a step "Smoke test the web API (fake provider)":
> start `python -m coursemate --provider fake serve --min-score 0.1 &`, wait until
> `curl -sf localhost:8000/api/health` works (up to 30 tries, 1 s apart), POST one question to /api/chat with
> `curl -sN`, save the output with tee, and `grep -q "^event: done"` on it. Linux CI uses `python`, not `py`.

**Prompt 2 (plan mode):**
> Update README.md. In "Run it" add: `serve` (and open http://localhost:8000), the three-format live tests, the
> Anthropic-format server command with ANTHROPIC_BASE_URL, and the compare command, with a "Hosted providers"
> subsection: how to point at the real APIs (base URL + key in a git-ignored .env, never committed), and that the
> app refuses a hosted URL without a key. In "Design decisions" add bullets: streaming over SSE with fetch + POST
> (why not EventSource; why not WebSockets); async FastAPI with blocking retrieval in asyncio.to_thread; plain-text
> answers with [n] citations checked at the end, and retracting to "I don't know"; StreamingLLM next to LLMClient
> and plain httpx instead of SDKs; history = last 5 turns, follow-ups searched with the previous question; tests
> with recorded responses + live tests through Ollama's compatible endpoints, so $0; my numbers from
> evals/runs/compare-local.txt in a small table (TTFT, total, tokens) and the hosted cost per 1,000 answers. In
> "Known limitations" remove "No conversation history" and add: sessions in memory (lost on restart, one process
> only); no login or rate limiting; the follow-up search trick can drift; no retries or fallback between providers;
> prices hard-coded and dated; token counts from the local tokenizer; some OpenAI reasoning models only accept the
> default temperature; the compatible endpoints use Ollama's default context size (num_ctx can't be set per
> request); the page is plain JavaScript (no TypeScript) and has no automated browser test; `ask` still has no
> history.

**Prompt 3 (plan mode):**
> Add 5 questions with model answers (5–8 sentences each, plain language, using my code and my numbers from
> evals/runs/compare-local.txt) to @INTERVIEW.md under "## API, streaming and providers (Phase 2 Mon)":
> 1. Why stream LLM responses, and how does SSE work? When would you use WebSockets or polling instead?
> 2. What does async buy a Python web server? What happens if you call blocking code inside `async def`, and how
>    did you handle retrieval?
> 3. How do you stream an answer and still enforce guardrails like valid citations? What does the user see when
>    a check fails?
> 4. What differs between the OpenAI and Anthropic APIs, how did you hide the differences, and how did you
>    compare latency and cost without paying for API calls?
> 5. LLM system design: design a chat API for 1,000 students at once, with streaming, conversation history, two
>    LLM providers with fallback, and a monthly budget. (Cover: stateless servers with sessions in Redis or Postgres,
>    async workers, timeouts, retries with backoff, fallback to the second provider, rate limits per student,
>    token budgets and cost alerts, caching, prompt-injection and data privacy, what to log and monitor.)

**You check:** fill in your own numbers wherever Claude left a placeholder. Read every interview answer and rewrite
Q2 in your own words, using the restaurant waiter from 3c.

**Done when:**

```bash
py -m ruff check . && py -m ruff format --check . && py -m pytest -q
```

all pass. **Commit:** `Document the API, streaming and providers`

```text
!git push -u origin api
!gh pr create --fill
!gh pr checks --watch
!gh pr merge --merge --delete-branch
!git switch main
!git pull
```

All CI steps should be green, including **Smoke test the web API**. If it fails, open the failed step on GitHub,
read the last lines, and ask Claude to fix it.

---

## Check your understanding (for the reading day)

Answer in your own words first, then open the answer.

1. An answer takes 4 s before the first word and 10 s in total. With streaming, what are the TTFT and the total
   time? Without streaming?
   <details><summary>Answer</summary>With streaming: TTFT 4 s, total 10 s. Without: the user sees nothing until
   10 s, so the effective TTFT is 10 s; the total is still 10 s. Streaming doesn't make the model faster, it shows
   the work earlier.</details>
2. Write the exact SSE text for `format_sse("done", {"found": False})`. Why do we put JSON in the data line instead
   of the raw answer text?
   <details><summary>Answer</summary><code>'event: done\ndata: {"found": false}\n\n'</code> (JSON writes Python's
   False as false). A blank line ends an SSE event, so raw text with a newline in it could end the event early or
   break the format. JSON turns newlines into the two characters <code>\n</code>, so each event stays on one data
   line.</details>
3. Inside `async def chat(...)`, someone calls `retrieve(...)` directly instead of `await asyncio.to_thread(retrieve,
   ...)`. Retrieval takes 0.3 s. What happens when 10 students ask at the same moment?
   <details><summary>Answer</summary>Retrieval blocks the event loop, so the server does nothing else for 0.3 s each
   time: the 10 retrievals run one after another (up to 3 s for the last student), and every other student's
   streaming tokens stop flowing meanwhile. With to_thread, the waits overlap and the loop keeps serving.</details>
4. The model streams "Projects lose 10% per day late [3]." but only 2 sources were retrieved. What does the student
   see during and after the stream, and why can't we prevent the "during" part?
   <details><summary>Answer</summary>During: the words appear as they arrive. After: the done event has found false,
   so the page replaces them with "I don't know...". We can't check [3] until the text is complete, because the
   citation comes at the end; preventing it would mean not streaming (or buffering a sentence at a time and
   checking each).</details>
5. Turn 1 sends 700 input tokens and each earlier turn adds 80. How many input tokens does turn 4 send, and what
   does turn 4 cost on Sonnet 5.5 with 60 output tokens?
   <details><summary>Answer</summary>700 + 3 × 80 = 940 input tokens. Cost = (940 × 2.00 + 60 × 10.00) / 1,000,000
   = (1,880 + 600) / 1,000,000 = $0.00248.</details>
6. Name three things that differ between the OpenAI and Anthropic streaming formats, and one thing that's the same.
   <details><summary>Answer</summary>Different: the auth header (Authorization: Bearer vs x-api-key +
   anthropic-version); where the system prompt goes (a message vs a separate field); max tokens optional vs
   required; where the text is (choices[0].delta.content vs content_block_delta → delta.text); where token counts
   come from (a final usage chunk you must request vs message_start and message_delta); the end marker ([DONE] vs
   message_stop). Same: both are SSE over one HTTP POST, and both take a list of user/assistant messages.</details>
7. Your three `compare` rows had almost the same times. Does that mean the Anthropic API is as fast as Ollama?
   <details><summary>Answer</summary>No. All three rows ran the same local model; only the packaging differed, and
   packaging costs almost nothing. The hosted APIs' speed depends on their models and servers (and the network),
   which we didn't measure because it costs money. What the rows prove is that the format isn't the bottleneck,
   the model is.</details>

---

## When you're done

Tell the roadmap chat **"done"**. It will review the repo, run the tests and tick this row.

## Level up

**Give Claude eyes on what the user sees, and turn what it finds into tests.** Unit tests prove the server sends the
right events; only a browser proves the page shows them right. With `claude --chrome`, ask Claude to test like a QA
person: a numbered checklist of happy and unhappy paths, the console read at the end, and pass/fail for each, before
it changes any code. Every bug it finds becomes a fix **and** a test (server bugs in `tests/test_api.py`), so the
browser run finds new things next time instead of old ones. Connect the browser only for sessions that need it
(`--chrome`, or `/chrome` when you get there): its tools take up context. And keep it on `localhost` and test
pages: it acts in your real, signed-in Chrome.
