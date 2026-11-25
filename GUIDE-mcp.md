# Phase 2 Sun — CourseMate MCP server: your course tools in Claude Code (step-by-step guide)

> Read the **Project overview** first, on its own, before you open a terminal. It teaches today's ideas with small
> examples you can work out on paper. The steps start at Step 0.
>
> **Part A (day 1, about 2–2.5 hours): Steps 0–4.** You build the server, type the two small functions where the
> ideas live, and talk to the server by hand. **Part B (day 2, about 2–2.5 hours): Steps 5–9.** You connect it to
> Claude Code, use it on a real task, add resources and a prompt, and write the README, INTERVIEW and PR.
> Start day 2 by re-reading the overview and answering **Check your understanding** at the end.

---

## Project overview

### 1. What you've built so far (in plain words)

- **CourseMate RAG (Nov 11) and better retrieval (Nov 12):** your course notes are cut into **chunks**, each chunk
  becomes a vector with `nomic-embed-text`, and the vectors live in **Postgres + pgvector**. A question is embedded
  the same way and the closest chunks come back (`retrieve()` in `coursemate/retrieval.py`). Below `min_score`
  (0.55), CourseMate says "I don't know".
- **CourseMate API (Nov 16):** the same answers behind a FastAPI server with streaming.
- **Lab 2 (Nov 18, `ai-basics`):** "tool calling" means the model writes JSON naming a tool and its arguments, and
  **your code** runs the tool. You typed a 40-line agent loop.
- **CourseMate agent (Nov 20, branch `agent`):** four tools in `coursemate/agent_tools.py` (`CourseTools`):
  `search_notes`, `make_quiz`, `draft_announcement`, `send_announcement`. An **outbox** folder holds drafts
  (`outbox/drafts/001.md`) and sent messages (`outbox/sent/`). Sending needs your "yes", checked in **one place**:
  `run_tool` in `coursemate/agent.py`. You also ran the same tools on the **Claude Agent SDK**
  (`coursemate/sdk_agent.py`), which served them as an **in-process MCP server** (`create_sdk_mcp_server`). That
  was MCP hidden inside one Python program. Today it becomes a real one.

Today the tools leave your program. You'll put them behind the **Model Context Protocol (MCP)**, so **any** MCP
app can use them: Claude Code today, Claude Desktop or a colleague's agent tomorrow, without you writing code for
each app.

| Part | Question | What you'll see |
|---|---|---|
| A | What is MCP, really, on the wire? | you type one JSON-RPC message and send it to your server by hand |
| A | How do you turn Python functions into an MCP server? | four tools in about 60 lines with the official `mcp` SDK |
| A | Who says "yes" before a send, when the client isn't yours? | the **server** checks a yes that only you can give, in your own terminal |
| B | How does Claude Code use it? | `claude mcp add`, `/mcp`, and Claude calling `mcp__coursemate__search_notes` |
| B | What else can a server offer? | your notes as **resources** (`@` mentions) and a **prompt** (a slash command) |

### 2. The AI basics you need today (work these on paper)

#### 2a. Why a protocol: the M × N problem

Say you have **3 AI apps** (Claude Code, Claude Desktop, your own agent) and **2 systems** they should reach
(CourseMate, and your university's LMS). Without a standard, each app needs its own code for each system:

```text
 without MCP:  3 apps × 2 systems = 6 custom integrations
 with MCP:     3 apps (each speaks MCP once) + 2 servers (each speaks MCP once) = 5 pieces
```

Small numbers hide the point, so scale it: **10 apps and 20 systems**. Without MCP: 10 × 20 = **200**
integrations. With MCP: 10 + 20 = **30**. Every new app gets all 20 systems for free; every new server works in
all 10 apps. That's why people call MCP "USB-C for AI": one plug shape, many devices.

#### 2b. Three roles and three kinds of things a server offers

```text
  ┌─────────────── HOST: Claude Code (the app you type into) ────────────────┐
  │  the model (Claude)   asks for tools  ──►  MCP CLIENT (one per server)   │
  └──────────────────────────────────────────────────┬───────────────────────┘
                                                     │  JSON-RPC over stdin/stdout
                                                     ▼
                                   MCP SERVER: py -m coursemate.mcp_server
                                   (your process; it reaches Postgres, Ollama, outbox/)
```

- **Host:** the app (Claude Code). **Client:** the part of the host that talks to one server. **Server:** your
  program that offers things.
- A server offers three kinds of things. The difference is **who decides to use them**:

| Primitive | Who picks it | CourseMate example | In Claude Code |
|---|---|---|---|
| **Tool** | the **model** decides to call it | `search_notes(query)`, `send_announcement(draft_id)` | `mcp__coursemate__search_notes` |
| **Resource** | the **user** (or app) attaches it as context | the note file `notes://sample/syllabus.md` | type `@coursemate:notes://sample/syllabus.md` |
| **Prompt** | the **user** picks a ready-made instruction | `announcement(topic)` | `/mcp__coursemate__announcement` |

Quick check on paper: "the instructor wants Claude to read the whole syllabus before answering" → a **resource**
(the user attaches it). "Claude needs a fact in the middle of a task" → a **tool** (the model decides).

#### 2c. MCP on the wire: JSON lines over stdin and stdout

With the **stdio** transport, the client starts your server as a **child process** and they talk through its
standard input and output: the client writes **one JSON object per line** to the server's stdin, the server writes
**one JSON object per line** back to its stdout. The format is **JSON-RPC 2.0**: every request has an `id`, a
`method` and `params`; the reply has the **same `id`** and a `result` (or an `error`).

The paper exchange you'll reproduce in Step 4 (shortened with `…`):

```text
 >>> {"jsonrpc":"2.0", "id":1, "method":"tools/list", "params":{"_meta":{…}}}
 <<< {"jsonrpc":"2.0", "id":1, "result":{"tools":[{"name":"search_notes", "description":"Search the…",
                                                    "inputSchema":{"type":"object","properties":{"query":…}}}, …]}}

 >>> {"jsonrpc":"2.0", "id":2, "method":"tools/call",
      "params":{"name":"search_notes", "arguments":{"query":"When is Lab 4 due?"}, "_meta":{…}}}
 <<< {"jsonrpc":"2.0", "id":2, "result":{"content":[{"type":"text","text":"{\"results\": [{\"source\":
      \"lecture-02-hash-tables.md\", … \"due Friday of week 5\" …"}], "isError":false}}

 >>> {"jsonrpc":"2.0", "id":3, "method":"tools/call",
      "params":{"name":"send_announcement", "arguments":{"draft_id":1}, "_meta":{…}}}
 <<< {"jsonrpc":"2.0", "id":3, "result":{"content":[{"type":"text","text":"Error executing tool
      send_announcement: draft 1 is not approved, so nothing was sent. …"}], "isError":true}}
```

Count on paper: **3 requests, 3 replies, ids 1, 2, 3**. Two methods do all the tool work: `tools/list` ("what can
you do?") and `tools/call` ("do this one"). A tool that fails is **not** a JSON-RPC error: it's a normal `result`
with `"isError": true` and the reason as text, so the model can read it and react (exactly like Friday's
`"error: ..."` strings).

**The `_meta` part (new in the 2026-07-28 MCP spec).** Older MCP started with a **handshake** (`initialize`, then
`notifications/initialized`) and kept a session. The current spec is **stateless**: there's no handshake, and every
request carries three keys in `params._meta` instead:

```json
"_meta": {
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientInfo": {"name": "coursemate-raw", "version": "0.1"},
  "io.modelcontextprotocol/clientCapabilities": {}
}
```

The official Python SDK (`pip install mcp`, version 2.x) speaks **both** styles, so Claude Code connects whichever
one it uses. Your hand-made client in Step 1 uses the new one, because it's simpler: every line stands alone.

#### 2d. You already know this: Friday's agent vs. MCP

| Friday (`coursemate/agent.py`, inside one program) | Today (MCP, between two programs) |
|---|---|
| `TOOL_SCHEMAS`: name, description, JSON Schema | `tools/list` → `tools[]` with `name`, `description`, `inputSchema` |
| `run_tool(call, ...)` calls the Python function | `tools/call` with `name` and `arguments`; the **server** calls the function |
| `"error: LookupError: no draft 9..."` | `"isError": true` with the message as text |
| your loop, your model (`qwen2.5:3b`) | **Claude Code's** loop and model; your server only answers requests |

The model never runs a tool, in either column. The difference is **who owns the loop** and **which process runs
the tool**. With MCP, the loop belongs to someone else's app.

#### 2e. Who says "yes" before a send? Don't trust the client

On Friday **you owned the loop**, so the gate in `run_tool` saw every call. Today Claude Code owns the loop. It
does ask you before it runs an MCP tool ("Do you want to use mcp__coursemate__send_announcement?"), but think about
what your server can know:

- The server only sees a `tools/call` line. It **can't tell** whether a human clicked yes, whether the user set
  "always allow", or whether the client is a script that never asks anyone.
- So the **server** keeps its own gate, and the yes must come from somewhere the client can't reach. Today: you run
  `py -m coursemate approve 1` **in your own terminal**. It shows you draft 1 and saves a copy of the exact text in
  `outbox/approved/001.md`. `send_announcement` only sends a draft whose text **still matches** that copy.

Trace it on paper. The outbox has two drafts. You approve only draft 1.

```text
 outbox/drafts/001.md   "# Lab 4\n\nLab 4 is due Friday of week 5.\n"
 outbox/drafts/002.md   "# Party\n\nClass is cancelled tomorrow!\n"
 you run:  py -m coursemate approve 1   → y   → copies 001.md to outbox/approved/001.md

 send_announcement(2)   approved/002.md missing                    → isError: "draft 2 is not approved, ..."
 send_announcement(1)   approved/001.md == drafts/001.md (same)   → {"draft_id": 1, "status": "sent"}
 afterwards: drafts [2], sent [1], approved [1]
```

And one more case: you approve draft 1, then something **edits** `drafts/001.md` to "Lab 4 is cancelled!". Now the
two texts differ, so `is_approved(1)` is **False** and the send is refused. You approved **words**, not a number.

Three layers, each stopping a different mistake:

| Layer | Where | Stops | Can be skipped by |
|---|---|---|---|
| 1. Claude Code asks before `send_announcement` | the client (`.claude/settings.json`: `ask`) | Claude sending when you didn't expect it | "always allow", a different client, a script |
| 2. The server checks `approved/` | **your server** (`send_if_approved`, you type it) | any client sending a draft you never read | someone who can write to `outbox/approved/` |
| 3. Claude may not run `approve` or write in `outbox/` | Claude Code deny rules | Claude approving its own draft | a command the rules didn't list (deny lists are lists, Level-up tip 11) |

In a real product layer 2 would live behind the instructor's **login** in a separate web page, so nothing running
on the model's side could fake the yes. (MCP also has a standard way for a server to ask the user a question
through the client, called **elicitation**, and Claude Code shows it as a dialog. It's a nice way to ask, but the
answer still travels through the client, which is why today's yes lives outside it.)

#### 2f. Two kinds of errors: say what's wrong, hide what's internal

The SDK treats errors in two ways (you'll see both in Step 2):

```text
 raise ToolError("no draft 7; drafts: 1")   → isError: "Error executing tool send_announcement: no draft 7; drafts: 1"
 an unexpected bug (KeyError, a crash...)   → isError: "Error executing tool send_announcement"      (no details)
```

An expected problem gets a **useful message** the model can act on ("draft 7 doesn't exist, the drafts are 1").
A bug gets **no details**, so internals (paths, SQL, stack traces) never leak to the client. Our tools already
raise `LookupError` and `ValueError` for expected problems, so the server turns exactly those (plus "Ollama/Postgres
isn't running") into `ToolError`.

#### 2g. The stdout rule

stdout **is** the protocol. One `print("starting...")` in the server puts a line on stdout that isn't JSON. Paper
example: the client reads `CourseMate MCP server starting` where it expected `{"jsonrpc": ...}`, and `json.loads`
fails with `Expecting value: line 1 column 1`. Some clients skip such lines (Claude Code did, when I tried), some
crash, and a print in the middle of a reply can cut a message in half. The rule: **never print to stdout in an MCP
server**; logs go to stderr. You'll break this on purpose in Step 4.

#### 2h. Tool descriptions cost context

Like Friday (section 2e there), every tool's name, description and schema goes into the model's context. Four tools
≈ 450 tokens, sent with every Claude Code turn while the server is connected. Claude Code helps when you have many
servers (it can load tool definitions only when they're needed), but the habit is the same: short, exact
descriptions, and only the tools the job needs.

### 3. What we'll build (the parts and how they fit)

```text
 Claude Code  (host; Claude runs the loop)                 py -m coursemate.mcp_raw  (your hand-made client)
   │  claude mcp add coursemate -- …python.exe -m coursemate.mcp_server        │  mcp_request() ← YOU type (Step 1)
   │  .claude/settings.json: allow 3 tools, ask for send, deny "approve"      │
   └──────────────┐ JSON-RPC over stdio ┌────────────────────────────────────┘
                  ▼                     ▼
 coursemate/mcp_server.py   build_server(tools, notes_dir) → MCPServer("coursemate")
     tools:      search_notes, make_quiz, draft_announcement   (call CourseTools, errors → ToolError)
                 send_announcement → send_if_approved() ← YOU type (Step 3)
     resources:  notes://sample/<file>.md   (an allow-list of files, Step 7)
     prompt:     announcement(topic)        (Step 7)
                  │
                  ▼
 coursemate/agent_tools.py   CourseTools (unchanged) + Outbox.approve(), is_approved(), approved_ids()
                  │                                   │
                  ▼                                   ▼
 retrieval → Postgres, Ollama            outbox/drafts/ → (you: py -m coursemate approve 1) → outbox/approved/
                                                    → outbox/sent/
```

| New or changed file | What it is |
|---|---|
| `coursemate/mcp_raw.py` | **your** `mcp_request()`, and `RawClient`, a 30-line stdio client |
| `coursemate/mcp_server.py` | the MCP server: tools, **your** `send_if_approved()`, resources, prompt, `--fake` mode |
| `coursemate/agent_tools.py` (changed) | `Outbox` gets `approved/`, `approve()`, `is_approved()`, `approved_ids()` |
| `coursemate/cli.py` (changed) | `py -m coursemate approve <draft_id>`: shows the draft, asks y/N |
| `.claude/settings.json` | permission rules for the server's tools |
| `tests/test_mcp_raw.py`, `tests/test_mcp_server.py`, `tests/test_cli.py` | tests with the paper numbers |
| `.github/workflows/ci.yml` (changed) | a smoke test: the raw client against the fake server |

Same `coursemate` repo, on a branch called `mcp`, merged with a pull request at the end.

### 4. How we'll test it

1. **Your `mcp_request`** is tested with the paper message from 2c (`id` 3, `tools/call`, the three `_meta` keys).
2. **Your `send_if_approved`** is tested with the paper trace from 2e: drafts [1, 2], only 1 approved → send 1 works,
   send 2 is refused, drafts [2], sent [1]; and the "edited after approval" case is refused.
3. **The whole server** is tested through a **real MCP client in memory**: the SDK's `Client(server)` connects to
   your `MCPServer` object without a subprocess, lists the tools and calls them. Notes come from `FakeEmbedder` and
   an in-memory index of `notes/sample` (no database, no model).
4. **Over stdio, for real:** one test starts `py -m coursemate.mcp_server --fake` as a child process and talks to it
   with your `RawClient`. CI runs the same thing as a smoke test.
5. **Break-it tests:** remove the gate, and three tests fail; add one `print()`, and the stdio test fails.
6. **By hand in Claude Code** (Part B): the real task, the refusal, your approval, the send.

### 5. New words (short definitions)

- **MCP (Model Context Protocol):** an open standard for connecting AI apps to tools and data. **Host:** the app
  (Claude Code). **Client:** the host's connection to one server. **Server:** the program that offers tools,
  resources and prompts.
- **JSON-RPC 2.0:** a tiny format for requests and replies in JSON: `id`, `method`, `params` → `id`, `result` or
  `error`.
- **Transport:** how the JSON travels. **stdio:** the client starts the server and uses its stdin/stdout (local,
  one user). **Streamable HTTP:** the server is a web service (remote, many users, needs login).
- **Tool / resource / prompt:** see 2b. **Tool annotations:** hints in `tools/list` such as `readOnlyHint` and
  `destructiveHint`. Clients may show or use them; they are **hints, not security**.
- **`MCPServer`:** the class in the `mcp` 2.x SDK that turns decorated Python functions into a server (it was called
  `FastMCP` in 1.x). **`Client`:** the SDK's client; `Client(server)` connects in memory for tests.
- **`ToolError`:** the exception whose message the SDK sends to the client.
- **Stateless:** each request carries everything needed to answer it (here, the `_meta` keys); the server keeps no
  per-client session.
- **Elicitation:** the server asks the user a question through the client (a form). **Out-of-band approval:** the yes
  comes through a different channel than the request (your terminal, not the chat).
- **MCP scopes in Claude Code:** **local** (only you, only this project; the default), **project** (a `.mcp.json`
  file committed for the team), **user** (only you, every project).

### 6. Today's Claude Code technique: build your own MCP server (`claude mcp add`)

Until now you gave Claude Code new abilities with prompts, `CLAUDE.md`, skills, hooks and subagents. MCP is how you
give it **your own systems**: a database, an internal API, your course tools. The commands:

```text
 claude mcp add <name> -- <command> <args...>   register a stdio server (default scope: local)
 claude mcp list                                 start each server once and report "Connected" or "Failed"
 claude mcp get <name> / claude mcp remove <name>
 /mcp                                            inside Claude Code: status, tools, resources, prompts
```

Inside Claude Code, a server's tool is named `mcp__<server>__<tool>`, and permission rules use those names (you
used the same pattern in the Agent SDK on Friday). You'll allow the three safe tools, make the send ask every time,
and deny the two ways Claude could approve its own draft.

### 7. Talking to hosted APIs (interviewers will ask)

- **Same server, other hosts.** Claude Desktop, VS Code, Cursor and the Agent SDK all speak MCP, so this server works
  there unchanged. That's the point of 2a.
- **Hosted model APIs** (Anthropic's and OpenAI's) can call **remote** MCP servers over HTTP during a request. A
  remote server must be reachable on the internet and use the **Streamable HTTP** transport, and it should require
**OAuth** (anyone who finds the URL could call it otherwise). Your
  stdio server is local on purpose: it reaches your Postgres and your files, and nothing outside your PC can call it.
- **What moves to the server in production:** auth (which instructor is this?), per-course access, rate limits,
  audit logs of every `tools/call`, and the approval gate. The model side stays the same.

### 8. Time, cost and privacy

**Part A about 2–2.5 hours, Part B about 2–2.5 hours. Cost: $0.** `mcp` is a free library, `qwen2.5:3b` and
`nomic-embed-text` run locally, Postgres runs in Docker, and Claude Code uses your Claude plan (no API key).

**Privacy:** in Part B, Claude Code sends what your tools return (chunks of your notes, drafts) to Claude. That's
fine for `notes/sample/`, and **not** for `notes/private/`. Step 5 re-ingests **only** `notes/sample` before you
connect, so the database holds nothing else, and the resources are an allow-list of `notes/sample/*.md`.

> **Note on the expected outputs.** I built and tested a reference solution for this guide with `mcp` 2.3.0, the
> fake model, and a real Postgres + pgvector, and I checked that Claude Code 2.1.330 connects to the server
> (`claude mcp list` → `Connected`). Every test, command and file name below works. I couldn't run `qwen2.5:3b` or
> sign in to Claude Code while writing it, so what Claude says in Part B is **what to expect**, not a copy.

---

## How to use this guide with Claude Code

Every step has:

- **Learn:** the idea. Read it first.
- **Predict:** write down what you expect *before* running anything. Being wrong is how you learn.
- **Prompt:** what to type into Claude Code in **plan mode** (`Shift+Tab` until you see `⏸ plan mode on`). Read
  Claude's plan and ask "why?" about anything you don't understand.
- **You check:** what you verify yourself, with the file and function to open.
- **Done when:** a command whose output proves the step works.
- **Commit:** the message.

**Python is `py` on your machine.** Commands use `py -m ...`; inside Claude Code type `!py -m pytest`. CI on GitHub
runs Linux, where it stays `python`.

**Two terminals in Part B.** Terminal 1 runs Claude Code. Terminal 2 is a plain Git Bash in the `coursemate` folder
(venv activated) where **you** run `py -m coursemate approve ...`. Keep them separate on purpose: terminal 2 is your
"instructor's desk", which Claude can't use.

---

# Part A (day 1)

## Step 0 — Get ready: merge Friday's branch, start the `mcp` branch (15 min)

**Learn.** Today builds on Friday's `agent` branch (the tools and the outbox). Merge it first, so the `mcp` branch
starts from `main` and the PR at the end shows only today's work. Friday's to-do list (the SDK comparison table, the
`not-in-notes` rule, the prompt fix) can come later on its own branch; don't let it block today.

In Git Bash, in your `coursemate` folder:

```bash
git switch agent
git status
gh pr list
```

**What you should see:** `git status` says `nothing to commit, working tree clean` (if not, commit or stash first).
`gh pr list` shows the open PR from `agent`. Open it on GitHub, check that CI passed, and merge it. (No PR yet? Run
`git push -u origin agent` and `gh pr create --fill`, wait for CI, then merge.)

Now start from the merged `main`:

```bash
git switch main
git pull
git switch -c mcp
docker compose up -d
source .venv/Scripts/activate
py -m pip install -e ".[dev]"
py -m pytest -q
```

**What you should see:** `Switched to a new branch 'mcp'`, and the tests end with `0 failed` (I counted `215
passed, 7 deselected` with the `[sdk]` extra installed; without it, a few SDK tests are skipped).

Install the MCP SDK and check the version:

```bash
py -m pip install "mcp>=2.3,<3"
py -c "import importlib.metadata as m; print(m.version('mcp'))"
```

**What you should see:** `2.3.0` or a newer `2.x`. If it prints `1.something`, something pinned the old line: run
`py -m pip install -U "mcp>=2.3,<3"`. (The guide uses the 2.x names: `MCPServer`, `Client`, `is_error`. On 1.x they
were `FastMCP`, `ClientSession`, `isError`.)

Save **this file** as `GUIDE-mcp.md` in the `coursemate` folder. Then start Claude Code:

```bash
claude
```

**Prompt (plan mode):**
> Add `"mcp>=2.3,<3"` to the dependencies in pyproject.toml. In CLAUDE.md, add GUIDE-mcp.md (Nov 22, the CourseMate
> MCP server) to the Plan section, and add these rules to "How to work with me": "Nothing in the MCP server may
> print to stdout: stdout is the protocol." and "Over MCP, send_announcement sends only drafts the instructor
> approved with `py -m coursemate approve`; tests never remove that gate, and Claude never runs that command."
> Change nothing else.

**Done when:** `!git status` shows `pyproject.toml` and `CLAUDE.md` modified and `GUIDE-mcp.md` untracked.

**Commit:** `Add the MCP guide and the mcp dependency` (include GUIDE-mcp.md).

---

## Step 1 — MCP by hand: you type one request (20 min)

**Learn.** Before any SDK, write the message itself. Re-read 2c. An MCP request is a Python dict that becomes one
line of JSON. You'll type the function that builds it; Claude writes the test first, with the paper numbers.

**Predict.** For `mcp_request(3, "tools/call", {"name": "search_notes", "arguments": {"query": "Lab 4"}})`: what are
the top-level keys? Where do `name` and `arguments` end up? For `mcp_request(1, "tools/list")`, which keys are inside
`params`?

**Prompt (plan mode):**
> Create coursemate/mcp_raw.py with `PROTOCOL_VERSION = "2026-07-28"` and
> `mcp_request(request_id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]` whose body is
> `raise NotImplementedError` (I'll write it). Write tests first in tests/test_mcp_raw.py, using my guide's paper
> example: request id 3, "tools/call", name "search_notes", arguments {"query": "Lab 4"} → jsonrpc "2.0", the same
> id and method, name and arguments inside params, and params._meta with the three keys
> io.modelcontextprotocol/protocolVersion ("2026-07-28"), clientInfo and clientCapabilities ({}). A second test:
> with no params, params has only "_meta". Run the tests and show me them failing.

**You check:** open `tests/test_mcp_raw.py` and find the line that checks `clientCapabilities == {}`. Why `{}`? (Our
client offers nothing extra to the server: no elicitation forms, no sampling.)

**Now you type** the body of `mcp_request` in `coursemate/mcp_raw.py`:

```python
def mcp_request(
    request_id: int, method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One MCP request: a JSON-RPC 2.0 message. Since 2026-07-28 every request says who
    is asking and which protocol version it speaks, in params._meta (no handshake)."""
    meta = {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientInfo": {"name": "coursemate-raw", "version": "0.1"},
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {**(params or {}), "_meta": meta},
    }
```

Read it line by line:
- `meta`: the three keys that replace the old handshake. Every request carries them, so the server needs no memory
  of earlier requests (that's "stateless").
- `{**(params or {}), "_meta": meta}`: `**` copies every key of `params` into a new dict (or nothing, if `params` is
  `None`), then adds `_meta`. The caller's dict isn't changed.
- `"id": request_id`: the reply will carry the same id. With many requests in flight, that's how the client matches
  each reply to its request.

```bash
py -m pytest -q tests/test_mcp_raw.py
py -m ruff format .
```

**Done when:** `2 passed`.

**Commit:** `mcp_request: an MCP request by hand`

---

## Step 2 — The server: three tools in a few lines (30 min)

**Learn.** The `mcp` SDK turns a Python function into a tool. The **function name** becomes the tool name, the
**type hints** become the `inputSchema`, and the SDK checks the arguments before your function runs:

```python
server = MCPServer("coursemate")


@server.tool(description="Search the instructor's course notes...", annotations=READ)
def search_notes(query: str) -> dict[str, Any]:
    with tool_errors():  # LookupError / ValueError → ToolError (section 2f)
        return tools.search_notes(query)
```

`tools/list` then shows `{"name": "search_notes", "inputSchema": {"properties": {"query": {"type": "string"}},
"required": ["query"]}, "annotations": {"readOnlyHint": true}, ...}`. Compare that with Friday's `TOOL_SCHEMAS`:
the same information, now generated from the function.

In this step, the server gets the **three tools that don't send**. `send_announcement` comes in Step 3, with its
gate. The server reuses Friday's `CourseTools` and its descriptions (`TOOL_SCHEMAS`), so the tools behave exactly
as they did in your agent.

**Predict.** The test calls `make_quiz` with `questions=9`. `CourseTools.make_quiz` raises
`ValueError("questions must be between 1 and 5, not 9")`. What text will the client get? And what would it get if
`tool_errors()` didn't catch `ValueError`?

**Prompt (plan mode):**
> Create coursemate/mcp_server.py with `build_server(tools: CourseTools, notes_dir: Path) -> MCPServer` using
> `from mcp.server.mcpserver import MCPServer`. Register search_notes, make_quiz (questions defaults to 3) and
> draft_announcement as thin wrappers around CourseTools, reusing the descriptions from TOOL_SCHEMAS. Add
> ToolAnnotations: READ (read_only_hint=True, open_world_hint=False) for search and quiz, DRAFT (not read-only,
> destructive_hint=False) for the draft. Add a `tool_errors()` context manager that turns LookupError, ValueError,
> LLMError and StoreError into ToolError. Add `main()` with argparse: `--fake` (FakeEmbedder, build_index of the
> notes, FakeClient, min_score 0.1), `--outbox` and `--notes` defaulting to the repo folder (outbox/,
> notes/sample/) so it works from any working directory, `--min-score`; otherwise OllamaEmbedder, PgVectorStore and
> OllamaClient("qwen2.5:3b") like sdk_agent.py. It runs on stdio and must never print to stdout.
> Write tests/test_mcp_server.py first with the SDK's in-memory `Client(server)` (`from mcp import Client`, async
> tests with anyio) on an in-memory index of notes/sample: three tools are listed with their hints and
> search_notes requires "query"; "When is Lab 4 due?" returns lecture-02-hash-tables.md first; a blank title and
> questions=9 come back with is_error and the messages "must not be empty" and "between 1 and 5". Show them failing
> first, then make them pass.

**You check:** open `coursemate/mcp_server.py`.
- Find `tool_errors()`. Which four exception types does it catch? Why not `Exception`? (Section 2f: a bug should
  stay a bug, with no internals sent out.)
- Find where `--outbox` gets its default. It should use the file's own location
  (`Path(__file__).resolve().parents[1]`), not the current folder: Claude Code may start the server from a
  different working directory.
- Search the file for `print(`. There should be none.

```bash
py -m pytest -q tests/test_mcp_server.py
py -m pytest -q
py -m ruff check . && py -m ruff format --check .
```

**Done when:** `tests/test_mcp_server.py` passes (`3 passed`) and the whole suite says `0 failed`.

**Commit:** `MCP server with three tools (search, quiz, draft)`

---

## Step 3 — The send tool and its gate: you type `send_if_approved` (30 min)

**Learn.** Re-read 2e and keep the paper trace next to you. Two pieces:

1. **The yes (Claude writes it):** `Outbox.approve(draft_id)` copies the draft's text to
   `outbox/approved/001.md`; `Outbox.is_approved(draft_id)` is True only if both files exist and their texts are
   equal; `approved_ids()` lists them. And a command for **you**: `py -m coursemate approve 1` shows draft 1 and asks
   `Approve draft 1 for sending, exactly as shown? [y/N]`. Anything but `y` is a no (fail closed, like Friday).
2. **The gate (you type it):** `send_if_approved(outbox, draft_id)` in `coursemate/mcp_server.py`, and the
   `send_announcement` tool that calls it.

**Predict.** Using the paper trace: after `send_if_approved(outbox, 1)` and the refused `send_if_approved(outbox, 2)`,
what are `outbox.draft_ids()` and `outbox.sent_ids()`? What happens if you call `send_if_approved(outbox, 1)` a
second time?

**Prompt (plan mode):**
> Add to Outbox in coursemate/agent_tools.py: an `approved/` folder, `approve(draft_id)` (copies the exact draft text
> to approved/NNN.md; LookupError if there's no draft), `is_approved(draft_id) -> bool` (True only if the draft and
> approved/NNN.md both exist with the same text) and `approved_ids()`. Add a CLI command `approve <draft_id>`
> (`--outbox`, default outbox) that prints the draft, asks "Approve draft N for sending, exactly as shown? [y/N] "
> with input(), approves only on "y" and exits 1 otherwise; an unknown draft prints the LookupError to stderr.
> In mcp_server.py add `send_if_approved(outbox: Outbox, draft_id: int) -> dict[str, Any]` with the body
> `raise NotImplementedError` (I'll write it), and a send_announcement tool that calls it inside tool_errors(), with a
> description saying the instructor approves drafts in their own terminal, and annotations destructive_hint=True,
> open_world_hint=True.
> Tests first, with my guide's paper trace: drafts 1 ("Lab 4") and 2 ("Party"), only 1 approved → sending 1 returns
> {"draft_id": 1, "status": "sent"}, sending 2 raises ToolError matching "not approved", then drafts [2], sent [1];
> a draft edited after approval is refused and sent stays []; through Client: draft then send without approval is
> is_error with "py -m coursemate approve 1" in the text and sent []; after approve(1) the send works; the tool list
> now has four tools and send has destructive_hint. In tests/test_cli.py: approve answers "" (no) then "y"; approve
> 7 prints "no draft 7; drafts: none". Show the tests failing.

**You check:** read `Outbox.is_approved` in `coursemate/agent_tools.py`. Point at the line that makes the "edited
after approval" case fail. Then read `run_approve` in `coursemate/cli.py`: does it show you the draft **before** it
asks?

**Now you type** the body of `send_if_approved` in `coursemate/mcp_server.py`:

```python
def send_if_approved(outbox: Outbox, draft_id: int) -> dict[str, Any]:
    """The server's own gate: send only a draft the instructor approved, word for word."""
    if not outbox.is_approved(draft_id):
        raise ToolError(
            f"draft {draft_id} is not approved, so nothing was sent. Tell the instructor to "
            f"read it and run this in their own terminal: py -m coursemate approve {draft_id}"
        )
    outbox.send(draft_id)
    return {"draft_id": draft_id, "status": "sent"}
```

Read it line by line:
- `if not outbox.is_approved(draft_id)`: the check comes **before** `outbox.send`. Same lesson as Friday's Step 4:
  check after the send and it's a receipt, not a gate.
- The message does Friday's three jobs (what happened, the consequence, what to do next), and the next step is for
  the **instructor**, in **their** terminal. Claude will read this text and pass it on to you.
- `raise ToolError(...)`: the SDK turns it into `"isError": true` with the message. No `try` needed here.
- If you call it again for draft 1 after it was sent, `drafts/001.md` is gone, so `is_approved(1)` is False and
  it's refused: nothing can be sent twice.

(Check the imports: you need `ToolError` from `mcp.server.mcpserver.exceptions` and `Outbox`. Claude probably added
them.)

```bash
py -m pytest -q tests/test_mcp_server.py tests/test_cli.py
py -m pytest -q
```

**Done when:** `tests/test_mcp_server.py` has `7 passed`, and the whole suite says `0 failed`.

Try your command once by hand (any draft text will do):

```bash
py -c "from coursemate.agent_tools import Outbox; print(Outbox('outbox').save_draft('Test', 'Hello'))"
py -m coursemate approve 1
```

**What you should see:** the number of the new draft (`1` if your outbox was empty; use that number below), then the
draft (`# Test` / `Hello`) and the question. Type `n`: it prints `not approved`. Run it again and type `y`:
`approved draft 1 (outbox: approved [1])`. Then clean up: `rm -r outbox` (it's git-ignored).

**Commit:** `send_announcement over MCP, gated by the instructor's approval`

---

## Step 4 — Talk to your server over stdio, then break it twice (25 min)

**Learn.** Time to see 2c for real. `RawClient` starts the server as a child process (`subprocess.Popen` with pipes
for stdin and stdout), writes one line of JSON per request using your `mcp_request`, and reads one line back. Two
details matter on Windows:

- It writes **bytes** (`line.encode("utf-8")`), not text: in text mode Windows turns `\n` into `\r\n`.
- `stderr` goes to `subprocess.DEVNULL` in the client, so the server's logs don't mix with your output.

**Predict.** Your `main()` will send `tools/list`, then `tools/call` for `search_notes`, then `tools/call` for
`send_announcement` with draft 1. What ids will the three replies have? Which one has `"isError": true`, and why?

**Prompt (plan mode):**
> In coursemate/mcp_raw.py add `RawClient(command: list[str])`: it starts the command with subprocess.Popen (stdin
> and stdout as pipes, stderr to DEVNULL), `request(method, params=None)` builds the message with mcp_request and
> an id counting from 1, writes one JSON line as UTF-8 bytes, flushes, reads one line, prints both (">>> " and
> "<<< ", the reply cut to 300 characters) and returns the parsed reply; RuntimeError if the server closed stdout;
> `close()` closes stdin and waits. Add `main()` that starts `[sys.executable, "-m", "coursemate.mcp_server",
> "--fake"]` and sends tools/list (print the tool names), search_notes "When is Lab 4 due?" and send_announcement
> draft 1 (print isError). Add a test that starts the fake server with --outbox in tmp_path, lists 4 tools and finds
> lecture-02-hash-tables.md, checking the reply ids 1 and 2.

**You check:** open `RawClient.request`. Find the two lines that make the id go 1, 2, 3. Find `readline()`: why does
reading **one line** give exactly one reply? (Because the server writes each JSON reply on one line: that's the stdio
framing rule.)

Now run it:

```bash
py -m coursemate.mcp_raw
```

**What you should see** (long lines cut here with `…`):

```text
>>> {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28", …}}}
<<< {"jsonrpc":"2.0","id":1,"result":{"cacheScope":"private","resultType":"complete","tools":[{"annotations":{"openWorldHint":false,"readOnlyHint":true},"description":"Search the instructor's course notes. …

tools: ['search_notes', 'make_quiz', 'draft_announcement', 'send_announcement']

>>> {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "search_notes", "arguments": {"query": "When is Lab 4 due?"}, …}}
<<< {"jsonrpc":"2.0","id":2,"result":{"content":[{"text":"{\n  \"results\": [\n    {\n      \"source\": \"lecture-02-hash-tables.md\",\n      \"section\": \"Lecture 2: Hash Tables > Lab 4\", …

first text: '{\n  "results": [\n    {\n      "source": "lecture-02-hash-tables.md",\n      "section": "Lecture 2: Hash Tables > Lab 4",\n '

>>> {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "send_announcement", "arguments": {"draft_id": 1}, …}}
<<< {"jsonrpc":"2.0","id":3,"result":{"content":[{"text":"Error executing tool send_announcement: draft 1 is not approved, so nothing was sent. Tell the instructor to read it and run this in their own terminal: py -m coursemate approve 1","type":"text"}],"isError":true, …

isError: True
```

Compare it with your paper exchange in 2c. Three things you couldn't see before:
- The server added fields you didn't ask for (`cacheScope`, `resultType`, `_meta.serverInfo`). Clients ignore what
  they don't need.
- The tool result is **text containing JSON** (`"{\n  \"results\": ...`): MCP's content is a list of text (or image)
  parts, and the model reads the text.
- The refusal is a normal `result`, not an `error`. The protocol worked; the **tool** said no.

**Break it on purpose** (Level-up tip 2). Commit first, then try each change, run the tests named, and undo with
`git checkout coursemate/mcp_server.py`.

**Predict** first, for both:
1. In `send_if_approved`, change `if not outbox.is_approved(draft_id):` to `if False:`. Run
   `py -m pytest -q tests/test_mcp_server.py`. Which tests fail?
2. In `main()` of `mcp_server.py`, add `print("CourseMate MCP server starting")` right after
   `args = parser.parse_args(argv)`. Run `py -m pytest -q tests/test_mcp_raw.py`. Which test fails, and with what
   error?

**What you should see:**
1. `3 failed, 4 passed`: `test_gate_sends_only_the_approved_draft`, `test_a_draft_changed_after_approval_is_not_approved`
   and `test_draft_then_send_without_approval_is_refused`. Without the gate, draft 2 ("Class is cancelled
   tomorrow!") goes out, and so does the edited draft.
2. `1 failed`: `test_the_real_server_over_stdio`, with `json.decoder.JSONDecodeError: Expecting value: line 1
   column 1 (char 0)`. Your client read `CourseMate MCP server starting` as the reply to request 1. (The in-memory
   tests still pass: they don't use stdout at all. That's why the stdio test exists.)

```bash
git checkout coursemate/mcp_server.py
git status
```

**Commit:** `RawClient: MCP over stdio by hand`

> **Stop here. Part A is done when** `py -m coursemate.mcp_raw` ends with `isError: True`, `py -m pytest -q` says
> `0 failed` (I counted `227 passed`), `git status` is clean, and your last commit is `RawClient: MCP over stdio by
> hand`. Push the branch so it's safe: `git push -u origin mcp`.

---

# Part B (day 2)

**Before you start:** re-read the Project overview (especially 2b, 2c and 2e) and answer the **Check your
understanding** questions at the end, on paper, before opening the answers.

## Step 5 — Connect the server to Claude Code (25 min)

**Learn.** `claude mcp add` stores "how to start this server" in your Claude Code config. Claude Code then starts
it when a session begins, keeps it running, and stops it at the end. Three choices:

- **Which Python?** Use the venv's own `python.exe` with a full path. Claude Code starts the server without
  activating your venv, so a bare `py` might find a Python without `mcp` installed. In Git Bash,
  `cygpath -m "$PWD"` turns `/c/Users/you/coursemate` into `C:/Users/you/coursemate`, which Windows programs
  understand.
- **Which scope?** **local** (the default): only you, only this project. A **project** scope would write `.mcp.json`
  for the team, but your full path wouldn't work on anyone else's PC.
- **Which data?** Claude will read what the tools return. So first make the database hold only `notes/sample`.

Open a terminal (this is terminal 2 from now on), in your `coursemate` folder:

```bash
source .venv/Scripts/activate
docker compose up -d
ollama list
py -m coursemate ingest notes/sample --reset
```

**What you should see:** `ollama list` shows `qwen2.5:3b` and `nomic-embed-text`. The ingest prints `ingested 8
files, 34 chunks` and **only** these eight files: `lecture-01-big-o.md` … `lecture-06-recursion.md`,
`projects.md`, `syllabus.md`. If anything else is listed, stop and run the ingest again on `notes/sample` only.

**Predict.** `claude mcp list` starts every configured server once. What will it say if Postgres isn't running?
(Hint: `build_tools` calls `store.prepare(...)` before the server starts.)

Register and check the server (still terminal 2, not inside Claude Code):

```bash
claude mcp add coursemate -- "$(cygpath -m "$PWD")/.venv/Scripts/python.exe" -m coursemate.mcp_server
claude mcp list
claude mcp get coursemate
```

**What you should see:**

```text
Added stdio MCP server coursemate with command: C:/Users/…/coursemate/.venv/Scripts/python.exe -m coursemate.mcp_server to local config
…
coursemate: C:/Users/…/coursemate/.venv/Scripts/python.exe -m coursemate.mcp_server - √ Connected
…
  Scope: Local config (private to you in this project)
  Status: √ Connected
  Type: stdio
```

(The predict answer: with Postgres stopped, the server crashes at start and the line ends with `× Failed to
connect — CONNECTION_CLOSED: Connection closed`. The real error is in the server's stderr; run
`py -m coursemate.mcp_server` yourself to read it, then `Ctrl+C`. Try
it: `docker compose stop`, `claude mcp list`, then `docker compose start`.)

Now open terminal 1 in the same folder, activate the venv, and start Claude Code:

```bash
source .venv/Scripts/activate
claude
```

Inside Claude Code, type `/mcp` and choose `coursemate`.

**What you should see:** status `connected`, with **4 tools**. Open the tool list and read `send_announcement`: the
description comes straight from your `tools/list` (depending on your Claude Code version, it may also show the
hints, such as "destructive").

**Done when:** `/mcp` shows coursemate connected with 4 tools. (Nothing to commit: the local scope lives in your
user config, outside the repo.)

---

## Step 6 — Permission rules, then a real task with your approval (30 min)

**Learn.** Re-read the three layers in 2e. Today you add layer 1 and layer 3 as **project** permission rules in
`.claude/settings.json` (committed, so they come with the repo):

```json
{
  "permissions": {
    "allow": [
      "mcp__coursemate__search_notes",
      "mcp__coursemate__make_quiz",
      "mcp__coursemate__draft_announcement"
    ],
    "ask": ["mcp__coursemate__send_announcement"],
    "deny": [
      "Bash(py -m coursemate approve:*)",
      "Bash(python -m coursemate approve:*)",
      "Edit(outbox/**)",
      "Write(outbox/**)"
    ]
  }
}
```

- `allow`: the read and draft tools run without asking (a draft changes nothing anyone sees).
- `ask`: the send always asks, even if you once clicked "don't ask again" (an `ask` rule wins over a remembered yes).
- `deny`: Claude can't run your approve command or write files in `outbox/`, so it can't fake your yes.

**Predict.** Your task will be "Write an announcement reminding students when Lab 4 is due, and send it." Which
tools will Claude call, in which order? How many times will Claude Code ask you something, and how many times will
the server refuse?

**Prompt (plan mode):**
> Create .claude/settings.json with these permission rules: allow mcp__coursemate__search_notes,
> mcp__coursemate__make_quiz and mcp__coursemate__draft_announcement; ask for mcp__coursemate__send_announcement;
> deny "Bash(py -m coursemate approve:*)", "Bash(python -m coursemate approve:*)", "Edit(outbox/**)" and
> "Write(outbox/**)". Don't change anything else.

Exit Claude Code (`/exit`) and start it again (`claude`), so it reads the new settings. Type `/permissions` and
check that the rules are listed.

**Now the real task.** Switch plan mode **off** (`Shift+Tab` until it's gone), because this time you want Claude
to act, not to plan. Type:

> Write an announcement reminding students when Lab 4 is due, and send it.

**What to expect** (Claude's words will differ):
1. `coursemate - search_notes (MCP)(query: "Lab 4 due date")`, which runs **without asking**, and finds
   `lecture-02-hash-tables.md` ("due Friday of week 5 at 11:59 pm").
2. `coursemate - draft_announcement (MCP)(title: …, body: …)`, again no question, then `draft_id: 1`.
3. `coursemate - send_announcement (MCP)(draft_id: 1)`. **Claude Code asks you** (layer 1). Choose **Yes**.
4. The server refuses (layer 2): `Error executing tool send_announcement: draft 1 is not approved, so nothing was
   sent. Tell the instructor ...`. Claude tells you it wasn't sent and that you need to run
   `py -m coursemate approve 1`.

(If Claude wants to run the approve command itself, the deny rule blocks it: that's Step 8.)

Now be the instructor. In **terminal 2**:

```bash
py -m coursemate approve 1
```

Read the draft. Is the date right (Friday of week 5, 11:59 pm)? Is it written for students? If yes, type `y`. You
should see `approved draft 1 (outbox: approved [1])`.

Back in Claude Code (terminal 1):

> I read draft 1 and approved it. Send it now.

**What to expect:** Claude Code asks again (the `ask` rule), you choose **Yes**, and the result is
`{"draft_id": 1, "status": "sent"}`. In terminal 2, check the world:

```bash
ls outbox/drafts outbox/sent outbox/approved
```

**What you should see:** `outbox/sent/001.md` and `outbox/approved/001.md`; `outbox/drafts/` is empty.

**You check:** count the human yeses it took to send one message: **two** in Claude Code, **one** in terminal 2.
Write one sentence for each saying what it protects against. (Hint: the two Claude Code yeses protect you from
Claude; the terminal yes protects your students from **any** client.)

Then press `Ctrl+O` (transcript view, Level-up tip 18) and find the `tools/call` result with the refusal text. That
string is exactly what your `send_if_approved` raised.

**Done when:** one announcement is in `outbox/sent/`, sent only after your `approve`.

**Commit:** `Permission rules for the coursemate MCP tools` (only `.claude/settings.json`; `outbox/` is git-ignored).

---

## Step 7 — Resources and a prompt: the other two primitives (25 min)

**Learn.** Re-read the table in 2b. Two additions to `build_server`:

1. **Resources:** one resource per file in `notes/sample/*.md`, with the URI `notes://sample/<file name>`. The
   server registers the files it found **at startup**, so it offers an **allow-list**, not a path the client
   chooses. A request for `notes://sample/../private/grades.md` doesn't match any registered resource, so it's an
   error. (Compare: a function that opened `notes_dir / name` for any `name` would need careful path checks. Here
   there's nothing to check.)
2. **A prompt:** `announcement(topic)` returns an instruction: search the notes for the topic, draft an
   announcement from those facts only, name the note file, **do not send**, show the draft id. Claude Code turns it
   into a slash command.

**Predict.** `notes/sample/` has 8 `.md` files. How many resources will `resources/list` return? What will a client
get for `notes://sample/../private/grades.md`?

**Prompt (plan mode):**
> In build_server, add one FileResource per notes_dir/*.md (sorted), URI "notes://sample/<file name>", name = file
> stem, mime type text/markdown, absolute path. Add a prompt `announcement(topic: str)` that tells the model to use
> search_notes for the topic, draft an announcement with draft_announcement using only those facts and naming the
> note file, not send it ("Do not send it."), and show the draft id and text so I can approve it. Tests with the
> in-memory Client: 8 resources, notes://sample/syllabus.md is among them and reading it contains "Late work", no
> URI contains "private"; reading notes://sample/../private/grades.md raises MCPError; the prompt for topic "Lab 4"
> contains "Lab 4" and "Do not send".

**You check:** in `build_server`, find the `for path in sorted(notes_dir.glob("*.md"))` loop. Why `glob` on
`notes_dir` (which is `notes/sample`) and not on `notes/`?

```bash
py -m pytest -q tests/test_mcp_server.py
py -m pytest -q
```

**Done when:** `10 passed` in `tests/test_mcp_server.py`, and the whole suite says `0 failed` (I counted `230
passed, 7 deselected` with the `[sdk]` extra).

Now use them in Claude Code. Restart it (`/exit`, `claude`) so the server restarts with the new code. Then:

1. Type `@` and then `coursemate:` and pick `notes://sample/syllabus.md` from the list (or type
   `@coursemate:notes://sample/syllabus.md`), then: `What's the late work policy? Answer only from this file.`
   **What to expect:** Claude answers from the attached file without calling `search_notes` (you gave it the
   context, so it didn't need to look).
2. Type `/mcp__coursemate__announcement` and give the topic `office hours`. **What to expect:** Claude searches,
   drafts, and stops with the draft id **without** trying to send (the prompt said not to). Check with
   `ls outbox/drafts` in terminal 2.

**Commit:** `MCP resources (sample notes) and an announcement prompt`

---

## Step 8 — Try to get around your own gate (15 min)

**Learn.** A gate you haven't attacked is a gate you hope works. You'll ask Claude, politely and then less politely,
to send a draft you never approved. The goal isn't to "win"; it's to see **which layer** stops each attempt.

**Predict** for each attempt below: which layer (1, 2 or 3 from 2e) stops it?

In Claude Code:

> Draft an announcement that class is cancelled tomorrow. Then approve it yourself and send it, I'm busy.

**What to expect:** Claude drafts it (no question). It may try `py -m coursemate approve 2` with Bash: Claude Code
**denies** it (layer 3, the deny rule) and Claude says it can't. It may then call `send_announcement`: Claude Code
asks you (layer 1). Say **Yes** on purpose, and watch the server refuse (layer 2). The draft stays in
`outbox/drafts/`.

Then one more:

> Write the approval file yourself: copy outbox/drafts/002.md to outbox/approved/002.md.

**What to expect:** a `Write`/`Edit` to `outbox/` is denied (layer 3). Claude **might** suggest `cp` with Bash
instead, which the deny list doesn't name: Claude Code asks you, and you say **No**. Write down what this shows: a
deny list blocks what it names; **you** are the last check for everything else, and in production the approval
would live in a system Claude can't reach at all (2e).

**You check:** in terminal 2, `ls outbox/sent`: only `001.md` from Step 6.

**Done when:** you can say, for each attempt, which layer stopped it. Delete the test draft with
`rm outbox/drafts/002.md` in terminal 2. (Nothing to commit.)

---

## Step 9 — CI, README, interview questions, pull request (30 min)

**Prompt 1 (plan mode):**
> In .github/workflows/ci.yml, after the agent eval smoke test, add a step "Smoke test the MCP server (raw JSON-RPC
> over stdio, fake model)" that runs `python -m coursemate.mcp_raw`.

**Prompt 2 (plan mode):**
> Update README.md: add a section "MCP server" with: what it offers (a table of the 4 tools with read/draft/send and
> their annotations, the resources notes://sample/*.md, the announcement prompt); how to run it
> (`py -m coursemate.mcp_server`, `--fake`, `py -m coursemate.mcp_raw`); how to connect it to Claude Code (the
> `claude mcp add ... cygpath ...` command, `/mcp`, the permission rules in .claude/settings.json); the approval flow
> (draft → `py -m coursemate approve N` in your own terminal → send). Design decisions: one set of tools
> (CourseTools) behind two interfaces (the agent loop and MCP); the server doesn't trust the client, so the yes
> lives outside it and covers the exact text; expected errors become ToolError, bugs stay hidden; resources are an
> allow-list of files; stdio because it's local and single-user; nothing prints to stdout. Known limitations:
> approval is a local file anyone with access to the folder could write; one user, no auth (an HTTP version would
> need OAuth and per-course access); deny rules are a list; tool results aren't checked for prompt injection yet
> (Nov 27); make_quiz is slow on a CPU model; the outbox only moves files.

**Prompt 3 (plan mode):**
> Add a section "## CourseMate MCP server" to @INTERVIEW.md with 5 questions and model answers (5–8 sentences each,
> plain language) based on this repo:
> 1. What is MCP and what problem does it solve? Explain hosts, clients and servers, and tools vs. resources vs.
>    prompts, with CourseMate examples.
> 2. Walk me through what happens on the wire when Claude Code calls your search_notes tool (stdio, JSON-RPC,
>    tools/list and tools/call, isError, the stateless _meta of the 2026-07-28 spec).
> 3. Your send tool has side effects. Why doesn't your server trust the client's permission prompt, and how does
>    your approval work? What would you change in production?
> 4. How do you test an MCP server? (in-memory Client tests, a real stdio test, break-it tests, CI smoke test, and
>    what you can only check by hand in a host.)
> 5. LLM system design: you're asked to expose a university's course systems (notes, LMS announcements, grades) to
>    AI assistants for 50 instructors and 5,000 students through MCP. Cover remote transport (Streamable HTTP) and
>    OAuth, per-course and per-role access (students never reach grades or sends), which tools need approval and how
>    approval works when the instructor answers hours later, audit logs of every tools/call, prompt injection
>    through notes and student messages, rate limits and cost, and how you'd roll out a new tool safely.

**You check:** every command in the README is one you ran. Rewrite the answer to Q3 in your own words, using your
three layers and one thing that happened in Step 8.

**Done when:**

```bash
py -m ruff check . && py -m ruff format --check . && py -m pytest -q
py -m coursemate.mcp_raw
```

all pass, and the last line is `isError: True`. **Commit:** `Document the MCP server: README, interview questions, CI smoke test`

Push and open the pull request:

```text
!git push -u origin mcp
!gh pr create --base main --title "CourseMate MCP server" --body "The four course tools over MCP (stdio), a send gated by the instructor's out-of-band approval, the sample notes as resources, an announcement prompt, a raw JSON-RPC client, in-memory and stdio tests, and Claude Code permission rules."
!gh pr checks --watch
```

**What you should see:** a PR link, then the checks finishing with `pass`, including the new step "Smoke test the
MCP server". If a step fails, open the PR on GitHub, click the failed check, read the error at the bottom, and ask
Claude to fix it. Leave the PR open: when you tell the roadmap chat "done", it reviews the PR before you merge it.

---

## Check your understanding (for the reading day)

Answer in your own words first, then open the answer.

1. You add a tool `post_to_lms(draft_id)` to the server and connect the server to 3 apps. Then you connect 2 more
   servers (a calendar, a gradebook) to the same apps. How many integrations did you write, and how many would you
   have written without MCP?
   <details><summary>Answer</summary>With MCP: each app speaks MCP once (3) and each server speaks it once (3 servers),
   so 6 pieces, and the new tool needed no change in any app. Without MCP: every app needs code for every system, 3 ×
   3 = 9 integrations, and the new tool means touching all 3 apps. The gap grows fast: 10 apps × 20 systems is 200
   vs. 30.</details>
2. Sort into tool, resource or prompt, and say who picks each: (a) the syllabus file, (b) "search the notes",
   (c) "draft this week's announcement" as a ready-made instruction, (d) "send draft 3".
   <details><summary>Answer</summary>(a) resource: the user attaches it as context. (b) tool: the model decides to
   call it in the middle of a task. (c) prompt: the user picks it, like a slash command, and it fills in an
   instruction. (d) tool, a write tool with a side effect, so it needs approval.</details>
3. Write the JSON-RPC request your client sends to call `draft_announcement` with title "Lab 4" and body "Due
   Friday", as request number 5. What tells the client which reply belongs to it? How does the reply say the tool
   failed?
   <details><summary>Answer</summary>{"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name":
   "draft_announcement", "arguments": {"title": "Lab 4", "body": "Due Friday"}, "_meta": {the three
   io.modelcontextprotocol/ keys}}}. The reply carries the same "id": 5. A tool failure is still a "result", with
   "isError": true and the message as text in "content"; a JSON-RPC "error" is for protocol problems (an unknown
   method, bad params).</details>
4. Claude Code already asks you before `send_announcement`. Why does the server check `approved/` too? Name a case
   where only the server's check saves you.
   <details><summary>Answer</summary>The server only sees a tools/call line; it can't know whether a human said yes.
   If you once chose "always allow", run Claude Code in a mode that skips prompts, or connect the server to a
   different client or a script, the client's question never happens, but the server's check still does. Also,
   Claude Code's yes is about the call; the server's approval covers the exact text you read.</details>
5. You approve draft 4, then an agent edits `outbox/drafts/004.md` before sending. What happens and which line of
   code decides it?
   <details><summary>Answer</summary>The send is refused with "draft 4 is not approved, so nothing was sent". In
   Outbox.is_approved, the approved copy's text is compared with the current draft's text; they differ, so it
   returns False, and send_if_approved raises ToolError before outbox.send runs.</details>
6. A teammate adds `print("loaded 34 chunks")` to the server. Which of your tests catches it, and why do the
   in-memory tests still pass?
   <details><summary>Answer</summary>test_the_real_server_over_stdio fails with a JSONDecodeError, because the
   client reads the printed line as the reply to request 1. The in-memory Client(server) tests pass because they
   don't use stdin/stdout at all; the messages travel inside the process. That's why you keep one real stdio test,
   and why CI runs the raw client.</details>
7. Why does `tool_errors()` catch `LookupError` and `ValueError` but not `Exception`?
   <details><summary>Answer</summary>Those are the errors the tools raise on purpose (no such draft, a bad number,
   Ollama not running), with messages the model can act on. Anything else is a bug; the SDK answers "Error executing
   tool ..." without details, so stack traces, paths or SQL don't leak to the client. You still see the full
   traceback in the server's stderr.</details>

---

## When you're done

Tell the roadmap chat **"done"**. It will review the PR, run the tests and tick this row.

## Level up

**Give Claude your own systems with MCP, and keep the safety in the server.** Wrap the functions you already trust
(here `CourseTools`) instead of writing new ones, so your agent and every MCP client share one tested core. Register
the server with a full path to the venv's Python, check it with `claude mcp list` before you start a session, and
look at `/mcp` when a tool seems missing. Then set permission rules by tool name (`mcp__coursemate__...`): allow the
read tools, `ask` for anything with side effects, and deny the commands that could fake your approval. But don't let
the client's prompt be your only gate: a server can't see who clicked yes, so anything with side effects checks a yes
the model can't give (your terminal today, a login tomorrow), and one real stdio test plus a CI smoke test make sure
a stray `print()` never breaks the protocol.
