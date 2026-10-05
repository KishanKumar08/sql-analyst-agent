# SQL Analyst Agent

You ask a question in plain English, point it at a SQLite file, and it works out the answer: it looks at the tables, writes SQL, runs it, checks what came back, and returns JSON. The model only picks the next step. The loop, tools, guardrails, budgets, context handling, tracing and evals around it are what this project is really about, and I wrote those myself.

```bash
agent ask "Which genre has the most tracks?" --db data/Chinook_Sqlite.sqlite
```
```json
{
  "answer": "Rock has the most tracks: 1,297 of 3,503 ...",
  "sql_used": ["SELECT g.Name, COUNT(*) AS n FROM Track t JOIN Genre g ..."],
  "assumptions": [],
  "confidence": "high",
  "answerable": true,
  "status": "complete",
  "stop_reason": "submitted"
}
```
The trace path goes to stderr: `trace: traces/run-20261005-....jsonl`

---

## Setup

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/). From a fresh clone this took me under 30 seconds.

```bash
uv sync                                  # install dependencies
uv run python scripts/fetch_db.py        # downloads Chinook v1.4.5 and checks its SHA-256
cp .env.example .env                     # then fill in your Azure OpenAI values
uv run agent ask "How many tracks are in the database?" --db data/Chinook_Sqlite.sqlite
```

If you'd rather type plain `agent ask ...` without `uv run`, run `source .venv/bin/activate` first.

Other things you can run:

```bash
uv run agent trace traces/run-XXXX.jsonl            # readable view of a trace (--full shows everything)
uv run agent chat --db data/Chinook_Sqlite.sqlite   # follow-up questions in one session
uv run python -m eval.run_eval                      # 10 questions x 3 runs, report goes to eval/results/
uv run pytest                                       # 89 tests, no API calls, about 2s
uv run python scripts/make_demo_db.py               # builds a second, non-Chinook database to try --db on
```

The four run budgets are flags: `--max-turns --max-tokens --max-cost --max-wall`. Everything else (and those too) can be set with `AGENT_<NAME>` env vars, e.g. `AGENT_QUERY_TIMEOUT_S=2`, `AGENT_MAX_ROWS=100`, `AGENT_REASONING_EFFORT=low`, `AGENT_SCHEMA_IN_PROMPT=false`. Defaults live in [agent/config.py](agent/config.py).

---

## How I approached it

I wanted to build this as a harness project more than a text-to-SQL one. Getting the model to write a query is the easy bit. What happens when the query is bad, when the run goes long, when the model is confidently wrong: that's the part I wanted to get right.

So the order I worked in was roughly:

1. **Look at the data before writing any agent code.** I answered all 10 questions myself in `sqlite3` first. This turned out to matter a lot. Three of the questions aren't really about SQL at all: Q8 is a 41-way tie, Q9 ("best customer") is ambiguous, and Q10 (profit margin) can't be answered because there's no cost data. I wouldn't have known how to grade the agent without finding that first.
2. **Safety layer next**, with tests that try to break it, before any model was involved.
3. **Loop, tools and budgets, tested against a fake model.** I wrote a scripted stand-in for the LLM so I could force every limit and every failure without spending money. It caught a real bug (more on that below).
4. **Then the real model, then evals, then a lot of reading traces.** Most of the actual improvements came from this step. I ran the eval 7 times in total and changed something each time based on what the traces showed.

---

## What's in the repo

```
agent/
  cli.py         ask / chat / trace commands, plus a watchdog
  loop.py        the agent loop: budgets, stop conditions, the forced final answer, fallback
  tools.py       the 4 tools: schemas, validation, dispatch (never raises)
  db.py          read-only, time-limited, size-capped SQLite access
  context.py     trims results before the model sees them, compacts old turns
  tracer.py      writes the JSONL trace
  trace_view.py  pretty-prints a trace
  llm.py         Azure OpenAI call + retries (the only file that knows about the provider)
  prompts.py     system prompt (generic, nothing about Chinook in it)
  config.py      every limit in one place
eval/            ground truth, graders, runner, results
tests/           safety attacks, loop tests with the fake model, grader tests
```

One run, roughly:

```
question ─► model ─► tool calls ─► tool results (trimmed) ─► model ─► … ─► submit_answer ─► JSON
              ▲                                                │
              └── before every call: check budgets, compact ───┘
                        │ budget (minus a 15% reserve) used up
                        ▼
              one forced "submit what you have" call, output capped to what's left
                        │ that fails too / no room left
                        ▼
              fallback answer built from the last query that worked  ─► JSON
```

| Piece | Where |
|---|---|
| the loop | [loop.py](agent/loop.py), hand-written |
| tools + validation | [tools.py](agent/tools.py) |
| read-only + limits | [db.py](agent/db.py) |
| budgets + stopping | [loop.py](agent/loop.py): `_limit_hit`, `_finalize`, `_fallback`; watchdog in [cli.py](agent/cli.py) |
| context | [context.py](agent/context.py), schema preload in `loop.py` |
| tracing | [tracer.py](agent/tracer.py), `agent trace` |
| evals | [eval/](eval/) |

---

## Decisions, and why

### No framework, my own loop
I wrote the loop against the Chat Completions API with the plain `openai` SDK. I did think about PydanticAI and the OpenAI Agents SDK, but the things I cared about most (when budgets get checked, what happens at a limit, what gets compacted, what goes in the trace) are exactly the things a framework decides for you. I'd have ended up working around its stop logic instead of designing my own, and I wanted to be able to explain and change every line. The loop itself (`Agent._loop`) ends up being about 40 lines; everything else is harness around it.

The SDK just does HTTP. I turned its retries off (`max_retries=0`) and retry myself in [llm.py](agent/llm.py), so every retry shows up in the trace and can't run past the run's deadline.

### Model: gpt-5-mini, reasoning effort medium
I picked gpt-5-mini because it's cheap ($0.25 per million input tokens, $2.00 per million output, $0.025 per million cached input), good at tool calling, supports strict tool schemas, and I had Azure credits. It's a reasoning model, so hidden reasoning tokens get billed as output; the tracer records them separately and the cost estimate includes them. Switching provider means changing `llm.py` and nothing else.

I started on `low` effort and only moved to `medium` after measuring it:

| Effort | Pass | Avg cost/run | Avg turns | Avg latency |
|---|---|---|---|---|
| low | 29/30 | $0.0028 | 4.9 | 12.7s |
| medium | 30/30 | $0.0043 | 4.4 | 18.0s |

The one failure on `low` wasn't random. It was Q10 again and again (about 1 run in 3): the model used the sale price as if it were the cost and reported "0% margin for every genre" with `answerable: true`. That's a plausible, confident, wrong number, which is the worst thing an analyst tool can do. Q10 alone on `medium` passed 5/5. Paying about $0.0015 and 5 seconds more per question for that felt like an easy call. `AGENT_REASONING_EFFORT=low` still works if speed matters more.

### Read-only: three layers, because one wasn't enough
My first thought was the usual: open the file read-only and set `PRAGMA query_only`. Before trusting that, I tested each layer on its own against a copy of the database. Both of them block DELETE and DROP, but **both let `ATTACH DATABASE` and `PRAGMA writable_schema` through**. ATTACH can open (or create) a different file with write access, so "read-only" wasn't actually read-only.

The thing that does block it is SQLite's authorizer callback: SQLite asks my function before every operation, and I only allow reads, functions, recursive CTEs and a short list of schema PRAGMAs (`table_info`, `foreign_key_list`, ...). So it's three layers now: `mode=ro`, `query_only`, and the authorizer.

I didn't go for "check the SQL starts with SELECT". It's easy to get around (`WITH x AS (...) DELETE`, `REPLACE`, comments), and SQLite already knows exactly what each statement does.

The other limits:
- **Timeout:** a progress handler checks the clock every 1,000 VM steps and aborts. An infinite recursive CTE stops on time.
- **Row cap:** `fetchmany(max_rows + 1)`, so a huge result never gets pulled into memory. Long text is clipped, and blobs become `<blob N bytes>`.
- **One statement per call:** `SELECT 1; DROP TABLE x` is rejected.

[tests/test_db_safety.py](tests/test_db_safety.py) tries 13 kinds of writes, a runaway query and a huge result, and checks the file's SHA-256 is unchanged after each write attempt.

### Only the four tools
I thought about adding `get_distinct_values`, `count_rows`, `explain_query` and `search_columns`. Every one of them is a `SELECT` the model can already write with `run_sql`, so they'd add choices without adding anything it couldn't do. And the traces kept showing that more ways to explore means more exploring, which means more turns. I'd rather have a few general tools with good error messages. Adding a tool is about 5 lines anyway (a schema in `SCHEMAS` plus a `_name` method); validation, strict mode, errors and tracing come with it.

### Errors are written for the model to recover from
A failing tool never crashes the run; it returns `{"ok": false, "error_type": ..., "message": ..., "hint": ...}`. I tried to make the hints actually useful: `SELECT * FROM Tracks` gets "Did you mean: Track?" (via `difflib`), a timeout gets "add filters or LIMIT, avoid cross joins". There's a catch-all at the end of `ToolBox.call()` so even a bug in my own tool code becomes an error result rather than a crash.

### Checking `submit_answer` in code, not just asking nicely
Some rules I put in the prompt first and then had to move into code because the model didn't follow them:
- **Every query in `sql_used` must actually have run** in this session (compared ignoring whitespace, case and a trailing `;`). If not, the answer is rejected and the model is told which ones. In the first eval this caught the model 4 times listing schema lookups (`"-- list tables ..."`) as if they were SQL. This is also my self-verification step: it's free when the answer is fine.
- **No email or phone number in the answer unless the question asked for one.** The model kept putting Helena Holý's email into "who is our best customer?". I added a prompt rule and it ignored it. Now it's enforced in code, and it still catches the model every eval. Honestly this was the clearest lesson of the whole thing: a rule in the prompt is a suggestion, a rule in code is a guarantee.
- **Strict tool schemas** (`strict: true`), so the API guarantees every required field is there. Before this, 6 of 30 runs left out `answerable` and needed an extra (expensive) turn. Strict mode doesn't support `maxLength`, so I strip it from the schema I send but still check it in my own validator.

### Budgets and stopping
Before every model call I check turns, total tokens, cost and wall-clock time. The check is a projection (last real prompt size, plus whatever's been added, plus the last output size) so one big step can't jump straight past the limit.

When the projection hits 85% of any budget, exploration stops. The 15% reserve pays for one last call that forces `submit_answer` with "you've hit the X limit, submit what you have". That call's output tokens are capped to exactly what's left. The answer comes back as `status: "partial"`, confidence capped at `medium`, and an assumption saying the run stopped early.

If even that fails, the harness writes an answer itself from the last query that worked (`confidence: low`, `status: "failed"`). LLM errors, a bad `--db` path, Ctrl-C and unexpected exceptions all end up there. On top of that there's a watchdog thread in the CLI that prints valid JSON and exits if a run somehow goes 30s past its wall-clock limit. Every way out prints valid JSON.

There's also a one-time reminder at 60% of the turn budget ("you've used X of Y turns, submit if you have the answer"), added after I saw runs find the answer at turn 4 and then keep re-checking it until they hit the limit.

I added `status` and `stop_reason` (`submitted`, `max_turns`, `max_tokens`, `max_cost`, `max_wall_time`, `llm_error`, `db_error`, `interrupted`, `internal_error`) to the required JSON shape, plus `stop_detail` when the run didn't end normally.

The first version of this had a bug: it stopped exploring at the limit but didn't budget the final call. A test with a 5,000-token budget ended at 6,600. That's what the projection and the output cap fix. [tests/test_loop.py](tests/test_loop.py) forces every limit with the fake model and checks it holds.

### Context: preload the schema, trim results, compact old turns
**Schema preload.** After the v4 eval I wanted to see where the turns were going. 46% of all turns were the model working out the schema (`list_tables` in 30/30 runs, `describe_table Track` in 15). I'd assumed caching repeated tool calls would help, but there were zero repeated calls within a run, and a SQLite query takes about 1 ms anyway. The cost is the model turn wrapped around the call.

So the harness now reads the schema once at the start (tables, columns, types, keys, row counts, 2 sample rows each) and puts it in the system prompt:
- the model starts already knowing the tables: 4.4 → 2.2 turns, 35% cheaper, 22% faster
- it sits in the unchanging start of every request, so Azure's prompt cache bills it at the cached rate; the cached share of input went from 30% to 79%
- it's cached in memory per file (path, size, mtime), so the eval and `agent chat` only read it once
- if it's bigger than `max_schema_chars` (12k) it falls back to the discovery tools; `AGENT_SCHEMA_IN_PROMPT=false` turns it off

The first version of this broke Q10, which I didn't see coming. Without sample rows, the model saw `Track.UnitPrice` and `InvoiceLine.UnitPrice` next to each other, decided one of them was "the cost", and wrote one query without ever looking at the data. When it had to call `describe_table`, it saw sample rows where both prices were 0.99, and that's what made it realise it was just a price. Putting 2 sample rows per table into the schema brought that back (Q10 3/3 in the full eval, 4/5 in a separate Q10-only run). It's not perfect: across 8 Q10 runs with the schema preloaded, 1 still made the mistake, against 0 of 8 in discovery mode. I decided the speed and cost win was worth it; `AGENT_SCHEMA_IN_PROMPT=false` is there for the more careful mode.

**Trimming.** SQLite returns up to 200 rows, but the model only sees the first 50, with a note when rows are hidden ("Showing 50 of more than 200 rows. Aggregate, filter or add LIMIT"). Wide rows get cut further to fit 6,000 characters. The full result still goes to the trace, along with exactly what the model saw.

**Compaction.** If the prompt goes past 16k tokens, older tool results get rewritten in place as one-line summaries; the newest 4 stay as they are. What it always keeps: the system prompt and question, every SQL query that was tried, schema facts (columns, keys, row counts), the shape and first 3 rows of old results, and every error message so the model doesn't repeat a mistake. What it drops: sample rows and the long tail of rows. I did this in plain code rather than asking the LLM to summarise, because an LLM summary costs a call, adds latency, and might drop the one number that matters. Rewriting old messages breaks the prompt cache once, which is why it only kicks in past a threshold. In practice it rarely triggers now that runs are about 2 turns.

### Tracing
One JSONL file per run, flushed after every line so a killed run still leaves something readable. It has: `run_start` (question, DB, config), `llm_request` (only the messages added since the last request, so it's complete without repeating everything), `llm_response` (content, tool calls, input/cached/output/reasoning tokens, latency, cost, retries), `tool_call` / `tool_result` (args, full result, what the model actually saw, latency), `compaction`, `limit_reached` and `run_end` (stop reason, output, totals). `agent trace <file>` shows it with highlighted SQL, small result tables and errors in red.

### Keeping it generic
The system prompt says nothing about Chinook. It describes how a careful analyst works: check the joins, look for ties, duplicates and NULLs, say what you assumed, and say `answerable=false` instead of quietly using different data. To check I wasn't fooling myself, [scripts/make_demo_db.py](scripts/make_demo_db.py) builds an unrelated SaaS database with a table name containing a space, no declared foreign keys, NULLs, a view, dates as text and money in cents. The agent handled it fine (see sample trace 06).

### Extras
I added four: parallel tool calls (all calls in a turn are run, and tests cover it), streaming progress (each step prints to stderr while stdout stays clean JSON), multi-turn follow-ups (`agent chat`), and self-verification (the `sql_used` check above).

I skipped two on purpose. The **sandboxed Python tool** needs real isolation (a separate process, no network or filesystem, CPU and memory limits); a half-done sandbox would undermine the safety work, and none of the 10 questions needed it. **Resume from trace** is doable since the trace has everything, but with runs at around 10 seconds, re-running is cheaper than resuming, so it went to the bottom of the list. I'd rather have a solid core than half-finished extras.

---

## Evals

### Ground truth
[eval/ground_truth.py](eval/ground_truth.py) computes every answer with SQL at eval time, rather than me typing numbers in, and has comments on what's tricky about each one:

| Q | Truth | What's tricky |
|---|---|---|
| Q1 | 3,503 | nothing |
| Q2 | Rock (1,297) | nothing |
| Q3 | Iron Maiden 21, Led Zeppelin 14, Deep Purple 11, Metallica 10, U2 10 | Metallica and U2 tie for 4th/5th, but 6th has 6, so the top 5 is clear |
| Q4 | USA, Canada, France, Brazil, Germany | billing country vs customer country? I checked: 0 mismatches. `Invoice.Total` equals the sum of the invoice lines too |
| Q5 | Jane Peacock ($833.04) | has to go Customer.SupportRepId → Employee |
| Q6 | 2010: $481.45, up $31.99 (+7.12%) on 2009 | two-part answer; revenue barely moves year to year |
| Q7 | video ≈ 39 min, audio ≈ 4.3–4.7 min | milliseconds → minutes |
| Q8 | 41 tracks tie at 5 playlists | playlist names are duplicated ("Music" is IDs 1 and 8); counting distinct names gives 4, same 41 tracks |
| Q9 | Helena Holý ($49.62) | "best" is ambiguous: by spend she's clearly first, by number of orders it's a 58-way tie, by tracks bought a 3-way tie |
| Q10 | can't be answered | there's no cost data anywhere; the only money column is the sale price |

### How I grade
Deterministic graders, no LLM judge, one per question:
- Q1, Q2, Q5: the right value or name appears. For Q5, if it ranks all three agents, the winner has to come first.
- Q3, Q4: all 5 names; Q4 also checks the order.
- Q6, Q7: numbers within a tolerance. Q6 needs the year, the previous year, and either the % change (±0.15) or the dollar change (±$0.05); Q7 needs all 5 averages within ±0.06 min.
- The ones without a single exact answer:
  - **Q8:** passes only if it gives the max (5, or 4 by distinct name), admits there's a tie, and names at least one of the tied tracks. Naming one track as "the" answer fails.
  - **Q9:** passes if it names Helena Holý and says what "best" meant, either in `assumptions` or by mentioning spend/revenue.
  - **Q10:** passes if `answerable=false`, or if it explains there's no cost data without putting a margin % on it. Making up a margin fails.
- A run that never submitted (`status: failed`) always fails.
- Separately, every run re-executes its `sql_used` to check the cited queries actually run.

The graders have their own tests ([tests/test_graders.py](tests/test_graders.py)) with realistic right and wrong answers, because a grader that passes bad answers makes the whole report pointless.

Why no LLM judge: it'd cope better with odd phrasing, but it adds cost, randomness, and a second model whose mistakes I'd then have to check too. The downside of my approach is that graders can be too strict about format (e.g. "4 min 26 s" instead of 4.43), so I read the failing runs by hand.

### Results
Final run: [eval/results/results-20261005-170956.md](eval/results/results-20261005-170956.md) (every answer, grade and a link to its trace). gpt-5-mini, effort medium, schema preloaded (the defaults), 3 runs per question.

**30/30 passed** · avg $0.0022 per run · 2.1 turns · 10.4s · $0.07 for the whole eval

| Q | Pass | Avg cost | Avg turns | Avg latency |
|---|---|---|---|---|
| Q1 tracks | 3/3 | $0.0007 | 2 | 4.5s |
| Q2 genre | 3/3 | $0.0018 | 2 | 7.8s |
| Q3 artists | 3/3 | $0.0028 | 2 | 12.3s |
| Q4 countries | 3/3 | $0.0022 | 2 | 11.3s |
| Q5 support agent | 3/3 | $0.0019 | 2 | 8.9s |
| Q6 best year | 3/3 | $0.0030 | 2.7 | 13.5s |
| Q7 media type | 3/3 | $0.0020 | 2 | 9.4s |
| Q8 playlists | 3/3 | $0.0029 | 2 | 13.2s |
| Q9 best customer | 3/3 | $0.0032 | 3 | 15.3s |
| Q10 profit margin | 3/3 | $0.0019 | 1 | 8.1s |

How it got there (every run is in `eval/results/`):

| Run | What changed | Pass | $/run | Turns | Latency | Tool errors | Limit hits |
|---|---|---|---|---|---|---|---|
| v1 | first real run (effort low) | 30/30 | $0.0033 | 6.1 | 16.3s | 11 | 0 |
| v2 | strict schemas, parallel describes, email check | 29/30 | $0.0037 | 6.4 | 17.2s | 19 | 2 |
| v3 | prompt fixes, turn reminder | 29/30 | $0.0028 | 4.9 | 12.7s | 0 | 0 |
| v4 | v3 on effort medium | 30/30 | $0.0043 | 4.4 | 18.0s | 0 | 0 |
| v5 | schema in the system prompt | 28/30 | $0.0024 | 2.1 | 11.8s | 0 | 0 |
| v6 | v5 + 2 sample rows per table | 30/30 | $0.0028 | 2.2 | 14.0s | 2 | 0 |
| v7 | handle non-data messages (#14 below) | 30/30 | $0.0022 | 2.1 | 10.4s | 3 | 0 |

I wouldn't read too much into "100%". These are 10 questions I studied closely, on the database I built against, so treat it as a regression suite, not a measure of how it does on new questions. Also, v1 scored 30/30 too, and its traces were full of problems the pass rate didn't show: wasted turns, 11 tool errors, SQL pasted into answers, an email address nobody asked for. Most of what I learned came from reading traces, not from the score. (The tool errors in v6 and v7 are the email check catching the model, which is it working as intended.)

Total API spend for everything (development plus all evals): $0.78, summed from the traces.

---

## What broke, and what I changed

Caught by tests while building:
1. **The final answer blew the budget.** The forced final call wasn't budgeted, so a 5,000-token run ended at 6,600. Fixed by projecting the next call from real token counts and capping the final call's output.
2. **ATTACH got through "read-only".** Neither `mode=ro` nor `query_only` stops it. The authorizer does (see above).
3. **The trace path wrapped across two lines** when printed with `rich`, so copy-pasting it didn't work. Now it's a plain print.

First real call:

4. **Every call returned 404.** The Azure portal shows the endpoint as `https://<res>.services.ai.azure.com/openai/v1/responses`, but the SDK wants just the base URL and adds its own path. [llm.py](agent/llm.py) now keeps only the scheme and host, so either form works. The harness handled it properly (valid JSON, `stop_reason: llm_error`, a trace), which is how I worked out what was wrong.

From reading traces (none of these showed up in the pass rate):

5. **6 of 30 answers were missing `answerable`**, got rejected and had to be resubmitted. Non-strict tool calling lets the model skip fields. Strict schemas fixed it; none since.
6. **4 runs listed `"-- list tables ..."` in `sql_used`.** The model counts schema lookups as "queries used". The provenance check caught every one; the tool description now says list/describe aren't SQL. None since.
7. **Q9 answers included the customer's email.** I added a prompt rule; it didn't work. Moved the check into code.
8. **SQL and confidence pasted into the `answer` text, and all 41 tied tracks listed for Q8.** The answer had no style rules. The prompt now says no SQL in the answer, and give a count plus examples for long lists.
9. **Every run spent 3 turns before writing any SQL** (list, describe, describe) because it described one table per turn. Told it to describe them all in one turn.
10. **My fix for #9 caused a regression.** 18 `unknown_table` errors (`tracks`, `genres`, `media_types`): it now called `list_tables` and `describe_table` in the same turn, guessing names before it had seen the list. Changed the prompt to "list_tables on its own first, then describe using exact names". None since.
11. **Q6 and Q7 hit the 15-turn limit after finding the answer at turn 4** ([trace](traces/samples/05-LIMIT-q6-reverification-loop-v2.jsonl)). My own "sanity-check your results" instruction plus low effort turned into endless re-checking (the same yearly totals, one year per query). Changed it to "at most one or two targeted checks, never re-run a result you already have" and added the 60% turn reminder. The forced final answer still got both right, which was nice to see.
12. **Q10 on low effort: "0% margin for every genre", `answerable: true`** ([trace](traces/samples/03-FAILED-q10-proxy-cost-low-effort.jsonl)). It used the sale price as the cost. A general rule in the `answerable` description helped a bit but didn't fix it. Medium effort did (5/5), so that's the default now.
13. **At a limit, the forced answer set `answerable: false` "because I could not run the query"** ([trace](traces/samples/04-LIMIT-max-turns-forced-answer.jsonl), recorded before the fix). It mixed up "I didn't finish" with "the data can't answer this". The final-answer prompt now says running out of turns doesn't make a question unanswerable; `status` and `stop_reason` already say the run stopped early.
14. **Typing "Hi" into `agent chat` cost 6 turns and $0.0064 and produced a "database inventory" nobody asked for** ("since you asked me to proceed"). There was no path for messages that aren't data questions: the model replied with text, my nudge said "otherwise continue investigating", and it took that as the user telling it to carry on. Now greetings, small talk and off-topic messages go straight to `submit_answer` with `answerable: false` and a short "here's what I can help with" reply, and the nudge says it's from the harness. "Hi" now takes 1 turn and $0.0012. I re-ran the full eval (v7) to make sure nothing else moved. One side effect: Q10 now answers in 1 turn straight from the schema (it lists the tables it checked for cost columns).

---

## Sample traces

In [traces/samples/](traces/samples/); open any of them with `uv run agent trace <file>`.

| File | What it shows |
|---|---|
| `01-q8-tie-handled.jsonl` | finds the 41-way tie and reports the count plus examples |
| `02-q10-unanswerable.jsonl` | checks for cost data, finds none, `answerable: false` |
| `03-FAILED-q10-proxy-cost-low-effort.jsonl` | **failure:** on low effort, treats the sale price as cost → "0% margin" |
| `04-LIMIT-max-turns-forced-answer.jsonl` | **limit:** `--max-turns 3`, the forced final answer, `status: partial` (before fix #13) |
| `05-LIMIT-q6-reverification-loop-v2.jsonl` | **limit:** the re-checking loop from #11 hits 15 turns; the forced answer is still right |
| `06-other-database-saas.jsonl` | the SaaS database: uses a window function for "current plan", drops cancelled companies, converts cents, states the per-seat assumption |

All the traces from the final eval are in `traces/eval-20261005-170956/`, and every earlier eval's traces are committed too, so the trace links in `eval/results/*.md` all work.

---

## Known limitations
- **The Q10-style "proxy" mistake still happens on low effort**, and nothing in code catches "used a price as a cost". The prompt helps; more reasoning is the only thing I found that really fixes it.
- **Budgets use a projection.** The final call's output is hard-capped, but its input size is an estimate, so a run can finish slightly over the token or cost budget (by how wrong one prompt estimate is).
- **Graders match text.** A right answer in an unusual format (durations as "4:26") would be marked wrong.
- **The query timeout doesn't cover one expensive function call.** Something like `randomblob(1e9)` allocates before the progress handler gets a chance. Small risk, but not zero.
- **In `agent chat`, a follow-up can't cite SQL from an earlier question** in `sql_used` without re-running it, because the provenance check is per question.
- **`list_tables` counts rows in every table**, which would be slow on a big database. The query timeout guards it, and the count shows `null` if it times out.
