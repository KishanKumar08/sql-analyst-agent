# SQL Analyst Agent

A command-line agent that answers business questions about any SQLite database. It explores the schema, writes and runs SQL, and returns a structured JSON answer. The model only decides *what to do next*. Everything around it is hand-written and is the actual subject of this project: the loop, tools, guardrails, budgets, context handling, tracing and evals.

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
The trace path is printed on stderr: `trace: traces/run-20261005-....jsonl`

---

## Setup (fresh clone)

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # install dependencies
uv run python scripts/fetch_db.py        # download Chinook v1.4.5 and verify its SHA-256
cp .env.example .env                     # then fill in your Azure OpenAI values
uv run agent ask "How many tracks are in the database?" --db data/Chinook_Sqlite.sqlite
```

To type `agent ask ...` exactly as in the brief, without `uv run`, activate the environment first: `source .venv/bin/activate`.

Other commands:

```bash
uv run agent trace traces/run-XXXX.jsonl            # human-readable trace (--full for everything)
uv run agent chat --db data/Chinook_Sqlite.sqlite   # follow-up questions in one session
uv run python -m eval.run_eval                      # 10 questions x 3 runs, writes eval/results/
uv run pytest                                       # 89 tests, no API calls, ~2s
uv run python scripts/make_demo_db.py               # a second, non-Chinook database for --db
```

The four run budgets are flags: `--max-turns --max-tokens --max-cost --max-wall`. Every setting, including those, can also be set as an environment variable `AGENT_<NAME>` (e.g. `AGENT_QUERY_TIMEOUT_S=2`, `AGENT_MAX_ROWS=100`, `AGENT_REASONING_EFFORT=low`, `AGENT_SCHEMA_IN_PROMPT=false`). Defaults are in [agent/config.py](agent/config.py).

---

## Architecture

```
agent/
  cli.py         ask / chat / trace commands, watchdog
  loop.py        the agent loop: budgets, stop conditions, finalize, fallback
  tools.py       4 tools: schemas, input validation, dispatch (never raises)
  db.py          read-only, time-limited, size-capped SQLite access
  context.py     result trimming before the model sees it + compaction
  tracer.py      JSONL trace writer
  trace_view.py  pretty-printer
  llm.py         Azure OpenAI wrapper + retries (only file that knows the provider)
  prompts.py     system prompt (generic: no Chinook knowledge)
  config.py      every limit in one place
eval/            ground truth, graders, runner, results
tests/           safety attacks, loop behaviour with a scripted fake model, grader tests
```

One run:

```
question ─► model ─► tool calls ─► tool results (trimmed) ─► model ─► … ─► submit_answer ─► JSON
              ▲                                                │
              └── before every call: budgets, compaction ──────┘
                        │ budget (minus 15% reserve) used up
                        ▼
              one forced "submit_answer now" call, output capped to what's left
                        │ fails or no room left
                        ▼
              deterministic fallback from the last successful query  ─► JSON
```

### What I built vs what a framework would do

I wrote the loop myself against the Chat Completions API using the plain `openai` SDK, with no agent framework. The assignment is about the harness, and I wanted to own every part of it: when budgets are checked, what happens at a limit, what compaction keeps, what goes in the trace. A framework would have given me the loop and tool dispatch, but I would then be working around its stop conditions and its retry behaviour rather than designing them. The loop itself (`Agent._loop`) is about 40 lines; the rest is the harness around it.

The SDK's only jobs are HTTP and response parsing. I turned off its built-in retries (`max_retries=0`) and retry in [llm.py](agent/llm.py) instead, so each retry shows up in the trace and can never run past the run's wall-clock deadline.

### Model

**Azure OpenAI `gpt-5-mini`**, reasoning effort **`medium`**. Why this model:
- **Tool use:** reliable at calling tools in parallel and following schemas, and it supports strict tool schemas.
- **Cost:** $0.25 per million input tokens, $2.00 per million output tokens, $0.025 per million cached input tokens. A full 30-run eval costs about $0.13.
- **Availability:** I have Azure credits.

It's a reasoning model, so hidden reasoning tokens are billed as output tokens. The tracer records them separately, and the cost estimate includes them. The deployment is configurable (`AZURE_OPENAI_DEPLOYMENT`), and switching provider means changing only `llm.py`.

**Why `medium` effort rather than `low`:** I measured it, with the same code on the same 30 runs:

| Effort | Pass | Avg cost/run | Avg turns | Avg latency |
|---|---|---|---|---|
| low | 29/30 | $0.0028 | 4.9 | 12.7s |
| **medium** | **30/30** | $0.0043 | 4.4 | 18.0s |

The one failure at `low` kept happening (Q10, about 1 run in 3 across two evals). The model used the sale price as a stand-in for cost and confidently reported "0% margin" with `answerable: true`. Q10 alone at `medium` passed 5 out of 5. A plausible but wrong number is the most damaging kind of failure for an analyst tool, so preventing it is worth about $0.0015 and 5 seconds per question. `AGENT_REASONING_EFFORT=low` is still available when latency matters more.

---

## Requirements → where they live

| | Requirement | Implementation |
|---|---|---|
| R1 | Agent loop | [loop.py](agent/loop.py), hand-written |
| R2 | Tools + validation | [tools.py](agent/tools.py): `list_tables`, `describe_table`, `run_sql`, `submit_answer`. Each has a JSON schema and a validator. Bad JSON, missing or unknown fields, wrong types and bad enums all come back as `{"ok": false, "error_type", "message", "hint"}` |
| R3 | Safety + limits | [db.py](agent/db.py): 3 write-protection layers, a per-query timeout, a row cap and a cell-length cap |
| R4 | Budgets | turns, tokens, USD and wall-clock limits; reserve + forced final answer + fallback + watchdog |
| R5 | Context | [context.py](agent/context.py): trimming + deterministic compaction |
| R6 | Tracing | [tracer.py](agent/tracer.py) + `agent trace` |
| R7 | Evals | [eval/](eval/): ground truth from SQL, per-question graders, 3 runs each |

---

## Key design decisions

### Read-only is enforced in three layers (R3)
1. The file is opened with `mode=ro`.
2. The connection runs `PRAGMA query_only=ON`.
3. **A SQLite authorizer callback** allows only `SELECT`, reads, functions, recursive CTEs, and an allowlist of introspection PRAGMAs (`table_info`, `foreign_key_list`, …). Everything else is denied before it executes.

I tested each layer on its own against a copy of the database. **`mode=ro` and `query_only` each block DELETE and DROP, but both allow `ATTACH DATABASE` and `PRAGMA writable_schema`.** `ATTACH` can open, or create, another file with write access. Only the authorizer blocks it. That experiment is why there are three layers rather than one.

**Rejected:** checking that the SQL "starts with SELECT" using a regex or a parser. It's easy to get around (`WITH x AS (...) DELETE ...`, `REPLACE`, `ATTACH`, comments), and it duplicates what SQLite already knows exactly.

The other limits:
- **Timeout:** a progress handler checks a deadline every 1,000 VM steps and aborts the statement. An infinite recursive CTE stops at the timeout.
- **Size:** `fetchmany(max_rows + 1)` reads at most one row past the cap, so a huge result never gets loaded into memory. Long text is clipped, and blobs are replaced by `<blob N bytes>`.
- **One statement per call:** `SELECT 1; DROP TABLE x` is rejected.

All of this is covered in [tests/test_db_safety.py](tests/test_db_safety.py). Every write test also checks the database file's SHA-256 is unchanged.

### Tool errors are designed for recovery (R2)
Errors carry a type and, where possible, a fix. `SELECT * FROM Tracks` returns `unknown_table` with the hint "Did you mean: Track?" (via `difflib`). A timeout comes back with "add filters or LIMIT, avoid cross joins". `ToolBox.call()` has a final catch-all, so even a bug in a tool turns into an error result rather than a crash.

### `submit_answer` is checked in code before it's accepted
- **Where its SQL came from:** every query in `sql_used` must match (ignoring whitespace, case and a trailing `;`) a query that **actually ran successfully in this session**. Otherwise the submission is rejected and the model is told which queries are unverified. In the first eval this caught the model 4 times listing schema lookups (`"-- list tables ..."`) as if they were SQL. It's the self-verification step I chose: cheap, deterministic, and it costs no extra model call when the answer is fine.
- **Personal data:** an answer containing an email address or phone number is rejected unless the question asked for contact details. The model kept adding Helena Holý's email to "who is our best customer?". When I added a rule to the prompt, it ignored it. So now it's enforced in code, and in the next eval the check caught it 3 times.
- **Schema:** tools are sent with `strict: true`, so the API guarantees every required field is present. Before this, 6 of 30 runs left `answerable` out of `submit_answer`, and each one cost an extra, expensive turn. Strict mode doesn't support `maxLength`, so it's removed from the schema sent to the model but still enforced by our own validator.

### Budgets and stopping (R4)
Four limits are checked **before every model call**: turns, total tokens, USD cost and wall-clock time. The check uses a *projection*: the last call's real prompt size, plus the new messages since then, plus the last output size. That way one big step can't jump past the limit.

Exploration stops when the projected spend reaches 85% of any budget (`finalize_reserve = 0.15`). The reserve pays for **one forced call** with `tool_choice = submit_answer` and the message "the run has hit its X limit, submit what you have". That call's `max_completion_tokens` is set to exactly the room left in the token and cost budgets, so it can't overshoot. The answer is marked `status: "partial"`, confidence is capped at `medium`, and an assumption explains that the run stopped early.

Before any limit is reached, a **one-time reminder** is added at 60% of the turn budget: "you've used X of Y turns; if you have the answer, submit". I added it after seeing runs that found the answer at turn 4 and then re-checked it until the turn limit (see Failure modes).

If even that call fails, the harness builds a **fallback answer without the model**: it reports what the last successful query returned, with `confidence: low` and `status: "failed"`. LLM errors, a bad `--db` path, Ctrl-C and any unexpected exception all end up on this path. There's also a watchdog thread in the CLI: if a run somehow passes its wall-clock limit by 30s, it prints a valid JSON failure and exits. **Every exit path prints valid JSON.**

I added two fields to the required output shape: `status` (`complete` / `partial` / `failed`) and `stop_reason` (`submitted`, `max_turns`, `max_tokens`, `max_cost`, `max_wall_time`, `llm_error`, `db_error`, `interrupted`, `internal_error`), plus `stop_detail` when the run didn't end normally.

**Bug I found while building this:** the first version stopped exploring at the limit, then made the final call with no cap. A test with a 5,000-token budget ended at 6,600 tokens. The fixes were the projection and the hard output cap on the final call. [tests/test_loop.py](tests/test_loop.py) now forces every limit with a scripted fake model and checks the budget holds.

### Context management (R5)
**0. Preload the schema, once.** After v4 I measured where the turns went: **46% of all turns were schema discovery** (`list_tables` in 30/30 runs, `describe_table Track` in 15), and there were **zero** repeated identical tool calls within a run. Caching tool *results* would save almost nothing, because a SQLite query takes about 1 ms. The cost is the model turn around the call. So the harness now reads the schema when the run starts (tables, columns, types, PK/FK, row counts, 2 sample rows each) and puts it at the end of the system prompt:
- **Fewer turns:** the model starts knowing the schema. 4.4 → 2.2 turns, 35% cheaper, 22% faster.
- **Cheap tokens:** it sits in the unchanging start of every request, so Azure's automatic prompt cache bills it at the cached rate (10× cheaper). Cached share of input went from 30% to 79%.
- **Read once per file:** it's cached in memory, keyed by path, size and mtime, so eval and `agent chat` read it once.
- **Size limit:** if the summary is larger than `max_schema_chars` (12k), the agent falls back to the discovery tools. `AGENT_SCHEMA_IN_PROMPT=false` turns it off.

**What went wrong first:** the schema *without* sample rows (v5) broke Q10 (1/3). The model saw `Track.UnitPrice` and `InvoiceLine.UnitPrice` side by side, decided one was "the cost", and wrote a single query without looking at the data. In v4, `describe_table`'s sample rows (both prices 0.99) were what led it to "this is a price, there's no cost data". Adding 2 sample rows per table brought that evidence back (v6: Q10 3/3 in the full eval, 4/5 in a separate Q10-only run). **Trade-off I accepted:** across 8 Q10 runs with the schema preloaded, 1 still made the proxy mistake, against 0 of 8 in discovery mode. For the most careful (and slower) mode, use `AGENT_SCHEMA_IN_PROMPT=false`.

**1. Trim before the model sees it.** SQLite returns up to `max_rows` (200) rows, but the model sees at most `rows_to_model` (50). When rows are hidden, a note says "Showing 50 of more than 200 rows. Aggregate, filter or add LIMIT". Wide rows are cut further to fit `max_tool_result_chars` (6,000). The **full** result still goes to the trace, along with the trimmed text the model actually saw, so you can always tell the two apart.

**2. Compact old turns.** When the prompt passes `compact_after_tokens` (16k), older tool results are rewritten in place as one-line summaries. The newest 4 stay verbatim.

| Kept | Dropped |
|---|---|
| system prompt and question (never touched) | sample rows from `describe_table` |
| every SQL query that was tried (assistant messages are never touched) | rows beyond the first 3 of old results |
| for old schema lookups: columns, types, PKs, FKs, row counts | |
| for old queries: row count, columns, first 3 rows | |
| every error message, so the model doesn't repeat a mistake | |

Compaction is deterministic code with no extra model call, so it's free, instant and predictable.
- **Rejected: LLM summarization.** It costs a call, adds latency, and can drop the one number that matters.
- **Trade-off:** rewriting old messages breaks the provider's prompt-cache prefix once. That's why it only triggers past a threshold rather than every turn.

### Observability (R6)
One JSONL file per run, flushed after every line, so a killed run still leaves a readable trace.

| Event | Contents |
|---|---|
| `run_start` | question, DB, full config |
| `llm_request` | only the messages *added since the last request*, which is a faithful but non-repetitive record |
| `llm_response` | content, tool calls, input / cached / output / reasoning tokens, latency, cost, retries |
| `tool_call`, `tool_result` | arguments, full result, the trimmed version the model saw, latency |
| `compaction` | how much was compacted |
| `limit_reached` | which limit |
| `run_end` | stop reason, final output, totals |

`agent trace <file>` renders it with highlighted SQL, small result tables, errors in red, and per-turn tokens and cost.

### Generic by design
The system prompt has no knowledge of Chinook. It describes a careful analyst's process: check joins, look for ties, duplicates and NULLs, state interpretations, and say `answerable=false` rather than substitute different data. [scripts/make_demo_db.py](scripts/make_demo_db.py) builds an unrelated SaaS database with a table name containing a space, no declared foreign keys, NULLs, a view, text dates and money in cents. The tests also cover views, empty tables and quoted names.

### Stretch goals done
- **Parallel tool calls:** all tool calls in a turn are executed, and the tests cover it.
- **Streaming progress:** each tool call and result, compaction and limit is printed to stderr while the agent works. stdout stays clean JSON.
- **Multi-turn follow-ups:** `agent chat` reuses the previous transcript, e.g. "now break that down by year".
- **Self-verification:** the `sql_used` provenance check described above.

### Why only the four required tools
I considered `get_distinct_values(column)`, `count_rows(table)`, `explain_query(sql)` and `search_columns(keyword)`. Each one is just a `SELECT` the model can already write with `run_sql`, so it would add choices without adding capability. The traces showed that extra choices cost turns: when the model had more ways to explore, it explored more. Fewer, general tools plus good error messages ("Did you mean: Track?") worked better. Adding a tool takes about 5 lines (a schema in `SCHEMAS` plus a `_name` method); validation, strict mode, errors and tracing come for free.


---

## Evaluation (R7)

### Ground truth
I explored the data by hand before writing any agent code. [eval/ground_truth.py](eval/ground_truth.py) computes every answer **with SQL at eval time** rather than hardcoding numbers, and comments explain the traps found in each question:

| Q | Truth | What makes it tricky |
|---|---|---|
| Q1 | 3,503 | none |
| Q2 | Rock (1,297) | none |
| Q3 | Iron Maiden 21, Led Zeppelin 14, Deep Purple 11, Metallica 10, U2 10 | Metallica and U2 tie at 4th/5th; 6th has 6, so the set is clear |
| Q4 | USA, Canada, France, Brazil, Germany | billing country vs customer country: I checked, there are 0 mismatches. `Invoice.Total` = sum of invoice lines |
| Q5 | Jane Peacock ($833.04) | the agent must find `Customer.SupportRepId` → `Employee` |
| Q6 | 2010: $481.45, +$31.99 (+7.12%) vs 2009 | two-part answer; revenue is nearly flat across years |
| Q7 | 5 media types (video ≈ 39 min, audio ≈ 4.3–4.7) | ms → minutes |
| Q8 | **41 tracks tie** at 5 playlists | playlist names are duplicated ("Music" is IDs 1 and 8); by distinct name the max is 4, same 41 tracks |
| Q9 | Helena Holý ($49.62) | "best" is ambiguous: by spend she's clearly first; by order count it's a 58-way tie; by tracks bought a 3-way tie |
| Q10 | **Unanswerable** | there's no cost data; the only money column is the sale price |

### How grading works
Grading is deterministic, with no LLM judge. It checks the structured output against the truth:

- **Exact facts (Q1, Q2, Q5):** the right value or name appears. For Q5, if all agents are ranked, the winner must be named first.
- **Sets (Q3, Q4):** all 5 names must appear. Q4 also checks rank order.
- **Numbers with tolerance (Q6, Q7):** Q6 needs the year, the previous year, and either the % change (±0.15 points) or the absolute change (±$0.05). Q7 needs all 5 averages within ±0.06 min.
- **Questions without one exact answer:**
  - **Q8 (ties):** passes only if the answer gives the max count (5, or 4 if counting distinct names), **acknowledges the tie**, and names at least one track from the tied set. Naming one track as "the" answer fails.
  - **Q9 (ambiguous):** passes if it names Helena Holý **and** states the interpretation, either in `assumptions` or by mentioning spend or revenue.
  - **Q10 (unanswerable):** passes if `answerable=false`, or if it explains the missing cost data without stating a margin %. Inventing a margin fails.
- A run that never submitted (`status: failed`) always fails.
- As a separate signal, every run also re-executes its `sql_used` to confirm the cited queries actually run.

The graders have their own tests ([tests/test_graders.py](tests/test_graders.py)), with realistic right and wrong answers for every question. A grader that wrongly passes bad answers would make the whole report meaningless.

**Rejected: an LLM judge.** It would handle unusual phrasing better, but it adds cost, randomness, and a second model whose mistakes I would then also need to evaluate. Deterministic graders can be too strict about phrasing (e.g. "4 min 26 s" instead of 4.43); I checked failing runs by hand for this.

### Results

Final run: [eval/results/results-20261005-152007.md](eval/results/results-20261005-152007.md) (every answer, grade and trace link). Model `gpt-5-mini`, effort `medium`, schema preloaded with sample rows (the defaults), 3 runs per question.

**30/30 passed (100%)** · avg **$0.0028**/run · **2.2** turns · **14.0s** · total $0.08

| Q | Pass | Avg cost | Avg turns | Avg latency |
|---|---|---|---|---|
| Q1 tracks | 3/3 | $0.0006 | 2 | 4.8s |
| Q2 genre | 3/3 | $0.0017 | 2 | 11.9s |
| Q3 artists | 3/3 | $0.0053 | 3 | 26.0s |
| Q4 countries | 3/3 | $0.0024 | 2 | 12.1s |
| Q5 support agent | 3/3 | $0.0021 | 2 | 11.9s |
| Q6 best year | 3/3 | $0.0027 | 2 | 12.5s |
| Q7 media type | 3/3 | $0.0019 | 2 | 10.1s |
| Q8 playlists | 3/3 | $0.0027 | 2 | 11.7s |
| Q9 best customer | 3/3 | $0.0031 | 2.7 | 15.7s |
| Q10 profit margin | 3/3 | $0.0053 | 2.7 | 23.2s |

**How it got here.** Every eval run is kept in `eval/results/`:

| Run | Change | Pass | $/run | Turns | Latency | Tool errors | Limit hits |
|---|---|---|---|---|---|---|---|
| v1 | baseline (effort low) | 30/30 | $0.0033 | 6.1 | 16.3s | 11 | 0 |
| v2 | strict schemas, parallel describes, email check | 29/30 | $0.0037 | 6.4 | 17.2s | 19 | 2 |
| v3 | prompt fixes, turn reminder | 29/30 | $0.0028 | 4.9 | 12.7s | **0** | 0 |
| v4 | v3 at effort medium | 30/30 | $0.0043 | 4.4 | 18.0s | 0 | 0 |
| v5 | schema preloaded in the system prompt | 28/30 | $0.0024 | 2.1 | 11.8s | 0 | 0 |
| v6 | v5 + 2 sample rows per table (**final**) | **30/30** | **$0.0028** | **2.2** | **14.0s** | 2 | 0 |

Read 100% with care:
- These are 10 questions I studied closely, on the database I developed against. It's a regression suite, not a measure of how the agent does on new questions.
- v1 also scored 30/30, but its traces showed real problems the pass rate hid: wasted turns, 11 tool errors, SQL pasted into answers, and an email address included that nobody asked for.

Total API spend across all development and evals: $0.70 (summed from every trace).

---

## Failure modes I observed, and what I changed

**While building, found by tests:**
1. **The final answer overshot the budget.** The final forced call wasn't budgeted: a 5,000-token run ended at 6,600. Fix: project the next call's size from the last real call, and cap the final call's output to the remaining budget.
2. **`ATTACH` gets past a read-only connection.** It's not stopped by `mode=ro` or `query_only`. Fix: the authorizer as the main guard (see above).
3. **The trace path printed by `rich` wrapped across two lines**, so copy-pasting it broke. Fix: print it as plain text.

**Before the first real call:**
4. **Every call failed with 404.** The Azure portal shows the endpoint as `https://<res>.services.ai.azure.com/openai/v1/responses`, but the SDK wants only the base URL and appends its own path. Fix: [llm.py](agent/llm.py) keeps only the scheme and host, so either form works. The harness handled the failure correctly (valid JSON, `stop_reason: llm_error`, a trace), which is also how I found it.

**From real traces** (all found by reading traces with `agent trace` and the analysis scripts, not from the pass rate):

| # | What I saw | Why | Change |
|---|---|---|---|
| 5 | 6/30 runs: `submit_answer` missing `answerable` → rejected → resubmitted | non-strict tool calling lets the model leave out fields | `strict: true` schemas. **0 since** |
| 6 | 4/30 runs: `sql_used` contained `"-- list tables ..."` pseudo-SQL | the model treats schema lookups as "queries used" | the provenance check caught them; the tool description now says list/describe calls aren't SQL. **0 since** |
| 7 | Q9 answers included the customer's email | the model selected `Email` and reported everything it had | a prompt rule **didn't work**, so the answer is checked in code (`personal_data` error) |
| 8 | SQL and confidence repeated inside `answer`; all 41 tied tracks listed in Q8 | the answer field had no style rules | prompt: no SQL in the answer; for long lists give the count plus examples |
| 9 | Every run spent 3 turns before any SQL: list → describe → describe | one describe per turn | prompt: describe all relevant tables in one turn |
| 10 | **My fix for #9 caused a regression:** 18 `unknown_table` errors (`tracks`, `genres`, `media_types`) | the model now called list_tables *and* describe_table in the same turn, guessing names before seeing the list | prompt: list_tables alone first, then describe using exact names. **0 since** |
| 11 | Q6/Q7 hit the 15-turn limit after finding the answer at turn 4 ([trace](traces/samples/05-LIMIT-q6-reverification-loop-v2.jsonl)) | my "sanity-check results" instruction plus low effort meant endless re-verification (the same yearly totals, one year per query) | prompt: at most 1–2 targeted checks, never re-run a result you already have; harness: turn reminder at 60%. The forced final answer still produced correct output in both runs |
| 12 | Q10 at effort low: "0% margin for every genre" with `answerable: true` ([trace](traces/samples/03-FAILED-q10-proxy-cost-low-effort.jsonl)) | the sale price was used as a stand-in for cost | a general rule in the `answerable` description helped but didn't fix it. **Effort medium did (5/5)** → default changed |
| 13 | Forced final answer at a limit set `answerable: false` "because I could not run the query" ([trace](traces/samples/04-LIMIT-max-turns-forced-answer.jsonl) shows the behaviour before the fix) | the model mixed up "didn't finish" with "the data can't answer this" | the final-answer prompt says running out of turns doesn't make a question unanswerable; status/stop_reason already record that |

---

## Sample traces

In [traces/samples/](traces/samples/). View any with `uv run agent trace <file>`.

| File | What it shows |
|---|---|
| `01-q8-tie-handled.jsonl` | Finds the 41-way tie and reports the count plus examples, not a single "top" track |
| `02-q10-unanswerable.jsonl` | Checks the schema for cost data, finds none, `answerable: false` |
| `03-FAILED-q10-proxy-cost-low-effort.jsonl` | **Failure:** at effort low, uses the sale price as cost → "0% margin" |
| `04-LIMIT-max-turns-forced-answer.jsonl` | **Limit:** `--max-turns 3`, the forced final answer, `status: partial` (recorded before fix #13) |
| `05-LIMIT-q6-reverification-loop-v2.jsonl` | **Limit:** the re-verification loop (#11) hits 15 turns; the forced final answer is still correct |
| `06-other-database-saas.jsonl` | Unseen database: uses a window function for "current plan", excludes cancellations, converts cents, states the per-seat assumption |

The full trace set for the final eval is in `traces/eval-20261005-152007/`. Traces from every earlier eval run are committed too (`traces/eval-*/`), so every trace link in `eval/results/*.md` works.

## Known limitations
- **Q10-style "proxy" mistakes still happen at effort low**, and nothing in code detects them. The prompt and the `answerable` description reduce them; the only reliable fix I found was more reasoning.
- **Budgets are enforced before each call using a projection.** The final call's output is hard-capped, but its input size is an estimate, so a run can end slightly over the token or cost budget (by the estimation error on one prompt).
- **Graders match text.** A correct answer in an unusual format (e.g. durations as "4:26") would be marked wrong.
- **The query timeout doesn't cover expensive single SQLite functions** (e.g. `randomblob(1e9)` allocates before the progress handler fires). The memory risk is small but not zero.
- In `agent chat`, a follow-up can't cite SQL from an *earlier* question in `sql_used` without re-running it (the provenance check works per question).
- `list_tables` counts rows in every table, which is slow on very large databases. It's guarded by the query timeout, and the count shows `null` if it times out.
---
