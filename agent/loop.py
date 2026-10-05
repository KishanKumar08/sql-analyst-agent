"""
The agent loop.

question ──► model call ──► tool calls ──► tool results ──► [model call] ...
                    │                                               │
                    └── budgets checked before every call ──────────┘
                                        │ limit hit
                                        ▼
                    one forced "submit_answer now" call (from the reserve)
                                        │ that fails too
                                        ▼
                    deterministic fallback built from the last good query

Whatever happens, `run()` returns a valid answer dict.
"""
import json
import time
from dataclasses import asdict, dataclass, field

from . import context, prompts
from .config import Config
from .db import Limits, SafeDB
from .llm import LLMError, LLMResponse, to_json
from .tools import SCHEMAS, ToolBox, normalize_sql, openai_tools, validate

_SCHEMA_CACHE: dict[tuple, str] = {}

LIMIT_TEXT = {
    "max_turns": "maximum number of turns",
    "max_tokens": "token budget",
    "max_cost": "cost budget",
    "max_wall_time": "wall-clock time limit",
}


@dataclass
class Stats:
    turns: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    compactions: int = 0
    elapsed_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class RunResult:
    output: dict
    stats: Stats
    trace_path: str | None
    messages: list = field(default_factory=list)


class Agent:
    def __init__(self, cfg: Config, llm, tracer, progress=None):
        self.cfg = cfg
        self.llm = llm
        self.tracer = tracer
        self.progress = progress or (lambda *_: None)

    def run(self, question: str, db_path: str, history: list | None = None) -> RunResult:
        cfg = self.cfg
        self.stats = Stats()
        self.t0 = time.monotonic()
        self.deadline = self.t0 + cfg.max_wall_s
        self.tools = None
        self.messages = []
        self._sent = 0
        self._last_input = self._last_output = 0
        self._msgs_at_last_call = 0
        output = None
        stop_reason = None
        detail = ""

        self.tracer.event("run_start", question=question, db=str(db_path),
                          config=asdict(cfg), model=cfg.deployment)
        try:
            db = SafeDB(db_path, Limits(cfg.query_timeout_s, cfg.max_rows, cfg.max_cell_chars))
        except (FileNotFoundError, ValueError) as e:
            output = self._fallback("db_error", str(e))
            return self._finish(output, "db_error", str(e))

        self.tools = ToolBox(db, question)
        # `history` lets a follow-up question reuse an earlier conversation.
        if not history:
            schema = self._load_schema(db) if cfg.schema_in_prompt else None
            history = [{"role": "system", "content": prompts.system_prompt(cfg, schema)}]
        self.messages = history
        self.messages.append({"role": "user", "content": question})
        try:
            output, stop_reason = self._loop()
            if output is None:
                detail = f"{LIMIT_TEXT.get(stop_reason, stop_reason)} reached"
                self.tracer.event("limit_reached", reason=stop_reason, stats=asdict(self._stats()))
                self.progress("limit", detail)
                output = self._finalize(stop_reason)
        except LLMError as e:
            stop_reason, detail = "llm_error", str(e)
        except KeyboardInterrupt:
            stop_reason, detail = "interrupted", "interrupted by user"
        except Exception as e:
            stop_reason, detail = "internal_error", f"{type(e).__name__}: {e}"
        finally:
            db.close()

        if output is None:
            output = self._fallback(stop_reason, detail)
        return self._finish(output, stop_reason, detail)

    def _load_schema(self, db: SafeDB) -> str | None:
        """
        Schema summary for the system prompt, or None to fall back to the
        discovery tools. Cached per database file (keyed by path, size and
        mtime), so eval runs and chat follow-ups read it once.
        """
        st = db.path.stat()
        key = (str(db.path.resolve()), st.st_size, st.st_mtime_ns, self.cfg.schema_sample_rows)
        if key not in _SCHEMA_CACHE:
            try:
                _SCHEMA_CACHE[key] = db.schema_summary(self.cfg.schema_sample_rows)
            except Exception as e:  # never fail a run over an optimisation
                self.tracer.event("schema_preload", used=False, reason=f"{type(e).__name__}: {e}")
                return None
        schema = _SCHEMA_CACHE[key]
        if len(schema) > self.cfg.max_schema_chars:
            self.tracer.event("schema_preload", used=False, chars=len(schema),
                              reason=f"larger than max_schema_chars ({self.cfg.max_schema_chars})")
            return None
        self.tracer.event("schema_preload", used=True, chars=len(schema), schema=schema)
        tables = sum(1 for line in schema.splitlines() if line.startswith("- "))
        self.progress("schema", f"schema preloaded ({tables} tables, {len(schema)} chars)")
        return schema

    # ------------------------------------------------------------------
    def _loop(self):
        """Explore until the model submits or a budget (minus reserve) runs out.
        Returns (answer, "submitted") or (None, limit_name)."""
        tool_defs = openai_tools()
        reminder_at = max(3, int(self.cfg.max_turns * 0.6))
        reminded = False
        while True:
            limit = self._limit_hit(reserve=True)
            if limit:
                return None, limit

            # One-time reminder: in testing the model sometimes found the answer
            # early and then kept re-verifying it until the turn limit.
            if not reminded and self.stats.turns >= reminder_at:
                reminded = True
                self.messages.append({"role": "user", "content": prompts.PROGRESS_PROMPT.format(
                    used=self.stats.turns, limit=self.cfg.max_turns)})
                self.progress("nudge", f"turn reminder ({self.stats.turns}/{self.cfg.max_turns})")

            resp = self._call_model(tool_defs, "auto")
            if not resp.tool_calls:
                why = "hit max_completion_tokens" if resp.finish_reason == "length" else "no tool call"
                self.progress("nudge", why)
                self.messages.append({"role": "user", "content": prompts.NUDGE_PROMPT})
                continue

            for tc in resp.tool_calls:
                result = self._run_tool(tc)
                if tc.name == "submit_answer" and result.get("ok"):
                    answer = json.loads(tc.arguments)
                    # Answer every remaining tool call id so the transcript
                    # stays valid if it is reused for a follow-up question.
                    for other in resp.tool_calls:
                        if not any(m.get("tool_call_id") == other.id for m in self.messages):
                            self.messages.append({"role": "tool", "tool_call_id": other.id,
                                                  "content": '{"ok":false,"message":"skipped: run ended"}'})
                    return self._shape(answer), "submitted"

    def _finalize(self, reason: str) -> dict | None:
        """One forced submit_answer call, paid for by the reserve. Its output
        is capped to whatever budget is left, so it cannot overshoot."""
        self.messages.append({"role": "user",
                              "content": prompts.FINALIZE_PROMPT.format(reason=LIMIT_TEXT[reason])})
        out_cap = self._output_room()
        if out_cap < 300 or self.stats.turns >= self.cfg.max_turns or self.deadline - time.monotonic() < 5:
            self.tracer.event("finalize_skipped", output_room=out_cap)
            return None  # not enough budget left for a useful final call
        try:
            resp = self._call_model(openai_tools(), {"type": "function", "function": {"name": "submit_answer"}},
                                    max_output=out_cap)
        except LLMError as e:
            self.tracer.event("finalize_failed", error=str(e))
            return None
        for tc in resp.tool_calls:  # keep the transcript valid for follow-ups
            self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": '{"ok":true}'})
        call = next((tc for tc in resp.tool_calls if tc.name == "submit_answer"), None)
        if not call:
            return None
        try:
            answer = json.loads(call.arguments)
        except json.JSONDecodeError:
            return None
        if validate("submit_answer", answer):
            return None
        self.tracer.event("tool_call", turn=self.stats.turns, tool="submit_answer",
                          arguments=answer, forced=True)

        # Keep only queries that really ran; we can't send it back to fix them.
        ran = [q for q in answer["sql_used"] if normalize_sql(q) in self.tools.successful_sql]
        out = self._shape(answer)
        out["sql_used"] = ran
        note = f"The run stopped early ({LIMIT_TEXT[reason]} reached); this answer is based on a partial investigation."
        out["assumptions"] = [note] + out["assumptions"]
        if out["confidence"] == "high":
            out["confidence"] = "medium"
        out["status"] = "partial"
        return out

    def _fallback(self, reason: str, detail: str) -> dict:
        """No model involved: build the best answer we can from what ran."""
        last = list(self.tools.successful_sql.values())[-1] if self.tools and self.tools.successful_sql else None
        msg = f"I could not complete the analysis ({detail or LIMIT_TEXT.get(reason, reason)})."
        if last:
            msg += (f" The last successful query returned {last['row_count']} row(s) with columns "
                    f"{last['columns']}; first rows: {json.dumps(last['rows'][:3], default=str, ensure_ascii=False)}.")
        else:
            msg += " No query completed successfully."
        return {
            "answer": msg,
            "sql_used": [last["sql"]] if last else [],
            "assumptions": ["This is not a verified answer; the run ended before the agent submitted one."],
            "confidence": "low",
            "answerable": False,
            "status": "failed",
        }

    # ------------------------------------------------------------------
    def _call_model(self, tool_defs, tool_choice, max_output: int | None = None) -> LLMResponse:
        if self.stats.input_tokens and context.estimate_tokens(self.messages) > self.cfg.compact_after_tokens:
            before = context.estimate_tokens(self.messages)
            n = context.compact(self.messages, self.cfg)
            if n:
                self.stats.compactions += 1
                after = context.estimate_tokens(self.messages)
                self.tracer.event("compaction", results_compacted=n, est_tokens_before=before,
                                  est_tokens_after=after)
                self.progress("compact", f"{n} old results summarized (~{before} → ~{after} tokens)")

        self.stats.turns += 1
        turn = self.stats.turns
        self.tracer.event("llm_request", turn=turn, n_messages=len(self.messages),
                          new_messages=self.messages[self._sent:],
                          tools=[t["function"]["name"] for t in tool_defs] if turn == 1 else None,
                          tool_choice=tool_choice,
                          est_prompt_tokens=context.estimate_tokens(self.messages))
        self._sent = len(self.messages)

        resp = self.llm.complete(self.messages, tool_defs, tool_choice, deadline=self.deadline,
                                 max_output=max_output)

        u = resp.usage
        self._last_input, self._last_output = u.input_tokens, u.output_tokens
        self._msgs_at_last_call = len(self.messages)
        cost = self.cfg.cost(u.input_tokens, u.cached_tokens, u.output_tokens)
        s = self.stats
        s.input_tokens += u.input_tokens
        s.cached_tokens += u.cached_tokens
        s.output_tokens += u.output_tokens
        s.reasoning_tokens += u.reasoning_tokens
        s.cost_usd += cost
        self.tracer.event("llm_response", turn=turn, latency_ms=resp.latency_ms,
                          finish_reason=resp.finish_reason, content=resp.content,
                          tool_calls=[{"id": t.id, "name": t.name, "arguments": t.arguments}
                                      for t in resp.tool_calls],
                          usage=asdict(u), cost_usd=round(cost, 6),
                          cumulative_cost_usd=round(s.cost_usd, 6), retries=resp.retries)
        self.progress("turn", f"turn {turn}: {u.input_tokens} in / {u.output_tokens} out tokens, "
                              f"{resp.latency_ms / 1000:.1f}s, ${s.cost_usd:.4f} so far")
        self.messages.append(resp.as_message())
        self._sent = len(self.messages)
        return resp

    def _run_tool(self, tc) -> dict:
        turn = self.stats.turns
        self.tracer.event("tool_call", turn=turn, tool=tc.name, call_id=tc.id, arguments=tc.arguments)
        self.progress("tool", _describe_call(tc))
        started = time.monotonic()
        result = self.tools.call(tc.name, tc.arguments)
        shown = context.render_tool_result(tc.name, result, self.cfg)
        self.stats.tool_calls += 1
        if not result.get("ok"):
            self.stats.tool_errors += 1
            self.progress("error", f"{result.get('error_type')}: {result.get('message', '')[:150]}")
        elif tc.name == "run_sql":
            self.progress("result", f"{result['row_count']} row(s)" + (" (truncated)" if result["truncated"] else ""))
        self.tracer.event("tool_result", turn=turn, tool=tc.name, call_id=tc.id, ok=bool(result.get("ok")),
                          latency_ms=round((time.monotonic() - started) * 1000, 1),
                          # Only stored when the model saw a trimmed version.
                          result=result, shown_to_model=shown if shown != to_json(result) else None)
        self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": shown})
        return result

    def _next_input(self) -> int:
        """Projected prompt size of the next call: the last real prompt size
        plus an estimate for messages added since. Better than estimating the
        whole transcript from characters, which misses tool schemas etc."""
        if not self._last_input:
            return context.estimate_tokens(self.messages)
        return self._last_input + context.estimate_tokens(self.messages[self._msgs_at_last_call:])

    def _limit_hit(self, reserve: bool) -> str | None:
        """Checked before every exploration call. Stops when the *projected*
        spend after the next call would eat into the finalize reserve."""
        cfg, s = self.cfg, self.stats
        frac = (1 - cfg.finalize_reserve) if reserve else 1.0
        nxt_in = self._next_input()
        nxt_out = self._last_output or 500
        if s.turns >= (cfg.max_turns - 1 if reserve else cfg.max_turns):
            return "max_turns"
        if s.total_tokens + nxt_in + nxt_out >= cfg.max_tokens * frac:
            return "max_tokens"
        if s.cost_usd + cfg.cost(nxt_in, 0, nxt_out) >= cfg.max_cost_usd * frac:
            return "max_cost"
        if time.monotonic() - self.t0 >= cfg.max_wall_s * frac:
            return "max_wall_time"
        return None

    def _output_room(self) -> int:
        """Output tokens the final call may use without breaking the token or
        cost budget, given its projected input."""
        cfg, s = self.cfg, self.stats
        nxt_in = self._next_input()
        by_tokens = cfg.max_tokens - s.total_tokens - nxt_in
        dollars_left = cfg.max_cost_usd - s.cost_usd - cfg.cost(nxt_in, 0, 0)
        by_cost = int(dollars_left / cfg.price_output * 1_000_000)
        return max(0, min(by_tokens, by_cost, cfg.max_completion_tokens))

    def _stats(self) -> Stats:
        self.stats.elapsed_s = round(time.monotonic() - self.t0, 2)
        return self.stats

    @staticmethod
    def _shape(answer: dict) -> dict:
        return {k: answer[k] for k in SCHEMAS["submit_answer"]["parameters"]["required"]} | {"status": "complete"}

    def _finish(self, output: dict, stop_reason: str, detail: str) -> RunResult:
        output["stop_reason"] = stop_reason
        if detail and stop_reason != "submitted":
            output["stop_detail"] = detail
        stats = self._stats()
        self.tracer.event("run_end", stop_reason=stop_reason, detail=detail, output=output, stats=asdict(stats))
        self.tracer.close()
        return RunResult(output=output, stats=stats,
                         trace_path=str(self.tracer.path) if self.tracer.path else None,
                         messages=self.messages)


def _describe_call(tc) -> str:
    try:
        args = json.loads(tc.arguments or "{}")
    except json.JSONDecodeError:
        return f"{tc.name}(<invalid json>)"
    if tc.name == "run_sql":
        return "run_sql: " + " ".join(str(args.get("sql", "")).split())[:160]
    if tc.name == "describe_table":
        return f"describe_table: {args.get('table_name')}"
    if tc.name == "submit_answer":
        return "submit_answer"
    return tc.name
