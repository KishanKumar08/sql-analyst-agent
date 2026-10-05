"""
Keep the model's context small and predictable.

Two mechanisms:

1. `render_tool_result` — shrink each tool result *before* the model sees it:
   only the first `rows_to_model` rows, a clear note when rows were hidden,
   and a hard character cap. The full result still goes to the trace.

2. `compact` — when the prompt grows past `compact_after_tokens`, rewrite
   older tool results into one-line summaries. Deterministic (no extra model
   call), so it is free, fast and easy to reason about.

What compaction always preserves:
  - the system prompt and the user's question (never touched)
  - every assistant message, including the SQL it ran (tool-call arguments
    are small and are the record of what was tried)
  - the newest `keep_recent_tool_results` tool results, verbatim
  - for older results: schema facts (columns, keys), the shape and first
    rows of each query result, and every error message
What it drops: sample rows from describe_table and the long tail of rows.
"""
import json

from .llm import to_json

COMPACTED_MARK = "[compacted]"


def render_tool_result(name: str, result: dict, cfg) -> str:
    if name == "run_sql" and result.get("ok"):
        rows = result["rows"]
        shown = rows[: cfg.rows_to_model]
        out = {"ok": True, "columns": result["columns"], "rows": shown,
               "row_count": result["row_count"]}
        if result.get("truncated") or len(shown) < len(rows):
            more = "more than " if result.get("truncated") else ""
            out["note"] = (f"Showing {len(shown)} of {more}{result['row_count']} rows. "
                           "Aggregate, filter or add LIMIT/ORDER BY if you need specific rows.")
        text = to_json(out)
        # If wide rows still blow the char budget, drop rows until it fits.
        while len(text) > cfg.max_tool_result_chars and len(shown) > 1:
            shown = shown[: max(1, len(shown) // 2)]
            out["rows"] = shown
            out["note"] = (f"Showing {len(shown)} of {result['row_count']} rows (rows are wide). "
                           "Select fewer columns or aggregate.")
            text = to_json(out)
        return text[: cfg.max_tool_result_chars]
    text = to_json(result)
    if len(text) > cfg.max_tool_result_chars:
        text = text[: cfg.max_tool_result_chars] + '..." [output truncated]'
    return text


def estimate_tokens(messages: list[dict]) -> int:
    """Rough count (~4 chars/token). Only used to decide when to compact."""
    return sum(len(to_json(m)) for m in messages) // 4


def _summarize(name: str, content: str) -> str:
    try:
        r = json.loads(content)
    except json.JSONDecodeError:
        return f"{COMPACTED_MARK} {name}: {content[:200]}"
    if not r.get("ok", False):
        return f"{COMPACTED_MARK} {name} error ({r.get('error_type')}): {r.get('message', '')[:300]}"
    if name == "run_sql":
        return (f"{COMPACTED_MARK} run_sql ok: {r.get('row_count')} rows, columns {r.get('columns')}; "
                f"first rows: {to_json(r.get('rows', [])[:3])}")
    if name == "describe_table":
        cols = ", ".join(f"{c['name']} {c['type']}{' PK' if c['primary_key'] else ''}"
                         for c in r.get("columns", []))
        fks = ", ".join(f"{f['column']}->{f['references']}" for f in r.get("foreign_keys", []))
        return (f"{COMPACTED_MARK} describe_table {r.get('table')} ({r.get('row_count')} rows): "
                f"{cols}" + (f"; FKs: {fks}" if fks else ""))
    return f"{COMPACTED_MARK} {name}: {content[:500]}"


def compact(messages: list[dict], cfg) -> int:
    """Summarize old tool results in place. Returns how many were compacted."""
    names = {tc["id"]: tc["function"]["name"]
             for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []}
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    old = tool_idx[: max(0, len(tool_idx) - cfg.keep_recent_tool_results)]
    n = 0
    for i in old:
        m = messages[i]
        if m["content"].startswith(COMPACTED_MARK):
            continue
        m["content"] = _summarize(names.get(m["tool_call_id"], "tool"), m["content"])
        n += 1
    return n
