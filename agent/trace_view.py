"""
Human-readable view of a JSONL trace:  agent trace traces/run-XXXX.jsonl
"""
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()

def _rows_table(columns, rows, limit):
    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    for c in columns:
        t.add_column(str(c))
    for r in rows[:limit]:
        t.add_row(*[str(v) for v in r])
    return t


def show(path: str, full: bool = False) -> int:
    p = Path(path)
    if not p.exists():
        console.print(f"[red]no such trace: {path}[/]")
        return 1
    events = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            console.print(f"[red]line {i}: not valid JSON (trace may be truncated)[/]")

    for e in events:
        t = e["type"]
        stamp = f"[dim]{e['t_ms'] / 1000:6.1f}s[/]"
        if t == "run_start":
            c = e["config"]
            console.print(Panel(
                f"[bold]{e['question']}[/]\n"
                f"db: {e['db']}   model: {e['model']} (effort {c['reasoning_effort']})\n"
                f"limits: {c['max_turns']} turns · {c['max_tokens']} tokens · ${c['max_cost_usd']} · "
                f"{c['max_wall_s']}s wall · {c['query_timeout_s']}s/query · {c['max_rows']} rows",
                title=f"run {e['run_id']}", expand=False))
        elif t == "llm_request":
            console.print(f"\n{stamp} [bold blue]── turn {e['turn']}[/] "
                          f"[dim]({e['n_messages']} messages, ~{e['est_prompt_tokens']} tokens"
                          + (f", forced: {e['tool_choice']['function']['name']}" if isinstance(e['tool_choice'], dict) else "")
                          + ")[/]")
            for m in e.get("new_messages") or []:
                if m["role"] == "user" and e["turn"] > 1:
                    console.print(f"   [yellow]harness → model:[/] {m['content'][:300]}")
        elif t == "llm_response":
            u = e["usage"]
            console.print(f"{stamp} [dim]model: {u['input_tokens']} in ({u['cached_tokens']} cached) / "
                          f"{u['output_tokens']} out ({u['reasoning_tokens']} reasoning) · "
                          f"{e['latency_ms'] / 1000:.1f}s · ${e['cost_usd']:.5f} "
                          f"(total ${e['cumulative_cost_usd']:.4f}) · {e['finish_reason']}[/]")
            for r in e.get("retries") or []:
                console.print(f"   [yellow]retry {r['attempt']}: {r['error']}, waited {r['wait_s']}s[/]")
            if e.get("content"):
                txt = e["content"] if full else e["content"][:400]
                console.print(f"   [italic]{txt}[/]")
        elif t == "tool_call":
            args = e["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    console.print(f"   [cyan]▶ {e['tool']}[/] [red]{args}[/]")
                    continue
            forced = " [yellow](forced)[/]" if e.get("forced") else ""
            if e["tool"] == "run_sql":
                console.print(f"   [cyan]▶ run_sql[/]{forced}")
                console.print(Syntax(args.get("sql", ""), "sql", theme="ansi_dark", padding=(0, 5), word_wrap=True))
            elif e["tool"] == "submit_answer":
                console.print(f"   [cyan]▶ submit_answer[/]{forced} [dim]{args.get('confidence')}, "
                              f"answerable={args.get('answerable')}[/]")
            else:
                console.print(f"   [cyan]▶ {e['tool']}[/] {json.dumps(args, ensure_ascii=False)}")
        elif t == "tool_result":
            r = e["result"]
            if not e["ok"]:
                console.print(f"   [red]✗ {r.get('error_type')}: {r.get('message')}[/]"
                              + (f" [dim]hint: {r['hint']}[/]" if r.get("hint") else ""))
            elif e["tool"] == "run_sql":
                console.print(f"   [green]✓ {r['row_count']} row(s){' (truncated)' if r['truncated'] else ''}"
                              f" in {r['elapsed_ms']}ms[/]")
                console.print(_rows_table(r["columns"], r["rows"], 50 if full else 5), style="dim")
            elif e["tool"] == "list_tables":
                console.print("   [green]✓[/] " + ", ".join(f"{x['name']}({x['rows']})" for x in r["tables"]))
            elif e["tool"] == "describe_table":
                cols = ", ".join(c["name"] for c in r["columns"])
                console.print(f"   [green]✓ {r['table']}[/] [dim]{cols}[/]")
            else:
                console.print(f"   [green]✓ {json.dumps(r)[:200]}[/]")
        elif t == "schema_preload":
            if e["used"]:
                n = sum(1 for line in e["schema"].splitlines() if line.startswith("- "))
                console.print(f"{stamp} [magenta]schema preloaded into the system prompt: {n} tables, {e['chars']} chars[/]")
                if full:
                    console.print(e["schema"], style="dim")
            else:
                console.print(f"{stamp} [magenta]schema not preloaded: {e['reason']}[/]")
        elif t == "compaction":
            console.print(f"{stamp} [magenta]⇣ compaction: {e['results_compacted']} old results summarized "
                          f"(~{e['est_tokens_before']} → ~{e['est_tokens_after']} tokens)[/]")
        elif t == "limit_reached":
            console.print(f"{stamp} [bold yellow]⚠ limit reached: {e['reason']}[/]")
        elif t == "finalize_failed":
            console.print(f"{stamp} [red]final forced answer failed: {e['error']}[/]")
        elif t == "run_end":
            o, s = e["output"], e["stats"]
            body = (f"[bold]{o['answer']}[/]\n\n"
                    f"confidence: {o['confidence']} · answerable: {o['answerable']} · status: {o.get('status')}\n")
            if o.get("assumptions"):
                body += "assumptions:\n" + "\n".join(f"  • {x}" for x in o["assumptions"]) + "\n"
            body += (f"\n[dim]{s['turns']} turns · {s['tool_calls']} tool calls ({s['tool_errors']} errors) · "
                     f"{s['input_tokens']}+{s['output_tokens']} tokens · ${s['cost_usd']:.4f} · "
                     f"{s['elapsed_s']}s · {s['compactions']} compactions[/]")
            colour = "green" if e["stop_reason"] == "submitted" else "yellow"
            console.print(Panel(body, title=f"[{colour}]stop: {e['stop_reason']}[/]"
                                + (f" — {e['detail']}" if e.get("detail") else ""), expand=False))
    return 0
