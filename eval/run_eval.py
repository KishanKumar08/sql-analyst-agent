"""
Run every eval question N times, grade each run, and write a report.

    uv run python -m eval.run_eval                  # all 10 questions x 3 runs
    uv run python -m eval.run_eval --runs 1 --only Q8 Q10
    uv run python -m eval.run_eval --workers 3      # parallel runs (watch rate limits)
    AGENT_REASONING_EFFORT=low uv run python -m eval.run_eval   # compare a setting

Writes eval/results/results-<stamp>.json and .md, and keeps every run's trace
under traces/eval-<stamp>/.
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent.config import Config
from agent.db import QueryError, SafeDB
from agent.llm import AzureLLM
from agent.loop import Agent
from agent.tracer import Tracer
from eval.ground_truth import DEFAULT_DB, compute
from eval.graders import QUESTIONS, grade

ROOT = Path(__file__).resolve().parent.parent


def sql_check(db_path, sql_used) -> bool:
    """
    Secondary signal (not part of pass/fail): do the cited queries run?
    """
    db = SafeDB(db_path)
    try:
        for q in sql_used:
            db.run(q)
        return True
    except QueryError:
        return False
    finally:
        db.close()


def run_one(qid, attempt, cfg, llm, db, truth, trace_dir):
    tracer = Tracer(trace_dir, run_id=f"{qid}-{attempt}")
    started = time.monotonic()
    res = Agent(cfg, llm, tracer).run(QUESTIONS[qid], os.path.relpath(db))  # relative: keeps home paths out of traces
    passed, reason = grade(qid, res.output, truth)
    row = {
        "qid": qid, "attempt": attempt, "passed": passed, "reason": reason,
        "stop_reason": res.output["stop_reason"], "status": res.output["status"],
        "confidence": res.output["confidence"], "answerable": res.output["answerable"],
        "answer": res.output["answer"], "assumptions": res.output["assumptions"],
        "sql_ok": sql_check(db, res.output["sql_used"]),
        "turns": res.stats.turns, "tool_errors": res.stats.tool_errors,
        "tokens": res.stats.total_tokens, "cost_usd": round(res.stats.cost_usd, 5),
        "latency_s": round(time.monotonic() - started, 2),
        "trace": str(Path(res.trace_path).relative_to(ROOT)),
    }
    mark = "PASS" if passed else "FAIL"
    print(f"  {qid} #{attempt} {mark}  {row['turns']}t ${row['cost_usd']:.4f} {row['latency_s']}s  {reason}",
          file=sys.stderr, flush=True)
    return row


def summarize(rows):
    def agg(rs):
        return {
            "runs": len(rs),
            "passed": sum(r["passed"] for r in rs),
            "pass_rate": round(sum(r["passed"] for r in rs) / len(rs), 3),
            "avg_cost_usd": round(statistics.mean(r["cost_usd"] for r in rs), 5),
            "avg_turns": round(statistics.mean(r["turns"] for r in rs), 2),
            "avg_latency_s": round(statistics.mean(r["latency_s"] for r in rs), 2),
        }
    per_q = {q: agg([r for r in rows if r["qid"] == q]) for q in dict.fromkeys(r["qid"] for r in rows)}
    return {"overall": agg(rows), "per_question": per_q,
            "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4)}


def to_markdown(summary, rows, meta):
    o = summary["overall"]
    lines = [
        f"# Eval results — {meta['stamp']}",
        "",
        f"Model: `{meta['model']}` (reasoning effort `{meta['reasoning_effort']}`) · "
        f"{meta['runs_per_question']} runs per question · total cost ${summary['total_cost_usd']}",
        "",
        f"**Overall: {o['passed']}/{o['runs']} passed ({o['pass_rate']:.0%})** · "
        f"avg ${o['avg_cost_usd']:.4f}/run · {o['avg_turns']} turns · {o['avg_latency_s']}s",
        "",
        "| Q | Question | Pass | Avg cost | Avg turns | Avg latency |",
        "|---|---|---|---|---|---|",
    ]
    for q, s in summary["per_question"].items():
        lines.append(f"| {q} | {QUESTIONS[q]} | {s['passed']}/{s['runs']} | ${s['avg_cost_usd']:.4f} | "
                     f"{s['avg_turns']} | {s['avg_latency_s']}s |")
    lines += ["", "## Every run", "", "| Run | Result | Why | Stop | Conf. | Trace |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['qid']} #{r['attempt']} | {'✅' if r['passed'] else '❌'} | {r['reason']} | "
                     f"{r['stop_reason']} | {r['confidence']} | `{r['trace']}` |")
    lines += ["", "## Answers", ""]
    for r in rows:
        lines.append(f"- **{r['qid']} #{r['attempt']}** ({'pass' if r['passed'] else 'FAIL'}): "
                     + r["answer"].replace("\n", " ")
                     + (f"  _Assumptions: {'; '.join(r['assumptions'])}_" if r["assumptions"] else ""))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--only", nargs="*", help="question ids, e.g. Q8 Q10")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--db", default=str(DEFAULT_DB))
    a = p.parse_args(argv)

    cfg = Config.from_env()  # e.g. AGENT_REASONING_EFFORT=low to compare settings
    llm = AzureLLM(cfg)
    truth = compute(a.db)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trace_dir = ROOT / "traces" / f"eval-{stamp}"
    qids = a.only or list(QUESTIONS)
    jobs = [(q, i) for q in qids for i in range(1, a.runs + 1)]
    print(f"running {len(jobs)} runs with {a.workers} worker(s)…", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(lambda j: run_one(j[0], j[1], cfg, llm, a.db, truth, trace_dir), jobs))

    summary = summarize(rows)
    meta = {"stamp": stamp, "model": cfg.deployment, "reasoning_effort": cfg.reasoning_effort,
            "runs_per_question": a.runs, "config": vars(cfg)}
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"results-{stamp}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "runs": rows}, indent=2, ensure_ascii=False))
    (out_dir / f"results-{stamp}.md").write_text(to_markdown(summary, rows, meta))

    o = summary["overall"]
    print(f"\n{o['passed']}/{o['runs']} passed ({o['pass_rate']:.0%}) · total ${summary['total_cost_usd']} · "
          f"report: eval/results/results-{stamp}.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
