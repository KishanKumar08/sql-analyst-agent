"""
Command line entry point.

  agent ask "question" --db file.sqlite     answer one question
  agent chat --db file.sqlite               ask follow-up questions in one session
  agent trace traces/run-XXXX.jsonl         show a trace in readable form

The answer JSON goes to stdout; progress and the trace path go to stderr.
The four run budgets have flags. Every other setting (query timeout, row cap,
reasoning effort, schema preload, ...) is an AGENT_<NAME> environment
variable, e.g. AGENT_QUERY_TIMEOUT_S=2. Defaults are in agent/config.py.
"""
import argparse
import json
import os
import sys
import threading

from dotenv import load_dotenv

from .config import Config
from .llm import AzureLLM
from .loop import Agent
from .trace_view import show
from .tracer import Tracer

ICONS = {"turn": "·", "error": "✗", "limit": "⚠", "nudge": "!"}


def log(text: str):
    print(text, file=sys.stderr, flush=True)


def print_progress(kind: str, text: str):
    log(f"  {ICONS.get(kind, '→')} {text}")


def start_watchdog(seconds: float) -> threading.Timer:
    """
    Last resort for "a run must never hang": if the run is still going well
    past its wall-clock limit, print a valid failure answer and exit.
    """
    def fire():
        print(json.dumps({"answer": "Stopped by the watchdog: the run exceeded its wall-clock limit.",
                          "sql_used": [], "assumptions": [], "confidence": "low", "answerable": False,
                          "status": "failed", "stop_reason": "watchdog"}, indent=2), flush=True)
        os._exit(3)
    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    return timer


def ask(question: str, db: str, cfg: Config, quiet: bool) -> int:
    watchdog = start_watchdog(cfg.max_wall_s + 30)
    agent = Agent(cfg, AzureLLM(cfg), Tracer(cfg.trace_dir), None if quiet else print_progress)
    result = agent.run(question, db)
    watchdog.cancel()

    print(json.dumps(result.output, indent=2, ensure_ascii=False))
    s = result.stats
    log(f"{s.turns} turns · {s.total_tokens} tokens · ${s.cost_usd:.4f} · {s.elapsed_s}s · "
        f"stop: {result.output['stop_reason']}")
    log(f"trace: {result.trace_path}")
    return 0 if result.output["status"] == "complete" else 1


def chat(db: str, cfg: Config, quiet: bool) -> int:
    """Follow-up questions reuse the previous conversation."""
    llm = AzureLLM(cfg)
    history = None
    log("Ask a question (empty line or Ctrl-D to quit).")
    while True:
        try:
            question = input("\n? ").strip()
        except EOFError:
            break
        if not question:
            break
        result = Agent(cfg, llm, Tracer(cfg.trace_dir), None if quiet else print_progress).run(question, db, history)
        print(json.dumps(result.output, indent=2, ensure_ascii=False))
        log(f"trace: {result.trace_path}")
        history = result.messages
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="agent", description="SQL analyst agent")
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("ask", "chat"):
        p = commands.add_parser(name)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("--db", required=True, help="path to a SQLite file")
        p.add_argument("--max-turns", type=int, help="max model calls")
        p.add_argument("--max-tokens", type=int, help="max input+output tokens for the run")
        p.add_argument("--max-cost", type=float, help="max cost in USD")
        p.add_argument("--max-wall", type=float, help="max seconds")
        p.add_argument("--quiet", action="store_true", help="no progress output")

    p = commands.add_parser("trace")
    p.add_argument("file")
    p.add_argument("--full", action="store_true", help="show full results and model text")

    args = parser.parse_args()
    if args.command == "trace":
        return show(args.file, full=args.full)

    # Flags left unset (None) keep the env/default value.
    cfg = Config.from_env(max_turns=args.max_turns, max_tokens=args.max_tokens,
                          max_cost_usd=args.max_cost, max_wall_s=args.max_wall)
    if args.command == "ask":
        return ask(args.question, args.db, cfg, args.quiet)
    return chat(args.db, cfg, args.quiet)


if __name__ == "__main__":
    sys.exit(main())
