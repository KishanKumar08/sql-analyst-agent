"""
JSONL trace writer. One file per run, one event per line, flushed after
every write so a crashed or killed run still leaves a readable trace.

Event types:
  run_start      question, db, config
  llm_request    turn, messages added since the previous request, prompt size estimate
  llm_response   turn, content, tool calls, tokens, latency, cost, retries
  tool_call      turn, tool, arguments
  tool_result    turn, tool, ok, full result, the text the model actually saw, latency
  compaction     how many results were summarized, token estimate before/after
  limit_reached  which budget ran out
  run_end        stop reason, final answer, totals
"""
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


class Tracer:
    def __init__(self, trace_dir: str | Path, run_id: str | None = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path = Path(trace_dir) / f"run-{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self._seq = 0
        self._t0 = time.monotonic()

    def event(self, type_: str, **data):
        self._seq += 1
        rec = {
            "seq": self._seq,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "t_ms": round((time.monotonic() - self._t0) * 1000, 1),
            "run_id": self.run_id,
            "type": type_,
            **data,
        }
        self._f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def close(self):
        if not self._f.closed:
            self._f.close()


class NullTracer:
    """
    Used in tests that don't care about traces.
    """
    path = None
    run_id = "null"

    def event(self, *a, **k):
        pass

    def close(self):
        pass
