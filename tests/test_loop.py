import json
from pathlib import Path

import pytest

from agent.config import Config
from agent.llm import LLMError
from agent.loop import Agent
from agent.tracer import NullTracer, Tracer
from tests.fake_llm import FakeLLM, call, raw_call

DB = Path(__file__).resolve().parent.parent / "data" / "Chinook_Sqlite.sqlite"
COUNT_SQL = "SELECT COUNT(*) FROM Track"

pytestmark = pytest.mark.skipif(not DB.exists(), reason="run scripts/fetch_db.py first")

REQUIRED = {"answer", "sql_used", "assumptions", "confidence", "answerable", "stop_reason", "status"}


def submit(sql_used=(COUNT_SQL,), answer="There are 3503 tracks.", **kw):
    args = {"answer": answer, "sql_used": list(sql_used), "assumptions": [],
            "confidence": "high", "answerable": True} | kw
    return [call("submit_answer", **args)]


def run(script, cfg=None, tracer=None, **llm_kw):
    cfg = cfg or Config()
    llm = FakeLLM(script, **llm_kw)
    res = Agent(cfg, llm, tracer or NullTracer()).run("How many tracks?", str(DB))
    assert REQUIRED <= res.output.keys()
    json.dumps(res.output)  # must always be serializable
    return res, llm


# ---- happy path -------------------------------------------------------------

def test_happy_path():
    res, _ = run([[call("list_tables")], [call("run_sql", sql=COUNT_SQL)], submit()])
    assert res.output["stop_reason"] == "submitted"
    assert res.output["status"] == "complete"
    assert res.output["sql_used"] == [COUNT_SQL]
    assert res.stats.turns == 3


def test_parallel_tool_calls_in_one_turn():
    res, _ = run([[call("describe_table", table_name="Track"), call("run_sql", sql=COUNT_SQL)], submit()])
    assert res.stats.tool_calls == 3
    assert res.stats.turns == 2


# ---- tool failures go back to the model --------------------------------------

@pytest.mark.parametrize("bad", [
    raw_call("run_sql", "{not json"),
    call("run_sql"),                              # missing field
    call("run_sql", sql=COUNT_SQL, extra=1),      # unknown field
    call("describe_table", table_name="Tracks"),  # unknown table
    call("run_sql", sql="DROP TABLE Track"),      # forbidden
    call("run_sql", sql="SELEC 1"),               # syntax error
    call("no_such_tool"),
])
def test_tool_errors_are_returned_not_raised(bad):
    res, llm = run([[bad], [call("run_sql", sql=COUNT_SQL)], submit()])
    assert res.output["stop_reason"] == "submitted"
    assert res.stats.tool_errors == 1
    error_msg = json.loads(llm.requests[1]["messages"][-1]["content"])
    assert error_msg["ok"] is False and error_msg["error_type"]


def test_submit_with_sql_never_run_is_rejected_then_fixed():
    res, llm = run([
        [call("run_sql", sql=COUNT_SQL)],
        submit(sql_used=["SELECT COUNT(*) FROM Album"]),   # never ran this
        submit(),
    ])
    assert res.stats.tool_errors == 1
    assert "unverified_sql" in llm.requests[2]["messages"][-1]["content"]
    assert res.output["sql_used"] == [COUNT_SQL]


def test_sql_used_matching_ignores_whitespace_and_semicolon():
    res, _ = run([[call("run_sql", sql=COUNT_SQL)], submit(sql_used=["select count(*)\n  from Track;"])])
    assert res.output["status"] == "complete"


def test_text_reply_without_tool_gets_nudged():
    res, llm = run(["I think it's 3503", [call("run_sql", sql=COUNT_SQL)], submit()])
    assert res.output["stop_reason"] == "submitted"
    assert "submit_answer" in llm.requests[1]["messages"][-1]["content"]


# ---- budgets: each ends gracefully with valid JSON ----------------------------

def _forced_submit(messages, tool_choice):
    assert tool_choice != "auto", "final call must force submit_answer"
    return submit(confidence="high")


def test_max_turns_forces_final_answer():
    cfg = Config(max_turns=4)
    loop = [[call("run_sql", sql=COUNT_SQL)]] * 3
    res, llm = run(loop + [_forced_submit], cfg)
    assert res.output["stop_reason"] == "max_turns"
    assert res.output["status"] == "partial"
    assert res.output["confidence"] == "medium"      # capped from high
    assert "stopped early" in res.output["assumptions"][0]
    assert res.stats.turns == 4                       # final call is within the limit


def test_final_call_fails_falls_back_to_last_query():
    cfg = Config(max_turns=3)
    res, _ = run([[call("run_sql", sql=COUNT_SQL)]] * 2 + [LLMError("boom")], cfg)
    assert res.output["stop_reason"] == "max_turns"
    assert res.output["status"] == "failed"
    assert res.output["confidence"] == "low"
    assert "3503" in res.output["answer"]            # partial evidence surfaced
    assert res.output["sql_used"] == [COUNT_SQL]


def test_forced_answer_drops_sql_that_never_ran():
    cfg = Config(max_turns=2)
    res, _ = run([[call("run_sql", sql=COUNT_SQL)],
                  lambda m, tc: submit(sql_used=[COUNT_SQL, "SELECT 42"])], cfg)
    assert res.output["sql_used"] == [COUNT_SQL]


def test_token_budget():
    cfg = Config(max_tokens=5000)
    res, _ = run([[call("list_tables")]], cfg, usage=(2000, 200), repeat_last=True)
    assert res.output["stop_reason"] == "max_tokens"
    assert res.stats.total_tokens <= 5000


def test_final_call_output_is_capped_to_remaining_budget():
    # Output-heavy calls: the forced final call must be told how little room is left.
    cfg = Config(max_tokens=12000)
    script = [[call("run_sql", sql=COUNT_SQL)]] * 3 + [_forced_submit]
    res, llm = run(script, cfg, usage=(1500, 1500))
    final = llm.requests[-1]
    assert final["tool_choice"] != "auto"
    assert final["max_output"] is not None and final["max_output"] < cfg.max_completion_tokens
    assert res.output["status"] == "partial"
    assert res.stats.total_tokens <= cfg.max_tokens


def test_cost_budget():
    cfg = Config(max_cost_usd=0.002)
    res, _ = run([[call("list_tables")]], cfg, usage=(3000, 200), repeat_last=True)
    assert res.output["stop_reason"] == "max_cost"
    assert res.stats.cost_usd <= 0.002


def test_wall_clock_limit():
    cfg = Config(max_wall_s=1.0)
    res, _ = run([[call("list_tables")]], cfg, latency_s=0.3, repeat_last=True)
    assert res.output["stop_reason"] == "max_wall_time"
    assert res.stats.elapsed_s < 2.0


def test_model_error_mid_run():
    res, _ = run([[call("run_sql", sql=COUNT_SQL)], LLMError("RateLimitError after 5 attempts")])
    assert res.output["stop_reason"] == "llm_error"
    assert res.output["status"] == "failed"
    assert "RateLimitError" in res.output["stop_detail"]


def test_bad_db_path_still_returns_json(tmp_path):
    llm = FakeLLM([])
    res = Agent(Config(), llm, NullTracer()).run("q", str(tmp_path / "missing.sqlite"))
    assert res.output["stop_reason"] == "db_error"
    assert llm.requests == []


# ---- context management ---------------------------------------------------------

def test_large_results_are_trimmed_before_model_sees_them():
    res, llm = run([[call("run_sql", sql="SELECT * FROM Track")], submit(sql_used=["SELECT * FROM Track"])])
    seen = json.loads(llm.requests[1]["messages"][-1]["content"])
    assert len(seen["rows"]) <= Config().rows_to_model
    assert "Showing" in seen["note"]


def test_compaction_keeps_sql_and_recent_results():
    cfg = Config(compact_after_tokens=3000, keep_recent_tool_results=2, max_turns=20)
    queries = [f"SELECT * FROM Track WHERE TrackId > {i}" for i in range(6)]
    res, llm = run([[call("run_sql", sql=q)] for q in queries] + [submit(sql_used=[queries[-1]])], cfg)
    assert res.stats.compactions >= 1
    last = llm.requests[-1]["messages"]
    tool_msgs = [m for m in last if m["role"] == "tool"]
    assert tool_msgs[0]["content"].startswith("[compacted] run_sql ok")
    assert not tool_msgs[-1]["content"].startswith("[compacted]")
    # The SQL that was tried is still in the transcript.
    sent_sql = [tc["function"]["arguments"] for m in last if m["role"] == "assistant"
                for tc in m.get("tool_calls", [])]
    assert all(any(q in s for s in sent_sql) for q in queries)
    assert last[1]["content"] == "How many tracks?"


# ---- trace -----------------------------------------------------------------------

def test_trace_has_required_events(tmp_path):
    tracer = Tracer(tmp_path)
    res, _ = run([[call("run_sql", sql=COUNT_SQL)], submit()], tracer=tracer)
    events = [json.loads(line) for line in Path(res.trace_path).read_text().splitlines()]
    types = [e["type"] for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    for t in ("llm_request", "llm_response", "tool_call", "tool_result"):
        assert t in types
    resp = next(e for e in events if e["type"] == "llm_response")
    assert {"usage", "latency_ms", "cost_usd"} <= resp.keys()
    assert events[-1]["stop_reason"] == "submitted"


# ---- answer checks enforced in code ----------------------------------------------

def test_unrequested_email_in_answer_is_rejected():
    res, llm = run([
        [call("run_sql", sql=COUNT_SQL)],
        submit(answer="Helena Holý (hholy@gmail.com) is the best customer."),
        submit(answer="Helena Holý (CustomerId 6) is the best customer."),
    ])
    assert "personal_data" in llm.requests[2]["messages"][-1]["content"]
    assert "@" not in res.output["answer"]


def test_email_allowed_when_question_asks_for_it():
    from agent.tools import ToolBox
    from agent.db import SafeDB
    tb = ToolBox(SafeDB(DB), question="What is the email of our best customer?")
    tb.call("run_sql", json.dumps({"sql": COUNT_SQL}))
    args = {"answer": "hholy@gmail.com", "sql_used": [COUNT_SQL], "assumptions": [],
            "confidence": "high", "answerable": True}
    assert tb.call("submit_answer", json.dumps(args))["ok"]


def test_tool_schemas_are_strict():
    from agent.tools import openai_tools
    for t in openai_tools():
        f = t["function"]
        assert f["strict"] is True
        assert f["parameters"]["additionalProperties"] is False
        assert set(f["parameters"]["required"]) == set(f["parameters"]["properties"])
        assert "maxLength" not in json.dumps(f)


def test_turn_reminder_is_sent_once():
    cfg = Config(max_turns=10)
    script = [[call("run_sql", sql=COUNT_SQL)]] * 7 + [submit()]
    res, llm = run(script, cfg)
    reminders = [m for m in llm.requests[-1]["messages"]
                 if m["role"] == "user" and "turns" in (m["content"] or "")]
    assert len(reminders) == 1
    assert "6 of 10" in reminders[0]["content"]


# ---- schema preloading ---------------------------------------------------------

def test_schema_is_preloaded_into_system_prompt():
    res, llm = run([[call("run_sql", sql=COUNT_SQL)], submit()])
    system = llm.requests[0]["messages"][0]["content"]
    assert "Track (3503 rows)" in system
    assert "GenreId INTEGER -> Genre.GenreId" in system
    assert "do not need list_tables" in system


def test_schema_preload_can_be_disabled():
    res, llm = run([[call("list_tables")], [call("run_sql", sql=COUNT_SQL)], submit()],
                   Config(schema_in_prompt=False))
    system = llm.requests[0]["messages"][0]["content"]
    assert "Track (3503 rows)" not in system
    assert "Call list_tables on its own first" in system


def test_oversized_schema_falls_back_to_discovery_tools():
    res, llm = run([[call("run_sql", sql=COUNT_SQL)], submit()], Config(max_schema_chars=100))
    system = llm.requests[0]["messages"][0]["content"]
    assert "Track (3503 rows)" not in system
    assert "Call list_tables on its own first" in system


def test_bool_config_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_SCHEMA_IN_PROMPT", "false")
    assert Config.from_env().schema_in_prompt is False
    monkeypatch.setenv("AGENT_SCHEMA_IN_PROMPT", "1")
    assert Config.from_env().schema_in_prompt is True
