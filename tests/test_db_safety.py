"""
Attack the safety layer. Each test tries something the agent must never be
able to do, and checks it fails cleanly with the right error kind.

Tests run against a copy of Chinook plus a small 'weird' database, so a
broken guard could never damage the real file.
"""
import hashlib
import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from agent.db import Limits, QueryError, SafeDB

CHINOOK = Path(__file__).resolve().parent.parent / "data" / "Chinook_Sqlite.sqlite"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def chinook(tmp_path):
    if not CHINOOK.exists():
        pytest.skip("run scripts/fetch_db.py first")
    copy = tmp_path / "chinook.sqlite"
    shutil.copy(CHINOOK, copy)
    return copy


@pytest.fixture
def weird(tmp_path):
    """
    Things Chinook doesn't have: spaces in names, a view, big + wide data.
    """
    p = tmp_path / "weird.sqlite"
    c = sqlite3.connect(p)
    c.executescript("""
        CREATE TABLE "Order Items" (id INTEGER PRIMARY KEY, "unit cost" REAL, note TEXT);
        CREATE TABLE big (n INTEGER);
        CREATE VIEW expensive AS SELECT * FROM "Order Items" WHERE "unit cost" > 10;
        CREATE TABLE empty (x INTEGER);
    """)
    c.executemany('INSERT INTO "Order Items" VALUES (?, ?, ?)',
                  [(i, i * 1.5, "x" * 1000) for i in range(1, 21)])
    c.executemany("INSERT INTO big VALUES (?)", [(i,) for i in range(5000)])
    c.commit()
    c.close()
    return p


# ---- writes must be impossible ----------------------------------------
WRITE_ATTEMPTS = [
    "INSERT INTO Genre (GenreId, Name) VALUES (999, 'Hacked')",
    "UPDATE Track SET UnitPrice = 0",
    "DELETE FROM Invoice",
    "DROP TABLE Customer",
    "CREATE TABLE evil (x)",
    "CREATE TEMP TABLE evil (x)",
    "ALTER TABLE Artist ADD COLUMN pwned TEXT",
    "ATTACH DATABASE '/tmp/evil.sqlite' AS evil",
    "PRAGMA writable_schema = 1",
    "PRAGMA query_only = OFF",
    "WITH x AS (SELECT 1) DELETE FROM Invoice",
    "REPLACE INTO Genre (GenreId, Name) VALUES (1, 'Hacked')",
    "VACUUM",
]


@pytest.mark.parametrize("sql", WRITE_ATTEMPTS)
def test_writes_are_rejected(chinook, sql):
    before = _sha(chinook)
    db = SafeDB(chinook)
    with pytest.raises(QueryError) as e:
        db.run(sql)
    assert e.value.kind in {"forbidden", "sql_error"}
    db.close()
    assert _sha(chinook) == before, "database file changed!"


def test_cannot_sneak_write_after_select(chinook):
    db = SafeDB(chinook)
    with pytest.raises(QueryError) as e:
        db.run("SELECT 1; DROP TABLE Customer")
    assert e.value.kind == "multiple_statements"


# ---- resource limits ----------------------------------------------------

def test_runaway_query_times_out(chinook):
    db = SafeDB(chinook, Limits(timeout_s=0.5))
    start = time.monotonic()
    with pytest.raises(QueryError) as e:
        db.run("WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) "
               "SELECT SUM(x) FROM r")
    assert e.value.kind == "timeout"
    assert time.monotonic() - start < 2.0


def test_huge_cross_join_times_out_or_is_capped(chinook):
    db = SafeDB(chinook, Limits(timeout_s=0.5, max_rows=50))
    try:
        res = db.run("SELECT * FROM Track a, Track b, Track c")
        assert res["row_count"] == 50 and res["truncated"]
    except QueryError as e:
        assert e.kind == "timeout"


def test_results_are_capped(weird):
    db = SafeDB(weird, Limits(max_rows=100))
    res = db.run("SELECT n FROM big")
    assert res["row_count"] == 100
    assert res["truncated"] is True


def test_small_result_not_marked_truncated(weird):
    db = SafeDB(weird, Limits(max_rows=100))
    res = db.run("SELECT COUNT(*) FROM big")
    assert res["rows"] == [(5000,)]
    assert res["truncated"] is False


def test_long_cells_are_clipped(weird):
    db = SafeDB(weird, Limits(max_cell_chars=50))
    res = db.run('SELECT note FROM "Order Items" LIMIT 1')
    assert len(res["rows"][0][0]) == 51  # 50 chars + ellipsis


# ---- errors come back as useful, typed results --------------------------

def test_unknown_table_suggests_fix(chinook):
    db = SafeDB(chinook)
    with pytest.raises(QueryError) as e:
        db.run("SELECT * FROM Tracks")
    assert e.value.kind == "unknown_table"
    assert "Track" in e.value.hint


def test_unknown_column(chinook):
    db = SafeDB(chinook)
    with pytest.raises(QueryError) as e:
        db.run("SELECT Nme FROM Track")
    assert e.value.kind == "unknown_column"


def test_syntax_error(chinook):
    db = SafeDB(chinook)
    with pytest.raises(QueryError) as e:
        db.run("SELEC * FROM Track")
    assert e.value.kind == "syntax_error"


def test_empty_query(chinook):
    with pytest.raises(QueryError) as e:
        SafeDB(chinook).run("   ")
    assert e.value.kind == "invalid_input"


# ---- introspection works on any database --------------------------------

def test_describe_table_case_insensitive(chinook):
    d = SafeDB(chinook).describe_table("track")
    assert d["table"] == "Track"
    assert d["row_count"] == 3503
    assert {"column": "GenreId", "references": "Genre.GenreId"} in d["foreign_keys"]
    assert len(d["sample_rows"]) == 3


def test_weird_names_views_and_empty_tables(weird):
    db = SafeDB(weird)
    tables = {t["name"]: t for t in db.list_tables()}
    assert tables["Order Items"]["rows"] == 20
    assert tables["expensive"]["type"] == "view"
    assert tables["empty"]["rows"] == 0
    d = db.describe_table("order items")
    assert d["columns"][1]["name"] == "unit cost"
    assert db.describe_table("empty")["sample_rows"] == []


def test_missing_and_non_sqlite_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        SafeDB(tmp_path / "nope.sqlite")
    junk = tmp_path / "junk.sqlite"
    junk.write_text("not a database")
    with pytest.raises(ValueError):
        SafeDB(junk)
