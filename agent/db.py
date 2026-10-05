"""
Read-only, time-limited, size-capped access to a SQLite database.

Safety is layered so that no single mistake lets the agent write:
  1. The file is opened with mode=ro (SQLite refuses writes at the file level).
  2. PRAGMA query_only=ON (SQLite refuses writes at the connection level).
  3. An authorizer callback allows only reads and a small allowlist of
     introspection PRAGMAs. Everything else (INSERT, DROP, ATTACH, writable
     PRAGMAs, temp tables, ...) is denied before it runs.
Plus resource limits:
  - a progress handler aborts any statement that runs past the timeout
  - results are fetched with fetchmany(max_rows + 1), so we never pull more
    than we need into memory, and long cell values are clipped.

Every failure is raised as QueryError with a short `kind` and a `hint`, so
the tool layer can hand it back to the model instead of crashing.
"""
import difflib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
# PRAGMAs that only read schema information.
_ALLOWED_PRAGMAS = {
    "table_info", "table_xinfo", "foreign_key_list", "index_list", "index_info",
}


class QueryError(Exception):
    def __init__(self, kind: str, message: str, hint: str = ""):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict:
        d = {"ok": False, "error_type": self.kind, "message": self.message}
        if self.hint:
            d["hint"] = self.hint
        return d


@dataclass
class Limits:
    timeout_s: float = 5.0
    max_rows: int = 200
    max_cell_chars: int = 200


def _authorizer(action, arg1, arg2, db_name, trigger):
    if action in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and arg1 and arg1.lower() in _ALLOWED_PRAGMAS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class SafeDB:
    def __init__(self, path: str | Path, limits: Limits | None = None):
        self.path = Path(path)
        self.limits = limits or Limits()
        if not self.path.is_file():
            raise FileNotFoundError(f"database not found: {self.path}")
        with open(self.path, "rb") as f:
            if f.read(16) != b"SQLite format 3\x00":
                raise ValueError(f"not a SQLite database: {self.path}")

        self.conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True,
                                    check_same_thread=False)
        self.conn.execute("PRAGMA query_only = ON")
        self.conn.set_authorizer(_authorizer)
        self._deadline = 0.0
        self.conn.set_progress_handler(self._check_deadline, 1000)

    def _check_deadline(self) -> int:
        # Non-zero return makes SQLite abort with "interrupted".
        return 1 if time.monotonic() > self._deadline else 0

    def close(self):
        self.conn.close()

    # ---- introspection -------------------------------------------------

    def table_names(self) -> list[str]:
        rows = self._execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name", max_rows=10_000)["rows"]
        return [r[0] for r in rows]

    def list_tables(self) -> list[dict]:
        out = []
        for name, kind in self._execute(
                "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name", max_rows=10_000)["rows"]:
            try:
                n = self._execute(f"SELECT COUNT(*) FROM {_quote_ident(name)}")["rows"][0][0]
            except QueryError:
                n = None  # e.g. a huge table that hits the timeout; not worth failing for
            out.append({"name": name, "type": kind, "rows": n})
        return out

    def schema_summary(self, sample_rows: int = 0) -> str:
        """Compact schema for the system prompt: one line per table with
        columns, types, keys and row count, plus `sample_rows` example rows
        so the model still sees what the data looks like."""
        lines = []
        for t in self.list_tables():
            q = _quote_ident(t["name"])
            fks = {f[3]: f"{f[2]}.{f[4]}"
                   for f in self._execute(f"PRAGMA foreign_key_list({q})", max_rows=10_000)["rows"]}
            cols = []
            for c in self._execute(f"PRAGMA table_info({q})", max_rows=10_000)["rows"]:
                col = f"{c[1]} {c[2]}".strip()
                if c[5]:
                    col += " PK"
                if c[1] in fks:
                    col += f" -> {fks[c[1]]}"
                cols.append(col)
            kind = " view" if t["type"] == "view" else ""
            rows = f"{t['rows']} rows" if t["rows"] is not None else "row count unknown"
            lines.append(f"- {t['name']}{kind} ({rows}): {', '.join(cols)}")
            if sample_rows:
                try:
                    sample = self._execute(f"SELECT * FROM {q} LIMIT {int(sample_rows)}")["rows"]
                except QueryError:
                    sample = []
                for r in sample:
                    lines.append("    e.g. " + repr(tuple(r)))
        return "\n".join(lines)

    def resolve_table(self, name: str) -> str:
        """Match a table name case-insensitively, or raise unknown_table."""
        names = self.table_names()
        for n in names:
            if n.lower() == name.lower():
                return n
        close = difflib.get_close_matches(name, names, n=3, cutoff=0.5)
        hint = f"Did you mean: {', '.join(close)}?" if close else "Call list_tables to see valid names."
        raise QueryError("unknown_table", f"No table or view named '{name}'.", hint)

    def describe_table(self, name: str, sample_rows: int = 3) -> dict:
        table = self.resolve_table(name)
        q = _quote_ident(table)
        cols = self._execute(f"PRAGMA table_info({q})", max_rows=10_000)["rows"]
        fks = self._execute(f"PRAGMA foreign_key_list({q})", max_rows=10_000)["rows"]
        sample = self._execute(f"SELECT * FROM {q} LIMIT {int(sample_rows)}")
        try:
            count = self._execute(f"SELECT COUNT(*) FROM {q}")["rows"][0][0]
        except QueryError:
            count = None
        return {
            "table": table,
            "row_count": count,
            "columns": [
                {"name": c[1], "type": c[2], "not_null": bool(c[3]), "primary_key": bool(c[5])}
                for c in cols
            ],
            # foreign_key_list rows: (id, seq, table, from, to, ...)
            "foreign_keys": [{"column": f[3], "references": f"{f[2]}.{f[4]}"} for f in fks],
            "sample_rows": sample["rows"],
        }

    # ---- query execution -----------------------------------------------

    def run(self, sql: str) -> dict:
        sql = (sql or "").strip()
        if not sql:
            raise QueryError("invalid_input", "Query is empty.")
        return self._execute(sql)

    def _execute(self, sql: str, max_rows: int | None = None) -> dict:
        max_rows = max_rows or self.limits.max_rows
        self._deadline = time.monotonic() + self.limits.timeout_s
        started = time.monotonic()
        try:
            cur = self.conn.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows + 1)
        except sqlite3.ProgrammingError as e:
            if "one statement at a time" in str(e):
                raise QueryError("multiple_statements", "Only one SQL statement per call is allowed.",
                                 "Remove extra statements; run them as separate calls.") from None
            raise QueryError("sql_error", str(e)) from None
        except sqlite3.DatabaseError as e:
            raise self._classify(e) from None

        truncated = len(rows) > max_rows
        rows = [tuple(self._clip(v) for v in r) for r in rows[:max_rows]]
        return {
            "ok": True,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        }

    def _clip(self, v):
        if isinstance(v, bytes):
            return f"<blob {len(v)} bytes>"
        if isinstance(v, str) and len(v) > self.limits.max_cell_chars:
            return v[: self.limits.max_cell_chars] + "…"
        return v

    def _classify(self, e: sqlite3.DatabaseError) -> QueryError:
        msg = str(e)
        low = msg.lower()
        if "interrupted" in low:
            return QueryError("timeout", f"Query exceeded the {self.limits.timeout_s}s time limit.",
                              "Add filters or LIMIT, avoid cross joins, or aggregate earlier.")
        if "not authorized" in low or "readonly" in low or "read-only" in low:
            return QueryError("forbidden", "The database is read-only; only SELECT queries are allowed.")
        if low.startswith("no such table"):
            bad = msg.split(":", 1)[-1].strip()
            try:
                self.resolve_table(bad)
            except QueryError as qe:
                return QueryError("unknown_table", msg, qe.hint)
            return QueryError("unknown_table", msg)
        if low.startswith("no such column"):
            return QueryError("unknown_column", msg,
                              "Call describe_table on the tables involved to check column names.")
        if "syntax error" in low or "incomplete input" in low:
            return QueryError("syntax_error", msg)
        return QueryError("sql_error", msg)
