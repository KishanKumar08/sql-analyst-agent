import json
import re

from .db import QueryError, SafeDB

CONFIDENCE = ["high", "medium", "low"]
_TYPES = {"string": str, "boolean": bool, "array": list, "object": dict}

SCHEMAS = {
    "list_tables": {
        "description": "List every table and view in the database with its row count. Start here.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "describe_table": {
        "description": "Show a table's columns, types, primary key, foreign keys, row count and a few sample rows.",
        "parameters": {
            "type": "object",
            "properties": {"table_name": {"type": "string", "maxLength": 200}},
            "required": ["table_name"],
        },
    },
    "run_sql": {
        "description": (
            "Run ONE read-only SQLite SELECT statement and get the rows back. "
            "Results are capped, so aggregate in SQL rather than pulling raw rows."
        ),
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string", "maxLength": 20000}},
            "required": ["sql"],
        },
    },
    "submit_answer": {
        "description": "Submit the final answer. This ends the run.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "maxLength": 4000,
                           "description": "Direct plain-language answer for a business stakeholder, with the "
                                          "key numbers. No SQL here; queries go in sql_used."},
                "sql_used": {"type": "array", "items": {"type": "string"},
                             "description": "Exact text of the run_sql queries whose results support the answer. "
                                            "Only queries you ran with run_sql; list_tables and describe_table "
                                            "calls are not SQL and must not be listed."},
                "assumptions": {"type": "array", "items": {"type": "string"},
                                "description": "Interpretations you had to make. Empty if none."},
                "confidence": {"type": "string", "enum": CONFIDENCE},
                "answerable": {"type": "boolean",
                               "description": "false if the database does not contain the data needed to answer. "
                                              "Using a different column as a stand-in for the missing data does "
                                              "not make it answerable."},
            },
            "required": ["answer", "sql_used", "assumptions", "confidence", "answerable"],
        },
    },
}


def _strict(schema: dict) -> dict:
    """
    Schema as sent to the model in strict mode: the API then guarantees
    every required field is present. Strict mode needs additionalProperties
    false and rejects some keywords (maxLength), which we still enforce
    ourselves in validate().
    """
    if not isinstance(schema, dict):
        return schema
    out = {k: _strict(v) for k, v in schema.items() if k != "maxLength"}
    if schema.get("type") == "object":
        out["additionalProperties"] = False
    if "properties" in schema:
        out["properties"] = {k: _strict(v) for k, v in schema["properties"].items()}
    return out


def openai_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": n, 
                "description": s["description"],
                "parameters": _strict(s["parameters"]), "strict": True
            }
        }
        for n, s in SCHEMAS.items()
    ]


def validate(name: str, args) -> str | None:
    """
    Return an error message, or None if args match the tool's schema.
    """
    schema = SCHEMAS[name]["parameters"]
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    props = schema["properties"]
    missing = [k for k in schema["required"] if k not in args]
    if missing:
        return f"missing required field(s): {', '.join(missing)}"
    unknown = [k for k in args if k not in props]
    if unknown:
        return f"unknown field(s): {', '.join(unknown)}; allowed: {', '.join(props) or 'none'}"
    for key, spec in props.items():
        if key not in args:
            continue
        v = args[key]
        if not isinstance(v, _TYPES[spec["type"]]):
            return f"'{key}' must be of type {spec['type']}"
        if spec["type"] == "string":
            if not v.strip():
                return f"'{key}' must not be empty"
            if "maxLength" in spec and len(v) > spec["maxLength"]:
                return f"'{key}' is too long (max {spec['maxLength']} characters)"
        if "enum" in spec and v not in spec["enum"]:
            return f"'{key}' must be one of {spec['enum']}"
        if spec["type"] == "array" and not all(isinstance(i, str) for i in v):
            return f"'{key}' must be a list of strings"
    return None


EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\+\d[\d\s().-]{7,}\d")
ASKS_FOR_CONTACT = re.compile(r"\b(e-?mail|phone|contact|address|fax)", re.I)

def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().rstrip(";").strip().lower()

# ---- dispatch ---------------------------------------------------------------
class ToolBox:
    def __init__(self, db: SafeDB, question: str = ""):
        self.db = db
        self.question = question
        self.successful_sql: dict[str, dict] = {}

    def call(self, name: str, raw_args: str) -> dict:
        if name not in SCHEMAS:
            return {"ok": False, "error_type": "unknown_tool",
                    "message": f"No tool named '{name}'. Available: {', '.join(SCHEMAS)}."}
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as e:
            return {"ok": False, "error_type": "invalid_arguments",
                    "message": f"Arguments are not valid JSON: {e}"}
        err = validate(name, args)
        if err:
            return {"ok": False, "error_type": "invalid_arguments", "message": err}
        try:
            return getattr(self, f"_{name}")(**args)
        except QueryError as e:
            return e.to_dict()
        except Exception as e:  # last line of defence: a tool bug must not kill the run
            return {"ok": False, "error_type": "internal_error", "message": f"{type(e).__name__}: {e}"}

    def _list_tables(self) -> dict:
        return {"ok": True, "tables": self.db.list_tables()}

    def _describe_table(self, table_name: str) -> dict:
        return {"ok": True, **self.db.describe_table(table_name)}

    def _run_sql(self, sql: str) -> dict:
        res = self.db.run(sql)
        self.successful_sql[normalize_sql(sql)] = {
            "sql": sql.strip(), "columns": res["columns"], "rows": res["rows"][:5],
            "row_count": res["row_count"], "truncated": res["truncated"],
        }
        return res

    def _submit_answer(self, **answer) -> dict:
        not_run = [q for q in answer["sql_used"] if normalize_sql(q) not in self.successful_sql]
        if not_run:
            return {"ok": False, "error_type": "unverified_sql",
                    "message": "These sql_used entries were never run successfully in this session: "
                               + json.dumps(not_run, ensure_ascii=False),
                    "hint": "Run them with run_sql first, or copy the exact text of queries you did run."}
        if answer["answerable"] and not answer["sql_used"]:
            return {"ok": False, "error_type": "invalid_arguments",
                    "message": "An answerable result must cite the queries it is based on in sql_used."}
        if not ASKS_FOR_CONTACT.search(self.question):
            found = EMAIL.findall(answer["answer"]) + PHONE.findall(answer["answer"])
            if found:
                return {
                    "ok": False, 
                    "error_type": "personal_data",
                    "message": f"The answer includes contact details the question did not ask for: {found}.",
                    "hint": "Remove them; identify people by name (and ID if useful)."
                }
        
        return {"ok": True, "accepted": True}
