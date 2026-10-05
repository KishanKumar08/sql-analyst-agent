"""
Graders: one function per question, each returns (passed, reason).

Grading is deterministic: we check the agent's structured output against
ground truth computed from the database (eval/ground_truth.py). No LLM judge.
That keeps grading free, repeatable and explainable, at the cost of some
brittleness to unusual phrasing (see README, "How grading works").

Rules shared by all graders:
  - a run that never submitted an answer (status "failed") fails
  - names are compared case- and accent-insensitively ("Holy" == "Holý")
  - numbers are pulled from the answer text with commas/$ removed and
    matched within a tolerance
"""
import re
import unicodedata

QUESTIONS = {
    "Q1": "How many tracks are in the database?",
    "Q2": "Which genre has the most tracks?",
    "Q3": "Which 5 artists have the most albums?",
    "Q4": "What are the top 5 countries by total revenue?",
    "Q5": "Which sales support agent's customers generated the most revenue?",
    "Q6": "Which year had the highest revenue, and how did it compare to the year before?",
    "Q7": "What is the average track length per media type, in minutes?",
    "Q8": "Which tracks appear in the most playlists?",
    "Q9": "Who is our best customer?",
    "Q10": "What is our profit margin by genre?",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("’", "'")
    return re.sub(r"\s+", " ", s).lower()


def numbers(text: str) -> list[float]:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def has_number(text: str, target: float, tol: float) -> bool:
    return any(abs(n - target) <= tol for n in numbers(text))


def mentions(text: str, name: str) -> bool:
    return norm(name) in norm(text)


COUNTRY_ALIASES = {"USA": ["usa", "united states", "u.s."]}


def mentions_country(text, c):
    return any(a in norm(text) for a in COUNTRY_ALIASES.get(c, [norm(c)]))


# ---------------------------------------------------------------------------

def g_q1(o, t):
    ok = has_number(o["answer"], t["value"], 0)
    return ok, f"expects {t['value']}"


def g_q2(o, t):
    return mentions(o["answer"], t["name"]), f"expects {t['name']}"


def g_q3(o, t):
    missing = [n for n in t["names"] if not mentions(o["answer"], n)]
    return not missing, f"missing: {missing}" if missing else "all 5 artists named"


def g_q4(o, t):
    missing = [c for c in t["names"] if not mentions_country(o["answer"], c)]
    if missing:
        return False, f"missing: {missing}"
    # Order matters for a top-5: check the countries appear in rank order.
    pos = [min(norm(o["answer"]).find(a) for a in COUNTRY_ALIASES.get(c, [norm(c)])
               if norm(o["answer"]).find(a) >= 0) for c in t["names"]]
    return pos == sorted(pos), "all 5 in order" if pos == sorted(pos) else "countries out of order"


def g_q5(o, t):
    a = norm(o["answer"])
    if t["name"].lower() not in a:
        return False, f"expects {t['name']}"
    # If the answer ranks all agents, the winner must be named first.
    first = min((a.find(n.lower()), n) for n in t["all"] if a.find(n.lower()) >= 0)[1]
    return first == t["name"], f"first agent named: {first}"


def g_q6(o, t):
    a = o["answer"]
    if not has_number(a, t["year"], 0) or not has_number(a, t["prev_year"], 0):
        return False, f"expects {t['year']} compared with {t['prev_year']}"
    cmp_ok = has_number(a, t["pct_change"], 0.15) or has_number(a, t["abs_change"], 0.05)
    return cmp_ok, ("change ok" if cmp_ok else
                    f"expects +{t['abs_change']} or +{t['pct_change']}%")


def g_q7(o, t):
    missing = [k for k, v in t["avg_minutes"].items() if not has_number(o["answer"], v, 0.06)]
    return not missing, f"wrong/missing: {missing}" if missing else "all 5 media types"


TIE_WORDS = ("tie", "tied", "share", "several", "multiple", "many", "each appear", "all appear")


def g_q8(o, t):
    a = norm(o["answer"])
    count_ok = has_number(o["answer"], t["max_playlists"], 0) or \
        has_number(o["answer"], t["max_playlists_by_distinct_name"], 0)
    tie_ok = has_number(o["answer"], t["tied_count"], 0) or any(w in a for w in TIE_WORDS)
    named = [n for n in t["tied_names"] if norm(n)[:25] in a]
    if not count_ok:
        return False, f"expects max of {t['max_playlists']} playlists"
    if not tie_ok:
        return False, f"does not acknowledge the {t['tied_count']}-way tie"
    if not named:
        return False, "names no track from the tied set"
    return True, f"tie acknowledged, {len(named)} tied track(s) named"


def g_q9(o, t):
    if not mentions(o["answer"], t["name"]):
        return False, f"expects {t['name']} (top by total spend)"
    stated = bool(o["assumptions"]) or any(w in norm(o["answer"]) for w in ("spend", "spent", "revenue"))
    return stated, "interpretation stated" if stated else "does not say what 'best' means"


MISSING_COST = re.compile(r"(no|not|lack|without|missing|doesn't|does not)[^.]{0,60}(cost|margin|expense)")


def g_q10(o, t):
    if o["answerable"] is False and MISSING_COST.search(norm(o["answer"])):
        return True, "answerable=false and explains missing cost data"
    if o["answerable"] is False:
        return True, "answerable=false"
    # Answerable=true is only acceptable if it clearly refuses to invent a margin.
    if MISSING_COST.search(norm(o["answer"])) and "%" not in o["answer"]:
        return True, "explains missing cost data (answerable flag set true)"
    return False, "claims a margin the data cannot support"


GRADERS = {f"Q{i}": g for i, g in enumerate(
    [g_q1, g_q2, g_q3, g_q4, g_q5, g_q6, g_q7, g_q8, g_q9, g_q10], start=1)}


def grade(qid: str, output: dict, truth: dict) -> tuple[bool, str]:
    if output.get("status") == "failed":
        return False, f"no answer submitted (stop: {output.get('stop_reason')})"
    try:
        return GRADERS[qid](output, truth[qid])
    except Exception as e:  # malformed output shouldn't crash the eval
        return False, f"grader error: {type(e).__name__}: {e}"
