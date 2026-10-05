from pathlib import Path

import pytest

from eval.ground_truth import compute
from eval.graders import grade

DB = Path(__file__).resolve().parent.parent / "data" / "Chinook_Sqlite.sqlite"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="run scripts/fetch_db.py first")


@pytest.fixture(scope="module")
def truth():
    return compute(DB)


def out(answer, answerable=True, assumptions=(), status="complete"):
    return {"answer": answer, "sql_used": ["SELECT 1"], "assumptions": list(assumptions),
            "confidence": "high", "answerable": answerable, "status": status}


CASES = [
    ("Q1", "There are 3,503 tracks in the database.", True),
    ("Q1", "There are 3,500 tracks.", False),
    ("Q2", "Rock has the most tracks, with 1,297.", True),
    ("Q2", "Latin has the most tracks.", False),
    ("Q3", "Iron Maiden (21), Led Zeppelin (14), Deep Purple (11), Metallica (10) and U2 (10).", True),
    ("Q3", "Iron Maiden, Led Zeppelin, Deep Purple, Metallica and Ozzy Osbourne.", False),
    ("Q4", "USA ($523.06), Canada ($303.96), France ($195.10), Brazil ($190.10), Germany ($156.48).", True),
    ("Q4", "The United States leads with $523.06, then Canada, France, Brazil and Germany.", True),
    ("Q4", "Canada, USA, France, Brazil, Germany.", False),
    ("Q5", "Jane Peacock's customers generated the most revenue: $833.04.", True),
    ("Q5", "Jane Peacock ($833.04) leads, ahead of Margaret Park ($775.40) and Steve Johnson ($720.16).", True),
    ("Q5", "Margaret Park leads with $775.40; Jane Peacock is second.", False),
    ("Q6", "2010 had the highest revenue at $481.45, up $31.99 (7.1%) from 2009.", True),
    ("Q6", "2010 was the best year with $481.45, about 7.12% above 2009 ($449.46).", True),
    ("Q6", "2010 had the highest revenue at $481.45.", False),
    ("Q6", "2012 had the highest revenue.", False),
    ("Q7", "MPEG audio 4.43, Protected AAC 4.70, Protected MPEG-4 video 39.05, Purchased AAC 4.35, AAC 4.61 min.", True),
    ("Q7", "MPEG audio 4.43, Protected AAC 4.70, video 39.05, Purchased AAC 4.35 min.", False),
    ("Q8", "41 tracks are tied, each appearing in 5 playlists, e.g. 'Koyaanisqatsi' and 'Ave Maria'.", True),
    ("Q8", "Several tracks tie at 5 playlists each, including Intoitus: Adorate Deum.", True),
    ("Q8", "The track in the most playlists is Koyaanisqatsi, which appears in 5 playlists.", False),
    ("Q8", "41 tracks are tied at 5 playlists each.", False),
    ("Q9", "Helena Holý is our best customer by total spend ($49.62).", True),
    ("Q9", "Helena Holy is the best customer.", False),  # no interpretation stated
    ("Q9", "Richard Cunningham spent the most ($47.62).", False),
    ("Q10", "Profit margin cannot be calculated: the database has sale prices but no cost data.", True),
    ("Q10", "Rock has a 30% margin, the highest of any genre.", False),
]


@pytest.mark.parametrize("qid,answer,expected", CASES)
def test_grader(truth, qid, answer, expected):
    answerable = not (qid == "Q10" and expected)
    passed, reason = grade(qid, out(answer, answerable=answerable), truth)
    assert passed == expected, reason


def test_q9_assumption_counts_as_interpretation(truth):
    o = out("Helena Holý is the best customer.", assumptions=["'Best' = highest total spend."])
    assert grade("Q9", o, truth)[0]


def test_q10_answerable_true_with_margin_fails(truth):
    o = out("Margins are unknown without cost data, but Rock is around 40%.", answerable=True)
    assert not grade("Q10", o, truth)[0]


def test_failed_runs_never_pass(truth):
    o = out("There are 3503 tracks.", status="failed")
    assert not grade("Q1", o, truth)[0]
