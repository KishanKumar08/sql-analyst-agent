"""
Ground truth for the 10 evaluation questions.

Run directly to print the truth:  python -m eval.ground_truth
"""
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "Chinook_Sqlite.sqlite"


def _rows(conn, sql):
    return conn.execute(sql).fetchall()


def compute(db_path=DEFAULT_DB) -> dict:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    t = {}

    # Q1 — no traps. 3503.
    t["Q1"] = {"value": _rows(conn, "SELECT COUNT(*) FROM Track")[0][0]}

    # Q2 — Rock by a wide margin (1297 vs 579). No NULL GenreIds.
    r = _rows(conn, """
        SELECT g.Name, COUNT(*) n FROM Track t JOIN Genre g USING(GenreId)
        GROUP BY g.GenreId ORDER BY n DESC LIMIT 1""")
    t["Q2"] = {"name": r[0][0], "count": r[0][1]}

    # Q3 — Metallica and U2 tie at 10 for 4th/5th, but 6th is 6, so the
    # top-5 *set* is unambiguous.
    r = _rows(conn, """
        SELECT ar.Name, COUNT(*) n FROM Album al JOIN Artist ar USING(ArtistId)
        GROUP BY ar.ArtistId ORDER BY n DESC LIMIT 5""")
    t["Q3"] = {"names": [x[0] for x in r], "counts": {x[0]: x[1] for x in r}}

    # Q4 — Considered BillingCountry vs Customer.Country: identical on this
    # data (0 mismatched invoices). Invoice.Total also equals
    # SUM(InvoiceLine.UnitPrice*Quantity), so either revenue definition works.
    r = _rows(conn, """
        SELECT BillingCountry, ROUND(SUM(Total), 2) rev FROM Invoice
        GROUP BY 1 ORDER BY rev DESC LIMIT 5""")
    t["Q4"] = {"names": [x[0] for x in r], "revenue": {x[0]: x[1] for x in r}}

    # Q5 — "Sales support agent" = Employee.Title 'Sales Support Agent',
    # linked via Customer.SupportRepId.
    r = _rows(conn, """
        SELECT e.FirstName || ' ' || e.LastName, ROUND(SUM(i.Total), 2) rev
        FROM Employee e JOIN Customer c ON c.SupportRepId = e.EmployeeId
        JOIN Invoice i USING(CustomerId) GROUP BY e.EmployeeId ORDER BY rev DESC""")
    t["Q5"] = {"name": r[0][0], "revenue": r[0][1], "all": dict(r)}

    # Q6 — Two-part answer: best year, plus change vs the prior year.
    r = _rows(conn, """
        SELECT CAST(strftime('%Y', InvoiceDate) AS INT) yr, ROUND(SUM(Total), 2)
        FROM Invoice GROUP BY yr ORDER BY yr""")
    by_year = dict(r)
    best = max(by_year, key=by_year.get)
    prev = by_year.get(best - 1)
    t["Q6"] = {
        "year": best,
        "revenue": by_year[best],
        "prev_year": best - 1,
        "prev_revenue": prev,
        "abs_change": round(by_year[best] - prev, 2) if prev else None,
        "pct_change": round((by_year[best] - prev) / prev * 100, 2) if prev else None,
        "by_year": by_year,
    }

    # Q7 — Milliseconds -> minutes. Video is ~39 min; audio ~4.3–4.7 min.
    r = _rows(conn, """
        SELECT m.Name, AVG(t.Milliseconds) / 60000.0 FROM Track t
        JOIN MediaType m USING(MediaTypeId) GROUP BY m.MediaTypeId""")
    t["Q7"] = {"avg_minutes": {name: round(v, 2) for name, v in r}}

    # Q8 — The hardest one. 41 tracks tie at 5 playlists each. Playlist
    # names are duplicated ("Music" is id 1 and 8, also "TV Shows", "Movies",
    # "Audiobooks"), so counting distinct playlist *names* gives 4 instead of
    # 5 — but the same 41 tracks either way. A correct answer must
    # acknowledge the tie rather than name a single "top" track.
    r = _rows(conn, """
        WITH c AS (SELECT TrackId, COUNT(DISTINCT PlaylistId) n
                   FROM PlaylistTrack GROUP BY TrackId)
        SELECT t.Name, c.n FROM c JOIN Track t USING(TrackId)
        WHERE c.n = (SELECT MAX(n) FROM c)""")
    by_name = _rows(conn, """
        SELECT MAX(n) FROM (SELECT COUNT(DISTINCT p.Name) n FROM PlaylistTrack pt
                            JOIN Playlist p USING(PlaylistId) GROUP BY pt.TrackId)""")
    t["Q8"] = {
        "max_playlists": r[0][1],
        "max_playlists_by_distinct_name": by_name[0][0],
        "tied_count": len(r),
        "tied_names": [x[0] for x in r],
    }

    # Q9 — "Best" is ambiguous. By spend: Helena Holý, a clear winner
    # ($49.62 vs $47.62). By invoice count: 58-way tie at 7 (meaningless).
    # By tracks bought: 3-way tie at 38. Spend is the only interpretation
    # that yields an answer, and the agent must state it as an assumption.
    r = _rows(conn, """
        SELECT c.FirstName || ' ' || c.LastName, ROUND(SUM(i.Total), 2) spend
        FROM Customer c JOIN Invoice i USING(CustomerId)
        GROUP BY c.CustomerId ORDER BY spend DESC LIMIT 2""")
    t["Q9"] = {"name": r[0][0], "spend": r[0][1], "runner_up": r[1][0]}

    # Q10 — Unanswerable. The only money columns are Track.UnitPrice and
    # InvoiceLine.UnitPrice (sale price; never differs from list price).
    # There is no cost data, so margin cannot be computed. Expect
    # answerable=false, or an explicit statement that cost data is missing.
    t["Q10"] = {"answerable": False}

    conn.close()
    return t


if __name__ == "__main__":
    import json
    print(json.dumps(compute(), indent=2, ensure_ascii=False))
