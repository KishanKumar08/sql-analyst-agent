"""
Build a small non-Chinook database to check the agent isn't Chinook-specific.

A SaaS subscriptions business, with deliberate awkwardness the agent has to
cope with: a table name with a space, no declared foreign keys, NULLs, a
view, dates stored as text, and money stored in cents.

python scripts/make_demo_db.py   ->  data/saas_demo.sqlite

Try:  agent ask "Which plan brings in the most monthly revenue?" --db data/saas_demo.sqlite
"""
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "data" / "saas_demo.sqlite"


def main():
    DEST.parent.mkdir(exist_ok=True)
    DEST.unlink(missing_ok=True)
    rnd = random.Random(42)
    c = sqlite3.connect(DEST)
    c.executescript("""
        CREATE TABLE companies (id INTEGER PRIMARY KEY, name TEXT, industry TEXT, country TEXT,
                                signup_date TEXT);
        CREATE TABLE plans (code TEXT PRIMARY KEY, label TEXT, monthly_price_cents INTEGER);
        CREATE TABLE "subscription events" (event_id INTEGER PRIMARY KEY, company_id INTEGER,
                                plan_code TEXT, event_type TEXT, event_date TEXT, seats INTEGER);
        CREATE TABLE support_tickets (id INTEGER PRIMARY KEY, company_id INTEGER, opened TEXT,
                                closed TEXT, priority TEXT);
    """)
    c.executemany("INSERT INTO plans VALUES (?,?,?)",
                  [("FREE", "Free", 0), ("TEAM", "Team", 2900), ("BIZ", "Business", 9900),
                   ("ENT", "Enterprise", 49900)])
    industries = ["Fintech", "Health", "Retail", "Education", None]
    countries = ["India", "USA", "Germany", "Brazil", "UK"]
    start = date(2023, 1, 1)
    ev = 0
    for cid in range(1, 121):
        signup = start + timedelta(days=rnd.randint(0, 600))
        c.execute("INSERT INTO companies VALUES (?,?,?,?,?)",
                  (cid, f"Company {cid:03d}", rnd.choice(industries), rnd.choice(countries), signup.isoformat()))
        plan = rnd.choices(["FREE", "TEAM", "BIZ", "ENT"], [40, 35, 20, 5])[0]
        d = signup
        ev += 1
        c.execute('INSERT INTO "subscription events" VALUES (?,?,?,?,?,?)',
                  (ev, cid, plan, "start", d.isoformat(), rnd.randint(1, 50)))
        for _ in range(rnd.randint(0, 3)):
            d += timedelta(days=rnd.randint(30, 200))
            kind = rnd.choice(["upgrade", "downgrade", "cancel"])
            ev += 1
            c.execute('INSERT INTO "subscription events" VALUES (?,?,?,?,?,?)',
                      (ev, cid, rnd.choice(["TEAM", "BIZ", "ENT"]), kind, d.isoformat(), rnd.randint(1, 80)))
            if kind == "cancel":
                break
        for _ in range(rnd.randint(0, 6)):
            opened = signup + timedelta(days=rnd.randint(0, 500))
            closed = None if rnd.random() < 0.15 else (opened + timedelta(days=rnd.randint(0, 20))).isoformat()
            c.execute("INSERT INTO support_tickets (company_id, opened, closed, priority) VALUES (?,?,?,?)",
                      (cid, opened.isoformat(), closed, rnd.choice(["low", "normal", "high", "urgent"])))
    c.execute("""CREATE VIEW open_tickets AS
                 SELECT * FROM support_tickets WHERE closed IS NULL""")
    c.commit()
    c.close()
    print(f"ok: {DEST}")


if __name__ == "__main__":
    main()
