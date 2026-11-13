import sqlite3
from contextlib import closing
from pathlib import Path
from datetime import datetime, timezone
from risk_engine.io import ROOT

DB = ROOT / "database" / "audit.db"

def record_decision(instrument, decision, reviewer, comment, evidence):
    DB.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB)) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, instrument TEXT,
            decision TEXT, reviewer TEXT, comment TEXT, evidence_json TEXT)''')
        import json
        conn.execute("INSERT INTO decisions(timestamp,instrument,decision,reviewer,comment,evidence_json) VALUES(?,?,?,?,?,?)",
                     (datetime.now(timezone.utc).isoformat(), instrument, decision,
                      reviewer, comment, json.dumps(evidence, default=str)))
        conn.commit()

def get_decisions():
    if not DB.exists():
        return []
    with closing(sqlite3.connect(DB)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM decisions ORDER BY id DESC")]
