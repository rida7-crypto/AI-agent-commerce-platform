"""
Audit trail: every step the agent takes gets logged here so it can be
shown to the user (and to judges) as an explainable, timestamped trail.
"""

import sqlite3
import json
import time
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "audit.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            step TEXT,
            detail TEXT,
            timestamp REAL
        )
        """
    )
    return conn


def log_step(session_id: str, step: str, detail: dict):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO audit_log (session_id, step, detail, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, step, json.dumps(detail, default=str), time.time()),
    )
    conn.commit()
    conn.close()


def get_trail(session_id: str):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT step, detail, timestamp FROM audit_log WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"step": r[0], "detail": json.loads(r[1]), "timestamp": r[2]} for r in rows]
