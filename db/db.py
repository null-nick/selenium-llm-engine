import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List

DB_FILE = Path(os.getenv("SELENIUM_LLM_DB", "./data/selenium_engine.db"))
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

DB_LOCK = threading.Lock()

CREATE_PROMPT_LOGS = """
CREATE TABLE IF NOT EXISTS prompt_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    engine TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    status TEXT NOT NULL,
    elapsed_ms INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_STATS = """
CREATE TABLE IF NOT EXISTS stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value INTEGER NOT NULL
)
"""


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_FILE), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(CREATE_PROMPT_LOGS)
            cur.execute(CREATE_STATS)
            conn.commit()
        finally:
            conn.close()


def log_prompt(
    engine: str, model: str, prompt: str, response: str, status: str, elapsed_ms: int
) -> None:
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO prompt_logs (engine, model, prompt, response, status, elapsed_ms) VALUES (?, ?, ?, ?, ?, ?)",
                (engine, model, prompt, response, status, elapsed_ms),
            )
            conn.commit()
        finally:
            conn.close()


def get_prompt_logs(
    limit: int = 100,
    offset: int = 0,
    engine: str | None = None,
    model: str | None = None,
    status: str | None = None,
) -> List[Dict[str, Any]]:
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            query = "SELECT * FROM prompt_logs"
            where_clauses = []
            params: list[Any] = []

            if engine:
                where_clauses.append("engine = ?")
                params.append(engine)
            if model:
                where_clauses.append("model = ?")
                params.append(model)
            if status:
                where_clauses.append("status = ?")
                params.append(status)

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            return [dict(x) for x in rows]
        finally:
            conn.close()


def _inc_stat(key: str, amount: int = 1) -> None:
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO stats (key, value) VALUES (?, ?)", (key, 0)
            )
            cur.execute(
                "UPDATE stats SET value = value + ? WHERE key = ?", (amount, key)
            )
            conn.commit()
        finally:
            conn.close()


def inc_requests() -> None:
    _inc_stat("requests", 1)


def inc_responses() -> None:
    _inc_stat("responses", 1)


def inc_errors() -> None:
    _inc_stat("errors", 1)


def get_stats() -> Dict[str, int]:
    with DB_LOCK:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT key, value FROM stats")
            return {row[0]: row[1] for row in cur.fetchall()}
        finally:
            conn.close()
