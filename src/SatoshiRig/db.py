"""Simple SQLite-backed key-value storage for SatoshiRig."""

import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.environ.get("STATE_DB", os.path.join(os.getcwd(), "data", "state.db"))


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv_store (
                section TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY(section, key)
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_value(section: str, key: str, default: Optional[str] = None) -> Optional[str]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT value FROM kv_store WHERE section = ? AND key = ?",
            (section, key),
        )
        row = cur.fetchone()
        return row[0] if row else default


def set_value(section: str, key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO kv_store(section, key, value)
            VALUES(?, ?, ?)
            ON CONFLICT(section, key)
            DO UPDATE SET value = excluded.value
            """,
            (section, key, value),
        )


def delete_value(section: str, key: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM kv_store WHERE section = ? AND key = ?",
            (section, key),
        )


def get_section(section: str) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT key, value FROM kv_store WHERE section = ?",
            (section,),
        )
        return {row[0]: row[1] for row in cur.fetchall()}
