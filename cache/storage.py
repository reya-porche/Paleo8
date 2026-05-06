"""
geo_tvt/cache/storage.py
SQLite-backed cache so scraped data survives between runs.
When offline, the system falls back to cached results automatically.
"""

import sqlite3
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DB, CACHE_TTL_DAYS


def _hash_key(source: str, params: dict) -> str:
    raw = f"{source}:{json.dumps(params, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key         TEXT PRIMARY KEY,
            source      TEXT NOT NULL,
            params      TEXT NOT NULL,
            data        TEXT NOT NULL,
            fetched_at  REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def cache_get(source: str, params: dict) -> Optional[Any]:
    """Return cached data if it exists and is not stale."""
    key = _hash_key(source, params)
    conn = _get_conn()
    row = conn.execute(
        "SELECT data, fetched_at FROM cache WHERE key = ?", (key,)
    ).fetchone()
    conn.close()

    if row is None:
        return None

    fetched_at = datetime.fromtimestamp(row[1])
    if datetime.now() - fetched_at > timedelta(days=CACHE_TTL_DAYS):
        return None  # stale

    return json.loads(row[0])


def cache_set(source: str, params: dict, data: Any) -> None:
    """Store data in cache."""
    key = _hash_key(source, params)
    conn = _get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO cache (key, source, params, data, fetched_at)
        VALUES (?, ?, ?, ?, ?)
    """, (key, source, json.dumps(params, sort_keys=True), json.dumps(data), time.time()))
    conn.commit()
    conn.close()


def cache_stats() -> dict:
    """Return cache statistics."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT source, COUNT(*), MIN(fetched_at), MAX(fetched_at)
        FROM cache GROUP BY source
    """).fetchall()
    conn.close()
    return {
        r[0]: {
            "entries": r[1],
            "oldest":  datetime.fromtimestamp(r[2]).isoformat() if r[2] else None,
            "newest":  datetime.fromtimestamp(r[3]).isoformat() if r[3] else None,
        }
        for r in rows
    }


def cache_clear(source: Optional[str] = None) -> int:
    """Clear cache entries. Pass source to clear only one source."""
    conn = _get_conn()
    if source:
        n = conn.execute("DELETE FROM cache WHERE source = ?", (source,)).rowcount
    else:
        n = conn.execute("DELETE FROM cache").rowcount
    conn.commit()
    conn.close()
    return n
