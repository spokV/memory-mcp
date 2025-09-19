"""SQLite storage helpers shared by memory servers."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "memories.db"


def connect(*, check_same_thread: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            metadata TEXT,
            tags TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def _serialise_row(row: sqlite3.Row) -> Dict[str, Any]:
    metadata = row["metadata"]
    tags = row["tags"]
    return {
        "id": row["id"],
        "content": row["content"],
        "metadata": json.loads(metadata) if metadata else None,
        "tags": json.loads(tags) if tags else [],
        "created_at": row["created_at"],
    }


def add_memory(
    conn: sqlite3.Connection,
    *,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cur = conn.execute(
        "INSERT INTO memories (content, metadata, tags) VALUES (?, ?, ?)",
        (
            content,
            json.dumps(metadata, ensure_ascii=False) if metadata else None,
            json.dumps(tags or [], ensure_ascii=False),
        ),
    )
    conn.commit()
    return get_memory(conn, cur.lastrowid)


def get_memory(conn: sqlite3.Connection, memory_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT id, content, metadata, tags, created_at FROM memories WHERE id = ?",
        (memory_id,),
    ).fetchone()
    return _serialise_row(row) if row else None


def delete_memory(conn: sqlite3.Connection, memory_id: int) -> bool:
    cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    return cur.rowcount > 0


def list_memories(conn: sqlite3.Connection, query: Optional[str] = None) -> List[Dict[str, Any]]:
    if query:
        pattern = f"%{query}%"
        rows = conn.execute(
            """
            SELECT id, content, metadata, tags, created_at
            FROM memories
            WHERE content LIKE ? OR tags LIKE ? OR metadata LIKE ?
            ORDER BY created_at DESC
            """,
            (pattern, pattern, pattern),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, content, metadata, tags, created_at FROM memories ORDER BY created_at DESC"
        ).fetchall()
    return [_serialise_row(row) for row in rows]
