"""SQLite storage helpers shared by memory servers."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    _ensure_fts(conn)


def _ensure_fts(conn: sqlite3.Connection) -> None:
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
    ).fetchone()
    if not table_exists:
        conn.execute(
            """
            CREATE VIRTUAL TABLE memories_fts
            USING fts5(content, metadata, tags)
            """
        )
        conn.commit()


def _build_metadata_dict(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Return metadata in a canonical form with optional hierarchy path."""

    normalised: Dict[str, Any] = {}

    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("Metadata keys must be strings")
        normalised[key] = value

    path: List[str] = []

    if "hierarchy" in metadata:
        hierarchy = metadata["hierarchy"]
        path_source: Optional[Sequence[Any]] = None

        if isinstance(hierarchy, Mapping):
            if "path" in hierarchy and hierarchy["path"] is not None:
                path_source = hierarchy["path"]
            else:
                collected: List[Any] = []
                for key in ("section", "subsection"):
                    if key in hierarchy and hierarchy[key] is not None:
                        collected.append(hierarchy[key])
                if collected:
                    path_source = collected
        elif isinstance(hierarchy, Sequence) and not isinstance(hierarchy, (str, bytes)):
            path_source = hierarchy
        else:
            raise ValueError("metadata['hierarchy'] must be a mapping or sequence")

        if path_source is None:
            raise ValueError("metadata['hierarchy'] must define a path")

        try:
            path = [str(part) for part in path_source if part is not None]
        except TypeError as exc:
            raise ValueError("metadata['hierarchy'] path must be iterable") from exc

    else:
        if "section" in metadata and metadata["section"] is not None:
            path.append(str(metadata["section"]))
        if "subsection" in metadata and metadata["subsection"] is not None:
            path.append(str(metadata["subsection"]))

    # Always rewrite hierarchy to the canonical form
    normalised.pop("hierarchy", None)

    if path:
        normalised["hierarchy"] = {"path": path}
        normalised["section"] = path[0]
        if len(path) > 1:
            normalised["subsection"] = path[1]
        else:
            normalised.pop("subsection", None)
    else:
        normalised.pop("section", None)
        normalised.pop("subsection", None)

    return normalised


def _prepare_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("Metadata must be a mapping")
    return _build_metadata_dict(metadata)


def _present_metadata(metadata: Optional[Any]) -> Optional[Any]:
    if metadata is None:
        return None
    if isinstance(metadata, Mapping):
        try:
            return _build_metadata_dict(metadata)
        except ValueError:
            # Surface legacy/invalid metadata without breaking callers
            return dict(metadata)
    return metadata


def _metadata_matches_filters(metadata: Optional[Any], filters: Mapping[str, Any]) -> bool:
    if not filters:
        return True

    canonical: Dict[str, Any] = {}
    if isinstance(metadata, Mapping):
        canonical = _present_metadata(metadata) or {}
    elif metadata is None:
        canonical = {}
    else:
        canonical = {"value": metadata}

    hierarchy_entry = canonical.get("hierarchy")
    hierarchy_path: List[str] = []
    if isinstance(hierarchy_entry, Mapping):
        path_value = hierarchy_entry.get("path")
        if isinstance(path_value, Sequence) and not isinstance(path_value, (str, bytes)):
            hierarchy_path = [str(part) for part in path_value]

    for key, expected in filters.items():
        if key == "section":
            if canonical.get("section") != expected:
                return False
        elif key == "subsection":
            if canonical.get("subsection") != expected:
                return False
        elif key in {"hierarchy", "hierarchy_path"}:
            if isinstance(expected, str):
                if expected not in hierarchy_path:
                    return False
            elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
                expected_list = [str(part) for part in expected]
                if hierarchy_path[: len(expected_list)] != expected_list:
                    return False
            else:
                return False
        else:
            if canonical.get(key) != expected:
                return False

    return True


def _validate_metadata_filters(metadata_filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if metadata_filters is None:
        return {}
    if not isinstance(metadata_filters, Mapping):
        raise ValueError("metadata_filters must be a mapping")
    validated: Dict[str, Any] = {}
    for key, value in metadata_filters.items():
        if not isinstance(key, str):
            raise ValueError("metadata_filters keys must be strings")
        validated[key] = value
    return validated


def _fts_enabled(conn: sqlite3.Connection) -> bool:
    return bool(
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone()
    )


def _fts_upsert(
    conn: sqlite3.Connection,
    memory_id: int,
    content: str,
    metadata_json: Optional[str],
    tags_json: Optional[str],
) -> None:
    if not _fts_enabled(conn):
        return
    conn.execute(
        "INSERT OR REPLACE INTO memories_fts(rowid, content, metadata, tags) VALUES (?, ?, ?, ?)",
        (
            memory_id,
            content,
            metadata_json or "",
            tags_json or "",
        ),
    )


def _fts_delete(conn: sqlite3.Connection, memory_id: int) -> None:
    if not _fts_enabled(conn):
        return
    conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (memory_id,))


def _serialise_row(row: sqlite3.Row) -> Dict[str, Any]:
    metadata = row["metadata"]
    tags = row["tags"]
    return {
        "id": row["id"],
        "content": row["content"],
        "metadata": _present_metadata(json.loads(metadata)) if metadata else None,
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
    prepared_metadata = _prepare_metadata(metadata)
    metadata_json = json.dumps(prepared_metadata, ensure_ascii=False) if prepared_metadata else None
    tags_json = json.dumps(tags or [], ensure_ascii=False)
    cur = conn.execute(
        "INSERT INTO memories (content, metadata, tags) VALUES (?, ?, ?)",
        (
            content,
            metadata_json,
            tags_json,
        ),
    )
    memory_id = cur.lastrowid
    _fts_upsert(conn, memory_id, content, metadata_json, tags_json)
    conn.commit()
    return get_memory(conn, memory_id)


def add_memories(
    conn: sqlite3.Connection,
    entries: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    prepared: List[tuple[str, Optional[str], Optional[str]]] = []

    for entry in entries:
        if "content" not in entry:
            raise ValueError("Each batch entry must include 'content'")
        content = str(entry["content"]).strip()
        metadata = entry.get("metadata")
        tags = entry.get("tags") or []
        prepared_metadata = _prepare_metadata(metadata)
        metadata_json = json.dumps(prepared_metadata, ensure_ascii=False) if prepared_metadata else None
        tags_json = json.dumps(tags, ensure_ascii=False)
        prepared.append((content, metadata_json, tags_json))
        rows.append({
            "content": content,
            "metadata_json": metadata_json,
            "tags_json": tags_json,
        })

    if not prepared:
        return []

    cur = conn.executemany(
        "INSERT INTO memories (content, metadata, tags) VALUES (?, ?, ?)",
        prepared,
    )

    # SQLite returns the cursor of the last execute; capture inserted IDs manually
    start_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    inserted: List[int] = list(range(start_id - len(prepared) + 1, start_id + 1))

    for memory_id, entry in zip(inserted, rows):
        _fts_upsert(conn, memory_id, entry["content"], entry["metadata_json"], entry["tags_json"])

    conn.commit()
    return [get_memory(conn, memory_id) for memory_id in inserted]


def get_memory(conn: sqlite3.Connection, memory_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT id, content, metadata, tags, created_at FROM memories WHERE id = ?",
        (memory_id,),
    ).fetchone()
    return _serialise_row(row) if row else None


def delete_memory(conn: sqlite3.Connection, memory_id: int) -> bool:
    _fts_delete(conn, memory_id)
    cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    return cur.rowcount > 0


def delete_memories(conn: sqlite3.Connection, memory_ids: Iterable[int]) -> int:
    ids = list(memory_ids)
    if not ids:
        return 0
    for memory_id in ids:
        _fts_delete(conn, memory_id)
    conn.execute(
        f"DELETE FROM memories WHERE id IN ({','.join('?' for _ in ids)})",
        ids,
    )
    conn.commit()
    return len(ids)


def list_memories(
    conn: sqlite3.Connection,
    query: Optional[str] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    validated_filters = _validate_metadata_filters(metadata_filters)

    rows: List[sqlite3.Row]

    if query and _fts_enabled(conn):
        # Use full-text search when available. Fall back to LIKE if the query fails.
        try:
            rows = conn.execute(
                """
                SELECT m.id, m.content, m.metadata, m.tags, m.created_at
                FROM memories m
                JOIN memories_fts f ON m.id = f.rowid
                WHERE f MATCH ?
                ORDER BY m.created_at DESC
                """,
                (query,),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    elif query:
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

    # If the FTS search yielded nothing because of an SQLite error (e.g. malformed query)
    # fall back to a LIKE search for resilience.
    if query and _fts_enabled(conn) and not rows:
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

    records: List[Dict[str, Any]] = []
    for row in rows:
        record = _serialise_row(row)
        if validated_filters and not _metadata_matches_filters(record.get("metadata"), validated_filters):
            continue
        records.append(record)

    return records
