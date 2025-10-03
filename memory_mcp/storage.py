"""SQLite storage helpers shared by memory servers."""
from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence as TypingSequence

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
    _ensure_embeddings_table(conn)
    _ensure_crossrefs_table(conn)


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


def _ensure_embeddings_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_embeddings (
            memory_id INTEGER PRIMARY KEY,
            embedding TEXT,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def _ensure_crossrefs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_crossrefs (
            memory_id INTEGER PRIMARY KEY,
            related TEXT,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def _build_metadata_dict(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Return metadata in a canonical form with optional hierarchy path."""

    normalised: Dict[str, Any] = {}

    for key in metadata.keys():
        if not isinstance(key, str):
            raise ValueError("Metadata keys must be strings")

    tasks_value = metadata.get("tasks")
    done_present = "done" in metadata
    done_value = metadata.get("done")

    for key, value in metadata.items():
        if key in {"tasks", "done", "hierarchy", "section", "subsection"}:
            continue
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

    if tasks_value is not None:
        normalised["tasks"] = _normalise_tasks(tasks_value)

    if done_present:
        normalised["done"] = _coerce_bool(done_value) if done_value is not None else False

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


TRUE_STRINGS = {"true", "1", "yes", "y", "on"}
FALSE_STRINGS = {"false", "0", "no", "n", "off"}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_STRINGS:
            return True
        if lowered in FALSE_STRINGS:
            return False
        raise ValueError("Boolean strings must be true/false, yes/no, on/off, or 1/0")
    raise ValueError("Boolean fields must be bool-like values")


def _normalise_tasks(tasks: Any) -> List[Dict[str, Any]]:
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, TypingSequence):
        raise ValueError("metadata['tasks'] must be a sequence of task entries")

    normalised: List[Dict[str, Any]] = []

    for index, item in enumerate(tasks):
        if isinstance(item, Mapping):
            if "title" not in item:
                raise ValueError(f"Task at index {index} must include a 'title'")
            title = str(item["title"]).strip()
            if not title:
                raise ValueError(f"Task at index {index} must provide a non-empty title")
            task_entry: Dict[str, Any] = {"title": title}
            if "done" in item and item["done"] is not None:
                try:
                    task_entry["done"] = _coerce_bool(item["done"])
                except ValueError as exc:
                    raise ValueError(
                        f"Task at index {index} has an invalid 'done' flag"
                    ) from exc
            else:
                task_entry["done"] = False
            for key, value in item.items():
                if key in {"title", "done"}:
                    continue
                task_entry[key] = value
        elif isinstance(item, str):
            title = item.strip()
            if not title:
                raise ValueError(f"Task at index {index} must provide a non-empty title")
            task_entry = {"title": title, "done": False}
        else:
            raise ValueError(
                "metadata['tasks'] entries must be mappings with 'title' or plain strings"
            )
        normalised.append(task_entry)

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


def _validate_tags(tags: Optional[Iterable[str]]) -> List[str]:
    if tags is None:
        return []
    validated: List[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError("Tags must be strings")
        stripped = tag.strip()
        if not stripped:
            raise ValueError("Tags cannot be empty strings")
        validated.append(stripped)
    return validated


def _enforce_tag_whitelist(tags: List[str]) -> None:
    from . import TAG_WHITELIST

    if not TAG_WHITELIST:
        return

    explicit = {tag for tag in TAG_WHITELIST if not tag.endswith('.*')}
    wildcards = [tag[:-2] for tag in TAG_WHITELIST if tag.endswith('.*')]

    for tag in tags:
        if tag in explicit:
            continue
        if any(tag == prefix or tag.startswith(prefix + '.') for prefix in wildcards):
            continue
        raise ValueError(f"Tag '{tag}' is not in the allowed tag list")


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _compute_embedding(
    content: str,
    metadata: Optional[Dict[str, Any]],
    tags: List[str],
) -> Dict[str, float]:
    parts: List[str] = [content]

    if metadata:
        try:
            metadata_str = json.dumps(metadata, ensure_ascii=False)
        except (TypeError, ValueError):
            metadata_str = str(metadata)
        parts.append(metadata_str)

    if tags:
        parts.append(" ".join(tags))

    joined = " \n ".join(parts).lower()
    tokens = _TOKEN_RE.findall(joined)
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = sum(counts.values())
    if not total:
        return {}

    return {token: count / total for token, count in counts.items()}


def _embedding_to_json(vector: Dict[str, float]) -> Optional[str]:
    if not vector:
        return None
    items = sorted(vector.items())
    return json.dumps(items, ensure_ascii=False)


def _json_to_embedding(data: Optional[str]) -> Dict[str, float]:
    if not data:
        return {}
    try:
        items = json.loads(data)
    except json.JSONDecodeError:
        return {}
    if isinstance(items, list):
        return {str(token): float(weight) for token, weight in items}
    return {}


def _embedding_norm(vector: Dict[str, float]) -> float:
    return math.sqrt(sum(weight * weight for weight in vector.values()))


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    for token, weight in vec_a.items():
        dot += weight * vec_b.get(token, 0.0)
    norm_a = _embedding_norm(vec_a)
    norm_b = _embedding_norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _upsert_embedding(
    conn: sqlite3.Connection,
    memory_id: int,
    vector: Dict[str, float],
) -> None:
    embedding_json = _embedding_to_json(vector)
    conn.execute(
        """
        INSERT INTO memories_embeddings(memory_id, embedding)
        VALUES(?, ?)
        ON CONFLICT(memory_id) DO UPDATE SET embedding=excluded.embedding
        """,
        (memory_id, embedding_json),
    )


def _delete_embedding(conn: sqlite3.Connection, memory_id: int) -> None:
    conn.execute("DELETE FROM memories_embeddings WHERE memory_id = ?", (memory_id,))


def _get_embeddings_for_ids(
    conn: sqlite3.Connection,
    memory_ids: List[int],
) -> Dict[int, Dict[str, float]]:
    if not memory_ids:
        return {}
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"SELECT memory_id, embedding FROM memories_embeddings WHERE memory_id IN ({placeholders})",
        memory_ids,
    ).fetchall()
    mapping: Dict[int, Dict[str, float]] = {}
    for row in rows:
        mapping[row["memory_id"]] = _json_to_embedding(row["embedding"])
    return mapping


def _search_by_vector(
    conn: sqlite3.Connection,
    vector_query: Dict[str, float],
    *,
    metadata_filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = 5,
    min_score: Optional[float] = None,
    exclude_ids: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    exclude_set = set(exclude_ids or [])

    candidates = list_memories(conn, query=None, metadata_filters=metadata_filters)
    filtered = [record for record in candidates if record["id"] not in exclude_set]

    ids = [record["id"] for record in filtered]
    embeddings = _get_embeddings_for_ids(conn, ids)

    results: List[Dict[str, Any]] = []
    for record in filtered:
        memory_id = record["id"]
        vector = embeddings.get(memory_id)
        if vector is None:
            vector = _compute_embedding(
                record["content"],
                record.get("metadata"),
                record.get("tags", []),
            )
            _upsert_embedding(conn, memory_id, vector)
        score = _cosine_similarity(vector_query, vector)
        if min_score is not None and score < min_score:
            continue
        results.append({"score": score, "memory": record})

    results.sort(key=lambda entry: entry["score"], reverse=True)
    if top_k is not None:
        results = results[: top_k]
    return results


def _store_crossrefs(
    conn: sqlite3.Connection,
    memory_id: int,
    related: List[Dict[str, Any]],
) -> None:
    related_json = json.dumps(related, ensure_ascii=False) if related else None
    conn.execute(
        """
        INSERT INTO memories_crossrefs(memory_id, related)
        VALUES(?, ?)
        ON CONFLICT(memory_id) DO UPDATE SET related=excluded.related
        """,
        (memory_id, related_json),
    )


def _clear_crossrefs(conn: sqlite3.Connection, memory_id: int) -> None:
    conn.execute("DELETE FROM memories_crossrefs WHERE memory_id = ?", (memory_id,))


def get_crossrefs(conn: sqlite3.Connection, memory_id: int) -> List[Dict[str, Any]]:
    row = conn.execute(
        "SELECT related FROM memories_crossrefs WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    if not row or not row["related"]:
        return []
    try:
        data = json.loads(row["related"])
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    return []


def _update_crossrefs_for_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    vector: Optional[Dict[str, float]] = None,
    top_k: int = 5,
    min_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if vector is None:
        embeddings = _get_embeddings_for_ids(conn, [memory_id])
        vector = embeddings.get(memory_id)
        if vector is None:
            record = get_memory(conn, memory_id)
            if record is None:
                return []
            vector = _compute_embedding(
                record["content"],
                record.get("metadata"),
                record.get("tags", []),
            )
            _upsert_embedding(conn, memory_id, vector)

    results = _search_by_vector(
        conn,
        vector,
        metadata_filters=None,
        top_k=top_k,
        min_score=min_score,
        exclude_ids=[memory_id],
    )

    related = [
        {"id": item["memory"]["id"], "score": item["score"]}
        for item in results
    ]
    _store_crossrefs(conn, memory_id, related)
    return related


def _update_crossrefs(conn: sqlite3.Connection, memory_id: int) -> None:
    related = _update_crossrefs_for_memory(conn, memory_id)
    for item in related:
        _update_crossrefs_for_memory(conn, item["id"])


def rebuild_crossrefs(conn: sqlite3.Connection) -> int:
    rows = conn.execute("SELECT id FROM memories").fetchall()
    total = 0
    for row in rows:
        memory_id = row["id"]
        _update_crossrefs_for_memory(conn, memory_id)
        total += 1
    conn.commit()
    return total


def update_crossrefs(conn: sqlite3.Connection, memory_id: int) -> None:
    _update_crossrefs(conn, memory_id)


def _remove_memory_from_crossrefs(conn: sqlite3.Connection, memory_id: int) -> None:
    rows = conn.execute("SELECT memory_id, related FROM memories_crossrefs").fetchall()
    for row in rows:
        related = []
        if row["related"]:
            try:
                related = json.loads(row["related"])
            except json.JSONDecodeError:
                related = []
        filtered = [entry for entry in related if entry.get("id") != memory_id]
        if len(filtered) != len(related):
            _store_crossrefs(conn, row["memory_id"], filtered)


def add_memory(
    conn: sqlite3.Connection,
    *,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    prepared_metadata = _prepare_metadata(metadata)
    validated_tags = _validate_tags(tags)
    _enforce_tag_whitelist(validated_tags)
    metadata_json = json.dumps(prepared_metadata, ensure_ascii=False) if prepared_metadata else None
    tags_json = json.dumps(validated_tags, ensure_ascii=False)
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
    vector = _compute_embedding(content, prepared_metadata, validated_tags)
    _upsert_embedding(conn, memory_id, vector)
    _update_crossrefs(conn, memory_id)
    conn.commit()
    return get_memory(conn, memory_id)


def add_memories(
    conn: sqlite3.Connection,
    entries: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    embeddings: List[Dict[str, float]] = []
    prepared: List[tuple[str, Optional[str], Optional[str]]] = []

    for entry in entries:
        if "content" not in entry:
            raise ValueError("Each batch entry must include 'content'")
        content = str(entry["content"]).strip()
        metadata = entry.get("metadata")
        tags = entry.get("tags") or []
        prepared_metadata = _prepare_metadata(metadata)
        validated_tags = _validate_tags(tags)
        _enforce_tag_whitelist(validated_tags)
        metadata_json = json.dumps(prepared_metadata, ensure_ascii=False) if prepared_metadata else None
        tags_json = json.dumps(validated_tags, ensure_ascii=False)
        prepared.append((content, metadata_json, tags_json))
        rows.append({
            "content": content,
            "metadata_json": metadata_json,
            "tags_json": tags_json,
        })
        embeddings.append(_compute_embedding(content, prepared_metadata, validated_tags))

    if not prepared:
        return []

    cur = conn.executemany(
        "INSERT INTO memories (content, metadata, tags) VALUES (?, ?, ?)",
        prepared,
    )

    # SQLite returns the cursor of the last execute; capture inserted IDs manually
    start_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    inserted: List[int] = list(range(start_id - len(prepared) + 1, start_id + 1))

    for memory_id, entry, vector in zip(inserted, rows, embeddings):
        _fts_upsert(conn, memory_id, entry["content"], entry["metadata_json"], entry["tags_json"])
        _upsert_embedding(conn, memory_id, vector)
        _update_crossrefs(conn, memory_id)

    conn.commit()
    return [get_memory(conn, memory_id) for memory_id in inserted]


def get_memory(conn: sqlite3.Connection, memory_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT id, content, metadata, tags, created_at FROM memories WHERE id = ?",
        (memory_id,),
    ).fetchone()
    if not row:
        return None
    record = _serialise_row(row)
    record["related"] = get_crossrefs(conn, memory_id)
    return record


def delete_memory(conn: sqlite3.Connection, memory_id: int) -> bool:
    _fts_delete(conn, memory_id)
    _delete_embedding(conn, memory_id)
    _clear_crossrefs(conn, memory_id)
    _remove_memory_from_crossrefs(conn, memory_id)
    cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    return cur.rowcount > 0


def delete_memories(conn: sqlite3.Connection, memory_ids: Iterable[int]) -> int:
    ids = list(memory_ids)
    if not ids:
        return 0
    for memory_id in ids:
        _fts_delete(conn, memory_id)
        _delete_embedding(conn, memory_id)
        _clear_crossrefs(conn, memory_id)
        _remove_memory_from_crossrefs(conn, memory_id)
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
    limit: Optional[int] = None,
    offset: Optional[int] = 0,
) -> List[Dict[str, Any]]:
    validated_filters = _validate_metadata_filters(metadata_filters)

    rows: List[sqlite3.Row]

    # Build LIMIT/OFFSET clause
    limit_clause = ""
    limit_params = []
    if limit is not None:
        limit_clause = " LIMIT ?"
        limit_params.append(limit)
        if offset:
            limit_clause += " OFFSET ?"
            limit_params.append(offset)

    if query and _fts_enabled(conn):
        # Use full-text search when available. Fall back to LIKE if the query fails.
        try:
            rows = conn.execute(
                f"""
                SELECT m.id, m.content, m.metadata, m.tags, m.created_at
                FROM memories m
                JOIN memories_fts f ON m.id = f.rowid
                WHERE f MATCH ?
                ORDER BY m.created_at DESC{limit_clause}
                """,
                (query, *limit_params),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    elif query:
        pattern = f"%{query}%"
        rows = conn.execute(
            f"""
            SELECT id, content, metadata, tags, created_at
            FROM memories
            WHERE content LIKE ? OR tags LIKE ? OR metadata LIKE ?
            ORDER BY created_at DESC{limit_clause}
            """,
            (pattern, pattern, pattern, *limit_params),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT id, content, metadata, tags, created_at FROM memories ORDER BY created_at DESC{limit_clause}",
            tuple(limit_params),
        ).fetchall()

    # If the FTS search yielded nothing because of an SQLite error (e.g. malformed query)
    # fall back to a LIKE search for resilience.
    if query and _fts_enabled(conn) and not rows:
        pattern = f"%{query}%"
        rows = conn.execute(
            f"""
            SELECT id, content, metadata, tags, created_at
            FROM memories
            WHERE content LIKE ? OR tags LIKE ? OR metadata LIKE ?
            ORDER BY created_at DESC{limit_clause}
            """,
            (pattern, pattern, pattern, *limit_params),
        ).fetchall()

    records: List[Dict[str, Any]] = []
    for row in rows:
        record = _serialise_row(row)
        if validated_filters and not _metadata_matches_filters(record.get("metadata"), validated_filters):
            continue
        records.append(record)

    return records


def collect_all_tags(conn: sqlite3.Connection) -> List[str]:
    tags: set[str] = set()
    rows = conn.execute("SELECT tags FROM memories")
    for (tags_json,) in rows:
        if not tags_json:
            continue
        try:
            parsed = json.loads(tags_json)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            for tag in parsed:
                if isinstance(tag, str) and tag.strip():
                    tags.add(tag.strip())
    return sorted(tags)


def find_invalid_tag_entries(
    conn: sqlite3.Connection,
    allowlist: Iterable[str],
) -> List[Dict[str, Any]]:
    allowed = set(allowlist)
    if not allowed:
        return []

    explicit = {tag for tag in allowed if not tag.endswith('.*')}
    wildcards = [tag[:-2] for tag in allowed if tag.endswith('.*')]

    invalid: List[Dict[str, Any]] = []
    rows = conn.execute("SELECT id, tags FROM memories")
    for memory_id, tags_json in rows:
        if not tags_json:
            continue
        try:
            parsed = json.loads(tags_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        bad: List[str] = []
        for tag in parsed:
            if not isinstance(tag, str):
                continue
            if tag in explicit:
                continue
            if any(tag == prefix or tag.startswith(prefix + '.') for prefix in wildcards):
                continue
            bad.append(tag)
        if bad:
            invalid.append({"id": memory_id, "invalid_tags": bad})
    return invalid


def semantic_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    metadata_filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = 5,
    min_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    vector_query = _compute_embedding(query, None, [])
    if not vector_query:
        return []
    return _search_by_vector(
        conn,
        vector_query,
        metadata_filters=metadata_filters,
        top_k=top_k,
        min_score=min_score,
    )


def rebuild_embeddings(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "SELECT id, content, metadata, tags FROM memories"
    ).fetchall()
    updated = 0
    for row in rows:
        memory_id = row["id"]
        metadata = json.loads(row["metadata"]) if row["metadata"] else None
        tags = json.loads(row["tags"]) if row["tags"] else []
        vector = _compute_embedding(row["content"], metadata, tags)
        _upsert_embedding(conn, memory_id, vector)
        updated += 1
    conn.commit()
    return updated
