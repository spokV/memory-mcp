"""SQLite storage helpers shared by memory servers."""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence as TypingSequence

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "memories.db"

# Embedding backend configuration
EMBEDDING_MODEL = os.getenv("MEMORY_MCP_EMBEDDING_MODEL", "tfidf")  # tfidf, sentence-transformers, openai

# Event notification configuration
EVENT_TRIGGER_TAG = "shared-cache"


def _emit_event(conn: sqlite3.Connection, memory_id: int, tags: List[str]) -> None:
    """Emit an event notification if memory has the trigger tag."""
    if EVENT_TRIGGER_TAG in tags:
        tags_json = json.dumps(tags, ensure_ascii=False)
        try:
            conn.execute(
                "INSERT INTO memories_events (memory_id, tags) VALUES (?, ?)",
                (memory_id, tags_json)
            )
            conn.commit()
        except Exception:
            # Don't fail memory operations if event emission fails
            pass


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
    _ensure_events_table(conn)


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


def _ensure_events_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            tags TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            consumed INTEGER DEFAULT 0,
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

# Cache for embedding models
_embedding_model_cache: Dict[str, Any] = {}


def _get_embedding_text(
    content: str,
    metadata: Optional[Dict[str, Any]],
    tags: List[str],
) -> str:
    """Combine content, metadata, and tags into a single text for embedding."""
    parts: List[str] = [content]

    if metadata:
        try:
            metadata_str = json.dumps(metadata, ensure_ascii=False)
        except (TypeError, ValueError):
            metadata_str = str(metadata)
        parts.append(metadata_str)

    if tags:
        parts.append(" ".join(tags))

    return " \n ".join(parts)


def _compute_embedding_tfidf(text: str) -> Dict[str, float]:
    """TF-IDF style bag-of-words embedding (default, no dependencies)."""
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = sum(counts.values())
    if not total:
        return {}

    return {token: count / total for token, count in counts.items()}


def _compute_embedding_sentence_transformers(text: str) -> Dict[str, float]:
    """Use sentence-transformers for better semantic embeddings."""
    try:
        if "sentence_transformers" not in _embedding_model_cache:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model by default
            model_name = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
            _embedding_model_cache["sentence_transformers"] = SentenceTransformer(model_name)

        model = _embedding_model_cache["sentence_transformers"]
        embedding = model.encode(text, convert_to_numpy=True)

        # Convert numpy array to dict for storage (use indices as keys)
        return {str(i): float(val) for i, val in enumerate(embedding)}

    except ImportError:
        # Fallback to TF-IDF if sentence-transformers not available
        return _compute_embedding_tfidf(text)
    except Exception:
        # Fallback on any error
        return _compute_embedding_tfidf(text)


def _compute_embedding_openai(text: str) -> Dict[str, float]:
    """Use OpenAI embeddings API."""
    try:
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback to TF-IDF if no API key
            return _compute_embedding_tfidf(text)

        if "openai_client" not in _embedding_model_cache:
            _embedding_model_cache["openai_client"] = openai.OpenAI(api_key=api_key)

        client = _embedding_model_cache["openai_client"]
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        response = client.embeddings.create(
            input=text,
            model=model_name,
        )

        embedding = response.data[0].embedding

        # Convert to dict for storage
        return {str(i): float(val) for i, val in enumerate(embedding)}

    except ImportError:
        # Fallback to TF-IDF if openai not available
        return _compute_embedding_tfidf(text)
    except Exception:
        # Fallback on any error (API error, rate limit, etc.)
        return _compute_embedding_tfidf(text)


def _compute_embedding(
    content: str,
    metadata: Optional[Dict[str, Any]],
    tags: List[str],
) -> Dict[str, float]:
    """Compute embedding using configured backend."""
    text = _get_embedding_text(content, metadata, tags)

    if EMBEDDING_MODEL == "sentence-transformers":
        return _compute_embedding_sentence_transformers(text)
    elif EMBEDDING_MODEL == "openai":
        return _compute_embedding_openai(text)
    else:  # Default to tfidf
        return _compute_embedding_tfidf(text)


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
    _emit_event(conn, memory_id, validated_tags)
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
            "validated_tags": validated_tags,
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

    # Emit events for memories with trigger tag
    for memory_id, entry in zip(inserted, rows):
        _emit_event(conn, memory_id, entry["validated_tags"])

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


def update_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    *,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Update an existing memory. Only provided fields are updated."""
    # First check if memory exists
    existing = get_memory(conn, memory_id)
    if not existing:
        return None

    # Determine what to update
    new_content = content.strip() if content is not None else existing["content"]
    new_metadata = _prepare_metadata(metadata) if metadata is not None else existing.get("metadata")
    new_tags = _validate_tags(tags) if tags is not None else existing.get("tags", [])

    if tags is not None:
        _enforce_tag_whitelist(new_tags)

    # Serialize for storage
    metadata_json = json.dumps(new_metadata, ensure_ascii=False) if new_metadata else None
    tags_json = json.dumps(new_tags, ensure_ascii=False)

    # Update the memory
    conn.execute(
        "UPDATE memories SET content = ?, metadata = ?, tags = ? WHERE id = ?",
        (new_content, metadata_json, tags_json, memory_id),
    )

    # Update FTS index
    _fts_upsert(conn, memory_id, new_content, metadata_json, tags_json)

    # Update embeddings
    vector = _compute_embedding(new_content, new_metadata, new_tags)
    _upsert_embedding(conn, memory_id, vector)

    # Update cross-references
    _update_crossrefs(conn, memory_id)

    conn.commit()
    _emit_event(conn, memory_id, new_tags)
    return get_memory(conn, memory_id)


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


def _parse_date_filter(date_str: str) -> str:
    """Parse date string to ISO format. Supports ISO dates and relative formats like '7d', '1m', '1y'."""
    if not date_str:
        return date_str

    # Try ISO format first
    try:
        parsed = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return parsed.strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass

    # Try relative formats: 7d, 1m, 1y, etc.
    match = re.match(r'^(\d+)([dmyDMY])$', date_str.strip())
    if match:
        value = int(match.group(1))
        unit = match.group(2).lower()

        now = datetime.utcnow()
        if unit == 'd':
            target = now - timedelta(days=value)
        elif unit == 'm':
            target = now - timedelta(days=value * 30)  # Approximate
        elif unit == 'y':
            target = now - timedelta(days=value * 365)  # Approximate
        else:
            raise ValueError(f"Unknown time unit: {unit}")

        return target.strftime('%Y-%m-%d %H:%M:%S')

    raise ValueError(f"Invalid date format: {date_str}")


def list_memories(
    conn: sqlite3.Connection,
    query: Optional[str] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = 0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
    tags_all: Optional[List[str]] = None,
    tags_none: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    validated_filters = _validate_metadata_filters(metadata_filters)

    rows: List[sqlite3.Row]

    # Parse date filters
    parsed_date_from = _parse_date_filter(date_from) if date_from else None
    parsed_date_to = _parse_date_filter(date_to) if date_to else None

    # Build date filter clauses (one with alias 'm.' for FTS, one without for regular queries)
    date_clause_fts = ""  # For FTS queries using alias 'm'
    date_clause_plain = ""  # For non-FTS queries
    date_params = []

    if parsed_date_from:
        date_clause_fts += " AND m.created_at >= ?"
        date_clause_plain += " AND created_at >= ?"
        date_params.append(parsed_date_from)
    if parsed_date_to:
        date_clause_fts += " AND m.created_at <= ?"
        date_clause_plain += " AND created_at <= ?"
        date_params.append(parsed_date_to)

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
                WHERE f MATCH ?{date_clause_fts}
                ORDER BY m.created_at DESC{limit_clause}
                """,
                (query, *date_params, *limit_params),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    elif query:
        pattern = f"%{query}%"
        rows = conn.execute(
            f"""
            SELECT id, content, metadata, tags, created_at
            FROM memories
            WHERE (content LIKE ? OR tags LIKE ? OR metadata LIKE ?){date_clause_plain}
            ORDER BY created_at DESC{limit_clause}
            """,
            (pattern, pattern, pattern, *date_params, *limit_params),
        ).fetchall()
    else:
        where_clause = " WHERE 1=1" + date_clause_plain if date_clause_plain else ""
        rows = conn.execute(
            f"SELECT id, content, metadata, tags, created_at FROM memories{where_clause} ORDER BY created_at DESC{limit_clause}",
            tuple([*date_params, *limit_params]),
        ).fetchall()

    # If the FTS search yielded nothing because of an SQLite error (e.g. malformed query)
    # fall back to a LIKE search for resilience.
    if query and _fts_enabled(conn) and not rows:
        pattern = f"%{query}%"
        rows = conn.execute(
            f"""
            SELECT id, content, metadata, tags, created_at
            FROM memories
            WHERE (content LIKE ? OR tags LIKE ? OR metadata LIKE ?){date_clause_plain}
            ORDER BY created_at DESC{limit_clause}
            """,
            (pattern, pattern, pattern, *date_params, *limit_params),
        ).fetchall()

    records: List[Dict[str, Any]] = []
    for row in rows:
        record = _serialise_row(row)
        if validated_filters and not _metadata_matches_filters(record.get("metadata"), validated_filters):
            continue

        # Apply tag filters
        record_tags = set(record.get("tags", []))

        # tags_any: match if ANY of the specified tags are present (OR logic)
        if tags_any:
            if not any(tag in record_tags for tag in tags_any):
                continue

        # tags_all: match only if ALL of the specified tags are present (AND logic)
        if tags_all:
            if not all(tag in record_tags for tag in tags_all):
                continue

        # tags_none: exclude if ANY of the specified tags are present (NOT logic)
        if tags_none:
            if any(tag in record_tags for tag in tags_none):
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


def get_statistics(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Gather statistics about stored memories."""
    stats: Dict[str, Any] = {}

    # Total count
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    stats["total_memories"] = total

    # Tag statistics
    tag_counts: Dict[str, int] = {}
    rows = conn.execute("SELECT tags FROM memories").fetchall()
    for (tags_json,) in rows:
        if tags_json:
            try:
                tags = json.loads(tags_json)
                if isinstance(tags, list):
                    for tag in tags:
                        if isinstance(tag, str):
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            except json.JSONDecodeError:
                pass

    stats["tag_counts"] = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))
    stats["unique_tags"] = len(tag_counts)

    # Section statistics
    section_counts: Dict[str, int] = {}
    subsection_counts: Dict[str, int] = {}
    rows = conn.execute("SELECT metadata FROM memories").fetchall()
    for (metadata_json,) in rows:
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
                if isinstance(metadata, dict):
                    section = metadata.get("section")
                    if section:
                        section_counts[section] = section_counts.get(section, 0) + 1
                    subsection = metadata.get("subsection")
                    if subsection:
                        subsection_counts[subsection] = subsection_counts.get(subsection, 0) + 1
            except json.JSONDecodeError:
                pass

    stats["section_counts"] = dict(sorted(section_counts.items(), key=lambda x: x[1], reverse=True))
    stats["subsection_counts"] = dict(sorted(subsection_counts.items(), key=lambda x: x[1], reverse=True))

    # Date-based statistics (memories per month)
    monthly_counts: Dict[str, int] = {}
    rows = conn.execute("SELECT created_at FROM memories").fetchall()
    for (created_at,) in rows:
        if created_at:
            try:
                # Extract YYYY-MM from timestamp
                month = created_at[:7]  # "2025-09"
                monthly_counts[month] = monthly_counts.get(month, 0) + 1
            except (IndexError, TypeError):
                pass

    stats["monthly_counts"] = dict(sorted(monthly_counts.items()))

    # Cross-reference statistics (most connected memories)
    crossref_counts: List[tuple[int, int]] = []
    rows = conn.execute("SELECT memory_id, related FROM memories_crossrefs").fetchall()
    for memory_id, related_json in rows:
        if related_json:
            try:
                related = json.loads(related_json)
                if isinstance(related, list):
                    crossref_counts.append((memory_id, len(related)))
            except json.JSONDecodeError:
                pass

    # Sort by count and take top 10
    crossref_counts.sort(key=lambda x: x[1], reverse=True)
    stats["most_connected"] = [
        {"memory_id": memory_id, "connections": count}
        for memory_id, count in crossref_counts[:10]
    ]

    # Date range
    date_range = conn.execute(
        "SELECT MIN(created_at), MAX(created_at) FROM memories"
    ).fetchone()
    if date_range and date_range[0]:
        stats["date_range"] = {
            "oldest": date_range[0],
            "newest": date_range[1],
        }

    return stats


def export_memories(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Export all memories to a JSON-serializable list."""
    rows = conn.execute(
        "SELECT id, content, metadata, tags, created_at FROM memories ORDER BY id"
    ).fetchall()

    exported: List[Dict[str, Any]] = []
    for row in rows:
        metadata = row["metadata"]
        tags = row["tags"]
        exported.append({
            "id": row["id"],
            "content": row["content"],
            "metadata": json.loads(metadata) if metadata else None,
            "tags": json.loads(tags) if tags else [],
            "created_at": row["created_at"],
        })

    return exported


def import_memories(
    conn: sqlite3.Connection,
    data: List[Dict[str, Any]],
    strategy: str = "append",
) -> Dict[str, Any]:
    """Import memories from a JSON list.

    Args:
        conn: Database connection
        data: List of memory dictionaries
        strategy: "replace" (clear all first), "merge" (skip duplicates), "append" (add all)

    Returns:
        Dictionary with import statistics
    """
    if strategy not in ("replace", "merge", "append"):
        raise ValueError("strategy must be 'replace', 'merge', or 'append'")

    # Replace: clear database first
    if strategy == "replace":
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM memories_fts")
        conn.execute("DELETE FROM memories_embeddings")
        conn.execute("DELETE FROM memories_crossrefs")
        conn.commit()

    imported = 0
    skipped = 0
    errors = []

    # Get existing content hashes for merge strategy
    existing_contents: set[str] = set()
    if strategy == "merge":
        rows = conn.execute("SELECT content FROM memories").fetchall()
        existing_contents = {row["content"] for row in rows}

    for idx, entry in enumerate(data):
        try:
            content = entry.get("content", "").strip()
            if not content:
                errors.append({"index": idx, "error": "Missing content"})
                continue

            # Skip duplicates in merge mode
            if strategy == "merge" and content in existing_contents:
                skipped += 1
                continue

            metadata = entry.get("metadata")
            tags = entry.get("tags", [])
            created_at = entry.get("created_at")

            # Prepare data
            prepared_metadata = _prepare_metadata(metadata) if metadata else None
            validated_tags = _validate_tags(tags)
            _enforce_tag_whitelist(validated_tags)

            metadata_json = json.dumps(prepared_metadata, ensure_ascii=False) if prepared_metadata else None
            tags_json = json.dumps(validated_tags, ensure_ascii=False)

            # Insert with optional created_at preservation
            if created_at:
                cur = conn.execute(
                    "INSERT INTO memories (content, metadata, tags, created_at) VALUES (?, ?, ?, ?)",
                    (content, metadata_json, tags_json, created_at),
                )
            else:
                cur = conn.execute(
                    "INSERT INTO memories (content, metadata, tags) VALUES (?, ?, ?)",
                    (content, metadata_json, tags_json),
                )

            memory_id = cur.lastrowid

            # Update FTS and embeddings
            _fts_upsert(conn, memory_id, content, metadata_json, tags_json)
            vector = _compute_embedding(content, prepared_metadata, validated_tags)
            _upsert_embedding(conn, memory_id, vector)

            imported += 1

        except Exception as exc:
            errors.append({"index": idx, "error": str(exc)})

    conn.commit()

    # Rebuild cross-references after import
    if imported > 0:
        rebuild_crossrefs(conn)

    return {
        "imported": imported,
        "skipped": skipped,
        "errors": errors[:10],  # Limit error list to first 10
        "total_errors": len(errors),
    }


def poll_events(
    conn: sqlite3.Connection,
    since_timestamp: Optional[str] = None,
    tags_filter: Optional[List[str]] = None,
    unconsumed_only: bool = True,
) -> List[Dict[str, Any]]:
    """Poll for memory events."""
    query = "SELECT id, memory_id, tags, timestamp, consumed FROM memories_events WHERE 1=1"
    params: List[Any] = []

    if unconsumed_only:
        query += " AND consumed = 0"

    if since_timestamp:
        query += " AND timestamp > ?"
        params.append(since_timestamp)

    if tags_filter:
        # Check if any of the filter tags are in the event's tags JSON array
        tag_conditions = " OR ".join(["json_extract(tags, '$') LIKE ?" for _ in tags_filter])
        query += f" AND ({tag_conditions})"
        for tag in tags_filter:
            params.append(f'%"{tag}"%')

    query += " ORDER BY timestamp DESC"

    rows = conn.execute(query, params).fetchall()

    events = []
    for row in rows:
        events.append({
            "id": row["id"],
            "memory_id": row["memory_id"],
            "tags": json.loads(row["tags"]) if row["tags"] else [],
            "timestamp": row["timestamp"],
            "consumed": bool(row["consumed"]),
        })

    return events


def clear_events(conn: sqlite3.Connection, event_ids: List[int]) -> int:
    """Mark events as consumed."""
    if not event_ids:
        return 0

    placeholders = ",".join(["?" for _ in event_ids])
    conn.execute(
        f"UPDATE memories_events SET consumed = 1 WHERE id IN ({placeholders})",
        event_ids
    )
    conn.commit()
    return len(event_ids)
