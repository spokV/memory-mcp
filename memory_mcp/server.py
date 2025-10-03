"""MCP-compatible memory server backed by SQLite."""
from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .storage import (
    add_memory,
    add_memories,
    clear_events,
    collect_all_tags,
    connect,
    delete_memory,
    delete_memories,
    export_memories,
    find_invalid_tag_entries,
    get_crossrefs,
    get_memory,
    get_statistics,
    import_memories,
    list_memories,
    poll_events,
    rebuild_embeddings,
    rebuild_crossrefs,
    semantic_search,
    update_crossrefs,
    update_memory,
)


def _read_int_env(var_name: str, fallback: int) -> int:
    try:
        return int(os.getenv(var_name, fallback))
    except (TypeError, ValueError):
        return fallback


VALID_TRANSPORTS = {"stdio", "sse", "streamable-http"}

_env_transport = os.getenv("MEMORY_MCP_TRANSPORT", "stdio")
DEFAULT_TRANSPORT = _env_transport if _env_transport in VALID_TRANSPORTS else "stdio"
DEFAULT_HOST = os.getenv("MEMORY_MCP_HOST", "127.0.0.1")
DEFAULT_PORT = _read_int_env("MEMORY_MCP_PORT", 8000)

mcp = FastMCP("Memory MCP Server", host=DEFAULT_HOST, port=DEFAULT_PORT)


def _with_connection(func):
    def wrapper(*args, **kwargs):
        conn = connect()
        try:
            return func(conn, *args, **kwargs)
        finally:
            conn.close()

    return wrapper


@_with_connection
def _create_memory(
    conn,
    content: str,
    metadata: Optional[Dict[str, Any]],
    tags: Optional[list[str]],
):
    return add_memory(conn, content=content.strip(), metadata=metadata, tags=tags or [])


@_with_connection
def _get_memory(conn, memory_id: int):
    return get_memory(conn, memory_id)


@_with_connection
def _update_memory(
    conn,
    memory_id: int,
    content: Optional[str],
    metadata: Optional[Dict[str, Any]],
    tags: Optional[list[str]],
):
    return update_memory(conn, memory_id, content=content, metadata=metadata, tags=tags)


@_with_connection
def _delete_memory(conn, memory_id: int):
    return delete_memory(conn, memory_id)


@_with_connection
def _list_memories(
    conn,
    query: Optional[str],
    metadata_filters: Optional[Dict[str, Any]],
    limit: Optional[int],
    offset: Optional[int],
    date_from: Optional[str],
    date_to: Optional[str],
    tags_any: Optional[List[str]],
    tags_all: Optional[List[str]],
    tags_none: Optional[List[str]],
):
    return list_memories(conn, query, metadata_filters, limit, offset, date_from, date_to, tags_any, tags_all, tags_none)


@_with_connection
def _create_memories(conn, entries: List[Dict[str, Any]]):
    return add_memories(conn, entries)


@_with_connection
def _delete_memories(conn, ids: List[int]):
    return delete_memories(conn, ids)


@_with_connection
def _collect_tags(conn):
    return collect_all_tags(conn)


@_with_connection
def _find_invalid_tags(conn):
    from . import TAG_WHITELIST

    return find_invalid_tag_entries(conn, TAG_WHITELIST)


@_with_connection
def _get_related(conn, memory_id: int, refresh: bool) -> List[Dict[str, Any]]:
    if refresh:
        update_crossrefs(conn, memory_id)
    refs = get_crossrefs(conn, memory_id)
    if not refs and not refresh:
        update_crossrefs(conn, memory_id)
        refs = get_crossrefs(conn, memory_id)
    return refs


@_with_connection
def _rebuild_crossrefs(conn):
    return rebuild_crossrefs(conn)


@_with_connection
def _semantic_search(
    conn,
    query: str,
    metadata_filters: Optional[Dict[str, Any]],
    top_k: Optional[int],
    min_score: Optional[float],
):
    return semantic_search(
        conn,
        query,
        metadata_filters=metadata_filters,
        top_k=top_k,
        min_score=min_score,
    )


@_with_connection
def _rebuild_embeddings(conn):
    return rebuild_embeddings(conn)


@_with_connection
def _get_statistics(conn):
    return get_statistics(conn)


@_with_connection
def _export_memories(conn):
    return export_memories(conn)


@_with_connection
def _import_memories(conn, data: List[Dict[str, Any]], strategy: str):
    return import_memories(conn, data, strategy)


def _build_tag_hierarchy(tags):
    root = {"name": "root", "path": [], "children": {}, "tags": []}
    for tag in tags:
        parts = tag.split('.')
        node = root
        if not parts:
            continue
        for idx, part in enumerate(parts):
            children = node.setdefault("children", {})
            if part not in children:
                children[part] = {
                    "name": part,
                    "path": node["path"] + [part],
                    "children": {},
                    "tags": []
                }
            node = children[part]
        node.setdefault("tags", []).append(tag)
    return _collapse_tag_tree(root)


def _collapse_tag_tree(node):
    children_map = node.get("children", {})
    children_list = [_collapse_tag_tree(child) for child in children_map.values()]
    node["children"] = children_list
    node["count"] = len(node.get("tags", [])) + sum(child["count"] for child in children_list)
    return {key: value for key, value in node.items() if key != "children" or value}


def _extract_hierarchy_path(metadata: Optional[Any]) -> List[str]:
    if not isinstance(metadata, Mapping):
        return []

    hierarchy = metadata.get("hierarchy")
    if isinstance(hierarchy, Mapping):
        raw_path = hierarchy.get("path")
        if isinstance(raw_path, Sequence) and not isinstance(raw_path, (str, bytes)):
            return [str(part) for part in raw_path if part is not None]

    path: List[str] = []
    section = metadata.get("section")
    if section is not None:
        path.append(str(section))
        subsection = metadata.get("subsection")
        if subsection is not None:
            path.append(str(subsection))
    return path


def _build_hierarchy_tree(memories: List[Dict[str, Any]], include_root: bool = False) -> Any:
    root: Dict[str, Any] = {
        "name": "root",
        "path": [],
        "memories": [],
        "children": {},
    }

    for memory in memories:
        path = _extract_hierarchy_path(memory.get("metadata"))
        node = root
        if not path:
            node["memories"].append(memory)
            continue

        for part in path:
            children: Dict[str, Any] = node.setdefault("children", {})
            if part not in children:
                children[part] = {
                    "name": part,
                    "path": node["path"] + [part],
                    "memories": [],
                    "children": {},
                }
            node = children[part]
        node["memories"].append(memory)

    def collapse(node: Dict[str, Any]) -> Dict[str, Any]:
        children_map: Dict[str, Any] = node.get("children", {})
        children_list = [collapse(child) for child in children_map.values()]
        node["children"] = children_list
        node["count"] = len(node.get("memories", [])) + sum(child["count"] for child in children_list)
        return node

    collapsed = collapse(root)
    if include_root:
        return collapsed
    return collapsed["children"]


@mcp.tool()
async def memory_create(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Create a new memory entry."""
    try:
        record = _create_memory(content=content.strip(), metadata=metadata, tags=tags or [])
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}
    return {"memory": record}


@mcp.tool()
async def memory_list(
    query: Optional[str] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = 0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
    tags_all: Optional[List[str]] = None,
    tags_none: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """List memories, optionally filtering by substring query or metadata.

    Args:
        query: Optional text search query
        metadata_filters: Optional metadata filters
        limit: Maximum number of results to return (default: unlimited)
        offset: Number of results to skip (default: 0)
        date_from: Optional date filter (ISO format or relative like "7d", "1m", "1y")
        date_to: Optional date filter (ISO format or relative like "7d", "1m", "1y")
        tags_any: Match memories with ANY of these tags (OR logic)
        tags_all: Match memories with ALL of these tags (AND logic)
        tags_none: Exclude memories with ANY of these tags (NOT logic)
    """
    try:
        items = _list_memories(query, metadata_filters, limit, offset, date_from, date_to, tags_any, tags_all, tags_none)
    except ValueError as exc:
        return {"error": "invalid_filters", "message": str(exc)}
    return {"count": len(items), "memories": items}


@mcp.tool()
async def memory_list_compact(
    query: Optional[str] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = 0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
    tags_all: Optional[List[str]] = None,
    tags_none: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """List memories in compact format (id, preview, tags only) to reduce context usage.

    Returns minimal fields: id, content preview (first 80 chars), tags, and created_at.
    This tool is useful for browsing memories without loading full content and metadata.

    Args:
        query: Optional text search query
        metadata_filters: Optional metadata filters
        limit: Maximum number of results to return (default: unlimited)
        offset: Number of results to skip (default: 0)
        date_from: Optional date filter (ISO format or relative like "7d", "1m", "1y")
        date_to: Optional date filter (ISO format or relative like "7d", "1m", "1y")
        tags_any: Match memories with ANY of these tags (OR logic)
        tags_all: Match memories with ALL of these tags (AND logic)
        tags_none: Exclude memories with ANY of these tags (NOT logic)
    """
    try:
        items = _list_memories(query, metadata_filters, limit, offset, date_from, date_to, tags_any, tags_all, tags_none)
    except ValueError as exc:
        return {"error": "invalid_filters", "message": str(exc)}

    # Convert to compact format
    compact_items = []
    for item in items:
        content = item.get("content", "")
        preview = content[:80] + "..." if len(content) > 80 else content
        compact_items.append({
            "id": item["id"],
            "preview": preview,
            "tags": item.get("tags", []),
            "created_at": item.get("created_at"),
        })

    return {"count": len(compact_items), "memories": compact_items}


@mcp.tool()
async def memory_create_batch(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create multiple memories in one call."""
    try:
        records = _create_memories(entries)
    except ValueError as exc:
        return {"error": "invalid_batch", "message": str(exc)}
    return {"count": len(records), "memories": records}


@mcp.tool()
async def memory_delete_batch(ids: List[int]) -> Dict[str, Any]:
    """Delete multiple memories by id."""
    deleted = _delete_memories(ids)
    return {"deleted": deleted}


@mcp.tool()
async def memory_get(memory_id: int) -> Dict[str, Any]:
    """Retrieve a single memory by id."""
    record = _get_memory(memory_id)
    if not record:
        return {"error": "not_found", "id": memory_id}
    return {"memory": record}


@mcp.tool()
async def memory_update(
    memory_id: int,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Update an existing memory. Only provided fields are updated."""
    try:
        record = _update_memory(memory_id, content, metadata, tags)
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}
    if not record:
        return {"error": "not_found", "id": memory_id}
    return {"memory": record}


@mcp.tool()
async def memory_delete(memory_id: int) -> Dict[str, Any]:
    """Delete a memory by id."""
    if _delete_memory(memory_id):
        return {"status": "deleted", "id": memory_id}
    return {"error": "not_found", "id": memory_id}


@mcp.tool()
async def memory_tags() -> Dict[str, Any]:
    """Return the allowlisted tags."""
    from . import list_allowed_tags

    return {"allowed": list_allowed_tags()}


@mcp.tool()
async def memory_tag_hierarchy(include_root: bool = False) -> Dict[str, Any]:
    """Return stored tags organised as a namespace hierarchy."""

    tags = _collect_tags()
    tree = _build_tag_hierarchy(tags)
    if not include_root and isinstance(tree, dict):
        tree = tree.get("children", [])
    return {"count": len(tags), "hierarchy": tree}


@mcp.tool()
async def memory_validate_tags(include_memories: bool = True) -> Dict[str, Any]:
    """Validate stored tags against the allowlist and report invalid entries."""
    from . import list_allowed_tags

    invalid_full = _find_invalid_tags()
    allowed = list_allowed_tags()
    existing = _collect_tags()
    response: Dict[str, Any] = {"allowed": allowed, "existing": existing, "invalid_count": len(invalid_full)}
    if include_memories:
        response["invalid"] = invalid_full
    return response


@mcp.tool()
async def memory_hierarchy(
    query: Optional[str] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    include_root: bool = False,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
    tags_all: Optional[List[str]] = None,
    tags_none: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return memories organised into a hierarchy derived from their metadata."""
    try:
        items = _list_memories(query, metadata_filters, None, 0, date_from, date_to, tags_any, tags_all, tags_none)
    except ValueError as exc:
        return {"error": "invalid_filters", "message": str(exc)}

    hierarchy = _build_hierarchy_tree(items, include_root=include_root)
    return {"count": len(items), "hierarchy": hierarchy}


@mcp.tool()
async def memory_semantic_search(
    query: str,
    top_k: int = 5,
    metadata_filters: Optional[Dict[str, Any]] = None,
    min_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Perform a semantic search using vector embeddings."""

    try:
        results = _semantic_search(
            query,
            metadata_filters,
            top_k,
            min_score,
        )
    except ValueError as exc:
        return {"error": "invalid_filters", "message": str(exc)}
    return {"count": len(results), "results": results}


@mcp.tool()
async def memory_rebuild_embeddings() -> Dict[str, Any]:
    """Recompute embeddings for all memories."""

    updated = _rebuild_embeddings()
    return {"updated": updated}


@mcp.tool()
async def memory_related(memory_id: int, refresh: bool = False) -> Dict[str, Any]:
    """Return cross-referenced memories for a given entry."""

    related = _get_related(memory_id, refresh)
    return {"id": memory_id, "related": related}


@mcp.tool()
async def memory_rebuild_crossrefs() -> Dict[str, Any]:
    """Recompute cross-reference links for all memories."""

    updated = _rebuild_crossrefs()
    return {"updated": updated}


@mcp.tool()
async def memory_stats() -> Dict[str, Any]:
    """Get statistics and analytics about stored memories."""

    return _get_statistics()


@mcp.tool()
async def memory_export() -> Dict[str, Any]:
    """Export all memories to JSON format for backup or transfer."""

    memories = _export_memories()
    return {"count": len(memories), "memories": memories}


@mcp.tool()
async def memory_import(
    data: List[Dict[str, Any]],
    strategy: str = "append",
) -> Dict[str, Any]:
    """Import memories from JSON format.

    Args:
        data: List of memory dictionaries with content, metadata, tags, created_at
        strategy: "replace" (clear all first), "merge" (skip duplicates), or "append" (add all)
    """
    try:
        result = _import_memories(data, strategy)
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}
    return result


@_with_connection
def _poll_events(
    conn,
    since_timestamp: Optional[str],
    tags_filter: Optional[List[str]],
    unconsumed_only: bool,
):
    return poll_events(conn, since_timestamp, tags_filter, unconsumed_only)


@_with_connection
def _clear_events(conn, event_ids: List[int]):
    return clear_events(conn, event_ids)


@mcp.tool()
async def memory_events_poll(
    since_timestamp: Optional[str] = None,
    tags_filter: Optional[List[str]] = None,
    unconsumed_only: bool = True,
) -> Dict[str, Any]:
    """Poll for memory events (e.g., shared-cache notifications).

    Args:
        since_timestamp: Only return events after this timestamp (ISO format)
        tags_filter: Only return events with these tags (e.g., ["shared-cache"])
        unconsumed_only: Only return unconsumed events (default: True)

    Returns:
        Dictionary with count and list of events
    """
    events = _poll_events(since_timestamp, tags_filter, unconsumed_only)
    return {"count": len(events), "events": events}


@mcp.tool()
async def memory_events_clear(event_ids: List[int]) -> Dict[str, Any]:
    """Mark events as consumed.

    Args:
        event_ids: List of event IDs to mark as consumed

    Returns:
        Dictionary with count of cleared events
    """
    cleared = _clear_events(event_ids)
    return {"cleared": cleared}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the memory MCP server")
    parser.add_argument(
        "--transport",
        choices=sorted(VALID_TRANSPORTS),
        default=DEFAULT_TRANSPORT,
        help="MCP transport to use (defaults to env MEMORY_MCP_TRANSPORT or 'stdio')",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Host interface for HTTP transports (defaults to env MEMORY_MCP_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for HTTP transports (defaults to env MEMORY_MCP_PORT or 8000)",
    )
    args = parser.parse_args(argv)

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
