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
    boost_memory,
    clear_events,
    collect_all_tags,
    connect,
    delete_memory,
    delete_memories,
    export_memories,
    find_invalid_tag_entries,
    get_backend_info,
    get_crossrefs,
    get_memory,
    get_statistics,
    hybrid_search,
    import_memories,
    list_memories,
    poll_events,
    rebuild_embeddings,
    rebuild_crossrefs,
    semantic_search,
    sync_to_cloud,
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


def _with_connection(func=None, *, writes=False):
    """Decorator that manages database connections and cloud sync.

    Opens a connection, runs the function, closes the connection,
    and syncs to cloud storage only after write operations.

    Args:
        writes: If True, syncs to cloud after operation. If False, skips sync (read-only).
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            conn = connect()
            try:
                result = func(conn, *args, **kwargs)
                # Only sync to cloud after write operations
                if writes:
                    sync_to_cloud()
                return result
            finally:
                conn.close()

        return wrapper

    # Allow using as @_with_connection or @_with_connection(writes=True)
    if func is not None:
        # Called as @_with_connection (default: read-only, no sync)
        return decorator(func)
    else:
        # Called as @_with_connection(writes=True)
        return decorator


@_with_connection(writes=True)
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


@_with_connection(writes=True)
def _update_memory(
    conn,
    memory_id: int,
    content: Optional[str],
    metadata: Optional[Dict[str, Any]],
    tags: Optional[list[str]],
):
    return update_memory(conn, memory_id, content=content, metadata=metadata, tags=tags)


@_with_connection(writes=True)
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
    sort_by_importance: bool = False,
):
    return list_memories(
        conn, query, metadata_filters, limit, offset,
        date_from, date_to, tags_any, tags_all, tags_none,
        sort_by_importance=sort_by_importance,
    )


@_with_connection(writes=True)
def _boost_memory(conn, memory_id: int, boost_amount: float):
    return boost_memory(conn, memory_id, boost_amount)


@_with_connection(writes=True)
def _create_memories(conn, entries: List[Dict[str, Any]]):
    return add_memories(conn, entries)


@_with_connection(writes=True)
def _delete_memories(conn, ids: List[int]):
    return delete_memories(conn, ids)


@_with_connection
def _collect_tags(conn):
    return collect_all_tags(conn)


@_with_connection
def _find_invalid_tags(conn):
    from . import TAG_WHITELIST

    return find_invalid_tag_entries(conn, TAG_WHITELIST)


@_with_connection(writes=True)  # May write crossrefs if refresh=True
def _get_related(conn, memory_id: int, refresh: bool) -> List[Dict[str, Any]]:
    if refresh:
        update_crossrefs(conn, memory_id)
    refs = get_crossrefs(conn, memory_id)
    if not refs and not refresh:
        update_crossrefs(conn, memory_id)
        refs = get_crossrefs(conn, memory_id)
    return refs


@_with_connection(writes=True)
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
def _hybrid_search(
    conn,
    query: str,
    semantic_weight: float,
    top_k: int,
    min_score: float,
    metadata_filters: Optional[Dict[str, Any]],
    date_from: Optional[str],
    date_to: Optional[str],
    tags_any: Optional[List[str]],
    tags_all: Optional[List[str]],
    tags_none: Optional[List[str]],
):
    return hybrid_search(
        conn,
        query,
        semantic_weight=semantic_weight,
        top_k=top_k,
        min_score=min_score,
        metadata_filters=metadata_filters,
        date_from=date_from,
        date_to=date_to,
        tags_any=tags_any,
        tags_all=tags_all,
        tags_none=tags_none,
    )


@_with_connection(writes=True)
def _rebuild_embeddings(conn):
    return rebuild_embeddings(conn)


@_with_connection
def _get_statistics(conn):
    return get_statistics(conn)


@_with_connection
def _export_memories(conn):
    return export_memories(conn)


@_with_connection(writes=True)
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
            memory_with_path = dict(memory)
            memory_with_path["hierarchy_path"] = node["path"]
            node["memories"].append(memory_with_path)
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
        memory_with_path = dict(memory)
        memory_with_path["hierarchy_path"] = node["path"]
        node["memories"].append(memory_with_path)

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
    sort_by_importance: bool = False,
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
        sort_by_importance: Sort results by importance score (default: False, sorts by date)
    """
    try:
        items = _list_memories(
            query, metadata_filters, limit, offset,
            date_from, date_to, tags_any, tags_all, tags_none,
            sort_by_importance,
        )
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
async def memory_hybrid_search(
    query: str,
    semantic_weight: float = 0.6,
    top_k: int = 10,
    min_score: float = 0.0,
    metadata_filters: Optional[Dict[str, Any]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
    tags_all: Optional[List[str]] = None,
    tags_none: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Perform a hybrid search combining keyword (FTS) and semantic (vector) search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both search methods,
    providing better results than either method alone.

    Args:
        query: Search query text
        semantic_weight: Weight for semantic results (0-1). Higher values favor semantic similarity.
                        Keyword weight = 1 - semantic_weight. Default: 0.6 (60% semantic, 40% keyword)
        top_k: Maximum number of results to return (default: 10)
        min_score: Minimum combined score threshold (default: 0.0)
        metadata_filters: Optional metadata filters
        date_from: Optional date filter (ISO format or relative like "7d", "1m", "1y")
        date_to: Optional date filter (ISO format or relative)
        tags_any: Match memories with ANY of these tags (OR logic)
        tags_all: Match memories with ALL of these tags (AND logic)
        tags_none: Exclude memories with ANY of these tags (NOT logic)

    Returns:
        Dictionary with count and list of results, each containing score and memory
    """
    try:
        results = _hybrid_search(
            query,
            semantic_weight,
            top_k,
            min_score,
            metadata_filters,
            date_from,
            date_to,
            tags_any,
            tags_all,
            tags_none,
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
async def memory_boost(
    memory_id: int,
    boost_amount: float = 0.5,
) -> Dict[str, Any]:
    """Boost a memory's importance score.

    Manually increase a memory's base importance to make it rank higher in
    importance-sorted searches. The boost is permanent and cumulative.

    Args:
        memory_id: ID of the memory to boost
        boost_amount: Amount to add to base importance (default: 0.5)
                      Common values: 0.25 (small), 0.5 (medium), 1.0 (large)

    Returns:
        Updated memory with new importance score, or error if not found
    """
    record = _boost_memory(memory_id, boost_amount)
    if not record:
        return {"error": "not_found", "id": memory_id}
    return {"memory": record, "boosted_by": boost_amount}


@mcp.tool()
async def memory_export() -> Dict[str, Any]:
    """Export all memories to JSON format for backup or transfer."""

    memories = _export_memories()
    return {"count": len(memories), "memories": memories}


@_with_connection
def _export_graph_html(conn, output_path: str, min_score: float) -> Dict[str, Any]:
    """Generate HTML knowledge graph visualization."""
    import json as json_lib
    from .storage import list_memories, get_crossrefs, rebuild_crossrefs

    memories = list_memories(conn, None, None, None, 0, None, None, None, None, None)
    if not memories:
        return {"error": "no_memories", "message": "No memories to visualize"}

    rebuild_crossrefs(conn)

    colors = ["#58a6ff", "#f78166", "#a371f7", "#7ee787", "#ffa657", "#ff7b72", "#79c0ff", "#d2a8ff"]
    tag_colors = {}

    nodes = []
    memories_data = {}
    for m in memories:
        tags = m.get("tags", [])
        primary_tag = tags[0] if tags else "untagged"
        if primary_tag not in tag_colors:
            tag_colors[primary_tag] = colors[len(tag_colors) % len(colors)]

        content = m["content"]
        label = content[:35].replace("\n", " ").replace('"', "'").replace("\\", "")

        nodes.append({
            "id": m["id"],
            "label": label + "..." if len(content) > 35 else label,
            "color": tag_colors[primary_tag],
        })

        memories_data[m["id"]] = {
            "id": m["id"],
            "tags": tags,
            "created": m.get("created_at", ""),
            "content": content
        }

    edges = []
    seen = set()
    for m in memories:
        for ref in get_crossrefs(conn, m["id"]):
            edge_key = tuple(sorted([m["id"], ref["id"]]))
            if edge_key not in seen and ref.get("score", 0) > min_score:
                seen.add(edge_key)
                edges.append({"from": m["id"], "to": ref["id"]})

    legend_html = "".join(
        f'<div class="legend-item"><span class="legend-color" style="background:{c}"></span>{t}</div>'
        for t, c in list(tag_colors.items())[:12]
    )

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Memory Knowledge Graph</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; display: flex; height: 100vh; }}
        #graph {{ flex: 1; height: 100%; }}
        #panel {{ width: 400px; background: #161b22; border-left: 1px solid #30363d; padding: 20px; overflow-y: auto; display: none; position: relative; }}
        #panel.active {{ display: block; }}
        #panel h2 {{ color: #58a6ff; margin-bottom: 10px; font-size: 16px; }}
        #panel .tags {{ margin-bottom: 15px; }}
        #panel .tag {{ display: inline-block; background: #30363d; padding: 3px 8px; border-radius: 12px; font-size: 12px; margin: 2px; }}
        #panel .meta {{ color: #8b949e; font-size: 12px; margin-bottom: 15px; }}
        #panel .content {{ white-space: pre-wrap; font-size: 13px; line-height: 1.6; background: #0d1117; padding: 15px; border-radius: 6px; max-height: calc(100vh - 200px); overflow-y: auto; }}
        #panel .close {{ position: absolute; top: 10px; right: 15px; cursor: pointer; font-size: 20px; color: #8b949e; }}
        #panel .close:hover {{ color: #fff; }}
        #legend {{ position: absolute; top: 10px; left: 10px; background: rgba(22,27,34,0.9); padding: 12px; border-radius: 6px; font-size: 12px; }}
        .legend-item {{ margin: 4px 0; display: flex; align-items: center; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
        #help {{ position: absolute; bottom: 10px; left: 10px; background: rgba(22,27,34,0.9); padding: 8px 12px; border-radius: 6px; font-size: 11px; color: #8b949e; }}
    </style>
</head>
<body>
    <div id="graph"></div>
    <div id="panel">
        <span class="close" onclick="closePanel()">&times;</span>
        <h2 id="panel-title">Memory #</h2>
        <div class="meta" id="panel-meta"></div>
        <div class="tags" id="panel-tags"></div>
        <div class="content" id="panel-content"></div>
    </div>
    <div id="legend"><b>Tags</b>{legend_html}</div>
    <div id="help">Click node to view | Scroll to zoom | Drag to pan</div>
    <script>
        var memoriesData = {json_lib.dumps(memories_data)};
        var nodes = new vis.DataSet({json_lib.dumps(nodes)});
        var edges = new vis.DataSet({json_lib.dumps(edges)});
        var container = document.getElementById("graph");
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            nodes: {{ shape: "dot", size: 16, font: {{ color: "#c9d1d9", size: 11 }}, borderWidth: 2 }},
            edges: {{ color: {{ color: "#30363d", opacity: 0.6 }}, smooth: {{ type: "continuous" }} }},
            physics: {{ barnesHut: {{ gravitationalConstant: -2500, springLength: 120, damping: 0.3 }} }},
            interaction: {{ hover: true, tooltipDelay: 100 }}
        }};
        var network = new vis.Network(container, data, options);
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var mem = memoriesData[nodeId];
                if (mem) {{
                    document.getElementById("panel-title").textContent = "Memory #" + mem.id;
                    document.getElementById("panel-meta").textContent = "Created: " + mem.created;
                    document.getElementById("panel-tags").innerHTML = mem.tags.map(t => '<span class="tag">' + t + '</span>').join("");
                    document.getElementById("panel-content").textContent = mem.content;
                    document.getElementById("panel").classList.add("active");
                }}
            }}
        }});
        function closePanel() {{ document.getElementById("panel").classList.remove("active"); }}
    </script>
</body>
</html>'''

    with open(output_path, "w") as f:
        f.write(html)

    return {
        "path": output_path,
        "nodes": len(nodes),
        "edges": len(edges),
        "tags": list(tag_colors.keys()),
    }


@mcp.tool()
async def memory_export_graph(
    output_path: Optional[str] = None,
    min_score: float = 0.25,
) -> Dict[str, Any]:
    """Export memories as interactive HTML knowledge graph.

    Args:
        output_path: Path to save HTML file (default: ~/memories_graph.html)
        min_score: Minimum similarity score for edges (default: 0.25)

    Returns:
        Dictionary with path, node count, edge count, and tags
    """
    import os
    if output_path is None:
        output_path = os.path.expanduser("~/memories_graph.html")

    return _export_graph_html(output_path, min_score)


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


@_with_connection(writes=True)
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
    parser = argparse.ArgumentParser(description="Memory MCP Server")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Default: start server (make it the default if no subcommand)
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

    # Subcommand: sync-pull
    sync_pull_parser = subparsers.add_parser(
        "sync-pull",
        help="Force pull database from cloud storage (ignore local cache)"
    )

    # Subcommand: sync-push
    sync_push_parser = subparsers.add_parser(
        "sync-push",
        help="Force push database to cloud storage"
    )

    # Subcommand: sync-status
    sync_status_parser = subparsers.add_parser(
        "sync-status",
        help="Show sync status and backend information"
    )

    # Subcommand: info
    info_parser = subparsers.add_parser(
        "info",
        help="Show storage backend information"
    )

    args = parser.parse_args(argv)

    # Handle subcommands
    if args.command == "sync-pull":
        _handle_sync_pull()
    elif args.command == "sync-push":
        _handle_sync_push()
    elif args.command == "sync-status":
        _handle_sync_status()
    elif args.command == "info":
        _handle_info()
    else:
        # Default: start server
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport=args.transport)


def _handle_sync_pull() -> None:
    """Handle sync-pull command."""
    import json
    from .backends import CloudSQLiteBackend
    from .storage import STORAGE_BACKEND

    if not isinstance(STORAGE_BACKEND, CloudSQLiteBackend):
        print("Error: sync-pull only works with cloud storage backends")
        print(f"Current backend: {STORAGE_BACKEND.__class__.__name__}")
        exit(1)

    print(f"Pulling database from {STORAGE_BACKEND.cloud_url}...")
    try:
        STORAGE_BACKEND.force_sync_pull()
        info = STORAGE_BACKEND.get_info()
        print("✓ Sync completed successfully")
        print(f"  Cache path: {info['cache_path']}")
        print(f"  Size: {info['cache_size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Last sync: {info.get('last_sync', 'N/A')}")
    except Exception as e:
        print(f"✗ Sync failed: {e}")
        exit(1)


def _handle_sync_push() -> None:
    """Handle sync-push command."""
    import json
    from .backends import CloudSQLiteBackend
    from .storage import STORAGE_BACKEND

    if not isinstance(STORAGE_BACKEND, CloudSQLiteBackend):
        print("Error: sync-push only works with cloud storage backends")
        print(f"Current backend: {STORAGE_BACKEND.__class__.__name__}")
        exit(1)

    print(f"Pushing database to {STORAGE_BACKEND.cloud_url}...")
    try:
        STORAGE_BACKEND.force_sync_push()
        info = STORAGE_BACKEND.get_info()
        print("✓ Push completed successfully")
        print(f"  Cloud URL: {info['cloud_url']}")
        print(f"  Size: {info['cache_size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Last sync: {info.get('last_sync', 'N/A')}")
    except Exception as e:
        print(f"✗ Push failed: {e}")
        exit(1)


def _handle_sync_status() -> None:
    """Handle sync-status command."""
    import json
    from .backends import CloudSQLiteBackend
    from .storage import STORAGE_BACKEND

    info = STORAGE_BACKEND.get_info()
    backend_type = info.get('backend_type', 'unknown')

    print(f"Storage Backend: {backend_type}")
    print()

    if backend_type == "cloud_sqlite":
        print(f"Cloud URL: {info.get('cloud_url', 'N/A')}")
        print(f"Bucket: {info.get('bucket', 'N/A')}")
        print(f"Key: {info.get('key', 'N/A')}")
        print()
        print(f"Cache Path: {info.get('cache_path', 'N/A')}")
        print(f"Cache Exists: {info.get('cache_exists', False)}")
        print(f"Cache Size: {info.get('cache_size_bytes', 0) / 1024 / 1024:.2f} MB")
        print()
        print(f"Is Dirty: {info.get('is_dirty', False)}")
        print(f"Last ETag: {info.get('last_etag', 'N/A')}")
        print(f"Last Sync: {info.get('last_sync', 'N/A')}")
        print(f"Auto Sync: {info.get('auto_sync', True)}")
        print(f"Encryption: {info.get('encrypt', False)}")
    elif backend_type == "local_sqlite":
        print(f"Database Path: {info.get('db_path', 'N/A')}")
        print(f"Exists: {info.get('exists', False)}")
        print(f"Size: {info.get('size_bytes', 0) / 1024 / 1024:.2f} MB")
    else:
        print(json.dumps(info, indent=2))


def _handle_info() -> None:
    """Handle info command."""
    import json
    from .storage import STORAGE_BACKEND

    info = STORAGE_BACKEND.get_info()
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
