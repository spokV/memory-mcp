"""MCP-compatible memory server backed by SQLite."""
from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

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

_env_transport = os.getenv("MEMORA_TRANSPORT", "stdio")
DEFAULT_TRANSPORT = _env_transport if _env_transport in VALID_TRANSPORTS else "stdio"
DEFAULT_HOST = os.getenv("MEMORA_HOST", "127.0.0.1")
DEFAULT_PORT = _read_int_env("MEMORA_PORT", 8000)
DEFAULT_GRAPH_PORT = _read_int_env("MEMORA_GRAPH_PORT", 8765)

mcp = FastMCP("Memory MCP Server", host=DEFAULT_HOST, port=DEFAULT_PORT)


@mcp.custom_route("/graph", methods=["GET"])
async def serve_graph(request: Request):
    """Serve the knowledge graph visualization via HTTP."""
    try:
        min_score = float(request.query_params.get("min_score", 0.25))
        result = _export_graph_html(output_path=None, min_score=min_score)
        if "error" in result:
            return JSONResponse(result, status_code=404)
        return HTMLResponse(result["html"])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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


def _compact_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact representation of a memory (id, preview, tags)."""
    content = memory.get("content", "")
    preview = content[:80] + "..." if len(content) > 80 else content
    return {
        "id": memory.get("id"),
        "preview": preview,
        "tags": memory.get("tags", []),
    }


def _build_hierarchy_tree(memories: List[Dict[str, Any]], include_root: bool = False, compact: bool = True) -> Any:
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
            mem_data = _compact_memory(memory) if compact else dict(memory)
            if not compact:
                mem_data["hierarchy_path"] = node["path"]
            node["memories"].append(mem_data)
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
        mem_data = _compact_memory(memory) if compact else dict(memory)
        if not compact:
            mem_data["hierarchy_path"] = node["path"]
        node["memories"].append(mem_data)

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
async def memory_get(memory_id: int, include_images: bool = False) -> Dict[str, Any]:
    """Retrieve a single memory by id.

    Args:
        memory_id: ID of the memory to retrieve
        include_images: If False, strip image data from metadata to reduce response size
    """
    record = _get_memory(memory_id)
    if not record:
        return {"error": "not_found", "id": memory_id}

    if not include_images and record.get("metadata", {}).get("images"):
        record["metadata"]["images"] = [
            {"caption": img.get("caption", "")} for img in record["metadata"]["images"]
        ]

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
    compact: bool = True,
) -> Dict[str, Any]:
    """Return memories organised into a hierarchy derived from their metadata.

    Args:
        compact: If True (default), return only id, preview (first 80 chars), and tags
                 per memory to reduce response size. Set to False for full memory data.
    """
    try:
        items = _list_memories(query, metadata_filters, None, 0, date_from, date_to, tags_any, tags_all, tags_none)
    except ValueError as exc:
        return {"error": "invalid_filters", "message": str(exc)}

    hierarchy = _build_hierarchy_tree(items, include_root=include_root, compact=compact)
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


@mcp.tool()
async def memory_upload_image(
    file_path: str,
    memory_id: int,
    image_index: int = 0,
    caption: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload an image file directly to R2 storage.

    Uploads a local image file to R2 and returns the r2:// reference URL
    that can be used in memory metadata.

    Args:
        file_path: Absolute path to the image file to upload
        memory_id: Memory ID this image belongs to (used for organizing in R2)
        image_index: Index of image within the memory (default: 0)
        caption: Optional caption for the image

    Returns:
        Dictionary with r2_url (the r2:// reference) and image object ready for metadata
    """
    import os
    import mimetypes

    from .image_storage import get_image_storage_instance

    image_storage = get_image_storage_instance()
    if not image_storage:
        return {
            "error": "r2_not_configured",
            "message": "R2 storage is not configured. Set MEMORA_STORAGE_URI to s3:// and configure AWS credentials.",
        }

    # Validate file exists
    if not os.path.isfile(file_path):
        return {"error": "file_not_found", "message": f"File not found: {file_path}"}

    # Determine content type
    content_type, _ = mimetypes.guess_type(file_path)
    if not content_type or not content_type.startswith("image/"):
        content_type = "image/png"  # Default to PNG for unknown types

    try:
        # Read file and upload
        with open(file_path, "rb") as f:
            image_data = f.read()

        r2_url = image_storage.upload_image(
            image_data=image_data,
            content_type=content_type,
            memory_id=memory_id,
            image_index=image_index,
        )

        # Build image object for metadata
        image_obj = {"src": r2_url}
        if caption:
            image_obj["caption"] = caption

        return {
            "r2_url": r2_url,
            "image": image_obj,
            "file_path": file_path,
            "content_type": content_type,
            "size_bytes": len(image_data),
        }

    except Exception as e:
        return {"error": "upload_failed", "message": str(e)}


@mcp.tool()
async def memory_migrate_images(dry_run: bool = False) -> Dict[str, Any]:
    """Migrate existing base64 images to R2 storage.

    Scans all memories and uploads any base64-encoded images to R2,
    replacing the data URIs with R2 URLs.

    Args:
        dry_run: If True, only report what would be migrated without making changes

    Returns:
        Dictionary with migration results including count of migrated images
    """
    return _migrate_images_to_r2(dry_run=dry_run)


@_with_connection(writes=True)
def _migrate_images_to_r2(conn, dry_run: bool = False) -> Dict[str, Any]:
    """Migrate all base64 images to R2 storage."""
    import json as json_lib
    from .image_storage import get_image_storage_instance, parse_data_uri
    from .storage import update_memory

    image_storage = get_image_storage_instance()
    if not image_storage:
        return {
            "error": "r2_not_configured",
            "message": "R2 storage is not configured. Set MEMORA_STORAGE_URI to s3:// and configure AWS credentials.",
        }

    # Find memories with base64 images
    rows = conn.execute(
        "SELECT id, metadata FROM memories WHERE metadata LIKE '%data:image%'"
    ).fetchall()

    if not rows:
        return {"migrated_memories": 0, "migrated_images": 0, "message": "No base64 images found"}

    results = {
        "dry_run": dry_run,
        "memories_scanned": len(rows),
        "migrated_memories": 0,
        "migrated_images": 0,
        "errors": [],
    }

    for row in rows:
        memory_id = row["id"]
        try:
            metadata = json_lib.loads(row["metadata"]) if row["metadata"] else {}
        except json_lib.JSONDecodeError:
            continue

        images = metadata.get("images", [])
        if not isinstance(images, list):
            continue

        updated = False
        for idx, img in enumerate(images):
            if not isinstance(img, dict):
                continue
            src = img.get("src", "")
            if not src.startswith("data:image"):
                continue

            if dry_run:
                results["migrated_images"] += 1
                updated = True
                continue

            # Upload to R2
            try:
                image_bytes, content_type = parse_data_uri(src)
                new_url = image_storage.upload_image(
                    image_data=image_bytes,
                    content_type=content_type,
                    memory_id=memory_id,
                    image_index=idx,
                )
                img["src"] = new_url
                results["migrated_images"] += 1
                updated = True
            except Exception as e:
                results["errors"].append({
                    "memory_id": memory_id,
                    "image_index": idx,
                    "error": str(e),
                })

        if updated:
            results["migrated_memories"] += 1
            if not dry_run:
                # Update the memory with new URLs
                update_memory(conn, memory_id, metadata=metadata)

    if dry_run:
        results["message"] = f"Would migrate {results['migrated_images']} images from {results['migrated_memories']} memories"
    else:
        results["message"] = f"Migrated {results['migrated_images']} images from {results['migrated_memories']} memories"

    return results


@_with_connection
def _export_graph_html(conn, output_path: Optional[str], min_score: float) -> Dict[str, Any]:
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
            "title": f"Memory #{m['id']}",
            "color": tag_colors[primary_tag],
        })

        # Expand R2 URLs in metadata for graph visualization
        meta = m.get("metadata") or {}
        if meta.get("images"):
            from .image_storage import expand_r2_url
            expanded_images = []
            for img in meta["images"]:
                if isinstance(img, dict) and img.get("src"):
                    src = img["src"]
                    # Expand r2:// or /r2/ URLs to full URLs
                    if src.startswith("r2://") or src.startswith("/r2/"):
                        src = expand_r2_url(src.replace("/r2/", "r2://") if src.startswith("/r2/") else src, use_proxy=True)
                    expanded_images.append({**img, "src": src})
                else:
                    expanded_images.append(img)
            meta = {**meta, "images": expanded_images}

        memories_data[m["id"]] = {
            "id": m["id"],
            "tags": tags,
            "created": m.get("created_at", ""),
            "content": content,
            "metadata": meta
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
        f'<div class="legend-item" data-tag="{t}" onclick="filterByTag(\'{t}\')"><span class="legend-color" style="background:{c}"></span>{t}</div>'
        for t, c in list(tag_colors.items())[:12]
    )

    # Build tag to node mapping
    tag_to_nodes = {}
    for m in memories:
        for tag in m.get("tags", []):
            if tag not in tag_to_nodes:
                tag_to_nodes[tag] = []
            tag_to_nodes[tag].append(m["id"])

    # Build section/subsection to node mapping (supports nested paths like "Development/Issues")
    section_to_nodes = {}
    path_to_nodes = {}  # Full path -> node ids
    for m in memories:
        meta = m.get("metadata") or {}
        section = meta.get("section", "Uncategorized")
        subsection = meta.get("subsection", "")

        if section not in section_to_nodes:
            section_to_nodes[section] = []
        section_to_nodes[section].append(m["id"])

        if subsection:
            # Build all intermediate paths (e.g., "Dev" and "Dev/Issues" from "Dev/Issues")
            # Add memory to ALL levels so parent shows cumulative count
            parts = subsection.split("/")
            for i in range(len(parts)):
                partial_path = "/".join(parts[:i+1])
                full_key = f"{section}/{partial_path}"
                if full_key not in path_to_nodes:
                    path_to_nodes[full_key] = []
                path_to_nodes[full_key].append(m["id"])

    # Build sections HTML with nested hierarchy
    sections_html = ""
    for section, node_ids in section_to_nodes.items():
        sections_html += f'<div class="section-item" data-section="{section}" onclick="filterBySection(\'{section}\')">{section} ({len(node_ids)})</div>'

        # Get all paths under this section and sort by depth
        section_paths = sorted([k for k in path_to_nodes.keys() if k.startswith(section + "/")])
        rendered_paths = set()

        for full_path in section_paths:
            sub_path = full_path[len(section)+1:]  # Remove "Section/" prefix
            parts = sub_path.split("/")

            # Render each level of the path
            for i, part in enumerate(parts):
                partial = "/".join(parts[:i+1])
                render_key = f"{section}/{partial}"

                if render_key not in rendered_paths:
                    rendered_paths.add(render_key)
                    indent = "&nbsp;&nbsp;" * i
                    count = len(path_to_nodes.get(render_key, []))
                    sections_html += f'<div class="subsection-item" data-subsection="{render_key}" onclick="filterBySubsection(\'{render_key}\')" style="padding-left:{8 + i*12}px;">{indent}└ {part} ({count})</div>'

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Memory Knowledge Graph</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; display: flex; height: 100vh; }}
        #graph {{ flex: 1; height: 100%; }}
        #panel {{ width: 400px; background: #161b22; border-left: 1px solid #30363d; padding: 20px; overflow-y: auto; display: none; position: relative; }}
        #panel.active {{ display: block; }}
        #panel h2 {{ color: #58a6ff; margin-bottom: 10px; font-size: 16px; }}
        #panel .tags {{ margin-bottom: 15px; }}
        #panel .tag {{ display: inline-block; background: #30363d; padding: 3px 8px; border-radius: 12px; font-size: 12px; margin: 2px; cursor: pointer; }}
        #panel .tag:hover {{ background: #484f58; }}
        #panel .meta {{ color: #8b949e; font-size: 12px; margin-bottom: 15px; }}
        #panel .content {{ font-size: 13px; line-height: 1.6; background: #0d1117; padding: 15px; border-radius: 6px; max-height: calc(100vh - 200px); overflow-y: auto; }}
        #panel .content h1, #panel .content h2, #panel .content h3 {{ color: #58a6ff; margin: 16px 0 8px 0; }}
        #panel .content h1 {{ font-size: 1.4em; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
        #panel .content h2 {{ font-size: 1.2em; }}
        #panel .content h3 {{ font-size: 1.1em; }}
        #panel .content p {{ margin: 8px 0; }}
        #panel .content ul, #panel .content ol {{ margin: 8px 0 8px 20px; }}
        #panel .content li {{ margin: 4px 0; }}
        #panel .content code {{ background: #30363d; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 12px; }}
        #panel .content pre {{ background: #0d1117; border: 1px solid #30363d; padding: 12px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }}
        #panel .content pre code {{ background: none; padding: 0; }}
        #panel .content a {{ color: #58a6ff; }}
        #panel .content table {{ border-collapse: collapse; margin: 8px 0; width: 100%; }}
        #panel .content th, #panel .content td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: left; }}
        #panel .content th {{ background: #21262d; }}
        #panel .content blockquote {{ border-left: 3px solid #30363d; padding-left: 12px; margin: 8px 0; color: #8b949e; }}
        #panel .content .mermaid {{ background: #161b22; padding: 16px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }}
        #panel .content .memory-images {{ margin-top: 16px; border-top: 1px solid #30363d; padding-top: 16px; }}
        #panel .content .memory-image {{ margin: 8px 0; }}
        #panel .content .memory-image img {{ max-width: 100%; border-radius: 6px; border: 1px solid #30363d; }}
        #panel .content .memory-image .caption {{ font-size: 11px; color: #8b949e; margin-top: 4px; text-align: center; }}
        #panel .content strong {{ color: #f0f6fc; }}
        #panel .close {{ position: absolute; top: 10px; right: 15px; cursor: pointer; font-size: 20px; color: #8b949e; }}
        #panel .close:hover {{ color: #fff; }}
        #resize-handle {{ width: 6px; background: #30363d; cursor: ew-resize; display: none; }}
        #resize-handle:hover, #resize-handle.dragging {{ background: #58a6ff; }}
        #resize-handle.active {{ display: block; }}
        #legend {{ position: absolute; top: 10px; left: 10px; background: rgba(22,27,34,0.9); padding: 12px; border-radius: 6px; font-size: 12px; }}
        .legend-item {{ margin: 4px 0; display: flex; align-items: center; cursor: pointer; padding: 2px 4px; border-radius: 4px; }}
        .legend-item:hover {{ background: rgba(255,255,255,0.1); }}
        .legend-item.active {{ background: rgba(88,166,255,0.3); }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
        #legend .reset {{ margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d; color: #58a6ff; cursor: pointer; }}
        #legend .reset:hover {{ text-decoration: underline; }}
        #sections {{ position: absolute; bottom: 50px; left: 10px; background: rgba(22,27,34,0.9); padding: 12px; border-radius: 6px; font-size: 12px; max-height: 40vh; overflow-y: auto; white-space: nowrap; }}
        #sections b {{ display: block; margin-bottom: 8px; }}
        .section-item {{ margin: 4px 0; cursor: pointer; padding: 3px 6px; border-radius: 4px; color: #7ee787; }}
        .section-item:hover {{ background: rgba(255,255,255,0.1); }}
        .section-item.active {{ background: rgba(126,231,135,0.3); }}
        .subsection-item {{ margin: 2px 0 2px 8px; cursor: pointer; padding: 2px 6px; border-radius: 4px; color: #8b949e; font-size: 11px; }}
        .subsection-item:hover {{ background: rgba(255,255,255,0.1); }}
        .subsection-item.active {{ background: rgba(88,166,255,0.3); color: #c9d1d9; }}
        #help {{ position: absolute; bottom: 10px; left: 10px; background: rgba(22,27,34,0.9); padding: 8px 12px; border-radius: 6px; font-size: 11px; color: #8b949e; }}
    </style>
</head>
<body>
    <div id="graph"></div>
    <div id="resize-handle"></div>
    <div id="panel">
        <span class="close" onclick="closePanel()">&times;</span>
        <h2 id="panel-title">Memory #</h2>
        <div class="meta" id="panel-meta"></div>
        <div class="tags" id="panel-tags"></div>
        <div class="content" id="panel-content"></div>
    </div>
    <div id="legend"><b>Tags</b>{legend_html}<div class="reset" onclick="resetFilter()">Show All</div></div>
    <div id="sections"><b>Sections</b>{sections_html}</div>
    <div id="help">Click tag/section to filter | Click node to view | Scroll to zoom</div>
    <script>
        var memoriesData = {json_lib.dumps(memories_data)};
        var tagToNodes = {json_lib.dumps(tag_to_nodes)};
        var sectionToNodes = {json_lib.dumps(section_to_nodes)};
        var subsectionToNodes = {json_lib.dumps(path_to_nodes)};
        var allNodes = {json_lib.dumps(nodes)};
        var allEdges = {json_lib.dumps(edges)};
        var nodes = new vis.DataSet(allNodes);
        var edges = new vis.DataSet(allEdges);
        var container = document.getElementById("graph");
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            nodes: {{ shape: "dot", size: 16, font: {{ color: "#c9d1d9", size: 11 }}, borderWidth: 2 }},
            edges: {{ color: {{ color: "#30363d", opacity: 0.6 }}, smooth: {{ type: "continuous" }} }},
            physics: {{ barnesHut: {{ gravitationalConstant: -2500, springLength: 120, damping: 0.3 }} }},
            interaction: {{ hover: true, tooltipDelay: 100 }}
        }};
        var network = new vis.Network(container, data, options);
        var currentFilter = null;

        function filterByTag(tag) {{
            document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
            var el = document.querySelector('.legend-item[data-tag="' + tag + '"]');
            if (el) el.classList.add('active');
            currentFilter = tag;
            var nodeIds = tagToNodes[tag] || [];
            var nodeSet = new Set(nodeIds);
            nodes.clear();
            edges.clear();
            var filteredNodes = allNodes.filter(n => nodeSet.has(n.id));
            var filteredEdges = allEdges.filter(e => nodeSet.has(e.from) && nodeSet.has(e.to));
            nodes.add(filteredNodes);
            edges.add(filteredEdges);
            network.fit({{ animation: true }});
        }}

        function resetFilter() {{
            document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
            currentFilter = null;
            nodes.clear();
            edges.clear();
            nodes.add(allNodes);
            edges.add(allEdges);
            network.fit({{ animation: true }});
        }}

        function filterBySection(section) {{
            document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
            document.querySelector('.section-item[data-section="' + section + '"]').classList.add('active');
            currentFilter = section;
            var nodeIds = sectionToNodes[section] || [];
            var nodeSet = new Set(nodeIds);
            nodes.clear();
            edges.clear();
            var filteredNodes = allNodes.filter(n => nodeSet.has(n.id));
            var filteredEdges = allEdges.filter(e => nodeSet.has(e.from) && nodeSet.has(e.to));
            nodes.add(filteredNodes);
            edges.add(filteredEdges);
            network.fit({{ animation: true }});
        }}

        function filterBySubsection(subsection) {{
            document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
            document.querySelector('.subsection-item[data-subsection="' + subsection + '"]').classList.add('active');
            currentFilter = subsection;
            var nodeIds = subsectionToNodes[subsection] || [];
            var nodeSet = new Set(nodeIds);
            nodes.clear();
            edges.clear();
            var filteredNodes = allNodes.filter(n => nodeSet.has(n.id));
            var filteredEdges = allEdges.filter(e => nodeSet.has(e.from) && nodeSet.has(e.to));
            nodes.add(filteredNodes);
            edges.add(filteredEdges);
            network.fit({{ animation: true }});
        }}

        // Configure marked for GitHub-flavored markdown
        marked.setOptions({{
            breaks: true,
            gfm: true
        }});

        // Initialize mermaid with dark theme
        mermaid.initialize({{
            startOnLoad: false,
            theme: 'dark',
            themeVariables: {{
                primaryColor: '#58a6ff',
                primaryTextColor: '#c9d1d9',
                primaryBorderColor: '#30363d',
                lineColor: '#8b949e',
                secondaryColor: '#21262d',
                tertiaryColor: '#161b22'
            }}
        }});

        function renderMarkdown(text) {{
            // Use marked to render markdown, links open in new tab
            var renderer = new marked.Renderer();
            renderer.link = function(href, title, text) {{
                return '<a href="' + href + '" target="_blank">' + text + '</a>';
            }};
            return marked.parse(text, {{ renderer: renderer }});
        }}

        async function renderMermaidBlocks() {{
            // Find mermaid code blocks and render them as diagrams
            var blocks = document.querySelectorAll('#panel-content pre code.language-mermaid');
            for (var block of blocks) {{
                var container = document.createElement('div');
                container.className = 'mermaid';
                container.textContent = block.textContent;
                block.parentElement.replaceWith(container);
            }}
            if (blocks.length > 0) {{
                await mermaid.run();
            }}
        }}

        function renderImages(metadata) {{
            var images = metadata && metadata.images;
            if (!images || images.length === 0) return '';

            var html = '<div class="memory-images">';
            for (var img of images) {{
                html += '<div class="memory-image">';
                html += '<img src="' + img.src + '" alt="' + (img.caption || '') + '">';
                if (img.caption) {{
                    html += '<div class="caption">' + img.caption + '</div>';
                }}
                html += '</div>';
            }}
            html += '</div>';
            return html;
        }}

        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var mem = memoriesData[nodeId];
                if (mem) {{
                    document.getElementById("panel-title").textContent = "Memory #" + mem.id;
                    document.getElementById("panel-meta").textContent = "Created: " + mem.created;
                    document.getElementById("panel-tags").innerHTML = mem.tags.map(t => '<span class="tag" onclick="filterByTag(\\''+t+'\\'); event.stopPropagation();">' + t + '</span>').join("");
                    document.getElementById("panel-content").innerHTML = renderMarkdown(mem.content);
                    renderMermaidBlocks();
                    document.getElementById("panel-content").innerHTML += renderImages(mem.metadata);
                    document.getElementById("panel").classList.add("active");
                    document.getElementById("resize-handle").classList.add("active");
                }}
            }}
        }});
        function closePanel() {{
            document.getElementById("panel").classList.remove("active");
            document.getElementById("resize-handle").classList.remove("active");
        }}

        // Resize handle logic
        var resizeHandle = document.getElementById("resize-handle");
        var panel = document.getElementById("panel");
        var isResizing = false;

        resizeHandle.addEventListener("mousedown", function(e) {{
            isResizing = true;
            resizeHandle.classList.add("dragging");
            document.body.style.cursor = "ew-resize";
            e.preventDefault();
        }});

        document.addEventListener("mousemove", function(e) {{
            if (!isResizing) return;
            var newWidth = window.innerWidth - e.clientX;
            if (newWidth >= 200 && newWidth <= 800) {{
                panel.style.width = newWidth + "px";
            }}
        }});

        document.addEventListener("mouseup", function() {{
            isResizing = false;
            resizeHandle.classList.remove("dragging");
            document.body.style.cursor = "";
        }});
    </script>
</body>
</html>'''

    result = {
        "nodes": len(nodes),
        "edges": len(edges),
        "tags": list(tag_colors.keys()),
    }

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(html)
        result["path"] = output_path
    else:
        result["html"] = html

    return result


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


def _is_port_in_use(host: str, port: int) -> bool:
    """Check if a port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _get_graph_data(min_score: float = 0.25, rebuild: bool = False) -> dict:
    """Get graph nodes, edges, and metadata for API response.

    Args:
        min_score: Minimum similarity score for edges
        rebuild: If True, rebuild crossrefs (slow). If False, use existing crossrefs.
    """
    import json as json_lib
    from .storage import list_memories, get_crossrefs, rebuild_crossrefs

    conn = connect()
    try:
        memories = list_memories(conn, None, None, None, 0, None, None, None, None, None)
        if not memories:
            return {"error": "no_memories", "message": "No memories to visualize"}

        # Only rebuild crossrefs if explicitly requested (expensive operation)
        if rebuild:
            rebuild_crossrefs(conn)

        colors = ["#58a6ff", "#f78166", "#a371f7", "#7ee787", "#ffa657", "#ff7b72", "#79c0ff", "#d2a8ff"]
        tag_colors = {}

        nodes = []
        tag_to_nodes = {}
        section_to_nodes = {}
        path_to_nodes = {}

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
                "title": f"Memory #{m['id']}",
                "color": tag_colors[primary_tag],
            })

            # Build tag mapping
            for tag in tags:
                if tag not in tag_to_nodes:
                    tag_to_nodes[tag] = []
                tag_to_nodes[tag].append(m["id"])

            # Build section mapping using hierarchy.path
            meta = m.get("metadata") or {}
            hierarchy = meta.get("hierarchy", {})
            hierarchy_path = hierarchy.get("path", []) if isinstance(hierarchy, dict) else []

            # Use hierarchy.path if available, otherwise fall back to section/subsection
            if hierarchy_path and len(hierarchy_path) >= 1:
                section = hierarchy_path[0]
                parts = hierarchy_path[1:]  # Remaining path elements after section
            else:
                section = meta.get("section", "Uncategorized")
                subsection = meta.get("subsection", "")
                parts = subsection.split("/") if subsection else []

            if section not in section_to_nodes:
                section_to_nodes[section] = []
            section_to_nodes[section].append(m["id"])

            if parts:
                for i in range(len(parts)):
                    partial_path = "/".join(parts[:i+1])
                    full_key = f"{section}/{partial_path}"
                    if full_key not in path_to_nodes:
                        path_to_nodes[full_key] = []
                    path_to_nodes[full_key].append(m["id"])

        edges = []
        seen = set()
        for m in memories:
            for ref in get_crossrefs(conn, m["id"]):
                edge_key = tuple(sorted([m["id"], ref["id"]]))
                if edge_key not in seen and ref.get("score", 0) > min_score:
                    seen.add(edge_key)
                    edges.append({"from": m["id"], "to": ref["id"]})

        return {
            "nodes": nodes,
            "edges": edges,
            "tagColors": tag_colors,
            "tagToNodes": tag_to_nodes,
            "sectionToNodes": section_to_nodes,
            "subsectionToNodes": path_to_nodes,
        }
    finally:
        conn.close()


def _get_memory_for_api(memory_id: int) -> dict:
    """Get a single memory with expanded R2 URLs for API response."""
    conn = connect()
    try:
        m = get_memory(conn, memory_id)
        if not m:
            return {"error": "not_found"}

        # Expand R2 URLs in metadata
        meta = m.get("metadata") or {}
        if meta.get("images"):
            from .image_storage import expand_r2_url
            expanded_images = []
            for img in meta["images"]:
                if isinstance(img, dict) and img.get("src"):
                    src = img["src"]
                    if src.startswith("r2://") or src.startswith("/r2/"):
                        src = expand_r2_url(src.replace("/r2/", "r2://") if src.startswith("/r2/") else src, use_proxy=True)
                    expanded_images.append({**img, "src": src})
                else:
                    expanded_images.append(img)
            meta = {**meta, "images": expanded_images}

        return {
            "id": m["id"],
            "tags": m.get("tags", []),
            "created": m.get("created_at", ""),
            "content": m["content"],
            "metadata": meta
        }
    finally:
        conn.close()


def _start_graph_server(host: str, port: int) -> None:
    """Start background HTTP server for graph visualization."""
    import threading
    import sys

    # Skip if port already in use (another instance running)
    if _is_port_in_use(host, port):
        print(f"Graph server port {port} already in use, skipping", file=sys.stderr)
        return

    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import Response

    # Static SPA HTML - fetches data from API
    GRAPH_HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Memory Knowledge Graph</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; display: flex; height: 100vh; }
        #graph { flex: 1; height: 100%; }
        #panel { width: 400px; background: #161b22; border-left: 1px solid #30363d; padding: 20px; overflow-y: auto; display: none; position: relative; }
        #panel.active { display: block; }
        #panel h2 { color: #58a6ff; margin-bottom: 10px; font-size: 16px; }
        #panel .tags { margin-bottom: 15px; }
        #panel .tag { display: inline-block; background: #30363d; padding: 3px 8px; border-radius: 12px; font-size: 12px; margin: 2px; cursor: pointer; }
        #panel .tag:hover { background: #484f58; }
        #panel .meta { color: #8b949e; font-size: 12px; margin-bottom: 15px; }
        #panel .content { font-size: 13px; line-height: 1.6; background: #0d1117; padding: 15px; border-radius: 6px; max-height: calc(100vh - 200px); overflow-y: auto; }
        #panel .content h1, #panel .content h2, #panel .content h3 { color: #58a6ff; margin: 16px 0 8px 0; }
        #panel .content h1 { font-size: 1.4em; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
        #panel .content h2 { font-size: 1.2em; }
        #panel .content h3 { font-size: 1.1em; }
        #panel .content p { margin: 8px 0; }
        #panel .content ul, #panel .content ol { margin: 8px 0 8px 20px; }
        #panel .content li { margin: 4px 0; }
        #panel .content code { background: #30363d; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 12px; }
        #panel .content pre { background: #0d1117; border: 1px solid #30363d; padding: 12px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }
        #panel .content pre code { background: none; padding: 0; }
        #panel .content a { color: #58a6ff; }
        #panel .content table { border-collapse: collapse; margin: 8px 0; width: 100%; }
        #panel .content th, #panel .content td { border: 1px solid #30363d; padding: 6px 10px; text-align: left; }
        #panel .content th { background: #21262d; }
        #panel .content blockquote { border-left: 3px solid #30363d; padding-left: 12px; margin: 8px 0; color: #8b949e; }
        #panel .content .mermaid { background: #161b22; padding: 16px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }
        #panel .content .memory-images { margin-top: 16px; border-top: 1px solid #30363d; padding-top: 16px; }
        #panel .content .memory-image { margin: 8px 0; }
        #panel .content .memory-image img { max-width: 100%; border-radius: 6px; border: 1px solid #30363d; }
        #panel .content .memory-image .caption { font-size: 11px; color: #8b949e; margin-top: 4px; text-align: center; }
        #panel .content strong { color: #f0f6fc; }
        #panel .close { position: absolute; top: 10px; right: 15px; cursor: pointer; font-size: 20px; color: #8b949e; }
        #panel .close:hover { color: #fff; }
        #resize-handle { width: 6px; background: #30363d; cursor: ew-resize; display: none; }
        #resize-handle:hover, #resize-handle.dragging { background: #58a6ff; }
        #resize-handle.active { display: block; }
        #legend { position: absolute; top: 10px; left: 10px; background: rgba(22,27,34,0.9); padding: 12px; border-radius: 6px; font-size: 12px; }
        .legend-item { margin: 4px 0; display: flex; align-items: center; cursor: pointer; padding: 2px 4px; border-radius: 4px; }
        .legend-item:hover { background: rgba(255,255,255,0.1); }
        .legend-item.active { background: rgba(88,166,255,0.3); }
        .legend-color { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        #legend .reset { margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d; color: #58a6ff; cursor: pointer; }
        #legend .reset:hover { text-decoration: underline; }
        #sections { position: absolute; bottom: 50px; left: 10px; background: rgba(22,27,34,0.9); padding: 12px; border-radius: 6px; font-size: 12px; max-height: 40vh; overflow-y: auto; white-space: nowrap; }
        #sections b { display: block; margin-bottom: 8px; }
        .section-item { margin: 4px 0; cursor: pointer; padding: 3px 6px; border-radius: 4px; color: #7ee787; }
        .section-item:hover { background: rgba(255,255,255,0.1); }
        .section-item.active { background: rgba(126,231,135,0.3); }
        .subsection-item { margin: 2px 0 2px 8px; cursor: pointer; padding: 2px 6px; border-radius: 4px; color: #8b949e; font-size: 11px; }
        .subsection-item:hover { background: rgba(255,255,255,0.1); }
        .subsection-item.active { background: rgba(88,166,255,0.3); color: #c9d1d9; }
        #help { position: absolute; bottom: 10px; left: 10px; background: rgba(22,27,34,0.9); padding: 8px 12px; border-radius: 6px; font-size: 11px; color: #8b949e; }
        #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #58a6ff; font-size: 16px; }
        #search-box { position: absolute; top: 10px; right: 10px; background: rgba(22,27,34,0.9); padding: 8px; border-radius: 6px; }
        #search-box input { background: #0d1117; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 10px; border-radius: 4px; width: 200px; }
        #search-box input:focus { outline: none; border-color: #58a6ff; }
    </style>
</head>
<body>
    <div id="graph"><div id="loading">Loading memories...</div></div>
    <div id="resize-handle"></div>
    <div id="panel">
        <span class="close" onclick="closePanel()">&times;</span>
        <h2 id="panel-title">Memory #</h2>
        <div class="meta" id="panel-meta"></div>
        <div class="tags" id="panel-tags"></div>
        <div class="content" id="panel-content"></div>
    </div>
    <div id="legend"><b>Tags</b><div id="legend-items"></div><div class="reset" onclick="resetFilter()">Show All</div></div>
    <div id="sections"><b>Sections</b><div id="section-items"></div></div>
    <div id="search-box"><input type="text" id="search" placeholder="Search memories..." oninput="searchMemories(this.value)"></div>
    <div id="help">Click tag/section to filter | Click node to view | Scroll to zoom | Type to search</div>
    <script>
        var graphData = null;
        var nodes, edges, network;
        var currentFilter = null;
        var memoryCache = {};

        // Configure marked for GitHub-flavored markdown
        marked.setOptions({ breaks: true, gfm: true });

        // Initialize mermaid with dark theme
        mermaid.initialize({
            startOnLoad: false,
            theme: 'dark',
            themeVariables: {
                primaryColor: '#58a6ff',
                primaryTextColor: '#c9d1d9',
                primaryBorderColor: '#30363d',
                lineColor: '#8b949e',
                secondaryColor: '#21262d',
                tertiaryColor: '#161b22'
            }
        });

        async function loadGraph() {
            try {
                const response = await fetch('/api/graph');
                graphData = await response.json();
                if (graphData.error) {
                    document.getElementById('loading').textContent = graphData.message || 'No memories found';
                    return;
                }
                initGraph();
            } catch (e) {
                document.getElementById('loading').textContent = 'Error loading graph: ' + e.message;
            }
        }

        function initGraph() {
            document.getElementById('loading').remove();

            // Build legend
            var legendHtml = '';
            var tagEntries = Object.entries(graphData.tagColors).slice(0, 12);
            for (var [tag, color] of tagEntries) {
                legendHtml += '<div class="legend-item" data-tag="' + tag + '" onclick="filterByTag(\\'' + tag + '\\')"><span class="legend-color" style="background:' + color + '"></span>' + tag + '</div>';
            }
            document.getElementById('legend-items').innerHTML = legendHtml;

            // Build sections
            var sectionsHtml = '';
            for (var [section, nodeIds] of Object.entries(graphData.sectionToNodes)) {
                sectionsHtml += '<div class="section-item" data-section="' + section + '" onclick="filterBySection(\\'' + section + '\\')">' + section + ' (' + nodeIds.length + ')</div>';
                var sectionPaths = Object.keys(graphData.subsectionToNodes).filter(k => k.startsWith(section + '/')).sort();
                var rendered = new Set();
                for (var fullPath of sectionPaths) {
                    var subPath = fullPath.slice(section.length + 1);
                    var parts = subPath.split('/');
                    for (var i = 0; i < parts.length; i++) {
                        var partial = parts.slice(0, i + 1).join('/');
                        var renderKey = section + '/' + partial;
                        if (!rendered.has(renderKey)) {
                            rendered.add(renderKey);
                            var indent = '&nbsp;&nbsp;'.repeat(i);
                            var count = (graphData.subsectionToNodes[renderKey] || []).length;
                            sectionsHtml += '<div class="subsection-item" data-subsection="' + renderKey + '" onclick="filterBySubsection(\\'' + renderKey + '\\')" style="padding-left:' + (8 + i*12) + 'px;">' + indent + '\\u2514 ' + parts[i] + ' (' + count + ')</div>';
                        }
                    }
                }
            }
            document.getElementById('section-items').innerHTML = sectionsHtml;

            // Init vis.js
            nodes = new vis.DataSet(graphData.nodes);
            edges = new vis.DataSet(graphData.edges);
            var container = document.getElementById('graph');
            var data = { nodes: nodes, edges: edges };
            var options = {
                nodes: { shape: 'dot', size: 16, font: { color: '#c9d1d9', size: 11 }, borderWidth: 2 },
                edges: { color: { color: '#30363d', opacity: 0.6 }, smooth: { type: 'continuous' } },
                physics: { barnesHut: { gravitationalConstant: -2500, springLength: 120, damping: 0.3 } },
                interaction: { hover: true, tooltipDelay: 100 }
            };
            network = new vis.Network(container, data, options);

            network.on('click', async function(params) {
                if (params.nodes.length > 0) {
                    var nodeId = params.nodes[0];
                    await showMemory(nodeId);
                }
            });
        }

        async function showMemory(nodeId) {
            // Fetch memory content (with caching)
            if (!memoryCache[nodeId]) {
                try {
                    const response = await fetch('/api/memories/' + nodeId);
                    memoryCache[nodeId] = await response.json();
                } catch (e) {
                    console.error('Error fetching memory:', e);
                    return;
                }
            }
            var mem = memoryCache[nodeId];
            if (mem.error) return;

            document.getElementById('panel-title').textContent = 'Memory #' + mem.id;
            document.getElementById('panel-meta').textContent = 'Created: ' + mem.created;
            document.getElementById('panel-tags').innerHTML = mem.tags.map(t => '<span class="tag" onclick="filterByTag(\\'' + t + '\\'); event.stopPropagation();">' + t + '</span>').join('');
            document.getElementById('panel-content').innerHTML = renderMarkdown(mem.content);
            await renderMermaidBlocks();
            document.getElementById('panel-content').innerHTML += renderImages(mem.metadata);
            document.getElementById('panel').classList.add('active');
            document.getElementById('resize-handle').classList.add('active');
        }

        function renderMarkdown(text) {
            var renderer = new marked.Renderer();
            renderer.link = function(href, title, text) {
                return '<a href="' + href + '" target="_blank">' + text + '</a>';
            };
            return marked.parse(text, { renderer: renderer });
        }

        async function renderMermaidBlocks() {
            var blocks = document.querySelectorAll('#panel-content pre code.language-mermaid');
            for (var block of blocks) {
                var container = document.createElement('div');
                container.className = 'mermaid';
                container.textContent = block.textContent;
                block.parentElement.replaceWith(container);
            }
            if (blocks.length > 0) await mermaid.run();
        }

        function renderImages(metadata) {
            var images = metadata && metadata.images;
            if (!images || images.length === 0) return '';
            var html = '<div class="memory-images">';
            for (var img of images) {
                html += '<div class="memory-image"><img src="' + img.src + '" alt="' + (img.caption || '') + '">';
                if (img.caption) html += '<div class="caption">' + img.caption + '</div>';
                html += '</div>';
            }
            return html + '</div>';
        }

        function filterByTag(tag) {
            document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
            var el = document.querySelector('.legend-item[data-tag="' + tag + '"]');
            if (el) el.classList.add('active');
            currentFilter = tag;
            var nodeIds = graphData.tagToNodes[tag] || [];
            applyFilter(nodeIds);
        }

        function filterBySection(section) {
            document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
            var el = document.querySelector('.section-item[data-section="' + section + '"]');
            if (el) el.classList.add('active');
            currentFilter = section;
            var nodeIds = graphData.sectionToNodes[section] || [];
            applyFilter(nodeIds);
        }

        function filterBySubsection(subsection) {
            document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
            var el = document.querySelector('.subsection-item[data-subsection="' + subsection + '"]');
            if (el) el.classList.add('active');
            currentFilter = subsection;
            var nodeIds = graphData.subsectionToNodes[subsection] || [];
            applyFilter(nodeIds);
        }

        function applyFilter(nodeIds) {
            var nodeSet = new Set(nodeIds);
            nodes.clear();
            edges.clear();
            var filteredNodes = graphData.nodes.filter(n => nodeSet.has(n.id));
            var filteredEdges = graphData.edges.filter(e => nodeSet.has(e.from) && nodeSet.has(e.to));
            nodes.add(filteredNodes);
            edges.add(filteredEdges);
            network.fit({ animation: true });
        }

        function resetFilter() {
            document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
            currentFilter = null;
            nodes.clear();
            edges.clear();
            nodes.add(graphData.nodes);
            edges.add(graphData.edges);
            network.fit({ animation: true });
        }

        function searchMemories(query) {
            if (!query || query.length < 2) {
                resetFilter();
                return;
            }
            query = query.toLowerCase();
            var matchingIds = graphData.nodes.filter(n => n.label.toLowerCase().includes(query)).map(n => n.id);
            applyFilter(matchingIds);
        }

        function closePanel() {
            document.getElementById('panel').classList.remove('active');
            document.getElementById('resize-handle').classList.remove('active');
        }

        // Resize handle logic
        var resizeHandle = document.getElementById('resize-handle');
        var panel = document.getElementById('panel');
        var isResizing = false;

        resizeHandle.addEventListener('mousedown', function(e) {
            isResizing = true;
            resizeHandle.classList.add('dragging');
            document.body.style.cursor = 'ew-resize';
            e.preventDefault();
        });

        document.addEventListener('mousemove', function(e) {
            if (!isResizing) return;
            var newWidth = window.innerWidth - e.clientX;
            if (newWidth >= 200 && newWidth <= 800) panel.style.width = newWidth + 'px';
        });

        document.addEventListener('mouseup', function() {
            isResizing = false;
            resizeHandle.classList.remove('dragging');
            document.body.style.cursor = '';
        });

        // Load graph on page load
        loadGraph();
    </script>
</body>
</html>'''

    async def graph_handler(request: Request):
        """Serve the static graph SPA."""
        return HTMLResponse(GRAPH_HTML)

    async def api_graph(request: Request):
        """API endpoint: Get graph nodes and edges."""
        try:
            min_score = float(request.query_params.get("min_score", 0.25))
            rebuild = request.query_params.get("rebuild", "").lower() == "true"
            result = _get_graph_data(min_score, rebuild=rebuild)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_memory(request: Request):
        """API endpoint: Get a single memory by ID."""
        try:
            memory_id = int(request.path_params.get("id"))
            result = _get_memory_for_api(memory_id)
            if result.get("error") == "not_found":
                return JSONResponse(result, status_code=404)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def r2_image_proxy(request: Request):
        """Proxy images from R2 storage."""
        try:
            from .image_storage import get_image_storage_instance

            image_storage = get_image_storage_instance()
            if not image_storage:
                return JSONResponse({"error": "R2 not configured"}, status_code=503)

            key = request.path_params.get("path", "")
            if not key:
                return JSONResponse({"error": "No path provided"}, status_code=400)

            try:
                response = image_storage.s3_client.get_object(
                    Bucket=image_storage.bucket,
                    Key=key,
                )
                image_data = response["Body"].read()
                content_type = response.get("ContentType", "image/jpeg")

                return Response(
                    content=image_data,
                    media_type=content_type,
                    headers={"Cache-Control": "public, max-age=86400"},
                )
            except Exception as e:
                return JSONResponse({"error": f"Image not found: {e}"}, status_code=404)

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    app = Starlette(routes=[
        Route("/graph", graph_handler),
        Route("/api/graph", api_graph),
        Route("/api/memories/{id:int}", api_memory),
        Route("/r2/{path:path}", r2_image_proxy),
    ])

    def run_server():
        import uvicorn
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f"Graph server started at http://{host}:{port}/graph", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Memory MCP Server")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Default: start server (make it the default if no subcommand)
    parser.add_argument(
        "--transport",
        choices=sorted(VALID_TRANSPORTS),
        default=DEFAULT_TRANSPORT,
        help="MCP transport to use (defaults to env MEMORA_TRANSPORT or 'stdio')",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Host interface for HTTP transports (defaults to env MEMORA_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for HTTP transports (defaults to env MEMORA_PORT or 8000)",
    )
    parser.add_argument(
        "--graph-port",
        type=int,
        default=DEFAULT_GRAPH_PORT,
        help="Port for graph visualization server (defaults to env MEMORA_GRAPH_PORT or 8765)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable the graph visualization server",
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

    # Subcommand: migrate-images
    migrate_parser = subparsers.add_parser(
        "migrate-images",
        help="Migrate base64 images to R2 storage"
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
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
    elif args.command == "migrate-images":
        _handle_migrate_images(dry_run=args.dry_run)
    else:
        # Default: start server
        mcp.settings.host = args.host
        mcp.settings.port = args.port

        # Start graph visualization server unless disabled
        if not args.no_graph:
            _start_graph_server(args.host, args.graph_port)

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


def _handle_migrate_images(dry_run: bool = False) -> None:
    """Handle migrate-images command."""
    import json

    print(f"{'[DRY RUN] ' if dry_run else ''}Migrating base64 images to R2 storage...")

    result = _migrate_images_to_r2(dry_run=dry_run)

    if "error" in result:
        print(f"Error: {result['message']}")
        return

    print(json.dumps(result, indent=2))

    if result.get("errors"):
        print(f"\nWarning: {len(result['errors'])} errors occurred during migration")


if __name__ == "__main__":
    main()
