"""MCP-compatible memory server backed by SQLite."""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .storage import (
    add_memory,
    add_memories,
    connect,
    delete_memory,
    delete_memories,
    get_memory,
    list_memories,
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
def _delete_memory(conn, memory_id: int):
    return delete_memory(conn, memory_id)


@_with_connection
def _list_memories(conn, query: Optional[str], metadata_filters: Optional[Dict[str, Any]]):
    return list_memories(conn, query, metadata_filters)


@_with_connection
def _create_memories(conn, entries: List[Dict[str, Any]]):
    return add_memories(conn, entries)


@_with_connection
def _delete_memories(conn, ids: List[int]):
    return delete_memories(conn, ids)


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
        return {"error": "invalid_metadata", "message": str(exc)}
    return {"memory": record}


@mcp.tool()
async def memory_list(
    query: Optional[str] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """List memories, optionally filtering by substring query or metadata."""
    try:
        items = _list_memories(query, metadata_filters)
    except ValueError as exc:
        return {"error": "invalid_filters", "message": str(exc)}
    return {"count": len(items), "memories": items}


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
async def memory_delete(memory_id: int) -> Dict[str, Any]:
    """Delete a memory by id."""
    if _delete_memory(memory_id):
        return {"status": "deleted", "id": memory_id}
    return {"error": "not_found", "id": memory_id}


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
