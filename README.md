# MCP Memory Server

This directory contains a lightweight Model Context Protocol (MCP) helper
that persists shared memories in an on-disk SQLite database. The
`mcp_server.py` entry point exposes a **true MCP server** (via `FastMCP`)
compatible with the Codex CLI and other MCP-aware clients.
## Features
- Zero external dependencies (built with the Python standard library)
- SQLite-backed storage that survives restarts (`.mcp/memory-mcp/memory_mcp/memories.db`)
- JSON payloads for easy integration with scripts or other agents

## Install
Install directly from this repository (editable install recommended during development):

```
pip install -e .mcp/memory-mcp
```

## Running the MCP Server
After installation you can launch the server via the console script or module entry point.

```bash
# stdio (spawned per-client)
memory-mcp-server

# shared HTTP endpoint
memory-mcp-server --transport streamable-http --host 127.0.0.1 --port 8765

# alternative module invocation
python -m memory_mcp --transport streamable-http --host 127.0.0.1 --port 8765
```

The CLI also honours environment variables if you'd rather configure it via
your process manager:

```
MEMORY_MCP_TRANSPORT=streamable-http
MEMORY_MCP_HOST=0.0.0.0
MEMORY_MCP_PORT=8765
```

To enable it in the Codex CLI, add the following to `~/.codex/config.toml` under
the `[mcp_servers]` section:

```toml
  [mcp_servers.memory]
  command = "memory-mcp-server"
  args = []
```

> Tip: ensure the `mcp` Python package is available in your environment
> (`pip install mcp`) before launching the CLI so the server can start.


## Schema Overview
The SQLite table is created on first launch with the structure:

| Column      | Type    | Notes                                      |
|-------------|---------|--------------------------------------------|
| `id`        | INTEGER | Primary key (autoincrement)                |
| `content`   | TEXT    | Memory text (required)                     |
| `metadata`  | TEXT    | Optional JSON blob stored as text          |
| `tags`      | TEXT    | JSON-encoded list of tags                  |
| `created_at`| TEXT    | UTC timestamp set automatically at insert  |

### Metadata shape

Metadata is always stored as JSON. Keys must be strings and values should be
JSON-serialisable primitives or objects. The server automatically normalises
hierarchical information into the following structure:

```json
{
  "hierarchy": {"path": ["Plan", "Research Initiatives"]},
  "section": "Plan",
  "subsection": "Research Initiatives",
  "project": "RL-Curriculum"
}
```

You can supply either `section`/`subsection`, a `hierarchy.path` array, or a
full `hierarchy` object when creating a memory. The server will derive the
canonical structure shown above so clients can rely on consistent field names.

### Filtering by metadata

`memory_list` now accepts an optional `metadata_filters` dictionary in addition
to the free-text `query`. Filters support direct matches on keys such as
`section` or `subsection`, and hierarchical queries via `hierarchy` or
`hierarchy_path`. Examples:

```json
{
  "section": "Plan"
}

{
  "hierarchy": ["Research Supplements"]
}

{
  "hierarchy_path": ["Plan", "Research Initiatives"],
  "project": "RL-Curriculum"
}
```

Filters are applied after any text query, so you can combine substring matching
with structured metadata selection.

### Full-text search

When SQLite is compiled with FTS5 support (default on most platforms), the
server maintains an auxiliary `memories_fts` index. This enables `memory_list`
queries to use full-text search semantics (including multi-term queries and
phrases). If FTS is unavailable or a query is malformed, the server falls back
to the legacy `LIKE`-based substring search automatically.

## Batch operations

Two additional tools simplify syncing large numbers of entries:

- `memory_create_batch(entries=[...])` accepts a list of objects with the same
  fields as `memory_create` (`content`, `metadata`, `tags`). The call returns
  the newly created records in order.
- `memory_delete_batch(ids=[...])` removes multiple memories by id and returns
  the deletion count.

## Notes
- Because the API is intentionally small, authentication is not provided.
  Run the server in trusted environments only.

Feel free to extend the endpoint set as future MCP requirements evolve.
