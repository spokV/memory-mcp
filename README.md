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

## Notes
- Because the API is intentionally small, authentication is not provided.
  Run the server in trusted environments only.

Feel free to extend the endpoint set as future MCP requirements evolve.
