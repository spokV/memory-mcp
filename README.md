# MCP Memory Server

This directory contains a lightweight Model Context Protocol (MCP) helper
that persists shared memories in an on-disk SQLite database. The
`mcp_server.py` entry point exposes a **true MCP server** (via `FastMCP`)
compatible with the Codex CLI and other MCP-aware clients.
## Features
- Semantic search (vector embeddings for concept queries)
- Cross-reference suggestions (auto-linked related memories)
- Tag validation (allowlist enforced on create/batch create)
- Zero external dependencies (built with the Python standard library)
- SQLite-backed storage that survives restarts (`.mcp/memory-mcp/memory_mcp/memories.db`)
- JSON payloads for easy integration with scripts or other agents
- Optional inline task lists with per-task completion flags
- Hierarchical listing tool to explore memories by section/subsection

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

### Task tracking metadata

Memories can now include lightweight checklists by supplying a `tasks` array in
the metadata. Each task requires a `title` and accepts an optional `done`
boolean:

```json
{
  "done": false,
  "tasks": [
    {"title": "compare osc_2_hi_res", "done": false},
    {"title": "prototype RL post-training", "done": true}
  ]
}
```

Tasks may also be provided as plain strings (they will be normalised to objects
with `done: false`). The top-level `done` field is optional and, when present,
is coerced into a boolean.


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

## Hierarchical views

Use the `memory_hierarchy` tool to explore memories grouped by their recorded
`section`/`subsection` (or any `hierarchy.path` supplied at creation time). The
endpoint accepts the same `query` and `metadata_filters` arguments as
`memory_list` plus an optional `include_root` flag:

```json
{
  "count": 4,
  "hierarchy": [
    {
      "name": "Plan",
      "path": ["Plan"],
      "count": 3,
      "memories": [...],
      "children": [
        {
          "name": "Immediate Actions",
          "path": ["Plan", "Immediate Actions"],
          "count": 2,
          "memories": [...],
          "children": []
        }
      ]
    }
  ]
}
```

Passing `include_root=true` returns a synthetic root node containing all
children plus any memories without hierarchy metadata.


## Semantic search
## Cross-reference suggestions

Every time a memory is created or updated, the server computes cosine-similarity against the existing corpus and stores the top matches. Use `memory_related` to retrieve the linked memories (optionally refreshing the links), and `memory_rebuild_crossrefs` to recompute them for the entire database.


Use `memory_semantic_search` to retrieve memories ranked by cosine similarity between lightweight bag-of-words embeddings derived from content, metadata, and tags. The tool accepts `top_k`, optional `metadata_filters`, and an optional `min_score` threshold.

If the embeddings fall out of sync (e.g., after manual DB edits), run `memory_rebuild_embeddings` to regenerate vectors for all entries.

## Tag utilities

Use `memory_tags` to retrieve the current allowlisted tags enforced by the server. This helps clients surface available categories or validate new entries before calling `memory_create`.
Use `memory_tag_hierarchy` to inspect the current tags arranged by dotted namespace.


Use `memory_validate_tags` to report stored entries carrying tags outside the allowlist. It returns the allowed set, an invalid count, and (optionally) the offending memory IDs. Configure the allowlist via `MEMORY_MCP_TAGS` (comma-separated) or `MEMORY_MCP_TAG_FILE` (JSON list).

Allowed tags can use wildcard namespaces (e.g. `beamng.*`) using `MEMORY_MCP_TAGS` or `MEMORY_MCP_TAG_FILE`. Wildcards permit any dotted suffix (like `beamng.plan` or `beamng.plan.rl`).

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
