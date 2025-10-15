# MCP Memory Server

This directory contains a lightweight Model Context Protocol (MCP) helper
that persists shared memories in an on-disk SQLite database. The
`mcp_server.py` entry point exposes a **true MCP server** (via `FastMCP`)
compatible with the Codex CLI and other MCP-aware clients.
## Features
- **Event Notifications** - Poll-based event system for inter-agent communication via shared-cache tag
- **Memory Updates** - Edit existing memories without recreating them
- **Date Range Filtering** - Query memories by creation date with ISO or relative formats (7d, 1m, 1y)
- **Advanced Tag Queries** - Filter with AND/OR/NOT logic using tags_any, tags_all, tags_none
- **Statistics & Analytics** - Get insights on tag usage, sections, monthly trends, and connections
- **Export/Import** - Backup and restore memories with merge, replace, or append strategies
- **Multiple Embedding Backends** - Choose between TF-IDF (default), sentence-transformers, or OpenAI
- **Semantic Search** - Vector embeddings for concept-based queries
- **Cross-reference Suggestions** - Auto-linked related memories
- **Tag Validation** - Allowlist enforced on create/batch create
- **Zero External Dependencies** - Built with Python standard library (optional backends available)
- **SQLite-backed Storage** - Persistent storage at `.mcp/memory-mcp/memory_mcp/memories.db`
- **JSON Payloads** - Easy integration with scripts or other agents
- **Task Lists** - Optional inline task lists with per-task completion flags
- **Hierarchical Listing** - Explore memories by section/subsection

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

### Automated Service Setup (macOS)

For multi-client workflows, use the automated setup script to run memory-mcp as a persistent service:

```bash
./setup-memory-mcp-service.sh
```

This creates a launchd service that:
- Auto-starts on system boot
- Runs as a shared HTTP endpoint (one process for all clients)
- Auto-configures Claude Code and Codex CLI
- Provides centralized logging and health monitoring

See [`docs/SHARED-SERVICE-SETUP.md`](docs/SHARED-SERVICE-SETUP.md) for detailed documentation.

### Client Configuration

To enable memory-mcp in the Codex CLI, add the following to `~/.codex/config.toml` under
the `[mcp_servers]` section:

**For stdio mode (default):**
```toml
  [mcp_servers.memory]
  command = "memory-mcp-server"
  args = []
```

**For shared HTTP service:**
```toml
  [mcp_servers.memory]
  url = "http://127.0.0.1:8765/sse"
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

### Pagination

Both `memory_list` and `memory_list_compact` support optional `limit` and
`offset` parameters for paginating large result sets:

```python
# Get first 10 memories
memory_list(limit=10)

# Get next 10 memories
memory_list(limit=10, offset=10)

# Get 5 memories matching a query
memory_list(query="beamng", limit=5)
```

### Compact listing

Use `memory_list_compact` to retrieve memories with minimal fields (id, 80-char
preview, tags, created_at) instead of full content and metadata. This reduces
context consumption by ~80-90% for browsing operations:

```python
# Browse recent memories efficiently
memory_list_compact(limit=20)

# Search with compact results
memory_list_compact(query="training", metadata_filters={"section": "Plan"})
```

Once you identify memories of interest, use `memory_get` to retrieve full details.

### Memory updates

Use `memory_update` to edit existing memories without recreating them:

```python
# Update only content
memory_update(memory_id=42, content="Updated content here")

# Update only tags
memory_update(memory_id=42, tags=["beamng", "training"])

# Update multiple fields
memory_update(
    memory_id=42,
    content="New content",
    metadata={"section": "Plan", "status": "completed"},
    tags=["beamng", "plan"]
)
```

Only the fields you provide are updated. The `created_at` timestamp is preserved, and embeddings and cross-references are automatically updated.

### Date range filtering

Both `memory_list` and `memory_list_compact` support date filtering with `date_from` and `date_to` parameters. Dates can be ISO format or relative:

```python
# ISO format
memory_list(date_from="2025-09-01", date_to="2025-09-30")

# Relative dates
memory_list(date_from="7d")  # Last 7 days
memory_list(date_from="1m")  # Last 1 month
memory_list(date_from="1y")  # Last 1 year

# Combine with other filters
memory_list(
    query="beamng",
    date_from="30d",
    tags_any=["training", "experiments"]
)
```

### Advanced tag queries

Filter memories with complex tag logic using `tags_any` (OR), `tags_all` (AND), and `tags_none` (NOT):

```python
# Match memories with ANY of these tags (OR logic)
memory_list(tags_any=["beamng", "laq"])

# Match memories with ALL of these tags (AND logic)
memory_list(tags_all=["beamng", "training"])

# Exclude memories with ANY of these tags (NOT logic)
memory_list(tags_none=["archived", "deprecated"])

# Combine multiple tag filters
memory_list(
    tags_all=["beamng", "plan"],
    tags_none=["completed"]
)
```

Tag filters work with all listing endpoints: `memory_list`, `memory_list_compact`, and `memory_hierarchy`.

### Statistics and analytics

Use `memory_stats` to get insights about your memory collection:

```python
stats = memory_stats()
```

Returns:
- `total_memories` - Total count
- `unique_tags` - Number of unique tags
- `tag_counts` - Tag usage frequency (sorted)
- `section_counts` - Memories per section
- `subsection_counts` - Memories per subsection
- `monthly_counts` - Memories created per month
- `most_connected` - Top 10 memories by cross-reference count
- `date_range` - Oldest and newest timestamps

### Export and import

Backup or transfer memories with `memory_export` and `memory_import`:

```python
# Export all memories
export = memory_export()
# Returns: {"count": 81, "memories": [...]}

# Import with different strategies
memory_import(data=export["memories"], strategy="append")   # Add all
memory_import(data=export["memories"], strategy="merge")    # Skip duplicates
memory_import(data=export["memories"], strategy="replace")  # Clear and replace
```

The export format preserves all fields including `created_at` timestamps. Import automatically rebuilds embeddings and cross-references.

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


## Semantic search and embeddings

Use `memory_semantic_search` to retrieve memories ranked by cosine similarity using vector embeddings:

```python
memory_semantic_search(
    query="steering control optimization",
    top_k=5,
    metadata_filters={"section": "Plan"},
    min_score=0.3
)
```

### Embedding backends

The server supports three embedding backends:

**TF-IDF** (default, no dependencies):
```bash
MEMORY_MCP_EMBEDDING_MODEL=tfidf
```
Lightweight bag-of-words approach. Fast and reliable for keyword matching.

**Sentence-Transformers** (better semantic understanding):
```bash
MEMORY_MCP_EMBEDDING_MODEL=sentence-transformers
SENTENCE_TRANSFORMERS_MODEL=all-MiniLM-L6-v2  # Optional, this is default
```
Requires: `pip install sentence-transformers`

Much better semantic understanding. Automatically falls back to TF-IDF if library unavailable.

**OpenAI** (highest quality):
```bash
MEMORY_MCP_EMBEDDING_MODEL=openai
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # Optional, this is default
```
Requires: `pip install openai`

Best semantic quality but requires API key and has cost. Falls back to TF-IDF if unavailable or errors occur.

After changing embedding backends, rebuild embeddings:
```python
memory_rebuild_embeddings()
```

## Cross-reference suggestions

Every time a memory is created or updated, the server computes cosine-similarity against the existing corpus and stores the top matches. Use `memory_related` to retrieve the linked memories (optionally refreshing the links), and `memory_rebuild_crossrefs` to recompute them for the entire database.

## Event Notifications

The event system enables asynchronous communication between agents (like codex-cli and Claude Code) by automatically tracking when memories with specific tags are created or updated.

### How it works

When a memory is created or updated with the `"shared-cache"` tag, an event is automatically written to the `memories_events` database table. Other agents can poll for these events to discover new notifications without constantly querying the memories table.

### Polling for events

Use `memory_events_poll` to check for new events:

```python
# Poll for all unconsumed events
events = memory_events_poll()

# Poll for specific tags
events = memory_events_poll(tags_filter=["shared-cache"])

# Poll for events since a specific time
events = memory_events_poll(since_timestamp="2025-10-03T10:00:00Z")

# Include already-consumed events
events = memory_events_poll(unconsumed_only=False)
```

Returns:
```python
{
  "count": 2,
  "events": [
    {
      "id": 5,
      "memory_id": 171,
      "tags": ["shared-cache"],
      "timestamp": "2025-10-03T17:25:53Z",
      "consumed": false
    },
    ...
  ]
}
```

### Marking events as consumed

After processing events, mark them as consumed to avoid re-processing:

```python
# Clear specific events
memory_events_clear(event_ids=[5, 6, 7])
# Returns: {"cleared": 3}
```

### Use case: Inter-agent notifications

**Agent 1 (codex-cli) writes a code review:**
```python
memory_create(
    content="Review findings: Bug in LAQ trainer...",
    tags=["shared-cache"]
)
# Event automatically created
```

**Agent 2 (Claude Code) checks for updates:**
```python
# User asks: "check for updates from codex"
events = memory_events_poll(tags_filter=["shared-cache"])

if events["count"] > 0:
    for event in events["events"]:
        memory = memory_get(memory_id=event["memory_id"])
        # Process the memory...

    # Mark as read
    event_ids = [e["id"] for e in events["events"]]
    memory_events_clear(event_ids=event_ids)
```

**Key points:**
- Events are **manual polling** - agents must explicitly call `memory_events_poll()`
- Event emission is **automatic** for memories with the `"shared-cache"` tag
- Events persist until marked as consumed with `memory_events_clear()`
- The system is **agent-agnostic** - works with any MCP-compatible tool

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
