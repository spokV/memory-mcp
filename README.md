# MCP Memory Server

A lightweight Model Context Protocol (MCP) server that persists shared memories in SQLite. Compatible with Claude Code, Codex CLI, and other MCP-aware clients.

## Features

- **Persistent Storage** - SQLite-backed database with optional cloud sync (S3, GCS, Azure)
- **Semantic Search** - Vector embeddings (TF-IDF, sentence-transformers, or OpenAI)
- **Event Notifications** - Poll-based system for inter-agent communication
- **Advanced Queries** - Full-text search, date ranges, tag filters (AND/OR/NOT)
- **Cross-references** - Auto-linked related memories based on similarity
- **Hierarchical Organization** - Explore memories by section/subsection
- **Export/Import** - Backup and restore with merge strategies
- **Knowledge Graph** - Interactive HTML visualization with filtering
- **Statistics & Analytics** - Tag usage, trends, and connection insights
- **Zero Dependencies** - Works out-of-box with Python stdlib (optional backends available)

## Install

```bash
# From GitHub
pip install git+https://github.com/spokV/memory-mcp.git

# For cloud storage (S3/R2/GCS/Azure), install boto3 separately
pip install boto3

# Or from local clone with extras
pip install -e ".mcp/src/memory-mcp[cloud]"  # includes boto3
pip install -e ".mcp/src/memory-mcp[all]"    # includes cloud + dev tools
```

## Usage

The server runs automatically when configured in Claude Code. Manual invocation:

```bash
# Default (stdio mode)
memory-mcp-server

# HTTP endpoint
memory-mcp-server --transport streamable-http --host 127.0.0.1 --port 8765
```

## Claude Code Config

Add to `.mcp.json` in your project root:

### Local DB
```json
{
  "mcpServers": {
    "memory": {
      "command": "{$HOME}/miniconda/bin/memory-mcp-server",
      "args": [],
      "env": {
        "MEMORY_MCP_DB_PATH": "{$HOME}/.local/share/memory-mcp/memories.db",
        "MEMORY_MCP_ALLOW_ANY_TAG": "1"
      }
    }
  }
}
```

### Cloud DB (S3/R2)
```json
{
  "mcpServers": {
    "memory": {
      "command": "/opt/conda/bin/memory-mcp-server",
      "args": [],
      "env": {
        "AWS_ENDPOINT_URL": "https://xxxxxxx.r2.cloudflarestorage.com",
        "MEMORY_MCP_STORAGE_URI": "s3://memories/memories.db",
        "MEMORY_MCP_CLOUD_ENCRYPT": "true",
        "MEMORY_MCP_ALLOW_ANY_TAG": "1"
      }
    }
  }
}
```

## Neovim Integration

Browse memories directly in Neovim with Telescope. Copy the plugin to your config:

```bash
# For kickstart.nvim / lazy.nvim
cp nvim/memory-mcp.lua ~/.config/nvim/lua/kickstart/plugins/
```

**Usage:** Press `<leader>sm` to open the memory browser with fuzzy search and preview.

Requires: `telescope.nvim`, `plenary.nvim`, and `memory-mcp` installed in your Python environment.

## Knowledge Graph Export

Export memories as an interactive HTML knowledge graph visualization:

```python
# Via MCP tool
memory_export_graph(output_path="~/memories_graph.html", min_score=0.25)
```

**Features:**
- Interactive node-based graph using vis.js
- Click nodes to view full memory content in side panel
- Filter by **tags** (left panel) - click any tag to show only related memories
- Filter by **sections/subsections** (right panel) - hierarchical organization
- Edges show cross-reference similarity between memories
- "Show All" button to reset filters
- Zoom, pan, and drag nodes to explore

**Parameters:**
- `output_path`: Where to save the HTML file (default: `~/memories_graph.html`)
- `min_score`: Minimum similarity score for edges (default: 0.25)
