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
