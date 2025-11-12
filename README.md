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
# Basic installation
pip install -e .mcp/src/memory-mcp

# With cloud storage support (S3, GCS, Azure)
pip install -e ".mcp/src/memory-mcp[cloud]"

# With all optional features
pip install -e ".mcp/src/memory-mcp[all]"
```

### Configure for Claude Code

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.memory]
command = "memory-mcp-server"
args = []
```

### Optional: Cloud Storage

Enable cloud sync with environment variables:

```bash
# AWS S3
export MEMORY_MCP_STORAGE_URI="s3://my-bucket/memories.db"
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"

# Local file (default: .mcp/memory-mcp/memory_mcp/memories.db)
export MEMORY_MCP_STORAGE_URI="file:///path/to/memories.db"
```

### Usage

The server runs automatically when configured in Claude Code. Manual invocation:

```bash
# Default (stdio mode)
memory-mcp-server

# HTTP endpoint
memory-mcp-server --transport streamable-http --host 127.0.0.1 --port 8765
```

### Claude Code Config
Add to `mcp.json`:
```
{
  "mcpServers": {
    "memory": {
      "command": "/home/spok/miniconda/bin/memory-mcp-server",
      "args": [],
      "env": {
        "MEMORY_MCP_DB_PATH": "/home/spok/.local/share/memory-mcp/memories.db",
        "MEMORY_MCP_ALLOW_ANY_TAG": "1"
      }
    }
  }
}
```
