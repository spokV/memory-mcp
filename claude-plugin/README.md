# Memora Context Plugin for Claude Code

Automatically injects relevant memories from Memora at session start.

## What it does

When you start a Claude Code session, this plugin:
1. Loads memora config from `.mcp.json` (supports cloud storage)
2. Extracts the project name from your working directory
3. Searches Memora for related memories
4. Injects relevant context into the session

## Installation

Symlink or copy to Claude Code plugins directory:

```bash
# Option 1: Symlink (recommended for development)
ln -s ~/repos/agentic-mcp-tools/memora/claude-plugin ~/.claude/plugins/memora-context

# Option 2: Copy
cp -r ~/repos/agentic-mcp-tools/memora/claude-plugin ~/.claude/plugins/memora-context
```

Restart Claude Code after installation.

## Requirements

- Memora with dependencies installed (venv at `~/repos/agentic-mcp-tools/.venv`)
- `.mcp.json` with memora server config (for cloud storage env vars)
- Claude Code with plugin support

## Configuration

The hook:
- Runs with a 5-second timeout to avoid blocking session start
- Loads env vars from `.mcp.json` to connect to same database as MCP server
- Searches for top 5 memories with minimum relevance score of 0.02
