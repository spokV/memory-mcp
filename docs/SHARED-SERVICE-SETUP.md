# Shared Service Setup for Memory MCP

This guide covers running memory-mcp as a **shared HTTP service** that multiple AI clients can connect to simultaneously.

## Understanding the Options

Memory-mcp can run in three modes:

### Option 1: stdio (Default - Per Client)
```bash
memory-mcp-server
```
- **How it works:** Each client spawns its own process
- **Good for:** Simple, single-client setups
- **Pros:** Standard MCP pattern, simple config
- **Cons:** Multiple processes if you use multiple clients

### Option 2: Manual HTTP Server
```bash
# Using the command directly
memory-mcp-server --transport streamable-http --host 127.0.0.1 --port 8765

# Or using module invocation
python -m memory_mcp --transport streamable-http --host 127.0.0.1 --port 8765

# Or with environment variables
MEMORY_MCP_TRANSPORT=streamable-http \
MEMORY_MCP_HOST=127.0.0.1 \
MEMORY_MCP_PORT=8765 \
memory-mcp-server
```
- **How it works:** Single HTTP server, all clients connect
- **Good for:** Testing, temporary multi-client use
- **Pros:** Simple one command, all clients share immediately
- **Cons:** Stops when terminal closes, manual start each session

### Option 3: Automated Service (This Guide)
```bash
./setup-memory-mcp-service.sh
```
- **How it works:** System service that auto-starts on boot
- **Good for:** Daily use with multiple clients
- **Pros:** Set-and-forget, auto-starts, managed service, centralized logs
- **Cons:** Initial setup time (automated though)

## Architecture Comparison

**Option 1 (stdio):**
```
Claude Desktop → spawns memory-mcp → DB
Claude Code    → spawns memory-mcp → DB
Codex CLI      → spawns memory-mcp → DB
```

**Options 2 & 3 (shared HTTP):**
```
              memory-mcp-server :8765
                       ↓
                     [DB]
                       ↑
    ┌──────────────────┼──────────────────┐
Claude Desktop   Claude Code        Codex CLI
```

Benefits of shared HTTP:
- ✅ Single process (lower memory)
- ✅ Simple HTTP/SSE configuration
- ✅ True real-time sync across clients
- ✅ Centralized logging

Additional benefits of Option 3 (service):
- ✅ Auto-starts on system boot
- ✅ Keeps running automatically
- ✅ Auto-configures clients
- ✅ Health monitoring via script

## Quick Start - Option 3 (Recommended)

### Setup

```bash
cd /path/to/memory-mcp
chmod +x setup-memory-mcp-service.sh
./setup-memory-mcp-service.sh
```

This will:
1. Create Python virtual environment (`.venv/`)
2. Install memory-mcp in editable mode
3. Create macOS launchd service (auto-starts on boot)
4. Start the service immediately
5. Auto-configure Claude Code (`~/.claude.json`)
6. Auto-configure Codex CLI (`~/.codex/config.toml`)
7. Show manual configuration steps for other clients

### Verify

Check status:
```bash
./check-status.sh
```

## Quick Start - Option 2 (Manual)

If you just want to test without the full service setup:

```bash
# Make sure memory-mcp is installed
pip install -e .

# Run the server
memory-mcp-server --transport streamable-http --host 127.0.0.1 --port 8765
```

Keep this terminal open. Then configure your clients (see below).

## Client Configuration

After starting the HTTP server (Options 2 or 3), configure your AI clients to connect to:
```
http://127.0.0.1:8765/sse
```

### Claude Desktop
Settings → Developer → Edit Config, add:
```json
"memory": {
  "type": "sse",
  "url": "http://127.0.0.1:8765/sse"
}
```

### Claude Code
Auto-configured by Option 3, or manually edit `~/.claude.json`:
```json
{
  "mcpServers": {
    "memory": {
      "type": "sse",
      "url": "http://127.0.0.1:8765/sse"
    }
  }
}
```

### Codex CLI
Auto-configured by Option 3, or manually edit `~/.codex/config.toml`:
```toml
[mcp_servers.memory]
url = "http://127.0.0.1:8765/sse"
```

### Cursor/VSCode
Add to settings:
```json
"claude.mcpServers": {
  "memory": {
    "type": "sse",
    "url": "http://127.0.0.1:8765/sse"
  }
}
```

## Service Management (Option 3 Only)

### Check Status
```bash
./check-status.sh
```

### Stop Service
```bash
launchctl unload ~/Library/LaunchAgents/com.memory-mcp.plist
```

### Start Service
```bash
launchctl load ~/Library/LaunchAgents/com.memory-mcp.plist
```

### Restart Service
```bash
launchctl kickstart -k gui/$(id -u)/com.memory-mcp
```

### View Logs
```bash
tail -f ~/.mcp/memory-mcp/logs/memory-mcp.log
tail -f ~/.mcp/memory-mcp/logs/memory-mcp-error.log
```

## File Locations

| File | Location |
|------|----------|
| Database | `~/.mcp/memory-mcp/memory_mcp/memories.db` |
| Service config | `~/Library/LaunchAgents/com.memory-mcp.plist` |
| Logs | `~/.mcp/memory-mcp/logs/` |
| Virtual env | `.venv/` (in repo directory) |

## Testing

After configuration, test from any client:
```
Create a memory: "Testing shared service" with tag "test"
```

Then from another client:
```
List memories with tag "test"
```

If you see the same memory, you're synced! 🎉

## Troubleshooting

### Service won't start (Option 3)
```bash
# Check logs
cat ~/.mcp/memory-mcp/logs/memory-mcp-error.log

# Verify installation
.venv/bin/memory-mcp-server --help

# Check port availability
lsof -i :8765
```

### Manual server won't start (Option 2)
```bash
# Check if something is using the port
lsof -i :8765

# Try a different port
memory-mcp-server --transport streamable-http --host 127.0.0.1 --port 8766
```

### Client can't connect
1. Verify service is running: `./check-status.sh` or `curl http://127.0.0.1:8765/`
2. Check client config syntax
3. Restart client application
4. Verify URL is exactly: `http://127.0.0.1:8765/sse`

### Different port
For Option 2, just use `--port XXXX`

For Option 3, edit `~/Library/LaunchAgents/com.memory-mcp.plist`:
- Change port in `<array>` section
- Update all client configs
- Restart: `launchctl kickstart -k gui/$(id -u)/com.memory-mcp`

## Comparison Table

| Feature | Option 1 (stdio) | Option 2 (manual HTTP) | Option 3 (service) |
|---------|------------------|------------------------|-------------------|
| **Multiple clients** | Each spawns process | ✅ Share one process | ✅ Share one process |
| **Auto-start** | N/A | ❌ Manual each time | ✅ On boot |
| **Memory usage** | High (N processes) | Low (1 process) | Low (1 process) |
| **Setup complexity** | Simple | Simple | Medium (automated) |
| **Centralized logs** | ❌ Per client | ❌ Terminal only | ✅ Log files |
| **Service management** | N/A | Ctrl+C to stop | ✅ launchctl commands |
| **Best for** | Single client | Testing/temporary | Daily multi-client use |

## Platform Support

Currently supports:
- ✅ macOS (launchd) - fully implemented

## Advanced: Custom Configuration

### Environment Variables (Options 2 & 3)

For Option 2:
```bash
export MEMORY_MCP_TRANSPORT=streamable-http
export MEMORY_MCP_HOST=127.0.0.1
export MEMORY_MCP_PORT=8765
memory-mcp-server
```

For Option 3, edit the plist file and add under `<dict>`:
```xml
<key>EnvironmentVariables</key>
<dict>
    <key>MEMORY_MCP_EMBEDDING_MODEL</key>
    <string>sentence-transformers</string>
    <key>LOG_LEVEL</key>
    <string>DEBUG</string>
</dict>
```

Then restart the service.

## Migration Guide

### From stdio to HTTP

1. Stop all clients
2. Choose Option 2 (testing) or Option 3 (permanent)
3. Update all client configs to use HTTP endpoint
4. Restart clients
5. Test that memories are shared

Your existing database at `~/.mcp/memory-mcp/memory_mcp/memories.db` will be used automatically.

