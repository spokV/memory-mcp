#!/bin/bash

# Memory MCP Universal Setup Script
# This script sets up memory-mcp as a shared HTTP service and configures all AI clients

set -e  # Exit on any error

echo "=================================================="
echo "Memory MCP Universal Setup"
echo "=================================================="
echo ""

# Configuration
MEMORY_MCP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$MEMORY_MCP_DIR/.venv"
MEMORY_MCP_PORT=8765
MEMORY_MCP_HOST="127.0.0.1"
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/com.memory-mcp.plist"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Create virtual environment and install
echo "Step 1: Setting up virtual environment..."
echo "----------------------------------------"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH"
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate and install
echo "Installing memory-mcp package..."
source "$VENV_PATH/bin/activate"
pip install -e "$MEMORY_MCP_DIR"
echo -e "${GREEN}✓ memory-mcp installed${NC}"
echo ""

# Step 2: Create launchd service for auto-start
echo "Step 2: Creating launchd service (auto-start on boot)..."
echo "----------------------------------------"

mkdir -p "$HOME/Library/LaunchAgents"

cat > "$LAUNCHD_PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.memory-mcp</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_PATH/bin/memory-mcp-server</string>
        <string>--transport</string>
        <string>streamable-http</string>
        <string>--host</string>
        <string>$MEMORY_MCP_HOST</string>
        <string>--port</string>
        <string>$MEMORY_MCP_PORT</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$HOME/.mcp/memory-mcp/logs/memory-mcp.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/.mcp/memory-mcp/logs/memory-mcp-error.log</string>
</dict>
</plist>
PLIST_EOF

# Create logs directory
mkdir -p "$HOME/.mcp/memory-mcp/logs"

echo -e "${GREEN}✓ launchd plist created at $LAUNCHD_PLIST${NC}"
echo ""

# Step 3: Load the launchd service
echo "Step 3: Starting memory-mcp service..."
echo "----------------------------------------"

# Unload if already loaded
launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true

# Load the service
launchctl load "$LAUNCHD_PLIST"
echo -e "${GREEN}✓ memory-mcp service started${NC}"

# Wait a moment for service to start
sleep 2

# Check if it's running
if curl -s "http://$MEMORY_MCP_HOST:$MEMORY_MCP_PORT/" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Service is responding on http://$MEMORY_MCP_HOST:$MEMORY_MCP_PORT${NC}"
else
    echo -e "${YELLOW}⚠ Service may still be starting... check logs at:${NC}"
    echo "   $HOME/.mcp/memory-mcp/logs/memory-mcp.log"
fi
echo ""

# Step 4: Configure AI clients
echo "Step 4: Configuring AI clients..."
echo "----------------------------------------"

# Configure Claude Code (~/.claude.json)
echo ""
echo "Configuring Claude Code..."
CLAUDE_CODE_CONFIG="$HOME/.claude.json"

if [ -f "$CLAUDE_CODE_CONFIG" ]; then
    cp "$CLAUDE_CODE_CONFIG" "$CLAUDE_CODE_CONFIG.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${GREEN}✓ Backed up existing config${NC}"
    
    # Add memory server using Python
    python3 <<PYTHON_SCRIPT
import json

config_path = "$CLAUDE_CODE_CONFIG"
with open(config_path, 'r') as f:
    config = json.load(f)

if 'mcpServers' not in config:
    config['mcpServers'] = {}

config['mcpServers']['memory'] = {
    'type': 'sse',
    'url': 'http://$MEMORY_MCP_HOST:$MEMORY_MCP_PORT/sse'
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("✓ Added memory MCP to Claude Code config")
PYTHON_SCRIPT
    
    echo -e "${GREEN}✓ Claude Code configured${NC}"
else
    # Create new config
    cat > "$CLAUDE_CODE_CONFIG" <<JSON_EOF
{
  "mcpServers": {
    "memory": {
      "type": "sse",
      "url": "http://$MEMORY_MCP_HOST:$MEMORY_MCP_PORT/sse"
    }
  }
}
JSON_EOF
    echo -e "${GREEN}✓ Claude Code config created${NC}"
fi

# Configure Codex CLI (~/.codex/config.toml)
echo ""
echo "Configuring Codex CLI..."
CODEX_CONFIG="$HOME/.codex/config.toml"

if [ -f "$CODEX_CONFIG" ]; then
    cp "$CODEX_CONFIG" "$CODEX_CONFIG.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Check if memory server already exists
    if grep -q "\[mcp_servers.memory\]" "$CODEX_CONFIG"; then
        echo -e "${YELLOW}⚠ Memory MCP already configured in Codex${NC}"
    else
        # Append to existing config
        cat >> "$CODEX_CONFIG" <<TOML_EOF

[mcp_servers.memory]
url = "http://$MEMORY_MCP_HOST:$MEMORY_MCP_PORT/sse"
TOML_EOF
        echo -e "${GREEN}✓ Codex CLI configured${NC}"
    fi
else
    # Create new config with memory server
    mkdir -p "$(dirname "$CODEX_CONFIG")"
    cat > "$CODEX_CONFIG" <<TOML_EOF
[mcp_servers.memory]
url = "http://$MEMORY_MCP_HOST:$MEMORY_MCP_PORT/sse"
TOML_EOF
    echo -e "${GREEN}✓ Codex CLI config created${NC}"
fi

# Instructions for other clients
echo ""
echo "=================================================="
echo "Manual Configuration Required"
echo "=================================================="
echo ""
echo -e "${YELLOW}The following clients need manual configuration:${NC}"
echo ""
echo "1. Claude Desktop:"
echo "   - Open Claude Desktop → Settings → Developer → Edit Config"
echo "   - Add this to mcpServers:"
echo '   "memory": {'
echo '     "type": "sse",'
echo "     \"url\": \"http://$MEMORY_MCP_HOST:$MEMORY_MCP_PORT/sse\""
echo '   }'
echo ""
echo "2. Cursor/VSCode with Claude extension:"
echo "   - Add to workspace settings or user settings:"
echo '   "claude.mcpServers": {'
echo '     "memory": {'
echo '       "type": "sse",'
echo "       \"url\": \"http://$MEMORY_MCP_HOST:$MEMORY_MCP_PORT/sse\""
echo '     }'
echo '   }'
echo ""

# Summary
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Memory MCP is now running as a service on:"
echo "  http://$MEMORY_MCP_HOST:$MEMORY_MCP_PORT"
echo ""
echo "Service will auto-start on system boot."
echo ""
echo "Service management commands:"
echo "  Stop:    launchctl unload $LAUNCHD_PLIST"
echo "  Start:   launchctl load $LAUNCHD_PLIST"
echo "  Restart: launchctl kickstart -k gui/\$(id -u)/com.memory-mcp"
echo ""
echo "Logs location:"
echo "  $HOME/.mcp/memory-mcp/logs/"
echo ""
echo -e "${GREEN}All configured clients now share the same memory database!${NC}"
echo ""
