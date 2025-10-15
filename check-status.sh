#!/bin/bash

# Memory MCP Service Status Checker

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

MEMORY_MCP_HOST="127.0.0.1"
MEMORY_MCP_PORT=8765
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/com.memory-mcp.plist"

echo "Memory MCP Service Status"
echo "========================="
echo ""

# Check if service is loaded
if launchctl list | grep -q "com.memory-mcp"; then
    echo -e "${GREEN}✓ Service is loaded${NC}"
else
    echo -e "${RED}✗ Service is not loaded${NC}"
    echo "  Run: launchctl load $LAUNCHD_PLIST"
    exit 1
fi

# Check if service is responding
if curl -s "http://$MEMORY_MCP_HOST:$MEMORY_MCP_PORT/" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Service is responding on http://$MEMORY_MCP_HOST:$MEMORY_MCP_PORT${NC}"
else
    echo -e "${RED}✗ Service is not responding${NC}"
    echo "  Check logs: tail -f $HOME/.mcp/memory-mcp/logs/memory-mcp.log"
    exit 1
fi

# Check configured clients
echo ""
echo "Configured Clients:"
echo "-------------------"

if [ -f "$HOME/.claude.json" ]; then
    if grep -q "memory" "$HOME/.claude.json"; then
        echo -e "${GREEN}✓ Claude Code${NC}"
    else
        echo -e "${YELLOW}⚠ Claude Code (not configured)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Claude Code (config not found)${NC}"
fi

if [ -f "$HOME/.codex/config.toml" ]; then
    if grep -q "memory" "$HOME/.codex/config.toml"; then
        echo -e "${GREEN}✓ Codex CLI${NC}"
    else
        echo -e "${YELLOW}⚠ Codex CLI (not configured)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Codex CLI (config not found)${NC}"
fi

echo ""
echo "Database location:"
echo "  $HOME/.mcp/memory-mcp/memory_mcp/memories.db"

# Check if database exists
if [ -f "$HOME/.mcp/memory-mcp/memory_mcp/memories.db" ]; then
    DB_SIZE=$(du -h "$HOME/.mcp/memory-mcp/memory_mcp/memories.db" | cut -f1)
    echo -e "  ${GREEN}✓ Database exists (size: $DB_SIZE)${NC}"
else
    echo -e "  ${YELLOW}⚠ Database not yet created (will be created on first use)${NC}"
fi

echo ""
echo "Recent logs:"
echo "------------"
if [ -f "$HOME/.mcp/memory-mcp/logs/memory-mcp.log" ]; then
    tail -5 "$HOME/.mcp/memory-mcp/logs/memory-mcp.log" | sed 's/^/  /'
else
    echo "  No logs yet"
fi

echo ""
