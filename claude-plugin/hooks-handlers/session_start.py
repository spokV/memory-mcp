#!/usr/bin/env python3
"""Memora SessionStart hook - inject relevant memories into Claude Code context.

This script:
1. Reads session info from stdin (cwd, session_id)
2. Extracts project context from working directory
3. Searches memora for relevant memories
4. Returns additionalContext for Claude's system prompt
"""

import json
import os
import sys
from pathlib import Path


def load_memora_env():
    """Load memora environment variables from .mcp.json if available."""
    # Try to find .mcp.json in common locations
    search_paths = [
        Path.home() / "repos" / "agentic-mcp-tools" / ".mcp.json",
        Path.home() / ".mcp.json",
        Path.cwd() / ".mcp.json",
    ]

    for mcp_path in search_paths:
        if mcp_path.exists():
            try:
                with open(mcp_path) as f:
                    config = json.load(f)
                memora_config = config.get("mcpServers", {}).get("memora", {})
                env_vars = memora_config.get("env", {})
                for key, value in env_vars.items():
                    if key not in os.environ:
                        os.environ[key] = value
                return True
            except Exception:
                pass
    return False


def extract_project_context(cwd: str) -> dict:
    """Extract project identifiers from working directory."""
    path = Path(cwd)

    # Get project name (directory name)
    project_name = path.name

    # Build search queries from path components
    queries = [project_name]

    # Add parent directory names that might be relevant
    for parent in list(path.parents)[:2]:
        if parent.name and parent.name not in ("", "Users", "home", "repos", "src"):
            queries.append(parent.name)

    return {
        "project_name": project_name,
        "cwd": cwd,
        "search_query": " ".join(queries[:3]),
    }


def search_memora(query: str, top_k: int = 5) -> list:
    """Search memora for relevant memories using direct storage import."""
    try:
        # Try importing memora (installed as uv tool or in path)
        from memora import storage

        conn = storage.connect()
        results = storage.hybrid_search(
            conn,
            query=query,
            top_k=top_k,
            min_score=0.02,
        )
        conn.close()

        return results

    except ImportError:
        # Try adding local repo to path
        memora_path = Path.home() / "repos" / "agentic-mcp-tools" / "memora"
        if memora_path.exists():
            sys.path.insert(0, str(memora_path))
            try:
                from memora import storage
                conn = storage.connect()
                results = storage.hybrid_search(conn, query=query, top_k=top_k, min_score=0.02)
                conn.close()
                return results
            except Exception:
                return []
        return []
    except Exception:
        return []


def format_memories(memories: list, max_chars: int = 1500) -> str:
    """Format memories concisely for context injection."""
    if not memories:
        return ""

    lines = ["## Relevant Memories\n"]
    total_chars = len(lines[0])

    for item in memories:
        memory = item.get("memory", item)
        score = item.get("score", 0)

        # Extract key info
        mid = memory.get("id", "?")
        content = memory.get("content", "")
        tags = memory.get("tags", [])

        # Truncate long content
        if len(content) > 150:
            content = content[:150] + "..."

        # Format as compact entry
        tags_str = ", ".join(tags[:3]) if tags else ""
        entry = f"- [#{mid}] {content}"
        if tags_str:
            entry += f" ({tags_str})"
        entry += "\n"

        # Check length limit
        if total_chars + len(entry) > max_chars:
            lines.append("- ... more available via `memory_hybrid_search`\n")
            break

        lines.append(entry)
        total_chars += len(entry)

    lines.append("\nUse memora tools to search for more context.\n")
    return "".join(lines)


def main():
    """Main entry point for SessionStart hook."""
    try:
        # Load memora env from .mcp.json
        load_memora_env()

        # Read input from stdin
        input_data = json.load(sys.stdin)

        cwd = input_data.get("cwd", os.getcwd())

        # Extract project context
        context = extract_project_context(cwd)

        # Search memora
        memories = search_memora(context["search_query"], top_k=5)

        # Format output
        if memories:
            additional_context = format_memories(memories)
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": additional_context
                }
            }
        else:
            output = {}

        print(json.dumps(output))

    except Exception:
        # On error, allow session to continue without blocking
        print(json.dumps({}))

    sys.exit(0)


if __name__ == "__main__":
    main()
