#!/usr/bin/env python3
"""Memora PostToolUse hook - auto-capture significant actions.

This script captures actions that have INHERENT CONTEXT:
- Git commits (commit message provides context)
- Test results (test output provides context)
- WebFetch research (URL and content provide context)
- Documentation edits (README, CLAUDE.md - content IS context)

It does NOT capture raw code edits (Edit/Write to source files) because:
- The hook only sees tool inputs/outputs, not conversation context
- Without knowing WHY a change was made, the capture is low-value noise
- Use manual memory_create for meaningful code change documentation
"""

import json
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List

# --- Configuration ---
SIGNIFICANCE_THRESHOLD = 0.6  # Raised to reduce false positives
CACHE_TTL_MINUTES = 30
MAX_CONTENT_LENGTH = 500

# --- Research Detection ---
RESEARCH_KEYWORDS = ["compare", "comparison", "difference", "vs", "versus", "alternative",
                     "features", "pros", "cons", "overview", "review", "analyze", "analysis"]
RESEARCH_URL_PATTERNS = [
    "github.com",      # GitHub repos
    "gitlab.com",      # GitLab repos
    "docs.",           # Documentation sites
    "documentation",   # Documentation pages
    "readme",          # README files
    "wiki",            # Wiki pages
    "blog",            # Blog posts
    "medium.com",      # Medium articles
    "dev.to",          # Dev.to articles
    "stackoverflow",   # Stack Overflow
    "arxiv.org",       # Academic papers
]
MAX_RESEARCH_CONTENT_LENGTH = 1500

# --- Excluded Tool Prefixes ---
EXCLUDED_PREFIXES = ["mcp__memora__"]


def load_memora_env() -> dict:
    """Load memora environment variables from .mcp.json if available."""
    search_paths = [
        Path.home() / "repos" / "agentic-mcp-tools" / ".mcp.json",
        Path.home() / ".mcp.json",
        Path.cwd() / ".mcp.json",
    ]
    env_vars = {}
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
                return env_vars
            except Exception:
                pass
    return env_vars


def is_enabled(env_vars: dict) -> bool:
    """Check if auto-capture is enabled."""
    flag = env_vars.get("MEMORA_AUTO_CAPTURE", os.environ.get("MEMORA_AUTO_CAPTURE", "false"))
    return flag.lower() in ("true", "1", "yes")


def get_memora_storage():
    """Import and return memora storage module."""
    try:
        from memora import storage
        return storage
    except ImportError:
        memora_path = Path.home() / "repos" / "agentic-mcp-tools" / "memora"
        if memora_path.exists():
            sys.path.insert(0, str(memora_path))
            try:
                from memora import storage
                return storage
            except Exception:
                return None
    return None


def is_excluded_tool(tool_name: str) -> bool:
    """Check if tool should be excluded from capture."""
    return any(tool_name.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def extract_content(tool_name: str, tool_input: dict, tool_result: dict) -> str:
    """Extract relevant content for analysis."""
    if tool_name == "Write":
        return tool_input.get("content", "")[:2000]
    elif tool_name == "Edit":
        old = tool_input.get("old_string", "")
        new = tool_input.get("new_string", "")
        return f"{old} -> {new}"[:2000]
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        # Handle tool_result which might be string or dict
        if isinstance(tool_result, dict):
            output = str(tool_result.get("output", tool_result.get("stdout", "")))[:1000]
        else:
            output = str(tool_result)[:1000]
        return f"{cmd}\n{output}"
    elif tool_name == "WebFetch":
        # Extract URL and result content
        url = tool_input.get("url", "")
        prompt = tool_input.get("prompt", "")
        if isinstance(tool_result, dict):
            content = str(tool_result.get("output", tool_result.get("content", "")))[:3000]
        else:
            content = str(tool_result)[:3000]
        return f"URL: {url}\nPrompt: {prompt}\n{content}"
    return ""


def is_research_url(url: str) -> bool:
    """Check if URL matches research patterns."""
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in RESEARCH_URL_PATTERNS)


def detect_webfetch_research(tool_input: dict, tool_result: dict) -> Tuple[Optional[str], float]:
    """Detect if WebFetch is research-worthy and calculate significance.

    Returns:
        (capture_type, significance_score) or (None, 0.0) if not significant
    """
    url = tool_input.get("url", "")
    prompt = tool_input.get("prompt", "")

    # Extract result content
    if isinstance(tool_result, dict):
        content = str(tool_result.get("output", tool_result.get("content", "")))
    else:
        content = str(tool_result)

    # Check if URL matches research patterns
    is_research = is_research_url(url)

    # Check prompt and content for research keywords
    combined_text = f"{prompt} {content}".lower()
    keyword_matches = count_keyword_matches(combined_text, RESEARCH_KEYWORDS)

    # Calculate significance score
    score = 0.0

    # GitHub repos are highly significant
    if "github.com" in url.lower() and "/blob/" not in url.lower():
        score += 0.5
        # Extra boost for README/main repo pages
        if url.rstrip("/").count("/") <= 4:  # Main repo page
            score += 0.2

    # Documentation sites
    elif any(p in url.lower() for p in ["docs.", "documentation", "wiki"]):
        score += 0.4

    # Other research URLs
    elif is_research:
        score += 0.3

    # Keyword matches boost
    if keyword_matches > 0:
        score += min(keyword_matches * 0.1, 0.3)

    # Content length indicates substantial findings
    if len(content) > 500:
        score += 0.1

    # Determine capture type based on content
    if "github.com" in url.lower():
        capture_type = "research-github"
    elif any(p in url.lower() for p in ["docs.", "documentation"]):
        capture_type = "research-docs"
    elif keyword_matches >= 2:
        capture_type = "research-comparison"
    else:
        capture_type = "research-general"

    # Only return if significant enough
    if score >= SIGNIFICANCE_THRESHOLD:
        return capture_type, min(score, 1.0)

    return None, 0.0


def summarize_research_content(url: str, prompt: str, content: str, max_length: int = MAX_RESEARCH_CONTENT_LENGTH) -> str:
    """Summarize WebFetch research content for storage.

    Extracts key information and truncates intelligently.
    """
    lines = []

    # Extract project/repo name from GitHub URLs
    if "github.com" in url.lower():
        parts = url.rstrip("/").split("/")
        if len(parts) >= 5:
            owner, repo = parts[3], parts[4]
            lines.append(f"**Repository:** {owner}/{repo}")

    lines.append(f"**URL:** {url}")

    if prompt:
        lines.append(f"**Query:** {prompt}")

    lines.append("")
    lines.append("**Key Findings:**")

    # Try to extract structured content (headings, bullet points)
    content_lines = content.split("\n")
    extracted = []
    current_section = None

    for line in content_lines:
        line = line.strip()
        if not line:
            continue

        # Capture headings
        if line.startswith("#"):
            current_section = line.lstrip("#").strip()
            if len(extracted) < 20:  # Limit sections
                extracted.append(f"\n**{current_section}**")

        # Capture bullet points and key info
        elif line.startswith(("-", "*", "•")) or ":" in line[:50]:
            if len(extracted) < 30:  # Limit bullet points
                extracted.append(line[:200])

        # Capture feature/capability mentions
        elif any(kw in line.lower() for kw in ["feature", "support", "provide", "include", "enable"]):
            if len(extracted) < 30:
                extracted.append(f"- {line[:200]}")

    # If no structured content found, take first N characters
    if not extracted:
        extracted = [content[:max_length]]

    lines.extend(extracted)

    result = "\n".join(lines)

    # Final truncation if still too long
    if len(result) > max_length:
        result = result[:max_length] + "\n\n[... truncated]"

    return result


def count_keyword_matches(content: str, keywords: List[str]) -> int:
    """Count keyword matches in content (case-insensitive)."""
    content_lower = content.lower()
    return sum(1 for kw in keywords if kw.lower() in content_lower)


def detect_capture_type(tool_name: str, tool_input: dict, tool_result: dict) -> Tuple[Optional[str], float]:
    """Detect capture type and calculate significance score.

    Only captures actions with INHERENT CONTEXT:
    - Git commits (commit message)
    - Test results (test output)
    - WebFetch research (URL + content)
    - Documentation edits (README, CLAUDE.md - content IS context)

    Does NOT capture raw Edit/Write to source code files.
    """
    # WebFetch research detection
    if tool_name == "WebFetch":
        return detect_webfetch_research(tool_input, tool_result)

    content = extract_content(tool_name, tool_input, tool_result)
    command = tool_input.get("command", "")

    # Git commit detection - commit message provides context
    if tool_name == "Bash" and "git commit" in command:
        return "git-commit", 0.8

    # Test result detection - test output provides context
    test_patterns = ["pytest", "npm test", "cargo test", "go test", "jest", "vitest", "make test"]
    if tool_name == "Bash" and any(p in command for p in test_patterns):
        if isinstance(tool_result, dict):
            output = str(tool_result.get("output", tool_result.get("stdout", "")))
        else:
            output = str(tool_result)
        if any(kw in output for kw in ["passed", "failed", "PASSED", "FAILED", "error", "Error"]):
            return "test-result", 0.7

    # For Edit/Write: ONLY capture documentation files where content IS context
    # Skip all source code edits - they lack conversation context
    if tool_name in ("Edit", "Write"):
        file_path = tool_input.get("file_path", "")
        file_name = Path(file_path).name if file_path else ""

        # Documentation files - content provides its own context
        doc_patterns = ["README", "CLAUDE.md", "CONTRIBUTING", "CHANGELOG", "LICENSE"]
        is_doc_file = any(p in file_name.upper() for p in doc_patterns)

        if is_doc_file:
            return "documentation", 0.7

        # Skip all other Edit/Write - no context available
        return None, 0.0

    # For other tools (Bash commands not covered above), skip
    return None, 0.0


def compute_content_hash(capture_type: str, tool_name: str, tool_input: dict) -> str:
    """Generate hash for deduplication."""
    if tool_name == "WebFetch":
        # Use URL for WebFetch deduplication
        key_parts = [
            capture_type,
            tool_name,
            tool_input.get("url", ""),
        ]
    else:
        key_parts = [
            capture_type,
            tool_name,
            tool_input.get("file_path", ""),
            tool_input.get("command", "")[:100],
        ]
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()[:16]


def load_cache(session_id: str) -> dict:
    """Load capture cache for session."""
    cache_file = Path(f"/tmp/memora_capture_cache_{session_id}.json")
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(session_id: str, cache: dict):
    """Save capture cache for session."""
    cache_file = Path(f"/tmp/memora_capture_cache_{session_id}.json")
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def is_duplicate(content_hash: str, session_id: str) -> bool:
    """Check if action was recently captured."""
    cache = load_cache(session_id)
    now = datetime.now()

    # Clean expired entries
    cache = {
        k: v for k, v in cache.items()
        if now - datetime.fromisoformat(v) < timedelta(minutes=CACHE_TTL_MINUTES)
    }

    if content_hash in cache:
        return True

    cache[content_hash] = now.isoformat()
    save_cache(session_id, cache)
    return False


def format_memory_content(
    capture_type: str,
    tool_name: str,
    tool_input: dict,
    tool_result: dict,
    cwd: str,
) -> str:
    """Format memory content for storage."""
    # Handle WebFetch research specially
    if tool_name == "WebFetch":
        url = tool_input.get("url", "")
        prompt = tool_input.get("prompt", "")
        if isinstance(tool_result, dict):
            content = str(tool_result.get("output", tool_result.get("content", "")))
        else:
            content = str(tool_result)

        titles = {
            "research-github": "GitHub Repository Research",
            "research-docs": "Documentation Research",
            "research-comparison": "Comparison Research",
            "research-general": "Web Research",
        }
        title = titles.get(capture_type, "Research")

        project = Path(cwd).name if cwd else "unknown"
        header = f"{title}\n\n**Project:** {project}\n"

        summary = summarize_research_content(url, prompt, content)
        return header + summary

    file_path = tool_input.get("file_path", "")
    command = tool_input.get("command", "")
    project = Path(cwd).name if cwd else "unknown"

    # Git commit - extract commit message
    if capture_type == "git-commit":
        # Extract commit message from command
        commit_msg = ""
        if "-m" in command:
            # Try to extract message after -m
            import re
            match = re.search(r'-m\s+["\'](.+?)["\']', command)
            if match:
                commit_msg = match.group(1)
            else:
                # Try without quotes
                match = re.search(r'-m\s+(\S+)', command)
                if match:
                    commit_msg = match.group(1)

        lines = ["Git Commit", ""]
        lines.append(f"**Project:** {project}")
        if commit_msg:
            lines.append(f"\n**Message:** {commit_msg}")
        lines.append(f"\n**Command:** `{command[:200]}`")
        return "\n".join(lines)

    # Test results
    if capture_type == "test-result":
        if isinstance(tool_result, dict):
            output = str(tool_result.get("output", tool_result.get("stdout", "")))[:MAX_CONTENT_LENGTH]
        else:
            output = str(tool_result)[:MAX_CONTENT_LENGTH]

        lines = ["Test Results", ""]
        lines.append(f"**Project:** {project}")
        lines.append(f"**Command:** `{command[:150]}`")
        if output:
            lines.append(f"\n**Output:**\n```\n{output}\n```")
        return "\n".join(lines)

    # Documentation edit
    if capture_type == "documentation":
        file_name = Path(file_path).name if file_path else "unknown"
        content = tool_input.get("content", "") or tool_input.get("new_string", "")

        lines = [f"Documentation Update: {file_name}", ""]
        lines.append(f"**Project:** {project}")
        lines.append(f"**File:** {file_path}")

        # For documentation, include more content since it IS the context
        if content:
            preview = content[:1000]
            lines.append(f"\n**Content:**\n```\n{preview}\n```")
            if len(content) > 1000:
                lines.append("\n[... truncated]")

        return "\n".join(lines)

    # Fallback
    return f"Auto-captured: {capture_type}\n\n**Project:** {project}"


def find_existing_memory(storage, conn, content: str, capture_type: str, project: str, file_path: str = "") -> Optional[dict]:
    """Search for existing memory that could be updated instead of creating new."""
    try:
        # First, try to find by same file path (most specific match)
        if file_path:
            results = storage.list_memories(
                conn,
                metadata_filters={"file_path": file_path, "capture_type": capture_type},
                limit=1,
            )
            if results:
                return results[0]

        # Then search for similar memories with same capture type and project
        results = storage.hybrid_search(
            conn,
            query=f"{project} {capture_type} {content[:100]}",
            top_k=5,
            min_score=0.15,  # Lower threshold for updates
            tags_any=[f"memora/auto-capture/{capture_type}"],
        )

        # Find best match in same project
        for result in results:
            memory = result.get("memory", {})
            mem_metadata = memory.get("metadata", {}) or {}
            if mem_metadata.get("project") == project:
                return memory

        return None
    except Exception:
        return None


def find_hierarchy_placement(storage, conn, capture_type: str, project: str) -> dict:
    """Find appropriate hierarchy placement based on existing memories.

    Priority:
    1. If project has existing hierarchy (e.g., memora/...), place under project/category
    2. If existing auto-capture memories have a hierarchy, reuse it
    3. Fall back to generic category/project structure
    """
    try:
        # Map capture types to categories for subsection placement
        category_mapping = {
            "git-commit": "commits",
            "test-result": "testing",
            "documentation": "docs",
            "research-github": "research",
            "research-docs": "research",
            "research-comparison": "research",
            "research-general": "research",
        }
        category = category_mapping.get(capture_type, "auto-capture")

        # First, check if project has an existing hierarchy (e.g., memora/knowledge)
        # by searching for memories with project as section
        project_memories = storage.hybrid_search(
            conn,
            query=project,
            top_k=5,
            min_score=0.1,
        )

        for result in project_memories:
            memory = result.get("memory", {})
            mem_metadata = memory.get("metadata", {}) or {}
            section = mem_metadata.get("section", "")

            # If we find memories where section == project, use project-first hierarchy
            if section == project or section.startswith(f"{project}/"):
                return {
                    "section": project,
                    "subsection": category,
                }

        # Second, check for existing auto-capture memories with same capture type
        results = storage.hybrid_search(
            conn,
            query=f"{project} {capture_type}",
            top_k=3,
            min_score=0.1,
            tags_any=["memora/auto-capture"],
        )

        for result in results:
            memory = result.get("memory", {})
            mem_metadata = memory.get("metadata", {}) or {}
            if mem_metadata.get("section"):
                return {
                    "section": mem_metadata.get("section"),
                    "subsection": mem_metadata.get("subsection", category),
                }

        # Default: use project-first hierarchy for consistency
        return {
            "section": project,
            "subsection": category,
        }
    except Exception:
        return {"section": project, "subsection": "auto-capture"}


def get_memory_type_config(capture_type: str, tool_result: dict) -> dict:
    """Determine the appropriate memory type and metadata based on capture type.

    Supported capture types:
    - git-commit: Git commits with commit message
    - test-result: Test runs (failures → issues)
    - documentation: README, CLAUDE.md edits
    - research-*: WebFetch research

    Returns:
        dict with: memory_type ("issue", "regular"), tags, and type-specific metadata
    """
    # Git commits → Regular memories with commit context
    if capture_type == "git-commit":
        return {
            "memory_type": "regular",
            "tags": ["memora/auto-capture", "memora/auto-capture/git-commit"],
            "metadata": {"type": "auto-capture", "capture_type": "git-commit"}
        }

    # Test results with failures → Open issues
    if capture_type == "test-result":
        if isinstance(tool_result, dict):
            output = str(tool_result.get("output", tool_result.get("stdout", "")))
        else:
            output = str(tool_result)

        has_failures = any(kw in output.lower() for kw in ["failed", "error", "failure"])

        if has_failures:
            return {
                "memory_type": "issue",
                "tags": ["memora/issues", "memora/auto-capture"],
                "metadata": {
                    "type": "issue",
                    "status": "open",
                    "severity": "major",
                    "category": "testing",
                }
            }
        else:
            # Tests passed - regular memory
            return {
                "memory_type": "regular",
                "tags": ["memora/auto-capture", "memora/auto-capture/test-result"],
                "metadata": {"type": "auto-capture", "capture_type": "test-result"}
            }

    # Documentation edits → Knowledge
    if capture_type == "documentation":
        return {
            "memory_type": "regular",
            "tags": ["memora/knowledge", "memora/auto-capture"],
            "metadata": {"type": "auto-capture", "capture_type": "documentation"}
        }

    # Research types → Regular memories with research tags
    if capture_type.startswith("research-"):
        return {
            "memory_type": "regular",
            "tags": ["memora/auto-capture", "memora/auto-capture/research"],
            "metadata": {"type": "auto-capture", "capture_type": capture_type}
        }

    # Fallback
    return {
        "memory_type": "regular",
        "tags": ["memora/auto-capture"],
        "metadata": {"type": "auto-capture", "capture_type": capture_type}
    }


def find_or_create_memory(
    storage,
    content: str,
    capture_type: str,
    tool_name: str,
    tool_input: dict,
    tool_result: dict,
    cwd: str,
    session_id: str,
    significance_score: float,
) -> tuple[Optional[dict], str]:
    """Find existing memory to update, or create new one with proper type and hierarchy.

    Returns:
        (memory, action) where action is "updated" or "created"
    """
    try:
        conn = storage.connect()
        project = Path(cwd).name if cwd else "unknown"

        # Get memory type configuration
        type_config = get_memory_type_config(capture_type, tool_result)

        # First, try to find existing memory to update
        file_path = tool_input.get("file_path", "")
        existing = find_existing_memory(storage, conn, content, capture_type, project, file_path)

        if existing:
            # Append to existing memory
            existing_content = existing.get("content", "")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            updated_content = f"{existing_content}\n\n---\n**[{timestamp}]**\n{content}"

            # Update existing memory
            storage.update_memory(
                conn,
                memory_id=existing["id"],
                content=updated_content,
            )
            conn.close()

            try:
                storage.sync_to_cloud()
            except Exception:
                pass

            return existing, "updated"

        # No existing memory found, create new one with proper type and hierarchy
        hierarchy = find_hierarchy_placement(storage, conn, capture_type, project)

        tags = type_config["tags"]
        metadata = type_config["metadata"].copy()

        # Add common metadata
        metadata.update({
            "tool_name": tool_name,
            "project": project,
            "cwd": cwd,
            "session_id": session_id,
            "significance_score": significance_score,
            "section": hierarchy["section"],
            "subsection": hierarchy["subsection"],
        })

        if file_path:
            metadata["file_path"] = file_path

        # For WebFetch, add URL to metadata
        url = tool_input.get("url", "")
        if url:
            metadata["url"] = url

        # For issues, add component from file path
        if type_config["memory_type"] == "issue" and file_path:
            # Extract component from file path (e.g., "auth" from "src/auth/login.py")
            path_parts = Path(file_path).parts
            if len(path_parts) > 1:
                metadata["component"] = path_parts[-2] if path_parts[-2] != "src" else path_parts[-1].replace(".py", "")

        memory = storage.add_memory(
            conn,
            content=content,
            metadata=metadata,
            tags=tags,
        )

        conn.close()

        try:
            storage.sync_to_cloud()
        except Exception:
            pass

        return memory, "created"
    except Exception:
        return None, "error"


def main():
    """Main entry point for PostToolUse hook."""
    try:
        # Load memora environment
        env_vars = load_memora_env()

        # Check if enabled
        if not is_enabled(env_vars):
            print(json.dumps({}))
            sys.exit(0)

        # Read input from stdin
        input_data = json.load(sys.stdin)

        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        tool_result = input_data.get("tool_result", {})
        session_id = input_data.get("session_id", "unknown")
        cwd = input_data.get("cwd", "")

        # Skip excluded tools
        if is_excluded_tool(tool_name):
            print(json.dumps({}))
            sys.exit(0)

        # Detect capture type and significance
        capture_type, significance = detect_capture_type(tool_name, tool_input, tool_result)

        if not capture_type or significance < SIGNIFICANCE_THRESHOLD:
            print(json.dumps({}))
            sys.exit(0)

        # Check for duplicates
        content_hash = compute_content_hash(capture_type, tool_name, tool_input)
        if is_duplicate(content_hash, session_id):
            print(json.dumps({}))
            sys.exit(0)

        # Get memora storage
        storage = get_memora_storage()
        if not storage:
            print(json.dumps({}))
            sys.exit(0)

        # Format content and find/create memory
        content = format_memory_content(capture_type, tool_name, tool_input, tool_result, cwd)
        memory, action = find_or_create_memory(
            storage, content, capture_type, tool_name, tool_input, tool_result,
            cwd, session_id, significance
        )

        if memory:
            if action == "updated":
                output = {
                    "systemMessage": f"[Memora] Updated: {capture_type} (#{memory.get('id', '?')})"
                }
            else:
                output = {
                    "systemMessage": f"[Memora] Captured: {capture_type} (#{memory.get('id', '?')})"
                }
            print(json.dumps(output))
        else:
            print(json.dumps({}))

    except Exception:
        print(json.dumps({}))

    sys.exit(0)


if __name__ == "__main__":
    main()
