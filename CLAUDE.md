# Memora

## Session Start

At the beginning of each session, proactively search memories for context related to the current project or working directory using `memory_hybrid_search`. Briefly summarize relevant findings to establish context.

## Memory Search

When the user asks about past work, stored knowledge, or previously discussed topics:
1. Use `memory_hybrid_search` to find relevant memories
2. Use `memory_semantic_search` for pure meaning-based lookup
3. Summarize findings and cite memory IDs (e.g., "Memory #51 shows...")

For research/recall questions, always search memories first before answering.
