"""Graph data generation and transformation logic."""

import json
from typing import Any, Dict, List, Optional

from ..storage import (
    connect,
    get_crossrefs,
    get_memory,
    list_memories,
    rebuild_crossrefs,
)
from .issues import (
    TAG_COLORS,
    build_status_to_nodes,
    get_issue_node_style,
    is_issue,
    build_issue_legend_html,
)
from .todos import (
    build_todo_status_to_nodes,
    get_todo_node_style,
    is_todo,
    build_todo_legend_html,
)
from .templates import build_static_html


def _expand_r2_urls(metadata: Optional[Dict]) -> Dict:
    """Expand R2 URLs in metadata for display."""
    if not metadata:
        return {}

    meta = dict(metadata)
    if meta.get("images"):
        from ..image_storage import expand_r2_url

        expanded_images = []
        for img in meta["images"]:
            if isinstance(img, dict) and img.get("src"):
                src = img["src"]
                if src.startswith("r2://") or src.startswith("/r2/"):
                    src = expand_r2_url(
                        src.replace("/r2/", "r2://") if src.startswith("/r2/") else src,
                        use_proxy=True,
                    )
                expanded_images.append({**img, "src": src})
            else:
                expanded_images.append(img)
        meta["images"] = expanded_images

    return meta


def _build_tag_colors(memories: List[Dict]) -> Dict[str, str]:
    """Build tag -> color mapping from memories."""
    tag_colors = {}
    for m in memories:
        tags = m.get("tags", [])
        primary_tag = tags[0] if tags else "untagged"
        if primary_tag not in tag_colors:
            tag_colors[primary_tag] = TAG_COLORS[len(tag_colors) % len(TAG_COLORS)]
    return tag_colors


def _build_nodes(memories: List[Dict], tag_colors: Dict[str, str]) -> List[Dict]:
    """Build vis.js node objects from memories."""
    nodes = []
    for m in memories:
        tags = m.get("tags", [])
        primary_tag = tags[0] if tags else "untagged"
        meta = m.get("metadata") or {}

        content = m["content"]
        label = content[:35].replace("\n", " ").replace('"', "'").replace("\\", "")

        node = {
            "id": m["id"],
            "label": label + "..." if len(content) > 35 else label,
            "title": f"Memory #{m['id']}",
            "color": tag_colors[primary_tag],
        }

        # Apply issue-specific styling
        issue_style = get_issue_node_style(meta)
        if issue_style:
            node.update(issue_style)

        # Apply TODO-specific styling
        todo_style = get_todo_node_style(meta)
        if todo_style:
            node.update(todo_style)

        nodes.append(node)

    return nodes


def _build_tag_to_nodes(memories: List[Dict]) -> Dict[str, List[int]]:
    """Build tag -> node IDs mapping."""
    tag_to_nodes: Dict[str, List[int]] = {}
    for m in memories:
        for tag in m.get("tags", []):
            if tag not in tag_to_nodes:
                tag_to_nodes[tag] = []
            tag_to_nodes[tag].append(m["id"])
    return tag_to_nodes


def _build_section_mappings(memories: List[Dict]) -> tuple:
    """Build section and subsection -> node IDs mappings.

    Returns (section_to_nodes, path_to_nodes) tuple.
    Issues are excluded since they have their own dedicated legend.
    """
    section_to_nodes: Dict[str, List[int]] = {}
    path_to_nodes: Dict[str, List[int]] = {}

    for m in memories:
        meta = m.get("metadata") or {}

        # Skip issues and TODOs - they have their own legends
        if is_issue(meta) or is_todo(meta):
            continue

        hierarchy = meta.get("hierarchy", {})
        hierarchy_path = hierarchy.get("path", []) if isinstance(hierarchy, dict) else []

        if hierarchy_path and len(hierarchy_path) >= 1:
            section = hierarchy_path[0]
            parts = hierarchy_path[1:]
        else:
            section = meta.get("section", "Uncategorized")
            subsection = meta.get("subsection", "")
            parts = subsection.split("/") if subsection else []

        if section not in section_to_nodes:
            section_to_nodes[section] = []
        section_to_nodes[section].append(m["id"])

        if parts:
            for i in range(len(parts)):
                partial_path = "/".join(parts[: i + 1])
                full_key = f"{section}/{partial_path}"
                if full_key not in path_to_nodes:
                    path_to_nodes[full_key] = []
                path_to_nodes[full_key].append(m["id"])

    return section_to_nodes, path_to_nodes


def _build_edges(conn, memories: List[Dict], min_score: float) -> List[Dict]:
    """Build vis.js edge objects from crossrefs."""
    edges = []
    seen = set()
    for m in memories:
        for ref in get_crossrefs(conn, m["id"]):
            edge_key = tuple(sorted([m["id"], ref["id"]]))
            if edge_key not in seen and ref.get("score", 0) > min_score:
                seen.add(edge_key)
                edges.append({"from": m["id"], "to": ref["id"]})
    return edges


def _build_legend_html(tag_colors: Dict[str, str]) -> str:
    """Build HTML for tag legend."""
    return "".join(
        f'<div class="legend-item" data-tag="{t}" onclick="filterByTag(\'{t}\')">'
        f'<span class="legend-color" style="background:{c}"></span>{t}</div>'
        for t, c in list(tag_colors.items())[:12]
    )


def _build_sections_html(
    section_to_nodes: Dict[str, List[int]], path_to_nodes: Dict[str, List[int]]
) -> str:
    """Build HTML for sections hierarchy."""
    sections_html = ""
    for section, node_ids in section_to_nodes.items():
        sections_html += (
            f'<div class="section-item" data-section="{section}" '
            f"onclick=\"filterBySection('{section}')\">{section} ({len(node_ids)})</div>"
        )

        section_paths = sorted(
            [k for k in path_to_nodes.keys() if k.startswith(section + "/")]
        )
        rendered_paths = set()

        for full_path in section_paths:
            sub_path = full_path[len(section) + 1 :]
            parts = sub_path.split("/")

            for i, part in enumerate(parts):
                partial = "/".join(parts[: i + 1])
                render_key = f"{section}/{partial}"

                if render_key not in rendered_paths:
                    rendered_paths.add(render_key)
                    indent = "&nbsp;&nbsp;" * i
                    count = len(path_to_nodes.get(render_key, []))
                    sections_html += (
                        f'<div class="subsection-item" data-subsection="{render_key}" '
                        f"onclick=\"filterBySubsection('{render_key}')\" "
                        f'style="padding-left:{8 + i*12}px;">{indent}└ {part} ({count})</div>'
                    )

    return sections_html


def get_graph_data(min_score: float = 0.25, rebuild: bool = False) -> Dict[str, Any]:
    """Get graph nodes, edges, and metadata for API response.

    Args:
        min_score: Minimum similarity score for edges
        rebuild: If True, rebuild crossrefs (slow). If False, use existing.

    Returns:
        Dict with nodes, edges, and various mappings.
    """
    conn = connect()
    try:
        memories = list_memories(conn, None, None, None, 0, None, None, None, None, None)
        if not memories:
            return {"error": "no_memories", "message": "No memories to visualize"}

        if rebuild:
            rebuild_crossrefs(conn)

        tag_colors = _build_tag_colors(memories)
        nodes = _build_nodes(memories, tag_colors)
        tag_to_nodes = _build_tag_to_nodes(memories)
        section_to_nodes, path_to_nodes = _build_section_mappings(memories)
        status_to_nodes = build_status_to_nodes(memories)
        todo_status_to_nodes = build_todo_status_to_nodes(memories)
        edges = _build_edges(conn, memories, min_score)

        return {
            "nodes": nodes,
            "edges": edges,
            "tagColors": tag_colors,
            "tagToNodes": tag_to_nodes,
            "sectionToNodes": section_to_nodes,
            "subsectionToNodes": path_to_nodes,
            "statusToNodes": status_to_nodes,
            "todoStatusToNodes": todo_status_to_nodes,
        }

    finally:
        conn.close()


def get_memory_for_api(memory_id: int) -> Dict[str, Any]:
    """Get a single memory with expanded R2 URLs for API response."""
    conn = connect()
    try:
        m = get_memory(conn, memory_id)
        if not m:
            return {"error": "not_found"}

        meta = _expand_r2_urls(m.get("metadata"))

        return {
            "id": m["id"],
            "content": m["content"],
            "tags": m.get("tags", []),
            "created": m.get("created_at", ""),
            "metadata": meta,
        }
    finally:
        conn.close()


def export_graph_html(
    output_path: Optional[str] = None, min_score: float = 0.25
) -> Dict[str, Any]:
    """Generate static HTML knowledge graph visualization.

    Args:
        output_path: Path to save HTML file, or None to return HTML in result.
        min_score: Minimum similarity score for edges.

    Returns:
        Dict with node/edge counts, tags, and optionally path or html.
    """
    conn = connect()
    try:
        memories = list_memories(conn, None, None, None, 0, None, None, None, None, None)
        if not memories:
            return {"error": "no_memories", "message": "No memories to visualize"}

        rebuild_crossrefs(conn)

        tag_colors = _build_tag_colors(memories)
        nodes = _build_nodes(memories, tag_colors)
        tag_to_nodes = _build_tag_to_nodes(memories)
        section_to_nodes, path_to_nodes = _build_section_mappings(memories)
        status_to_nodes = build_status_to_nodes(memories)
        todo_status_to_nodes = build_todo_status_to_nodes(memories)
        edges = _build_edges(conn, memories, min_score)

        # Build memories data for inline display
        memories_data = {}
        for m in memories:
            meta = _expand_r2_urls(m.get("metadata"))
            memories_data[m["id"]] = {
                "id": m["id"],
                "tags": m.get("tags", []),
                "created": m.get("created_at", ""),
                "content": m["content"],
                "metadata": meta,
            }

        # Build HTML components
        legend_html = _build_legend_html(tag_colors)
        sections_html = _build_sections_html(section_to_nodes, path_to_nodes)
        issues_legend_html = build_issue_legend_html(status_to_nodes)
        todos_legend_html = build_todo_legend_html(todo_status_to_nodes)

        html = build_static_html(
            nodes_json=json.dumps(nodes),
            edges_json=json.dumps(edges),
            memories_json=json.dumps(memories_data),
            tag_to_nodes_json=json.dumps(tag_to_nodes),
            section_to_nodes_json=json.dumps(section_to_nodes),
            path_to_nodes_json=json.dumps(path_to_nodes),
            status_to_nodes_json=json.dumps(status_to_nodes),
            todo_status_to_nodes_json=json.dumps(todo_status_to_nodes),
            legend_html=legend_html,
            sections_html=sections_html,
            issues_legend_html=issues_legend_html,
            todos_legend_html=todos_legend_html,
        )

        result = {
            "nodes": len(nodes),
            "edges": len(edges),
            "tags": list(tag_colors.keys()),
        }

        if output_path is not None:
            with open(output_path, "w") as f:
                f.write(html)
            result["path"] = output_path
        else:
            result["html"] = html

        return result

    finally:
        conn.close()
