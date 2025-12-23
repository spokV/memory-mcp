"""TODO-specific visualization logic for the knowledge graph."""

from typing import Any, Dict, List, Optional

# Status colors for TODO nodes
TODO_STATUS_COLORS = {
    "open": "#58a6ff",        # Blue
    "in_progress": "#ffa657", # Orange
    "completed": "#7ee787",   # Green
    "blocked": "#f85149",     # Red
}

# Priority colors (used for border/accent)
PRIORITY_COLORS = {
    "high": "#f85149",     # Red border
    "medium": "#d29922",   # Yellow border
    "low": "#8b949e",      # Gray border
}


def is_todo(metadata: Optional[Dict[str, Any]]) -> bool:
    """Check if a memory is a TODO based on metadata."""
    if not metadata:
        return False
    return metadata.get("type") == "todo"


def get_todo_status(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Get TODO status from metadata, or None if not a TODO."""
    if not is_todo(metadata):
        return None
    return metadata.get("status", "open")


def get_todo_priority(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Get TODO priority from metadata."""
    if not is_todo(metadata):
        return None
    return metadata.get("priority", "medium")


def get_todo_node_style(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Get vis.js node style properties for a TODO.

    Returns dict with 'shape', 'color', and optionally 'borderWidth' for TODOs.
    Returns empty dict for non-TODOs.
    """
    if not is_todo(metadata):
        return {}

    status = get_todo_status(metadata)
    priority = get_todo_priority(metadata)

    style = {
        "shape": "dot",
        "color": TODO_STATUS_COLORS.get(status, TODO_STATUS_COLORS["open"]),
    }

    # High priority TODOs get thicker border
    if priority == "high":
        style["borderWidth"] = 4
        style["shapeProperties"] = {"borderDashes": False}

    return style


def build_todo_status_to_nodes(memories: List[Dict]) -> Dict[str, List[int]]:
    """Build mapping of TODO status -> list of node IDs.

    Only includes memories that are TODOs (metadata.type == 'todo').
    """
    status_to_nodes: Dict[str, List[int]] = {}

    for m in memories:
        meta = m.get("metadata") or {}
        if is_todo(meta):
            status = get_todo_status(meta)
            if status not in status_to_nodes:
                status_to_nodes[status] = []
            status_to_nodes[status].append(m["id"])

    return status_to_nodes


def build_todo_category_to_nodes(memories: List[Dict]) -> Dict[str, List[int]]:
    """Build mapping of TODO category -> list of node IDs.

    Only includes memories that are TODOs (metadata.type == 'todo').
    """
    category_to_nodes: Dict[str, List[int]] = {}

    for m in memories:
        meta = m.get("metadata") or {}
        if is_todo(meta):
            category = meta.get("category", "uncategorized")
            if category not in category_to_nodes:
                category_to_nodes[category] = []
            category_to_nodes[category].append(m["id"])

    return category_to_nodes


def build_todo_legend_html(
    status_to_nodes: Dict[str, List[int]],
    category_to_nodes: Optional[Dict[str, List[int]]] = None,
) -> str:
    """Build HTML for TODO status and category legend section."""
    if not status_to_nodes:
        return ""

    html_parts = ['<div id="todos-legend"><b>TODOs</b>']

    # Status items
    for status, color in TODO_STATUS_COLORS.items():
        count = len(status_to_nodes.get(status, []))
        if count > 0:
            display_name = status.replace("_", " ").title()
            html_parts.append(
                f'<div class="legend-item todo-status" data-todo-status="{status}" '
                f'onclick="filterByTodoStatus(\'{status}\')">'
                f'<span class="legend-color" style="background:{color}"></span>'
                f'{display_name} ({count})</div>'
            )

    # Category items
    if category_to_nodes:
        html_parts.append('<div class="todo-categories collapsed"><b>Categories</b><span class="legend-toggle" onclick="toggleSection(this)">[+]</span><div class="section-items">')
        for category in sorted(category_to_nodes.keys()):
            count = len(category_to_nodes[category])
            html_parts.append(
                f'<div class="legend-item todo-category" data-todo-category="{category}" '
                f'onclick="filterByTodoCategory(\'{category}\')">'
                f'<span class="legend-color small" style="background:#8b949e"></span>'
                f'{category} ({count})</div>'
            )
        html_parts.append('</div></div>')

    html_parts.append('</div>')
    return "\n".join(html_parts)


# CSS for TODO badges (to be included in templates)
TODO_BADGE_CSS = """
.todo-badges { margin-bottom: 12px; }
.todo-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 4px;
    color: #fff;
}
.todo-badge.category { background: #30363d; color: #c9d1d9; }
.todo-badge.priority-high { background: #f85149; }
.todo-badge.priority-medium { background: #d29922; }
.todo-badge.priority-low { background: #8b949e; }
#todos-legend {
    margin-top: 16px;
    padding: 12px;
    border-top: 2px solid #58a6ff;
    border-left: 3px solid #58a6ff;
    background: rgba(88, 166, 255, 0.05);
    border-radius: 0 6px 6px 0;
}
#todos-legend b { display: block; margin-bottom: 8px; color: #58a6ff; cursor: pointer; }
#todos-legend b:hover { text-decoration: underline; }
#todos-legend b.active { background: rgba(88,166,255,0.2); padding: 2px 6px; border-radius: 4px; margin: -2px -6px 6px -6px; }
.legend-item.todo-category .legend-color {
    width: 8px !important;
    height: 8px !important;
}
.todo-categories { margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d; }
.todo-categories b { font-size: 11px; color: #8b949e; margin-bottom: 4px; display: inline !important; }
.todo-categories .legend-toggle { margin-left: 4px; }
.todo-categories.collapsed .section-items { display: none; }
.legend-item.todo-category { font-size: 11px; padding-left: 8px; }
"""


# JavaScript for TODO filtering (to be included in templates)
TODO_FILTER_JS = """
function filterAllTodos() {
    document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
    document.querySelector('#todos-legend b').classList.add('active');
    currentFilter = 'all-todos';
    var nodeIds = [];
    Object.values(graphData.todoStatusToNodes || {}).forEach(ids => nodeIds.push(...ids));
    applyFilter(nodeIds);
}

function filterByTodoStatus(status) {
    document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
    var el = document.querySelector('.legend-item[data-todo-status="' + status + '"]');
    if (el) el.classList.add('active');
    currentFilter = 'todo-status:' + status;
    var nodeIds = graphData.todoStatusToNodes[status] || [];
    applyFilter(nodeIds);
}

function filterByTodoCategory(category) {
    document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
    var el = document.querySelector('.legend-item[data-todo-category="' + category + '"]');
    if (el) el.classList.add('active');
    currentFilter = 'todo-category:' + category;
    var nodeIds = graphData.todoCategoryToNodes[category] || [];
    applyFilter(nodeIds);
}
"""


def get_todo_panel_html(metadata: Dict[str, Any]) -> str:
    """Generate HTML for TODO-specific fields in the detail panel."""
    if not is_todo(metadata):
        return ""

    status = metadata.get("status", "open")
    priority = metadata.get("priority", "medium")
    category = metadata.get("category", "")

    status_color = TODO_STATUS_COLORS.get(status, "#8b949e")

    html = f'''<div class="todo-badges">
        <span class="todo-badge" style="background:{status_color}">{status.upper()}</span>
        <span class="todo-badge priority-{priority}">{priority}</span>'''

    if category:
        html += f'<span class="todo-badge category">{category}</span>'

    html += '</div>'
    return html
