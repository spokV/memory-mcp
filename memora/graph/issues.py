"""Issue-specific visualization logic for the knowledge graph."""

from typing import Any, Dict, List, Optional

# Status colors for issue nodes
STATUS_COLORS = {
    "open": "#ff7b72",        # Red
    "in_progress": "#ffa657", # Orange
    "resolved": "#7ee787",    # Green
    "wontfix": "#8b949e",     # Gray
}

# Severity colors (used for border/accent)
SEVERITY_COLORS = {
    "critical": "#f85149",    # Bright red
    "major": "#d29922",       # Yellow
    "minor": "#8b949e",       # Gray
}

# Default tag colors (purple palette for general/knowledge memories)
TAG_COLORS = [
    "#a855f7", "#c084fc", "#d8b4fe", "#9333ea",
    "#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd"
]


def is_issue(metadata: Optional[Dict[str, Any]]) -> bool:
    """Check if a memory is an issue based on metadata."""
    if not metadata:
        return False
    return metadata.get("type") == "issue"


def get_issue_status(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Get issue status from metadata, or None if not an issue."""
    if not is_issue(metadata):
        return None
    return metadata.get("status", "open")


def get_issue_severity(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Get issue severity from metadata."""
    if not is_issue(metadata):
        return None
    return metadata.get("severity", "minor")


def get_issue_node_style(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Get vis.js node style properties for an issue.

    Returns dict with 'shape', 'color', and optionally 'borderWidth' for issues.
    Returns empty dict for non-issues.
    """
    if not is_issue(metadata):
        return {}

    status = get_issue_status(metadata)
    severity = get_issue_severity(metadata)

    style = {
        "shape": "dot",
        "color": STATUS_COLORS.get(status, STATUS_COLORS["open"]),
    }

    # Critical issues get thicker border
    if severity == "critical":
        style["borderWidth"] = 4
        style["shapeProperties"] = {"borderDashes": False}

    return style


def build_status_to_nodes(memories: list) -> Dict[str, list]:
    """Build mapping of issue status -> list of node IDs.

    Only includes memories that are issues (metadata.type == 'issue').
    """
    status_to_nodes: Dict[str, list] = {}

    for m in memories:
        meta = m.get("metadata") or {}
        if is_issue(meta):
            status = get_issue_status(meta)
            if status not in status_to_nodes:
                status_to_nodes[status] = []
            status_to_nodes[status].append(m["id"])

    return status_to_nodes


def build_issue_category_to_nodes(memories: List[Dict]) -> Dict[str, List[int]]:
    """Build mapping of issue component/category -> list of node IDs.

    Only includes memories that are issues (metadata.type == 'issue').
    """
    category_to_nodes: Dict[str, List[int]] = {}

    for m in memories:
        meta = m.get("metadata") or {}
        if is_issue(meta):
            category = meta.get("component", "uncategorized")
            if category not in category_to_nodes:
                category_to_nodes[category] = []
            category_to_nodes[category].append(m["id"])

    return category_to_nodes


def build_issue_legend_html(
    status_to_nodes: Dict[str, list],
    category_to_nodes: Optional[Dict[str, List[int]]] = None,
) -> str:
    """Build HTML for issue status and category legend section."""
    if not status_to_nodes:
        return ""

    html_parts = ['<div id="issues-legend"><b>Issues</b>']

    # Status items
    for status, color in STATUS_COLORS.items():
        count = len(status_to_nodes.get(status, []))
        if count > 0:
            display_name = status.replace("_", " ").title()
            html_parts.append(
                f'<div class="legend-item issue-status" data-status="{status}" '
                f'onclick="filterByStatus(\'{status}\')">'
                f'<span class="legend-color" style="background:{color};border-radius:2px"></span>'
                f'{display_name} ({count})</div>'
            )

    # Category items (components)
    if category_to_nodes:
        html_parts.append('<div class="issue-categories"><b>Components</b>')
        for category in sorted(category_to_nodes.keys()):
            count = len(category_to_nodes[category])
            html_parts.append(
                f'<div class="legend-item issue-category" data-issue-category="{category}" '
                f'onclick="filterByIssueCategory(\'{category}\')">'
                f'<span class="legend-color" style="background:#8b949e;border-radius:2px"></span>'
                f'{category} ({count})</div>'
            )
        html_parts.append('</div>')

    html_parts.append('</div>')
    return "\n".join(html_parts)


def get_issue_panel_html(metadata: Dict[str, Any]) -> str:
    """Generate HTML for issue-specific fields in the detail panel."""
    if not is_issue(metadata):
        return ""

    status = metadata.get("status", "open")
    severity = metadata.get("severity", "unknown")
    component = metadata.get("component", "")
    commit = metadata.get("commit", "")

    status_color = STATUS_COLORS.get(status, "#8b949e")
    severity_color = SEVERITY_COLORS.get(severity, "#8b949e")

    html = f'''<div class="issue-badges">
        <span class="issue-badge" style="background:{status_color}">{status.upper()}</span>
        <span class="issue-badge" style="background:{severity_color}">{severity}</span>'''

    if component:
        html += f'<span class="issue-badge component">{component}</span>'

    if commit:
        html += f'<span class="issue-badge commit">#{commit[:7]}</span>'

    html += '</div>'
    return html


# CSS for issue badges (to be included in templates)
ISSUE_BADGE_CSS = """
.issue-badges { margin-bottom: 12px; }
.issue-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 4px;
    color: #fff;
}
.issue-badge.component { background: #30363d; color: #c9d1d9; }
.issue-badge.commit { background: #21262d; color: #8b949e; font-family: monospace; }
#issues-legend {
    margin-top: 16px;
    padding: 12px;
    border-top: 2px solid #58a6ff;
    border-left: 3px solid #58a6ff;
    background: rgba(88, 166, 255, 0.05);
    border-radius: 0 6px 6px 0;
}
#issues-legend b { display: block; margin-bottom: 8px; color: #58a6ff; cursor: pointer; }
#issues-legend b:hover { text-decoration: underline; }
#issues-legend b.active { background: rgba(88,166,255,0.2); padding: 2px 6px; border-radius: 4px; margin: -2px -6px 6px -6px; }
.legend-item.issue-status .legend-color {
    width: 0 !important;
    height: 0 !important;
    border-radius: 0 !important;
    border-left: 6px solid transparent !important;
    border-right: 6px solid transparent !important;
    border-bottom-width: 10px !important;
    border-bottom-style: solid !important;
    background: none !important;
}
.legend-item.issue-category .legend-color {
    width: 5px !important;
    height: 5px !important;
    border-radius: 50% !important;
}
.issue-categories { margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d; }
.issue-categories b { font-size: 11px; color: #8b949e; margin-bottom: 4px; }
.legend-item.issue-category { font-size: 11px; padding-left: 8px; }
"""


# JavaScript for issue filtering (to be included in templates)
ISSUE_FILTER_JS = """
function filterAllIssues() {
    document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
    document.querySelector('#issues-legend b').classList.add('active');
    currentFilter = 'all-issues';
    var nodeIds = [];
    Object.values(graphData.statusToNodes || {}).forEach(ids => nodeIds.push(...ids));
    applyFilter(nodeIds);
}

function filterByStatus(status) {
    document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
    var el = document.querySelector('.legend-item[data-status="' + status + '"]');
    if (el) el.classList.add('active');
    currentFilter = 'status:' + status;
    var nodeIds = graphData.statusToNodes[status] || [];
    applyFilter(nodeIds);
}

function filterByIssueCategory(category) {
    document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
    var el = document.querySelector('.legend-item[data-issue-category="' + category + '"]');
    if (el) el.classList.add('active');
    currentFilter = 'issue-category:' + category;
    var nodeIds = graphData.issueCategoryToNodes[category] || [];
    applyFilter(nodeIds);
}
"""
