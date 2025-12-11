"""Issue-specific visualization logic for the knowledge graph."""

from typing import Any, Dict, Optional

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

# Default tag colors
TAG_COLORS = [
    "#58a6ff", "#f78166", "#a371f7", "#7ee787",
    "#ffa657", "#ff7b72", "#79c0ff", "#d2a8ff"
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
        "shape": "square",
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


def build_issue_legend_html(status_to_nodes: Dict[str, list]) -> str:
    """Build HTML for issue status legend section."""
    if not status_to_nodes:
        return ""

    html_parts = ['<div id="issues-legend"><b>Issues</b>']

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
#issues-legend { margin-top: 12px; padding-top: 8px; border-top: 1px solid #30363d; }
#issues-legend b { display: block; margin-bottom: 8px; }
.legend-item.issue-status .legend-color { border-radius: 2px !important; }
"""


# JavaScript for issue filtering (to be included in templates)
ISSUE_FILTER_JS = """
function filterByStatus(status) {
    document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
    var el = document.querySelector('.legend-item[data-status="' + status + '"]');
    if (el) el.classList.add('active');
    currentFilter = 'status:' + status;
    var nodeIds = graphData.statusToNodes[status] || [];
    applyFilter(nodeIds);
}
"""
