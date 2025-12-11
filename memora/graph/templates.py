"""HTML/CSS/JS templates for the knowledge graph visualization."""

from .issues import ISSUE_BADGE_CSS, ISSUE_FILTER_JS
from .todos import TODO_BADGE_CSS, TODO_FILTER_JS

# Base CSS styles shared by both static and dynamic graph
BASE_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; display: flex; height: 100vh; }
#graph { flex: 1; height: 100%; }
#panel { width: 400px; background: #161b22; border-left: 1px solid #30363d; padding: 20px; overflow-y: auto; display: none; position: relative; }
#panel.active { display: block; }
#panel h2 { color: #58a6ff; margin-bottom: 10px; font-size: 16px; }
#panel .tags { margin-bottom: 15px; }
#panel .tag { display: inline-block; background: #30363d; padding: 3px 8px; border-radius: 12px; font-size: 12px; margin: 2px; cursor: pointer; }
#panel .tag:hover { background: #484f58; }
#panel .meta { color: #8b949e; font-size: 12px; margin-bottom: 15px; }
#panel .content { font-size: 13px; line-height: 1.6; background: #0d1117; padding: 15px; border-radius: 6px; max-height: calc(100vh - 200px); overflow-y: auto; }
#panel .content h1, #panel .content h2, #panel .content h3 { color: #58a6ff; margin: 16px 0 8px 0; }
#panel .content h1 { font-size: 1.4em; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
#panel .content h2 { font-size: 1.2em; }
#panel .content h3 { font-size: 1.1em; }
#panel .content p { margin: 8px 0; }
#panel .content ul, #panel .content ol { margin: 8px 0 8px 20px; }
#panel .content li { margin: 4px 0; }
#panel .content code { background: #30363d; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 12px; }
#panel .content pre { background: #0d1117; border: 1px solid #30363d; padding: 12px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }
#panel .content pre code { background: none; padding: 0; }
#panel .content a { color: #58a6ff; }
#panel .content table { border-collapse: collapse; margin: 8px 0; width: 100%; }
#panel .content th, #panel .content td { border: 1px solid #30363d; padding: 6px 10px; text-align: left; }
#panel .content th { background: #21262d; }
#panel .content blockquote { border-left: 3px solid #30363d; padding-left: 12px; margin: 8px 0; color: #8b949e; }
#panel .content .mermaid { background: #161b22; padding: 16px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }
#panel .content .memory-images { margin-top: 16px; border-top: 1px solid #30363d; padding-top: 16px; }
#panel .content .memory-image { margin: 8px 0; }
#panel .content .memory-image img { max-width: 100%; border-radius: 6px; border: 1px solid #30363d; }
#panel .content .memory-image .caption { font-size: 11px; color: #8b949e; margin-top: 4px; text-align: center; }
#panel .content strong { color: #f0f6fc; }
#panel .close { position: absolute; top: 10px; right: 15px; cursor: pointer; font-size: 20px; color: #8b949e; }
#panel .close:hover { color: #fff; }
#resize-handle { width: 6px; background: #30363d; cursor: ew-resize; display: none; }
#resize-handle:hover, #resize-handle.dragging { background: #58a6ff; }
#resize-handle.active { display: block; }
#legend { position: absolute; top: 10px; left: 10px; background: rgba(22,27,34,0.9); padding: 12px; border-radius: 6px; font-size: 12px; }
.legend-item { margin: 4px 0; display: flex; align-items: center; cursor: pointer; padding: 2px 4px; border-radius: 4px; }
.legend-item:hover { background: rgba(255,255,255,0.1); }
.legend-item.active { background: rgba(88,166,255,0.3); }
.legend-color { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
.legend-color.triangle { width: 0; height: 0; border-radius: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-bottom: 10px solid currentColor; background: none !important; }
#legend .reset { margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d; color: #58a6ff; cursor: pointer; }
#legend-items { max-height: 0; overflow: hidden; transition: max-height 0.3s ease; }
#legend-items.expanded { max-height: 300px; }
.legend-toggle { cursor: pointer; color: #8b949e; font-size: 11px; margin-left: 4px; }
.legend-toggle:hover { color: #c9d1d9; }
#legend .reset:hover { text-decoration: underline; }
#sections { position: absolute; bottom: 50px; left: 10px; background: rgba(22,27,34,0.9); padding: 12px; border-radius: 6px; font-size: 12px; max-height: 40vh; overflow-y: auto; white-space: nowrap; }
#sections b { display: block; margin-bottom: 8px; }
.section-item { margin: 4px 0; cursor: pointer; padding: 3px 6px; border-radius: 4px; color: #7ee787; }
.section-item:hover { background: rgba(255,255,255,0.1); }
.section-item.active { background: rgba(126,231,135,0.3); }
.subsection-item { margin: 2px 0 2px 8px; cursor: pointer; padding: 2px 6px; border-radius: 4px; color: #8b949e; font-size: 11px; }
.subsection-item:hover { background: rgba(255,255,255,0.1); }
.subsection-item.active { background: rgba(88,166,255,0.3); color: #c9d1d9; }
#help { position: absolute; bottom: 10px; left: 10px; background: rgba(22,27,34,0.9); padding: 8px 12px; border-radius: 6px; font-size: 11px; color: #8b949e; }
"""

# Additional CSS for dynamic (SPA) graph
SPA_CSS = """
#loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #58a6ff; font-size: 16px; }
#search-box { position: absolute; top: 10px; right: 10px; background: rgba(22,27,34,0.9); padding: 8px; border-radius: 6px; }
#search-box input { background: #0d1117; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 10px; border-radius: 4px; width: 200px; }
#search-box input:focus { outline: none; border-color: #58a6ff; }
"""

# JavaScript for markdown/mermaid rendering
RENDER_JS = """
// Configure marked for GitHub-flavored markdown
marked.setOptions({ breaks: true, gfm: true });

// Initialize mermaid with dark theme
mermaid.initialize({
    startOnLoad: false,
    theme: 'dark',
    themeVariables: {
        primaryColor: '#58a6ff',
        primaryTextColor: '#c9d1d9',
        primaryBorderColor: '#30363d',
        lineColor: '#8b949e',
        secondaryColor: '#21262d',
        tertiaryColor: '#161b22'
    }
});

function renderMarkdown(text) {
    var renderer = new marked.Renderer();
    renderer.link = function(href, title, text) {
        return '<a href="' + href + '" target="_blank">' + text + '</a>';
    };
    return marked.parse(text, { renderer: renderer });
}

async function renderMermaidBlocks() {
    var blocks = document.querySelectorAll('#panel-content pre code.language-mermaid');
    for (var block of blocks) {
        var container = document.createElement('div');
        container.className = 'mermaid';
        container.textContent = block.textContent;
        block.parentElement.replaceWith(container);
    }
    if (blocks.length > 0) await mermaid.run();
}

function renderImages(metadata) {
    var images = metadata && metadata.images;
    if (!images || images.length === 0) return '';
    var html = '<div class="memory-images">';
    for (var img of images) {
        html += '<div class="memory-image"><img src="' + img.src + '" alt="' + (img.caption || '') + '">';
        if (img.caption) html += '<div class="caption">' + img.caption + '</div>';
        html += '</div>';
    }
    return html + '</div>';
}

function renderIssueBadges(metadata) {
    if (!metadata || metadata.type !== 'issue') return '';
    var status = metadata.status || 'open';
    var severity = metadata.severity || 'unknown';
    var component = metadata.component || '';
    var commit = metadata.commit || '';

    var statusColors = {open: '#ff7b72', in_progress: '#ffa657', resolved: '#7ee787', wontfix: '#8b949e'};
    var severityColors = {critical: '#f85149', major: '#d29922', minor: '#8b949e'};

    var html = '<div class="issue-badges">';
    html += '<span class="issue-badge" style="background:' + (statusColors[status] || '#8b949e') + '">' + status.toUpperCase() + '</span>';
    html += '<span class="issue-badge" style="background:' + (severityColors[severity] || '#8b949e') + '">' + severity + '</span>';
    if (component) html += '<span class="issue-badge component">' + component + '</span>';
    if (commit) html += '<span class="issue-badge commit">#' + commit.slice(0,7) + '</span>';
    html += '</div>';
    return html;
}
"""

# JavaScript for filtering
FILTER_JS = """
function filterByTag(tag) {
    document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
    var el = document.querySelector('.legend-item[data-tag="' + tag + '"]');
    if (el) el.classList.add('active');
    currentFilter = tag;
    var nodeIds = (typeof graphData !== 'undefined' ? graphData.tagToNodes : tagToNodes)[tag] || [];
    applyFilter(nodeIds);
}

function filterBySection(section) {
    document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
    var el = document.querySelector('.section-item[data-section="' + section + '"]');
    if (el) el.classList.add('active');
    currentFilter = section;
    var nodeIds = (typeof graphData !== 'undefined' ? graphData.sectionToNodes : sectionToNodes)[section] || [];
    applyFilter(nodeIds);
}

function filterBySubsection(subsection) {
    document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
    var el = document.querySelector('.subsection-item[data-subsection="' + subsection + '"]');
    if (el) el.classList.add('active');
    currentFilter = subsection;
    var nodeIds = (typeof graphData !== 'undefined' ? graphData.subsectionToNodes : subsectionToNodes)[subsection] || [];
    applyFilter(nodeIds);
}

function applyFilter(nodeIds) {
    var nodeSet = new Set(nodeIds);
    var sourceNodes = typeof graphData !== 'undefined' ? graphData.nodes : allNodes;
    var sourceEdges = typeof graphData !== 'undefined' ? graphData.edges : allEdges;
    nodes.clear();
    edges.clear();
    var filteredNodes = sourceNodes.filter(n => nodeSet.has(n.id));
    var filteredEdges = sourceEdges.filter(e => nodeSet.has(e.from) && nodeSet.has(e.to));
    nodes.add(filteredNodes);
    edges.add(filteredEdges);
    network.fit({ animation: true });
}

function resetFilter() {
    document.querySelectorAll('.legend-item, .section-item, .subsection-item').forEach(el => el.classList.remove('active'));
    currentFilter = null;
    var sourceNodes = typeof graphData !== 'undefined' ? graphData.nodes : allNodes;
    var sourceEdges = typeof graphData !== 'undefined' ? graphData.edges : allEdges;
    nodes.clear();
    edges.clear();
    nodes.add(sourceNodes);
    edges.add(sourceEdges);
    network.fit({ animation: true });
}
"""

# JavaScript for resize handle
RESIZE_JS = """
var resizeHandle = document.getElementById('resize-handle');
var panel = document.getElementById('panel');
var isResizing = false;

resizeHandle.addEventListener('mousedown', function(e) {
    isResizing = true;
    resizeHandle.classList.add('dragging');
    document.body.style.cursor = 'ew-resize';
    e.preventDefault();
});

document.addEventListener('mousemove', function(e) {
    if (!isResizing) return;
    var newWidth = window.innerWidth - e.clientX;
    if (newWidth >= 200 && newWidth <= 800) panel.style.width = newWidth + 'px';
});

document.addEventListener('mouseup', function() {
    isResizing = false;
    resizeHandle.classList.remove('dragging');
    document.body.style.cursor = '';
});
"""

# JavaScript for panel display
PANEL_JS = """
function closePanel() {
    document.getElementById('panel').classList.remove('active');
    document.getElementById('resize-handle').classList.remove('active');
}

function showPanel(mem) {
    document.getElementById('panel-title').textContent = 'Memory #' + mem.id;

    // Show issue badges if applicable
    var badgesHtml = renderIssueBadges(mem.metadata);
    document.getElementById('panel-meta').innerHTML = badgesHtml + 'Created: ' + mem.created;

    document.getElementById('panel-tags').innerHTML = mem.tags.map(function(t) {
        return '<span class="tag" onclick="filterByTag(\\'' + t + '\\'); event.stopPropagation();">' + t + '</span>';
    }).join('');

    document.getElementById('panel-content').innerHTML = renderMarkdown(mem.content);
    renderMermaidBlocks();
    document.getElementById('panel-content').innerHTML += renderImages(mem.metadata);
    document.getElementById('panel').classList.add('active');
    document.getElementById('resize-handle').classList.add('active');
}
"""


def get_full_css() -> str:
    """Get complete CSS including issue and TODO styles."""
    return BASE_CSS + "\n" + ISSUE_BADGE_CSS + "\n" + TODO_BADGE_CSS


def get_spa_css() -> str:
    """Get CSS for SPA (dynamic) graph."""
    return BASE_CSS + "\n" + SPA_CSS + "\n" + ISSUE_BADGE_CSS + "\n" + TODO_BADGE_CSS


def get_full_js() -> str:
    """Get complete JavaScript for graph functionality."""
    return "\n".join([RENDER_JS, FILTER_JS, ISSUE_FILTER_JS, TODO_FILTER_JS, PANEL_JS, RESIZE_JS])


def build_static_html(
    nodes_json: str,
    edges_json: str,
    memories_json: str,
    tag_to_nodes_json: str,
    section_to_nodes_json: str,
    path_to_nodes_json: str,
    status_to_nodes_json: str,
    todo_status_to_nodes_json: str,
    legend_html: str,
    sections_html: str,
    issues_legend_html: str,
    todos_legend_html: str,
) -> str:
    """Build complete static HTML for export."""
    css = get_full_css()
    js = get_full_js()

    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Memory Knowledge Graph</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" rel="stylesheet">
    <style>{css}</style>
</head>
<body>
    <div id="graph"></div>
    <div id="resize-handle"></div>
    <div id="panel">
        <span class="close" onclick="closePanel()">&times;</span>
        <h2 id="panel-title">Memory #</h2>
        <div class="meta" id="panel-meta"></div>
        <div class="tags" id="panel-tags"></div>
        <div class="content" id="panel-content"></div>
    </div>
    <div id="legend"><b>Tags</b>{legend_html}{issues_legend_html}{todos_legend_html}<div class="reset" onclick="resetFilter()">Show All</div></div>
    <div id="sections"><b>Sections</b>{sections_html}</div>
    <div id="help">Click tag/section to filter | Click node to view | Scroll to zoom</div>
    <script>
        var memoriesData = {memories_json};
        var tagToNodes = {tag_to_nodes_json};
        var sectionToNodes = {section_to_nodes_json};
        var subsectionToNodes = {path_to_nodes_json};
        var statusToNodes = {status_to_nodes_json};
        var todoStatusToNodes = {todo_status_to_nodes_json};
        var allNodes = {nodes_json};
        var allEdges = {edges_json};
        var currentFilter = null;
        var graphData = {{ statusToNodes: statusToNodes, todoStatusToNodes: todoStatusToNodes }};

        {js}

        // Initialize graph
        var nodes = new vis.DataSet(allNodes);
        var edges = new vis.DataSet(allEdges);
        var container = document.getElementById("graph");
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            nodes: {{ shape: "dot", size: 16, font: {{ color: "#c9d1d9", size: 11 }}, borderWidth: 2 }},
            edges: {{ color: {{ color: "#30363d", opacity: 0.6 }}, smooth: {{ type: "continuous" }} }},
            physics: {{ barnesHut: {{ gravitationalConstant: -2500, springLength: 120, damping: 0.3 }} }},
            interaction: {{ hover: true, tooltipDelay: 100 }}
        }};
        var network = new vis.Network(container, data, options);

        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var mem = memoriesData[nodeId];
                if (mem) showPanel(mem);
            }}
        }});
    </script>
</body>
</html>'''


def get_spa_html() -> str:
    """Get SPA HTML template for dynamic graph server."""
    css = get_spa_css()
    js = get_full_js()

    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Memory Knowledge Graph</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" rel="stylesheet">
    <style>{css}</style>
</head>
<body>
    <div id="graph"><div id="loading">Loading memories...</div></div>
    <div id="resize-handle"></div>
    <div id="panel">
        <span class="close" onclick="closePanel()">&times;</span>
        <h2 id="panel-title">Memory #</h2>
        <div class="meta" id="panel-meta"></div>
        <div class="tags" id="panel-tags"></div>
        <div class="content" id="panel-content"></div>
    </div>
    <div id="legend"><b>Tags</b><span class="legend-toggle" onclick="toggleTags()">[+]</span><div id="legend-items"></div><div id="issues-legend-items"></div><div id="todos-legend-items"></div><div class="reset" onclick="resetFilter()">Show All</div></div>
    <div id="sections"><b>Sections</b><div id="section-items"></div></div>
    <div id="search-box"><input type="text" id="search" placeholder="Search memories..." oninput="searchMemories(this.value)"></div>
    <div id="help">Click tag/section to filter | Click node to view | Scroll to zoom | Type to search</div>
    <script>
        var graphData = null;
        var nodes, edges, network;
        var currentFilter = null;
        var memoryCache = {{}};

        {js}

        async function loadGraph() {{
            try {{
                const response = await fetch('/api/graph');
                graphData = await response.json();
                if (graphData.error) {{
                    document.getElementById('loading').textContent = graphData.message || 'No memories found';
                    return;
                }}
                initGraph();
            }} catch (e) {{
                document.getElementById('loading').textContent = 'Error loading graph: ' + e.message;
            }}
        }}

        function initGraph() {{
            document.getElementById('loading').remove();

            // Build tag legend
            var legendHtml = '';
            var tagEntries = Object.entries(graphData.tagColors).slice(0, 12);
            for (var [tag, color] of tagEntries) {{
                legendHtml += '<div class="legend-item" data-tag="' + tag + '" onclick="filterByTag(\\'' + tag + '\\')"><span class="legend-color" style="background:' + color + '"></span>' + tag + '</div>';
            }}
            document.getElementById('legend-items').innerHTML = legendHtml;

            // Build issues legend
            var issuesHtml = '';
            if (graphData.statusToNodes && Object.keys(graphData.statusToNodes).length > 0) {{
                issuesHtml = '<div style="margin-top:8px;padding-top:8px;border-top:1px solid #30363d"><b>Issues</b></div>';
                var statusColors = {{open: '#ff7b72', in_progress: '#ffa657', resolved: '#7ee787', wontfix: '#8b949e'}};
                for (var [status, nodeIds] of Object.entries(graphData.statusToNodes)) {{
                    var color = statusColors[status] || '#8b949e';
                    var displayName = status.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                    issuesHtml += '<div class="legend-item" data-status="' + status + '" onclick="filterByStatus(\\'' + status + '\\')"><span class="legend-color" style="background:' + color + ';border-radius:2px"></span>' + displayName + ' (' + nodeIds.length + ')</div>';
                }}
            }}
            document.getElementById('issues-legend-items').innerHTML = issuesHtml;

            // Build TODOs legend
            var todosHtml = '';
            if (graphData.todoStatusToNodes && Object.keys(graphData.todoStatusToNodes).length > 0) {{
                todosHtml = '<div style="margin-top:8px;padding-top:8px;border-top:1px solid #30363d"><b>TODOs</b></div>';
                var todoStatusColors = {{open: '#58a6ff', in_progress: '#ffa657', completed: '#7ee787', blocked: '#f85149'}};
                for (var [status, nodeIds] of Object.entries(graphData.todoStatusToNodes)) {{
                    var color = todoStatusColors[status] || '#8b949e';
                    var displayName = status.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                    todosHtml += '<div class="legend-item" data-todo-status="' + status + '" onclick="filterByTodoStatus(\\'' + status + '\\')"><span class="legend-color triangle" style="border-bottom-color:' + color + '"></span>' + displayName + ' (' + nodeIds.length + ')</div>';
                }}
            }}
            document.getElementById('todos-legend-items').innerHTML = todosHtml;

            function toggleTags() {{
                var items = document.getElementById('legend-items');
                var toggle = document.querySelector('.legend-toggle');
                if (items.classList.contains('expanded')) {{
                    items.classList.remove('expanded');
                    toggle.textContent = '[+]';
                }} else {{
                    items.classList.add('expanded');
                    toggle.textContent = '[-]';
                }}
            }}
            window.toggleTags = toggleTags;

            // Build sections
            var sectionsHtml = '';
            for (var [section, nodeIds] of Object.entries(graphData.sectionToNodes)) {{
                sectionsHtml += '<div class="section-item" data-section="' + section + '" onclick="filterBySection(\\'' + section + '\\')">' + section + ' (' + nodeIds.length + ')</div>';
                var sectionPaths = Object.keys(graphData.subsectionToNodes).filter(k => k.startsWith(section + '/')).sort();
                var rendered = new Set();
                for (var fullPath of sectionPaths) {{
                    var subPath = fullPath.slice(section.length + 1);
                    var parts = subPath.split('/');
                    for (var i = 0; i < parts.length; i++) {{
                        var partial = parts.slice(0, i + 1).join('/');
                        var renderKey = section + '/' + partial;
                        if (!rendered.has(renderKey)) {{
                            rendered.add(renderKey);
                            var indent = '&nbsp;&nbsp;'.repeat(i);
                            var count = (graphData.subsectionToNodes[renderKey] || []).length;
                            sectionsHtml += '<div class="subsection-item" data-subsection="' + renderKey + '" onclick="filterBySubsection(\\'' + renderKey + '\\')" style="padding-left:' + (8 + i*12) + 'px;">' + indent + '\\u2514 ' + parts[i] + ' (' + count + ')</div>';
                        }}
                    }}
                }}
            }}
            document.getElementById('section-items').innerHTML = sectionsHtml;

            // Init vis.js
            nodes = new vis.DataSet(graphData.nodes);
            edges = new vis.DataSet(graphData.edges);
            var container = document.getElementById('graph');
            var data = {{ nodes: nodes, edges: edges }};
            var options = {{
                nodes: {{ shape: 'dot', size: 16, font: {{ color: '#c9d1d9', size: 11 }}, borderWidth: 2 }},
                edges: {{ color: {{ color: '#30363d', opacity: 0.6 }}, smooth: {{ type: 'continuous' }} }},
                physics: {{ barnesHut: {{ gravitationalConstant: -2500, springLength: 120, damping: 0.3 }} }},
                interaction: {{ hover: true, tooltipDelay: 100 }}
            }};
            network = new vis.Network(container, data, options);

            network.on('click', async function(params) {{
                if (params.nodes.length > 0) {{
                    var nodeId = params.nodes[0];
                    await showMemoryAsync(nodeId);
                }}
            }});
        }}

        async function showMemoryAsync(nodeId) {{
            if (!memoryCache[nodeId]) {{
                try {{
                    const response = await fetch('/api/memories/' + nodeId);
                    memoryCache[nodeId] = await response.json();
                }} catch (e) {{
                    console.error('Error fetching memory:', e);
                    return;
                }}
            }}
            var mem = memoryCache[nodeId];
            if (mem.error) return;
            showPanel(mem);
        }}

        function searchMemories(query) {{
            if (!query || query.length < 2) {{
                resetFilter();
                return;
            }}
            query = query.toLowerCase();
            var matchingIds = graphData.nodes.filter(n => n.label.toLowerCase().includes(query)).map(n => n.id);
            applyFilter(matchingIds);
        }}

        // Load graph on page load
        loadGraph();
    </script>
</body>
</html>'''
