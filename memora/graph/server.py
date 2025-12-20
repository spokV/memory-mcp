"""HTTP server and routes for graph visualization."""

import os
import socket
import sys
import threading
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from .data import export_graph_html, get_graph_data, get_memory_for_api
from .templates import get_spa_html

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _is_port_in_use(host: str, port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def register_graph_routes(mcp: "FastMCP") -> None:
    """Register graph-related routes on the MCP server.

    Args:
        mcp: The FastMCP server instance.
    """

    @mcp.custom_route("/graph", methods=["GET"])
    async def serve_graph(request: Request):
        """Serve the knowledge graph visualization via HTTP."""
        try:
            min_score = float(request.query_params.get("min_score", 0.25))
            result = export_graph_html(output_path=None, min_score=min_score)
            if "error" in result:
                return JSONResponse(result, status_code=404)
            return HTMLResponse(result["html"])
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)


def start_graph_server(host: str, port: int) -> None:
    """Start background HTTP server for graph visualization.

    This server provides:
    - /graph: SPA HTML page
    - /api/graph: Graph data API
    - /api/memories/{id}: Individual memory API
    - /r2/{path}: R2 image proxy

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    if _is_port_in_use(host, port):
        print(f"Graph server port {port} already in use, skipping", file=sys.stderr)
        return

    from starlette.applications import Starlette
    from starlette.routing import Route

    GRAPH_HTML = get_spa_html()

    async def graph_handler(request: Request):
        """Serve the static graph SPA."""
        return HTMLResponse(GRAPH_HTML)

    async def api_graph(request: Request):
        """API endpoint: Get graph nodes and edges."""
        try:
            min_score = float(request.query_params.get("min_score", 0.25))
            rebuild = request.query_params.get("rebuild", "").lower() == "true"
            result = get_graph_data(min_score, rebuild=rebuild)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_memory(request: Request):
        """API endpoint: Get a single memory by ID."""
        try:
            memory_id = int(request.path_params.get("id"))
            result = get_memory_for_api(memory_id)
            if result.get("error") == "not_found":
                return JSONResponse(result, status_code=404)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def r2_image_proxy(request: Request):
        """Proxy images from R2 storage."""
        try:
            from ..image_storage import get_image_storage_instance

            image_storage = get_image_storage_instance()
            if not image_storage:
                return JSONResponse({"error": "R2 not configured"}, status_code=503)

            key = request.path_params.get("path", "")
            if not key:
                return JSONResponse({"error": "No path provided"}, status_code=400)

            try:
                response = image_storage.s3_client.get_object(
                    Bucket=image_storage.bucket,
                    Key=key,
                )
                image_data = response["Body"].read()
                content_type = response.get("ContentType", "image/jpeg")

                return Response(
                    content=image_data,
                    media_type=content_type,
                    headers={"Cache-Control": "public, max-age=86400"},
                )
            except Exception as e:
                return JSONResponse({"error": f"Image not found: {e}"}, status_code=404)

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    app = Starlette(
        routes=[
            Route("/graph", graph_handler),
            Route("/api/graph", api_graph),
            Route("/api/memories/{id:int}", api_memory),
            Route("/r2/{path:path}", r2_image_proxy),
        ]
    )

    def run_server():
        import uvicorn

        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Get bucket name for unique URL
    bucket_name = ""
    try:
        from ..storage import STORAGE_BACKEND
        if hasattr(STORAGE_BACKEND, 'bucket'):
            bucket_name = STORAGE_BACKEND.bucket
    except Exception:
        pass

    bucket_param = f"?bucket={bucket_name}" if bucket_name else ""
    print(f"Graph visualization available at http://{host}:{port}/graph{bucket_param}", file=sys.stderr)
