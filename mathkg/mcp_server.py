"""A thin MCP server over stdio, proxying tool calls to the warm HTTP viewer.

The viewer (``run.py serve``, port 8765) is the single source of truth: this
MCP server holds **no** graph of its own. Every tool call is one HTTP request
to the viewer, which updates its resident ``ViewerState`` and pushes the change
to the open browser tab over SSE. That is what makes "pilot the graph" fluid:
an action is a ~100 ms round-trip, never a rebuild.

Usage::

    MATHKG_VIEWER_URL=http://127.0.0.1:8765 python -m mathkg.mcp_server
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "mathkg", "version": "2.0.0"}
DEFAULT_VIEWER = os.environ.get("MATHKG_VIEWER_URL", "http://127.0.0.1:8765")

#: 12 hue-spaced colours for categorical colouring (matches the 04 field spec).
CATEGORICAL_PALETTE = [
    "#c24d5d", "#c0571e", "#ac6900", "#897d00", "#508d23", "#009563",
    "#009496", "#008abc", "#337bd0", "#736ace", "#9b5bb6", "#b6508e",
]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_state",
        "description": "Current view: node/edge counts, spec and a readable summary.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_spec",
        "description": (
            "Patch the view spec with dotted key=value strings, e.g. "
            "['node.size=6 + 2.2 * degree', 'edge.opacity=0.4']. Expressions may "
            "use node columns (degree, sitelinks, kind, field, betweenness, "
            "closeness, eigenvector, pagerank, community)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "set": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["set"],
        },
    },
    {
        "name": "filter",
        "description": "Restrict the graph: min_degree, max_degree, kinds, largest component.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_degree": {"type": "integer"},
                "max_degree": {"type": "integer"},
                "kinds": {"type": "array", "items": {"type": "string"}},
                "component": {"type": "string", "enum": ["largest"]},
            },
        },
    },
    {
        "name": "layout",
        "description": "Switch the layout: installed, forceatlas2, layered, spring, physics, circular, radial.",
        "inputSchema": {
            "type": "object",
            "properties": {"layout": {"type": "string"}},
            "required": ["layout"],
        },
    },
    {
        "name": "color_by",
        "description": (
            "Colour nodes by a column. mode 'categorical' assigns one colour per "
            "distinct value; otherwise values are percentile-ranked along the "
            "palette. Column examples: degree, field, betweenness, pagerank, "
            "community."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "mode": {"type": "string", "enum": ["categorical", "numeric"]},
                "palette": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["column"],
        },
    },
    {
        "name": "ego",
        "description": "Replace the view with everything within N hops of a notion.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node": {"type": "string"},
                "radius": {"type": "integer", "default": 2},
            },
            "required": ["node"],
        },
    },
    {
        "name": "path",
        "description": "Prerequisite chain between two notions (directed or undirected).",
        "inputSchema": {
            "type": "object",
            "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
            "required": ["from", "to"],
        },
    },
    {
        "name": "describe",
        "description": "Counts, immediate prerequisites/enables, longest chain through a notion.",
        "inputSchema": {
            "type": "object",
            "properties": {"node": {"type": "string"}},
            "required": ["node"],
        },
    },
    {
        "name": "reset",
        "description": "Restore the full graph after an ego narrowing.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "centrality",
        "description": (
            "Compute a centrality per node and expose it as a column (usable by "
            "set_spec/color_by). kind: betweenness, closeness, eigenvector, pagerank."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["betweenness", "closeness", "eigenvector", "pagerank"]},
                "topk": {"type": "integer", "default": 10},
            },
            "required": ["kind"],
        },
    },
    {
        "name": "community",
        "description": "Detect communities (Louvain) and expose them as the 'community' column.",
        "inputSchema": {
            "type": "object",
            "properties": {"resolution": {"type": "number", "default": 1.0}},
        },
    },
    {
        "name": "select",
        "description": "List the neighbours of a notion within a radius (highlight support).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node": {"type": "string"},
                "radius": {"type": "integer", "default": 1},
            },
            "required": ["node"],
        },
    },
    {
        "name": "render_png",
        "description": "Render the current graph and spec to a PNG, and return its path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "mcp_render"},
                "figsize": {"type": "number", "default": 16},
            },
        },
    },
    {
        "name": "reload_graph",
        "description": "Rebuild the graph from the data files (after re-running the edges stage).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _post(base: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read())
        except Exception:
            return {"error": f"HTTP {exc.code}"}
    except urllib.error.URLError as exc:
        return {"error": f"viewer not reachable ({exc.reason}). Start it with `run.py serve`."}


def _get(base: str, path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=10) as response:
            return json.loads(response.read())
    except urllib.error.URLError as exc:
        return {"error": f"viewer not reachable ({exc.reason}). Start it with `run.py serve`."}


class MathKGClient:
    """A proxy: one method per MCP tool, each one HTTP call to the viewer."""

    def __init__(self, base_url: str = DEFAULT_VIEWER, post=None, get=None):
        self.base = base_url.rstrip("/")
        self._post = post or _post
        self._get = get or _get

    # -- tools --------------------------------------------------------------- #

    def get_state(self, **_kwargs) -> dict:
        spec = self._get(self.base, "/state")
        health = self._get(self.base, "/health")
        return {
            "version": spec.get("version"),
            "spec": spec.get("spec"),
            "nodes": health.get("nodes"),
            "edges": health.get("edges"),
        }

    def set_spec(self, set: list[str] | None = None, **_kwargs) -> dict:  # noqa: A002
        return self._post(self.base, "/spec", {"set": set or []})

    def filter(self, min_degree: int = 0, max_degree: int = 0,
               kinds: list[str] | None = None, component: str | None = None,
               **_kwargs) -> dict:
        patch = []
        if min_degree:
            patch.append(f"filter.min_degree={min_degree}")
        if max_degree:
            patch.append(f"filter.max_degree={max_degree}")
        if kinds:
            patch.append(f"filter.kinds={json.dumps(kinds)}")
        if component:
            patch.append(f"filter.component={component}")
        if not patch:
            return {"error": "no filter argument given"}
        return self._post(self.base, "/spec", {"set": patch})

    def layout(self, layout: str, **_kwargs) -> dict:
        return self._post(self.base, "/spec", {"set": [f"layout={layout}"]})

    def color_by(self, column: str, mode: str = "numeric",
                 palette: list[str] | None = None, **_kwargs) -> dict:
        rule: dict[str, Any] = {"column": column}
        if mode:
            rule["mode"] = mode
        if palette:
            rule["palette"] = palette
        elif mode == "categorical":
            rule["palette"] = CATEGORICAL_PALETTE
        return self._post(self.base, "/spec", {"set": [f"node.color_by={json.dumps(rule)}"]})

    def ego(self, node: str, radius: int = 2, **_kwargs) -> dict:
        return self._post(self.base, "/query", {"kind": "ego", "node": node, "radius": radius})

    def path(self, source: str | None = None, target: str | None = None,
             **kwargs: Any) -> dict:
        source = source or kwargs.get("from") or ""
        target = target or kwargs.get("to") or ""
        return self._post(self.base, "/query", {"kind": "path", "from": source, "to": target})

    def describe(self, node: str, **_kwargs) -> dict:
        return self._post(self.base, "/query", {"kind": "describe", "node": node})

    def reset(self, **_kwargs) -> dict:
        return self._post(self.base, "/query", {"kind": "reset"})

    def centrality(self, kind: str = "betweenness", topk: int = 10, **_kwargs) -> dict:
        return self._post(self.base, "/compute", {"kind": kind, "topk": topk})

    def community(self, resolution: float = 1.0, **_kwargs) -> dict:
        return self._post(self.base, "/compute", {"kind": "communities", "resolution": resolution})

    def select(self, node: str, radius: int = 1, **_kwargs) -> dict:
        return self._post(self.base, "/select", {"action": "neighbors", "node": node, "radius": radius})

    def render_png(self, name: str = "mcp_render", figsize: float = 16.0, **_kwargs) -> dict:
        return self._post(self.base, "/render_png", {"name": name, "figsize": figsize})

    def reload_graph(self, **_kwargs) -> dict:
        return self._post(self.base, "/reload", {})

    def call(self, name: str, arguments: dict) -> dict:
        handler = {
            "get_state": self.get_state,
            "set_spec": self.set_spec,
            "filter": self.filter,
            "layout": self.layout,
            "color_by": self.color_by,
            "ego": self.ego,
            "path": self.path,
            "describe": self.describe,
            "reset": self.reset,
            "centrality": self.centrality,
            "community": self.community,
            "select": self.select,
            "render_png": self.render_png,
            "reload_graph": self.reload_graph,
        }.get(name)
        if handler is None:
            raise ValueError(f"unknown tool {name!r}")
        return handler(**arguments)


# --------------------------------------------------------------------------- #
# JSON-RPC plumbing (unchanged)
# --------------------------------------------------------------------------- #

def handle(client: MathKGClient, request: dict) -> dict | None:
    method = request.get("method")
    request_id = request.get("id")

    if request_id is None:  # notification: no response
        return None

    def reply(result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def error(code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    try:
        if method == "initialize":
            return reply({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            })
        if method == "tools/list":
            return reply({"tools": TOOLS})
        if method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name", "")
            arguments = params.get("arguments") or {}
            result = client.call(name, arguments)
            return reply({"content": [{"type": "text", "text": json.dumps(result)}], "isError": False})
        return error(-32601, f"unknown method {method!r}")
    except Exception as exc:
        return error(-32603, str(exc))


def main() -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    client = MathKGClient()
    log.warning("mathkg MCP (thin client) -> %s", client.base)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "bad JSON"}}
            ) + "\n")
            sys.stdout.flush()
            continue
        response = handle(client, request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
