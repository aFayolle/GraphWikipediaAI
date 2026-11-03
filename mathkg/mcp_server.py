"""A minimal MCP server over stdio, wrapping the same core as the HTTP viewer.

Deliberately hand-rolled and dependency-free: the protocol surface needed here
is `initialize`, `tools/list` and `tools/call`, all of which are newline-delimited
JSON-RPC 2.0. Pulling a framework in to expose eight tools would be more code to
keep working than the tools themselves.

Everything routes through :class:`~mathkg.server.ViewerState`, so the MCP path
and the browser path cannot drift apart: a spec patch here is the same spec patch
there, and `render_png` writes an image an agent can actually look at.

Usage::

    python -m mathkg.mcp_server          # speaks MCP on stdin/stdout
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import networkx as nx

from . import config, dependency, view_render
from . import edges as edges_module
from . import view as view_module
from .server import ViewerState

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "mathkg", "version": "1.0.0"}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_state",
        "description": "Current spec, graph size and a short description of the view.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_spec",
        "description": (
            "Patch the view spec. Keys are dotted paths, values may be expressions over "
            "node columns (degree, sitelinks, kind, is_anchor) or edge columns (confidence, "
            "popularity, occurrences, relative_position, in_lead, in_proof). "
            "Example: {'node.size': '6 + 2.2 * degree'}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "set": {"type": "array", "items": {"type": "string"},
                        "description": "list of key=value patches"},
            },
            "required": ["set"],
        },
    },
    {
        "name": "filter",
        "description": "Restrict the graph: min_degree, kinds, largest component only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_degree": {"type": "integer"},
                "kinds": {"type": "array", "items": {"type": "string"}},
                "component": {"type": "string", "enum": ["largest"]},
            },
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
        "description": (
            "Prerequisite chain between two notions. Reports how it was found: "
            "directed (a real chain), undirected (direction mixed) or none."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
            "required": ["from", "to"],
        },
    },
    {
        "name": "describe",
        "description": "Counts, immediate prerequisites and enables, longest chain through a notion.",
        "inputSchema": {
            "type": "object",
            "properties": {"node": {"type": "string"}},
            "required": ["node"],
        },
    },
    {
        "name": "render_png",
        "description": "Render the current graph and spec to a PNG file, and return its path.",
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
        "description": "Rebuild the graph from edges.csv (after re-running the edges stage).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class MathKGServer:
    """Holds the graph and spec, and answers tool calls."""

    def __init__(self, graph: nx.Graph | None = None, spec: dict | None = None):
        self.spec = spec or view_module.load_spec()
        self.graph = graph if graph is not None else self._build_graph()
        mentions_path = config.RAW_DIR / "mentions.csv"
        self.mentions = (
            __import__("pandas").read_csv(mentions_path) if mentions_path.exists() else None
        )
        self.state = ViewerState(self.graph, self.spec, self.mentions)

    @staticmethod
    def _build_graph() -> nx.DiGraph:
        frame = dependency.load_edges()
        _, titles = edges_module.load_summaries()
        graph = dependency.build_dag(frame, titles=set(titles))
        graph, _ = dependency.break_cycles(graph)
        return dependency.enrich_with_attributes(graph)

    # -- tools --------------------------------------------------------------- #

    def get_state(self, **_kwargs) -> dict:
        return {
            "nodes": self.state.graph.number_of_nodes(),
            "edges": self.state.graph.number_of_edges(),
            "spec": self.state.spec,
            "detail": self.state.detail(),
        }

    def set_spec(self, set: list[str] | None = None, **_kwargs) -> dict:  # noqa: A002
        message = self.state.patch_spec(set or [])
        return {"version": message["version"], "detail": message["detail"]}

    def filter(self, min_degree: int = 0, kinds: list[str] | None = None,
               component: str | None = None, **_kwargs) -> dict:
        patch = []
        if min_degree:
            patch.append(f"filter.min_degree={min_degree}")
        if kinds:
            patch.append(f"filter.kinds={json.dumps(kinds)}")
        if component:
            patch.append(f"filter.component={component}")
        if patch:
            self.state.patch_spec(patch)
        return {"nodes": self.state.graph.number_of_nodes(), "filter": self.state.spec["filter"]}

    def ego(self, node: str, radius: int = 2, **_kwargs) -> dict:
        if node not in self.state.graph:
            return {"error": f"{node!r} is not in the graph"}
        subgraph = nx.ego_graph(self.state.graph, node, radius=radius, undirected=True)
        self.state.replace_graph(subgraph.copy())
        distances = nx.single_source_shortest_path_length(
            self.state.graph.to_undirected(), node
        )
        by_hop: dict[int, int] = {}
        for distance in distances.values():
            by_hop[distance] = by_hop.get(distance, 0) + 1
        return {"node": node, "hops": by_hop, "nodes": self.state.graph.number_of_nodes()}

    def path(self, source: str | None = None, target: str | None = None,
             **kwargs: Any) -> dict:
        source = source or kwargs.get("from")
        target = target or kwargs.get("to")
        chain, how = dependency.path_between(self.state.graph, source or "", target or "")
        return {"chain": chain, "how": how, "steps": len(chain) if chain else 0}

    def describe(self, node: str, **_kwargs) -> dict:
        return dependency.describe(self.state.graph, node)

    def render_png(self, name: str = "mcp_render", figsize: float = 16.0, **_kwargs) -> dict:
        out = config.ARTIFACT_DIR / "mcp" / f"{name}.png"
        view_module.validate_spec(self.state.spec)
        view_render.render_spec(
            self.state.graph,
            self.state.spec,
            out,
            mentions=self.mentions,
            figsize=(float(figsize), float(figsize) * 0.7),
        )
        return {"path": str(out), "bytes": out.stat().st_size}

    def reload_graph(self, **_kwargs) -> dict:
        self.state.replace_graph(self._build_graph())
        return {"nodes": self.state.graph.number_of_nodes(),
                "edges": self.state.graph.number_of_edges()}

    def call(self, name: str, arguments: dict) -> dict:
        handler = {
            "get_state": self.get_state,
            "set_spec": self.set_spec,
            "filter": self.filter,
            "ego": self.ego,
            "path": self.path,
            "describe": self.describe,
            "render_png": self.render_png,
            "reload_graph": self.reload_graph,
        }.get(name)
        if handler is None:
            raise ValueError(f"unknown tool {name!r}")
        try:
            return handler(**arguments)
        except view_module.ExpressionError as exc:
            raise ValueError(str(exc)) from exc


# --------------------------------------------------------------------------- #
# JSON-RPC plumbing
# --------------------------------------------------------------------------- #

def handle(server: MathKGServer, request: dict) -> dict | None:
    """One JSON-RPC message in, one response out (or None for notifications)."""
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
            return reply(
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                }
            )
        if method == "tools/list":
            return reply({"tools": TOOLS})
        if method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name", "")
            arguments = params.get("arguments") or {}
            result = server.call(name, arguments)
            return reply(
                {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                    "isError": False,
                }
            )
        return error(-32601, f"unknown method {method!r}")
    except Exception as exc:  # surfaced to the client, never swallowed
        return error(-32603, str(exc))


def main() -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    server = MathKGServer()
    log.warning(
        "mathkg MCP ready: %d nodes, %d edges",
        server.state.graph.number_of_nodes(),
        server.state.graph.number_of_edges(),
    )
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
        response = handle(server, request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
