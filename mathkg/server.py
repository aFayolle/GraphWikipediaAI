"""A tiny live-view server: holds the graph in memory, pushes style deltas.

Why a server at all: rebuilding the graph and the layout costs seconds, but a
*style* change costs a redraw. So the process keeps the graph, the layout and
the spec resident, and a spec patch becomes a small JSON message over SSE that
the open browser tab applies in place -- no reload, no re-fetch, no re-layout.

Stdlib only (`http.server` + SSE), because a live viewer should not need a web
framework to exist. `vis-network` is served straight out of the installed
`pyvis` package, so nothing has to be vendored or fetched from a CDN.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from . import view as view_module

log = logging.getLogger(__name__)

VIEWER_HTML = Path(__file__).with_name("viewer.html")
STATIC_FILES = {
    "/lib/vis-network.min.js": ("vis-9.1.2/vis-network.min.js", "application/javascript"),
    "/lib/vis-network.css": ("vis-9.1.2/vis-network.css", "text/css"),
}


def _pyvis_lib(relative: str) -> Path | None:
    """Locate a vis.js file inside the installed pyvis package."""
    try:
        import pyvis
    except ImportError:  # pragma: no cover - pyvis is a hard dependency here
        return None
    candidate = Path(pyvis.__file__).parent / "templates" / "lib" / relative
    return candidate if candidate.exists() else None


class ViewerState:
    """The graph, the spec, and the set of connected clients."""

    def __init__(self, graph: nx.Graph, spec: dict, mentions: pd.DataFrame | None = None):
        self.source_graph = graph      # everything
        self._full_graph = graph       # never narrowed: queries answer against this
        self.graph = graph             # what the spec currently shows
        self.spec = spec
        self.mentions = mentions
        self.version = 0
        self.structure_version = 0
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._last_edge_style: str | None = None
        self._refresh_graph()

    def _refresh_graph(self) -> None:
        """Rebuild the visible graph from source_graph + the spec's filter block.

        The filter used to be honoured by the PNG renderer only, so
        `set --set filter.min_degree=4` changed the spec text and nothing else.
        """
        from .view_render import apply_filter

        rows = view_module.node_rows(self.source_graph)
        self.graph = apply_filter(self.source_graph, self.spec, rows)
        self._trim_stragglers()
        self._node_columns = view_module.node_rows(self.graph)
        self._edge_columns = view_module.edge_rows(self.graph, self.mentions)
        self._positions = self._compute_positions()
        self._last_edge_style: str | None = None

    def _trim_stragglers(self) -> None:
        """Drop far, low-degree nodes.

        They carry no structure, but they sit so far out that fitting the view to
        them shrinks every other node to a dot. Only weakly-connected nodes far
        from the median position are removed, and never more than a fifth of the
        graph, so this cannot quietly eat the content.
        """
        import numpy as np

        pct = (self.spec.get("filter") or {}).get("trim_radius_pct")
        if not pct:
            return
        try:
            raw = self._installed_positions()
        except Exception:
            return
        nodes = [node for node in self.graph.nodes if node in raw]
        if len(nodes) < 50:
            return
        points = np.array([raw[node] for node in nodes], dtype=float)
        centre = np.median(points, axis=0)
        distance = np.linalg.norm(points - centre, axis=1)
        cutoff = float(np.percentile(distance, float(pct)))
        drop = [
            node for node, dist in zip(nodes, distance)
            if dist > cutoff and self.graph.degree(node) <= 3
        ]
        if not drop or len(drop) > 0.2 * len(nodes):
            return
        log.info("trim_radius_pct=%s: dropping %d stragglers", pct, len(drop))
        self.graph.remove_nodes_from(drop)

    # -- layout ------------------------------------------------------------- #

    def _compute_positions(self) -> dict[str, tuple[float, float]]:
        """Positions for the requested layout. The spec value always wins.

        `installed` is a *choice*, not an override: an earlier version let any
        imported layout beat `layered`/`forceatlas2`, so changing the layout
        looked like it did nothing while the log said "using installed layout".
        """
        layout = self.spec.get("layout")

        if layout == "physics":
            log.info("viewer: physics mode (no server positions)")
            return {}

        if layout in {"installed", "gephi"}:
            raw = self._installed_positions()
            if raw:
                return self._scale(raw)
            log.warning("viewer: layout=installed but no layout is installed; using forceatlas2")
            layout = "forceatlas2"

        if layout == "forceatlas2":
            from .render import forceatlas2_positions

            positions = forceatlas2_positions(self.graph)
            # Normalise onto the canvas. Scaling by a constant factor instead
            # produced coordinates in the hundreds of thousands, so every node
            # was sub-pixel and the tab looked empty.
            return self._scale(positions)

        if layout != "layered":
            log.info("viewer: layout=%s has no server positions (physics on)", layout)
            return {}

        from .dependency import break_cycles, layered_positions

        try:
            working, _ = break_cycles(nx.DiGraph(self.graph))
            positions = layered_positions(working)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("layered layout failed (%s); falling back to physics", exc)
            return {}
        return self._scale(positions)

    def _installed_positions(self) -> dict[str, tuple[float, float]]:
        from .render import _installed_path, _read_positions_file

        path = _installed_path(self.graph)
        if not path.exists():
            return {}
        log.info("viewer: using installed layout (%s)", path.name)
        return _read_positions_file(path)

    @staticmethod
    def _scale(
        positions: dict[str, tuple[float, float]], factor: float | None = None
    ) -> dict[str, tuple[float, float]]:
        """Map arbitrary coordinates onto the pixel canvas vis-network expects."""
        import numpy as np

        if factor is not None:
            return {node: (x * factor, y * factor) for node, (x, y) in positions.items()}
        if not positions:
            return {}

        values = np.array([p for p in positions.values()], dtype=float)
        # Robust, not min/max: a single distant outlier used to define the whole
        # span, squeezing every other node until it was sub-pixel.
        lo_x, hi_x = np.percentile(values[:, 0], [1, 99])
        lo_y, hi_y = np.percentile(values[:, 1], [1, 99])
        centre_x, centre_y = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
        span_x = max(hi_x - lo_x, 1e-9)
        span_y = max(hi_y - lo_y, 1e-9)
        scale = min(1600.0 / span_x, 1000.0 / span_y) * 0.92

        out = {}
        for node, (x, y) in positions.items():
            # Outliers are pulled back to the border instead of dragging the view out.
            out[node] = (
                float(np.clip((x - centre_x) * scale, -770.0, 770.0)),
                float(np.clip((y - centre_y) * scale, -470.0, 470.0)),
            )
        return out

    # -- derived attributes -------------------------------------------------- #

    def _size_for(self, node: str) -> float:
        expression = self.spec.get("node", {}).get("size", 8)
        try:
            value = view_module.evaluate(expression, self._node_columns[node], {"degree", "sitelinks", "kind", "is_anchor"})
        except view_module.ExpressionError as exc:
            log.warning("bad node.size expression: %s", exc)
            value = 8
        # No sqrt here: the expression already yields a radius. Applying a second
        # sqrt turned "9 + 11*sqrt(degree)" into sqrt(9 + 11*sqrt(degree)), so a
        # degree-40 hub rendered at ~6 px instead of ~78 px.
        return max(2.0, float(value))

    def _font_colour(self, node: str) -> str:
        """Dim label for leaves, near-white for hubs: a second visual channel."""
        degree = self.graph.degree(node) if node in self.graph else 0
        weight = min(1.0, (degree ** 0.5) / 5.5)
        low, high = (0x7E, 0x94, 0xB2), (0xFF, 0xFF, 0xFF)
        red, green, blue = (
            int(low[i] + (high[i] - low[i]) * weight) for i in range(3)
        )
        return f"#{red:02x}{green:02x}{blue:02x}"

    def _font_for(self, node: str) -> float:
        """Label size tied to the node's own size, so hubs read at a glance.

        A flat 9-12 px was drawn but unreadable: at the fitted zoom a 12 px glyph
        with a thick dark halo collapses into an invisible smudge.
        """
        base = float((self.spec.get("label") or {}).get("size", 9))
        try:
            size = float(self._size_for(node))
        except Exception:
            size = 10.0
        return float(min(max(base, size * 0.60), 46.0))

    def _font_size(self) -> float:
        return float(self.spec.get("label", {}).get("size", 7.0)) * 1.3

    def style_payload(self, sent: float | None = None) -> dict[str, Any]:
        """Node/edge styling only -- cheap enough to send on every change."""
        colours = view_module.node_colors(self.graph, self.spec, self._node_columns)
        labelled = self._labelled()
        edges = [
            {"id": f"{source}\u0000{target}", "width": self._edge_width(source, target)}
            for source, target in self.graph.edges
        ]
        # Edge widths only change when the weight formula does, so omit them
        # otherwise: that halves the message for a pure node-size change.
        fingerprint = json.dumps([edge["width"] for edge in edges])
        changed_edges = edges if fingerprint != self._last_edge_style else []
        self._last_edge_style = fingerprint

        return {
            "type": "style",
            "version": self.version,
            "sent": sent or time.time() * 1000,
            "spec": self.spec,
            "detail": self.detail(),
            "nodes": [
                {"id": node, "size": self._size_for(node), "color": colours.get(node, "#4E79A7"),
                 "font_size": self._font_for(node),
                 "font_color": self._font_colour(node),
                 "label": str(self.graph.nodes[node].get("label", node))[:48]
                          if node in labelled else ""}
                for node in self.graph.nodes
            ],
            "edges": changed_edges,
        }

    def _edge_width(self, source: str, target: str) -> float:
        row = self._edge_columns.get((source, target))
        if row is None:
            return 0.5
        try:
            weight = float(view_module.evaluate(
                self.spec.get("edge_weight", {}).get("formula", "confidence"),
                row,
                {"confidence", "popularity", "occurrences", "relative_position", "in_lead", "in_proof"},
            ))
        except view_module.ExpressionError:
            return 0.5
        return max(0.2, min(6.0, math.log1p(max(0.0, weight)) * 2))

    def graph_payload(self) -> dict[str, Any]:
        colours = view_module.node_colors(self.graph, self.spec, self._node_columns)
        labelled = self._labelled()
        nodes = []
        for node, data in self.graph.nodes(data=True):
            x, y = self._positions.get(node, (None, None))
            nodes.append(
                {
                    "id": node,
                    "label": (str(data.get("label", node))[:48]
                              if node in labelled else ""),
                    "title": f"{node} — degree {self.graph.degree(node)}, sitelinks {int(data.get('sitelinks', 0))}",
                    "x": x,
                    "y": y,
                    "size": self._size_for(node),
                    "color": colours.get(node, "#4E79A7"),
                    "degree": int(self.graph.degree(node)),
                    "font_size": self._font_for(node),
                 "font_color": self._font_colour(node),
                }
            )
        edges = [
            {
                "id": f"{source}\u0000{target}",
                "from": source,
                "to": target,
                "width": self._edge_width(source, target),
                "evidence": str(data.get("evidence", "")),
            }
            for source, target, data in self.graph.edges(data=True)
        ]
        return {
            "version": self.structure_version,
            "spec": self.spec,
            "detail": self.detail(),
            "physics": not self._positions,
            "nodes": nodes,
            "edges": edges,
        }

    def _labelled(self) -> set[str]:
        """Which nodes carry a visible label: the top `label.count` by degree."""
        limit = int((self.spec.get("label") or {}).get("count", 40))
        ranked = sorted(self.graph.degree, key=lambda pair: (-pair[1], str(pair[0])))
        return {node for node, _ in ranked[:limit]}

    def detail(self) -> str:
        """Live parameter readout. Values only -- no rationale, deliberately."""
        node = self.spec.get("node", {})
        label = self.spec.get("label", {})
        weight = self.spec.get("edge_weight", {})
        edge = self.spec.get("edge", {})
        filters = self.spec.get("filter", {})
        colour_rule = node.get("color_by")
        colour = (
            f"by {colour_rule.get('column')}" if colour_rule else str(node.get("color"))
        )
        parts = [
            f"layout={self.spec.get('layout')}",
            f"size={node.get('size')}",
            f"colour={colour}",
            f"edge_w={weight.get('formula')} >= {weight.get('minimum')}",
            f"edge_opacity={edge.get('opacity')}",
            f"labels={label.get('count')}@{label.get('size')}px",
            f"min_degree={filters.get('min_degree')}",
        ]
        return "  ·  ".join(parts)

    # -- broadcasting -------------------------------------------------------- #

    def subscribe(self) -> queue.Queue:
        channel: queue.Queue = queue.Queue(maxsize=8)
        with self._lock:
            self._subscribers.append(channel)
        return channel

    def unsubscribe(self, channel: queue.Queue) -> None:
        with self._lock:
            if channel in self._subscribers:
                self._subscribers.remove(channel)

    def broadcast(self, message: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for channel in subscribers:
            # A slow client misses a frame rather than blocking the server.
            with contextlib.suppress(queue.Full):
                channel.put_nowait(message)

    # -- mutations ----------------------------------------------------------- #

    def patch_spec(self, overrides: list[str]) -> dict[str, Any]:
        """Validate on a copy first, so a bad patch leaves the view untouched."""
        candidate = copy.deepcopy(self.spec)
        previous_layout = self.spec.get("layout")
        previous_filter = self.spec.get("filter")
        view_module.apply_overrides(candidate, overrides)
        view_module.validate_spec(candidate)
        layout_changed = candidate.get("layout") != previous_layout
        filter_changed = candidate.get("filter") != previous_filter
        self.spec = candidate
        self.version += 1

        if layout_changed or filter_changed:
            # Positions have to be recomputed and the client has to re-seat its
            # DataSets: a style delta cannot move nodes.
            self._refresh_graph()
            self.broadcast({"type": "graph"})
            return {"type": "graph", "version": self.version, "detail": self.detail()}

        message = self.style_payload()
        self.broadcast(message)
        return message

    def replace_graph(self, graph: nx.Graph) -> None:
        self.source_graph = graph
        self._refresh_graph()
        self.structure_version += 1
        self.broadcast({"type": "graph"})


class Handler(BaseHTTPRequestHandler):
    state: ViewerState
    server_version = "MathKGViewer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        log.debug("%s - %s", self.address_string(), fmt % args)

    # -- helpers ------------------------------------------------------------- #

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    # -- routes -------------------------------------------------------------- #

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, VIEWER_HTML.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/graph":
            self._send_json(self.state.graph_payload())
            return
        if path == "/state":
            self._send_json({"version": self.state.version, "spec": self.state.spec})
            return
        if path == "/events":
            self._stream_events()
            return
        if path in STATIC_FILES:
            relative, content_type = STATIC_FILES[path]
            found = _pyvis_lib(relative)
            if found is None:
                self._send(404, b"missing vis.js asset", "text/plain")
                return
            self._send(200, found.read_bytes(), content_type)
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?")[0]
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, 400)
            return

        if path == "/animate":
            self._handle_animate(body)
            return

        if path == "/query":
            self._handle_query(body)
            return

        if path == "/spec":
            overrides = body.get("set") or []
            try:
                message = self.state.patch_spec(overrides)
            except (ValueError, view_module.ExpressionError) as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            self._send_json({"ok": True, "version": message["version"], "detail": message["detail"]})
            return
        self._send_json({"error": "not found"}, 404)



    def _handle_animate(self, body: dict) -> None:
        """Run Gephi's ForceAtlas2 and stream the frames to the browser.

        This is the "watch it open" path, inside the same server that serves the
        tab: one graph, one renderer, no separate artifact to diverge.
        """
        from . import animation

        iterations = int(body.get("iterations", 400))
        stride = int(body.get("stride", 40))
        scaling = float(body.get("scaling", 10.0))
        try:
            result = animation.build_frames(
                self.state.graph, iterations=iterations, stride=stride, scaling=scaling
            )
        except Exception as exc:  # surfaced, never swallowed
            log.error("animate failed: %s", exc)
            self._send_json({"error": str(exc)}, 500)
            return

        self.state.broadcast({"type": "positions", **result})
        self._send_json(
            {
                "ok": True,
                "frames": len(result["frames"]),
                "nodes": len(result["nodes"]),
                "scaling": scaling,
                "iterations": iterations,
            }
        )

    def _handle_query(self, body: dict) -> None:
        """ego / path / describe / reset, answered against the *full* graph.

        ``ego`` narrows the *displayed* graph, but the full graph is kept, so a
        later ``describe``/``path`` still works and ``reset`` restores the view.
        """
        from . import dependency

        full = self.state._full_graph
        kind = str(body.get("kind", "")).strip().lower()
        if kind == "describe":
            node = str(body.get("node", ""))
            self._send_json(dependency.describe(full, node))
            return
        if kind == "reset":
            self.state.replace_graph(full.copy())
            self._send_json({"kind": "reset", "nodes": full.number_of_nodes()})
            return
        if kind == "ego":
            node = str(body.get("node", ""))
            radius = int(body.get("radius", 2))
            if node not in full:
                self._send_json({"error": f"{node!r} is not in the graph"}, 400)
                return
            subgraph = nx.ego_graph(full, node, radius=radius, undirected=True)
            distances = nx.single_source_shortest_path_length(
                full.to_undirected(), node, cutoff=radius
            )
            by_hop: dict[int, int] = {}
            for distance in distances.values():
                by_hop[distance] = by_hop.get(distance, 0) + 1
            self.state.replace_graph(subgraph.copy())
            self._send_json({"kind": "ego", "node": node, "hops": by_hop})
            return
        if kind == "path":
            source = str(body.get("from", ""))
            target = str(body.get("to", ""))
            chain, how = dependency.path_between(full, source, target)
            if chain is None:
                self._send_json(
                    {"kind": "path", "chain": None, "how": "none",
                     "error": f"no path from {source!r} to {target!r}"},
                    404,
                )
                return
            self._send_json({"kind": "path", "chain": chain, "how": how, "steps": len(chain)})
            return
        self._send_json({"error": f"unknown query {kind!r}"}, 400)

    def _stream_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        channel = self.state.subscribe()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                try:
                    message = channel.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                payload = json.dumps(message)
                self.wfile.write(f"event: {message['type']}\ndata: {payload}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.state.unsubscribe(channel)


def make_server(graph: nx.Graph, spec: dict, mentions: pd.DataFrame | None, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"state": ViewerState(graph, spec, mentions)})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def serve(graph: nx.Graph, spec: dict, mentions: pd.DataFrame | None, port: int = 8765) -> None:
    server = make_server(graph, spec, mentions, port)
    log.info("live view: http://127.0.0.1:%d  (Ctrl-C to stop)", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        server.server_close()
