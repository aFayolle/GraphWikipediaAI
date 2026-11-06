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
VIEWER3D_HTML = Path(__file__).with_name("viewer3d.html")
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


def build_dependency_graph() -> nx.DiGraph:
    """Rebuild the prerequisite DAG from disk, with the LLM field attached.

    Used by ``POST /reload`` so the running viewer can refresh after the edges
    stage re-runs, without going through run.py.
    """
    from . import config, dependency, edges as edges_module

    frame = dependency.load_edges()
    _, titles = edges_module.load_summaries()
    dag = dependency.build_dag(frame, titles=set(titles))
    dag, _ = dependency.break_cycles(dag)
    dag = dependency.enrich_with_attributes(dag)
    dag = dependency.connect_components(dag)
    field_csv = config.PROJECT_ROOT / "reports" / "llm_fields.csv"
    if field_csv.exists():
        fields = pd.read_csv(field_csv, dtype=str, keep_default_na=False)
        field = dict(zip(fields["title"], fields["llm_field"], strict=False))
        for node in dag.nodes:
            dag.nodes[node]["field"] = field.get(node, "") or "unknown"
    return dag


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

        if layout == "circular":
            return self._scale(nx.circular_layout(self.graph))

        if layout == "radial":
            return self._scale(nx.shell_layout(self.graph))

        if layout == "fields":
            return self._scale(self._grouped_positions())

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

    def _grouped_positions(self) -> dict[str, tuple[float, float]]:
        """Cluster by field: each domain gets a region on a circle, its nodes
        spread inside it. Different-coloured nodes therefore end up apart (the
        'repulsion between different colours' request)."""
        import math as _math

        fields: dict[str, list[str]] = {}
        for node in self.graph.nodes:
            field = str(self.graph.nodes[node].get("field", "")) or "unknown"
            fields.setdefault(field, []).append(node)

        positions: dict[str, tuple[float, float]] = {}
        count = len(fields)
        for index, (field, nodes) in enumerate(sorted(fields.items())):
            angle = 2 * _math.pi * index / count
            cx, cy = 8.0 * _math.cos(angle), 8.0 * _math.sin(angle)
            if len(nodes) == 1:
                positions[nodes[0]] = (cx, cy)
                continue
            sub = self.graph.subgraph(nodes)
            try:
                local = nx.spring_layout(sub, seed=0, scale=0.6, iterations=40)
            except Exception:
                local = {node: (0.0, 0.0) for node in nodes}
            for node, (x, y) in local.items():
                positions[node] = (cx + x, cy + y)
        return positions

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

    def _highlight_field(self) -> str | None:
        value = (self.spec.get("highlight") or {}).get("field")
        return str(value) if value else None

    def _is_highlighted(self, node: str) -> bool:
        field = self._highlight_field()
        if not field:
            return True  # no highlight -> everything is "highlighted"
        return str(self._node_columns.get(node, {}).get("field", "")) == field

    def _size_for(self, node: str) -> float:
        expression = self.spec.get("node", {}).get("size", 8)
        try:
            value = view_module.evaluate(expression, self._node_columns[node], view_module.NODE_COLUMNS)
        except view_module.ExpressionError as exc:
            log.warning("bad node.size expression: %s", exc)
            value = 8
        # No sqrt here: the expression already yields a radius. Applying a second
        # sqrt turned "9 + 11*sqrt(degree)" into sqrt(9 + 11*sqrt(degree)), so a
        # degree-40 hub rendered at ~6 px instead of ~78 px.
        size = max(2.0, float(value))
        if not self._is_highlighted(node):
            size *= 0.45
        return size

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

    def _legend(self, colours: dict[str, str]) -> list[dict[str, Any]]:
        """field -> colour -> count, for the categorical colour legend."""
        rule = (self.spec.get("node") or {}).get("color_by") or {}
        if rule.get("mode") != "categorical":
            return []
        column = rule.get("column", "field")
        colour_by_field: dict[str, str] = {}
        counts: dict[str, int] = {}
        for node in self.graph.nodes:
            value = str(self._node_columns.get(node, {}).get(column, "") or "unknown")
            colour_by_field.setdefault(value, colours.get(node, "#4E79A7"))
            counts[value] = counts.get(value, 0) + 1
        return [
            {"name": name, "color": colour_by_field[name], "count": counts[name]}
            for name in sorted(counts)
        ]

    def style_payload(self, sent: float | None = None) -> dict[str, Any]:
        """Node/edge styling only -- cheap enough to send on every change."""
        colours = view_module.node_colors(self.graph, self.spec, self._node_columns)
        if self._highlight_field():
            for node in self.graph.nodes:
                if not self._is_highlighted(node):
                    colours[node] = "#33415c"
        labelled = self._labelled()
        edges = [
            {"id": f"{source}\u0000{target}", "width": self._edge_width(source, target),
             "color": self._edge_color(), "opacity": self._edge_opacity()}
            for source, target in self.graph.edges
        ]
        # Edge style only changes when the weight formula or colour does, so omit
        # them otherwise: that halves the message for a pure node-size change.
        fingerprint = json.dumps([(edge["width"], edge["color"], edge["opacity"]) for edge in edges])
        changed_edges = edges if fingerprint != self._last_edge_style else []
        self._last_edge_style = fingerprint

        return {
            "type": "style",
            "version": self.version,
            "sent": sent or time.time() * 1000,
            "spec": self.spec,
            "detail": self.detail(),
            "legend": self._legend(colours),
            "physics_params": self._physics_params(),
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

    def _edge_color(self) -> str:
        """Edge colour from the spec (hex). Sent to the viewer for every edge."""
        return str((self.spec.get("edge") or {}).get("color", "") or "")

    def _edge_opacity(self) -> float:
        return float((self.spec.get("edge") or {}).get("opacity", 0.10) or 0.10)

    def _physics_params(self) -> dict[str, float]:
        p = self.spec.get("physics") or {}
        return {
            "repulsion": float(p.get("repulsion", 120)),
            "gravity": float(p.get("gravity", 0.05)),
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
        if self._highlight_field():
            for node in self.graph.nodes:
                if not self._is_highlighted(node):
                    colours[node] = "#33415c"
        labelled = self._labelled()
        nodes = []
        for node, data in self.graph.nodes(data=True):
            x, y = self._positions.get(node, (None, None))
            nodes.append(
                {
                    "id": node,
                    "label": (str(data.get("label", node))[:48]
                              if node in labelled else ""),
                    "title": f"{node} — {data.get('field', '')} · degree {self.graph.degree(node)}, sitelinks {int(data.get('sitelinks', 0))}",
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
                "color": self._edge_color(),
                "opacity": self._edge_opacity(),
                "evidence": str(data.get("evidence", "")),
            }
            for source, target, data in self.graph.edges(data=True)
        ]
        return {
            "version": self.structure_version,
            "spec": self.spec,
            "detail": self.detail(),
            "legend": self._legend(colours),
            "physics_params": self._physics_params(),
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
        if path == "/3d":
            self._send(200, VIEWER3D_HTML.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/graph":
            self._send_json(self.state.graph_payload())
            return
        if path == "/state":
            self._send_json({"version": self.state.version, "spec": self.state.spec})
            return
        if path == "/health":
            self._send_json({
                "ok": True,
                "nodes": self.state.graph.number_of_nodes(),
                "edges": self.state.graph.number_of_edges(),
                "full_nodes": self.state._full_graph.number_of_nodes(),
            })
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

        if path == "/reload":
            self._handle_reload()
            return

        if path == "/render_png":
            self._handle_render_png(body)
            return

        if path == "/compute":
            self._handle_compute(body)
            return

        if path == "/select":
            self._handle_select(body)
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

    def _handle_reload(self) -> None:
        try:
            new_graph = build_dependency_graph()
        except Exception as exc:
            log.error("reload failed: %s", exc)
            self._send_json({"error": str(exc)}, 500)
            return
        self.state._full_graph = new_graph
        self.state.replace_graph(new_graph.copy())
        self._send_json({
            "ok": True,
            "nodes": new_graph.number_of_nodes(),
            "edges": new_graph.number_of_edges(),
        })

    def _handle_render_png(self, body: dict) -> None:
        from . import config, view_render

        name = str(body.get("name", "mcp_render"))
        safe = "".join(c for c in name if c.isalnum() or c in "._-") or "mcp_render"
        figsize = float(body.get("figsize", 16))
        out = config.ARTIFACT_DIR / "mcp" / f"{safe}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            view_render.render_spec(
                self.state.graph, self.state.spec, out,
                mentions=self.state.mentions, figsize=(figsize, figsize * 0.7),
            )
        except Exception as exc:
            log.error("render_png failed: %s", exc)
            self._send_json({"error": str(exc)}, 500)
            return
        self._send_json({"path": str(out), "bytes": out.stat().st_size})

    def _refresh_style(self) -> None:
        """Recompute node columns and push a style delta (after attaching metrics)."""
        self.state._node_columns = view_module.node_rows(self.state.graph)
        self.state.broadcast(self.state.style_payload())

    def _handle_compute(self, body: dict) -> None:
        from . import analysis

        kind = str(body.get("kind", "")).strip().lower()
        full = self.state._full_graph

        if kind in ("communities", "community"):
            partition, modularity = analysis.communities(
                full, resolution=float(body.get("resolution", 1.0)))
            for node, cid in partition.items():
                full.nodes[node]["community"] = cid
                if node in self.state.graph:
                    self.state.graph.nodes[node]["community"] = cid
            sizes: dict[int, int] = {}
            for cid in partition.values():
                sizes[cid] = sizes.get(cid, 0) + 1
            self._refresh_style()
            self._send_json({
                "ok": True, "kind": kind, "modularity": modularity,
                "communities": len(sizes),
                "largest": sorted(sizes.values(), reverse=True)[:10],
            })
            return

        if kind not in analysis.CENTRALITIES:
            self._send_json({"error": f"unknown compute {kind!r}"}, 400)
            return
        values = analysis.centrality(full, kind)
        for node, value in values.items():
            full.nodes[node][kind] = value
            if node in self.state.graph:
                self.state.graph.nodes[node][kind] = value
        self._refresh_style()
        self._send_json({
            "ok": True, "kind": kind,
            "top": analysis.top_k(values, int(body.get("topk", 10))),
            "count": len(values),
        })

    def _handle_select(self, body: dict) -> None:
        action = str(body.get("action", "neighbors")).strip().lower()
        node = str(body.get("node", ""))
        radius = int(body.get("radius", 1))
        full = self.state._full_graph
        if action in ("neighbors", "highlight"):
            if node not in full:
                self._send_json({"error": f"{node!r} is not in the graph"}, 400)
                return
            subgraph = nx.ego_graph(full, node, radius=radius, undirected=True)
            neighbors = sorted(n for n in subgraph.nodes if n != node)
            self._send_json({"node": node, "neighbors": neighbors, "count": len(neighbors)})
            return
        self._send_json({"error": f"unknown select {action!r}"}, 400)

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
