"""Static rendering: PNG images that a human *or an agent* can actually look at.

The interactive `graph.html` is great for you but useless for me -- I cannot run
a browser here (`firefox` is a snap and snap-confine has no `cap_dac_override`
in this environment, so `--headless --screenshot` fails outright). A PNG is the
only channel by which I can see the graph and argue about it with you, which is
the whole point of this stage.

Everything is deterministic: the layout is seeded and cached, the figure size is
fixed, so two renders of the same graph are pixel-comparable and a diff between
two runs means something.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from . import config

# Must be set before matplotlib is imported; the default ~/.config path is not
# writable in this environment and matplotlib then falls back to a slow temp dir
# on every process start.
os.environ.setdefault("MPLCONFIGDIR", str(config.DATA_DIR / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

log = logging.getLogger(__name__)

BACKGROUND = "#101018"
FOREGROUND = "#f0f0f0"

# One marker per node kind, so a reader can tell a theorem from a definition
# without relying on colour (which is already carrying the field).
TYPE_MARKERS = {
    "theorem": "o",
    "definition": "s",
    "concept": "^",
    "notion": "D",
    "person": "x",
    "other": ".",
}

LAYOUT_ITERATIONS = 50
LAYOUT_CACHE = config.CACHE_DIR / "layouts"


# --------------------------------------------------------------------------- #
# Layout (seeded + cached)
# --------------------------------------------------------------------------- #

def _graph_fingerprint(graph: nx.Graph, iterations: int, seed: int) -> str:
    digest = hashlib.sha1()
    digest.update(f"{iterations}|{seed}|".encode())
    for node in sorted(graph.nodes):
        digest.update(str(node).encode())
    digest.update(b"|")
    for edge in sorted(graph.edges):
        digest.update(f"{edge[0]}\x00{edge[1]}".encode())
    return digest.hexdigest()[:16]


def layout_graph(
    graph: nx.Graph,
    seed: int = config.LAYOUT_SEED,
    iterations: int = LAYOUT_ITERATIONS,
    use_cache: bool = True,
) -> dict[str, tuple[float, float]]:
    """Seeded spring layout, cached on disk.

    Full scale is ~75 s for 5,968 nodes / 55,796 edges (sparse Früchterman-
    Reingold needs scipy); caching makes the second render instantaneous, which
    is what makes the iterate-and-look loop bearable.
    """
    LAYOUT_CACHE.mkdir(parents=True, exist_ok=True)
    key = _graph_fingerprint(graph, iterations, seed)
    cache_file = LAYOUT_CACHE / f"{key}.json"

    if use_cache and cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        log.info("layout: cache hit (%s)", cache_file.name)
        return {node: (float(x), float(y)) for node, (x, y) in raw.items()}

    log.info(
        "layout: computing for %d nodes / %d edges (this takes ~1 min at full scale)",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    positions = nx.spring_layout(graph, seed=seed, iterations=iterations)
    if use_cache:
        cache_file.write_text(
            json.dumps({node: [float(x), float(y)] for node, (x, y) in positions.items()}),
            encoding="utf-8",
        )
    return {node: (float(x), float(y)) for node, (x, y) in positions.items()}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _node_sizes(graph: nx.Graph, scale: float = 0.35) -> dict[str, float]:
    """Scatter area per node, from the `size` attribute (log of sitelinks)."""
    return {
        node: max(4.0, (float(data.get("size", 10.0)) ** 2) * scale)
        for node, data in graph.nodes(data=True)
    }


def _draw_edges(ax, graph: nx.Graph, positions, alpha: float = 0.10) -> None:
    segments = [
        (positions[source], positions[target])
        for source, target in graph.edges()
        if source in positions and target in positions
    ]
    if not segments:
        return
    ax.add_collection(
        LineCollection(segments, colors="#8899aa", linewidths=0.25, alpha=alpha, zorder=1)
    )


def _draw_nodes(ax, graph: nx.Graph, positions, type_attr: str | None) -> None:
    nodes = [node for node in graph.nodes if node in positions]
    if not nodes:
        return
    xs = [positions[node][0] for node in nodes]
    ys = [positions[node][1] for node in nodes]
    colors = [graph.nodes[node].get("color", config.UNREACHABLE_COLOR) for node in nodes]
    size_by_node = _node_sizes(graph)
    sizes = [size_by_node[node] for node in nodes]

    if not type_attr:
        ax.scatter(xs, ys, s=sizes, c=colors, linewidths=0, alpha=0.9, zorder=2)
        return

    by_type: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        kind = str(graph.nodes[node].get(type_attr) or "other")
        by_type.setdefault(kind, []).append(index)

    for kind, indices in sorted(by_type.items()):
        marker = TYPE_MARKERS.get(kind, ".")
        ax.scatter(
            [xs[i] for i in indices],
            [ys[i] for i in indices],
            s=[sizes[i] for i in indices],
            c=[colors[i] for i in indices],
            marker=marker,
            linewidths=0.4,
            edgecolors=BACKGROUND,
            alpha=0.92,
            zorder=2,
            label=kind,
        )


def _draw_labels(
    ax, graph: nx.Graph, positions, top_n: int, min_separation: float = 0.11
) -> None:
    """Label the top-degree nodes, but only where labels will not collide.

    Naive top-N labelling is useless on this graph: the hubs all sit in the same
    dense core, so 40 labels land on top of each other and the figure becomes
    unreadable. We instead walk the ranking and keep a label only if it is far
    enough (in a normalised coordinate frame) from every label already placed.
    """
    if top_n <= 0 or not positions:
        return

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0

    def unit(node: str) -> tuple[float, float]:
        x, y = positions[node]
        return ((x - min(xs)) / span_x, (y - min(ys)) / span_y)

    ranked = sorted(graph.degree, key=lambda pair: (-pair[1], str(pair[0])))
    placed: list[tuple[float, float]] = []

    for node, _degree in ranked:
        if node not in positions:
            continue
        ux, uy = unit(node)
        if any((ux - px) ** 2 + (uy - py) ** 2 < min_separation ** 2 for px, py in placed):
            continue
        placed.append((ux, uy))

        x, y = positions[node]
        label = str(graph.nodes[node].get("label", node))
        if len(label) > 34:
            label = label[:32] + "…"
        text = ax.annotate(
            label,
            (x, y),
            fontsize=7.0,
            color=FOREGROUND,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            zorder=3,
        )
        text.set_path_effects(
            [path_effects.withStroke(linewidth=2.2, foreground=BACKGROUND)]
        )


def largest_component(graph: nx.Graph) -> nx.Graph:
    """The biggest connected component, as an induced subgraph."""
    undirected = graph.to_undirected()
    if undirected.number_of_nodes() == 0:
        return graph
    biggest = max(nx.connected_components(undirected), key=len)
    return graph.subgraph(biggest).copy()


def render_graph(
    graph: nx.Graph,
    out_path: Path,
    title: str,
    subtitle: str = "",
    legend: list[tuple[str, str]] | None = None,
    type_attr: str | None = None,
    label_top: int = 40,
    only_largest_component: bool = False,
    figsize: tuple[float, float] = (15.0, 10.5),
    dpi: int = 110,
    seed: int = config.LAYOUT_SEED,
    positions: dict | None = None,
) -> Path:
    """Render a graph to PNG. Returns the path written.

    ``positions`` lets a caller supply a layout -- the prerequisite DAG uses
    layered positions, which a force layout would destroy.
    """
    if only_largest_component:
        graph = largest_component(graph)

    if positions is None:
        positions = layout_graph(graph, seed=seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    figure, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    _draw_edges(ax, graph, positions)
    _draw_nodes(ax, graph, positions, type_attr)
    _draw_labels(ax, graph, positions, label_top)

    ax.set_title(title, color=FOREGROUND, fontsize=14, pad=14, loc="left")
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0.0, 1.005),
            xycoords="axes fraction",
            color="#9aa4b2",
            fontsize=9,
            va="bottom",
            ha="left",
        )

    handles: list = []
    if legend:
        handles += [Patch(facecolor=colour, label=label) for label, colour in legend]
    if type_attr:
        for kind in sorted({str(d.get(type_attr) or "other") for _, d in graph.nodes(data=True)}):
            handles.append(
                Line2D(
                    [0], [0],
                    marker=TYPE_MARKERS.get(kind, "."),
                    color="none",
                    markerfacecolor=FOREGROUND,
                    markeredgecolor=FOREGROUND,
                    markersize=7,
                    label=kind,
                )
            )
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.01),
            ncols=min(6, len(handles)),
            frameon=False,
            labelcolor=FOREGROUND,
            fontsize=8,
        )

    ax.set_axis_off()
    figure.tight_layout()
    figure.savefig(out_path, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)
    log.info("render: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path
