"""Render a graph from a declarative spec.

Separated from :mod:`mathkg.view` so the spec/evaluator can be tested without
matplotlib, and so the live server can reuse the same drawing code.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import view as view_module  # noqa: E402
from .render import (  # noqa: E402
    BACKGROUND,
    FOREGROUND,
    _draw_labels,
    forceatlas2_positions,
    largest_component,
    layout_graph,
)

log = logging.getLogger(__name__)

NODE_COLUMNS = {"degree", "sitelinks", "kind", "is_anchor"}
EDGE_COLUMNS = {
    "confidence", "popularity", "occurrences", "relative_position", "in_lead", "in_proof",
}


def apply_filter(
    graph: nx.Graph, spec: dict, rows: dict[str, dict]
) -> nx.Graph:
    """Apply the spec's filter block. Reported, never silent."""
    filters = spec.get("filter") or {}
    working = graph

    ego = filters.get("ego")
    if ego and ego.get("node") in working:
        working = nx.ego_graph(
            working, ego["node"], radius=int(ego.get("radius", 2)), undirected=True
        )

    kinds = set(filters.get("kinds") or [])
    if kinds:
        keep = {node for node, row in rows.items() if row["kind"] in kinds}
        working = working.subgraph(set(working.nodes) & keep).copy()

    min_degree = int(filters.get("min_degree") or 0)
    if min_degree > 0:
        working = working.subgraph(
            [node for node, degree in working.degree() if degree >= min_degree]
        ).copy()

    if filters.get("component") == "largest":
        working = largest_component(working)

    log.info(
        "filter: %d -> %d nodes, %d edges",
        graph.number_of_nodes(),
        working.number_of_nodes(),
        working.number_of_edges(),
    )
    return working


def render_spec(
    graph: nx.Graph,
    spec: dict,
    out_path: Path,
    mentions: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (16.0, 11.0),
    dpi: int = 110,
) -> Path:
    """Draw ``graph`` according to ``spec`` and write a PNG."""
    node_columns = view_module.node_rows(graph)
    working = apply_filter(graph, spec, node_columns)
    node_columns = {node: row for node, row in node_columns.items() if node in working}

    if spec.get("layout") in {"installed", "gephi"}:
        from .render import _installed_path, _read_positions_file

        path = _installed_path(working)
        positions = _read_positions_file(path) if path.exists() else None
        if positions:
            log.info("layout: installed (%s)", path.name)
        else:
            log.warning("layout=installed but nothing is installed; using forceatlas2")
            positions = forceatlas2_positions(working)
    elif spec.get("layout") == "physics":
        # A PNG needs coordinates, so a physics view is rendered from the
        # force-directed layout rather than left blank.
        log.info("layout: physics requested for a static render; using forceatlas2")
        positions = forceatlas2_positions(working)
    elif spec.get("layout") == "layered":
        from .dependency import break_cycles, layered_positions

        acyclic, _ = break_cycles(nx.DiGraph(working))
        positions = layered_positions(acyclic, x_spacing=1.0, y_spacing=0.06)
        # nodes dropped by cycle breaking still need somewhere to sit
        missing = [node for node in working if node not in positions]
        for index, node in enumerate(missing):
            positions[node] = (0.0, -index * 0.06)
    elif spec.get("layout") == "forceatlas2":
        positions = forceatlas2_positions(working)
    else:
        positions = layout_graph(working)

    edge_columns = view_module.edge_rows(working, mentions)

    node_spec = spec.get("node") or {}
    edge_spec = spec.get("edge") or {}
    label_spec = spec.get("label") or {}
    weight_spec = spec.get("edge_weight") or {}

    # --- edge weight ------------------------------------------------------- #
    formula = weight_spec.get("formula") or "confidence"
    weight_expr = view_module.compile_expression(formula, EDGE_COLUMNS)
    minimum = float(weight_spec.get("minimum") or 0.0)
    weights = {
        key: max(0.0, float(weight_expr(row)))
        for key, row in edge_columns.items()
        if not isinstance(row.get("kind"), str)
    }
    kept_edges = [
        (source, target)
        for source, target in working.edges()
        if weights.get((source, target), 0.0) >= minimum
    ]
    peak = max((weights.get(edge, 0.0) for edge in kept_edges), default=0.0) or 1.0
    log.info("edge weight: %d/%d edges kept above %.2f", len(kept_edges), working.number_of_edges(), minimum)

    # --- draw -------------------------------------------------------------- #
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    base_width = float(view_module.evaluate(edge_spec.get("width", 0.25), {}, EDGE_COLUMNS) or 0.25)
    base_alpha = float(edge_spec.get("opacity", 0.1) or 0.1)
    segments, widths, alphas = [], [], []
    for source, target in kept_edges:
        if source not in positions or target not in positions:
            continue
        relative = weights.get((source, target), 0.0) / peak
        segments.append((positions[source], positions[target]))
        widths.append(base_width * (0.3 + 1.7 * relative))
        alphas.append(min(1.0, base_alpha * (0.4 + 2.6 * relative)))
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors=edge_spec.get("color", "#8899aa"),
                linewidths=widths,
                zorder=1,
            )
        )
        # per-edge alpha needs one collection per alpha bucket; approximate with
        # the mean so a 4k-edge draw stays cheap
        ax.collections[-1].set_alpha(sum(alphas) / len(alphas))

    size_expr = view_module.compile_expression(str(node_spec.get("size", 8)), NODE_COLUMNS)
    color_value = node_spec.get("color", "#4E79A7")
    if view_module.is_expression(color_value):
        raise view_module.ExpressionError(
            "node.color must be a literal hex string for now; use kinds for grouping"
        )
    opacity = float(node_spec.get("opacity", 0.9) or 0.9)

    nodes = [node for node in working if node in positions]
    xs = [positions[node][0] for node in nodes]
    ys = [positions[node][1] for node in nodes]
    sizes = [max(1.0, float(size_expr(node_columns[node]))) for node in nodes]
    ax.scatter(
        xs, ys, s=sizes, c=color_value, linewidths=0, alpha=opacity, zorder=2
    )

    _draw_labels(
        ax,
        working,
        positions,
        int(label_spec.get("count", 40)),
        float(label_spec.get("min_separation", 0.11)),
    )

    ax.set_title(
        str(spec.get("title") or spec.get("name") or out_path.stem),
        color=FOREGROUND,
        fontsize=14,
        pad=14,
        loc="left",
    )
    subtitle = str(spec.get("subtitle") or "")
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

    legend = spec.get("legend") or []
    if legend:
        ax.legend(
            handles=[Patch(facecolor=colour, label=label) for label, colour in legend],
            loc="upper left",
            bbox_to_anchor=(0.0, -0.01),
            ncols=min(6, len(legend)),
            frameon=False,
            labelcolor=FOREGROUND,
            fontsize=8,
        )

    ax.set_axis_off()
    figure.tight_layout()
    figure.savefig(out_path, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)
    log.info("render: %s", out_path)
    return out_path
