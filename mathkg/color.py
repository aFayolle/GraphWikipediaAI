"""Anchor-based colouring.

The idea from the original project, kept and repaired: instead of running
Louvain and showing opaque community ids, place the 12 "fields of mathematics"
as anchors and colour every node by its distance to them. A node's colour then
means "2 hops from number theory, 3 from topology", which is explainable.

What was broken in ``BDD.color_graph``:

* ``sorted(dist)[-3]`` took the three *furthest* anchors, contradicting its own
  comment ("les trois plus proches"). Fixed to the nearest ``k``.
* ``sorted()`` and ``>=`` raise ``TypeError`` as soon as one anchor is
  unreachable (``None``). Unreachable anchors are now skipped, and a node with
  no reachable anchor is greyed out instead of crashing the run.
* ``(1 - poids) / norm(poids)`` divided by the norm of the *raw* integer vector
  and produced negative weights, hence out-of-range RGB. Weights are now
  positive, normalised, and the result is clipped to ``[0, 1]``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import networkx as nx

from . import config

log = logging.getLogger(__name__)

RGB = tuple[float, float, float]


def hex_to_rgb(value: str) -> RGB:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"not a 6-digit hex colour: {value!r}")
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: Iterable[float]) -> str:
    channels = []
    for channel in rgb:
        clipped = min(1.0, max(0.0, float(channel)))
        channels.append(round(clipped * 255))
    return "#{:02X}{:02X}{:02X}".format(*channels)


def anchor_distances(
    graph: nx.Graph, anchors: Iterable[str]
) -> tuple[list[str], dict[str, list[int | None]]]:
    """BFS distance from every anchor to every node.

    Returns ``(anchor_order, {node: [d_anchor0, d_anchor1, ...]})`` where each
    entry is ``None`` when the node is unreachable from that anchor.
    """
    order = [anchor for anchor in anchors if anchor in graph]
    missing = [anchor for anchor in anchors if anchor not in graph]
    if missing:
        log.warning("anchors absent from the graph: %s", ", ".join(sorted(missing)))

    undirected = graph.to_undirected() if graph.is_directed() else graph
    per_anchor = {
        anchor: nx.single_source_shortest_path_length(undirected, anchor)
        for anchor in order
    }

    distances: dict[str, list[int | None]] = {
        node: [per_anchor[anchor].get(node) for anchor in order] for node in undirected
    }
    return order, distances


def blend_color(
    distance_vector: list[int | None],
    palette: list[str] = config.ANCHOR_PALETTE,
    k: int = config.ANCHOR_NEIGHBOURS,
    unreachable: str = config.UNREACHABLE_COLOR,
) -> str:
    """Colour one node from its distances to the anchors.

    Weighting is inverse-*square*, not the original ``1/(1+d)``: with twelve
    anchors a node is typically one hop from several of them, so a linear decay
    blended everything into the same mud (the shipped figure had 1,561 nodes
    sharing one brown). The decay is also what the original comment claimed
    ("décroit exponentiellement").

    A node at distance 0 from an anchor is an anchor: it gets that anchor's pure
    palette colour, so the twelve reference points stay recognisable instead of
    averaging into their own neighbours.
    """
    candidates = [
        (distance, index)
        for index, distance in enumerate(distance_vector)
        if distance is not None
    ]
    if not candidates:
        return unreachable

    # Nearest k, ties broken by anchor order so results are reproducible.
    candidates.sort(key=lambda item: (item[0], item[1]))
    nearest = candidates[:k]

    if nearest[0][0] == 0:
        return palette[nearest[0][1] % len(palette)]

    weights = [1.0 / (1.0 + distance) ** 2 for distance, _ in nearest]
    total = sum(weights)
    if total <= 0:
        return unreachable

    rgb = [0.0, 0.0, 0.0]
    for weight, (_, index) in zip(weights, nearest, strict=False):
        anchor_rgb = hex_to_rgb(palette[index % len(palette)])
        for channel in range(3):
            rgb[channel] += (weight / total) * anchor_rgb[channel]
    return rgb_to_hex(rgb)


def color_graph(
    graph: nx.Graph,
    anchors: Iterable[str],
    palette: list[str] = config.ANCHOR_PALETTE,
    k: int = config.ANCHOR_NEIGHBOURS,
) -> tuple[list[str], dict[str, list[int | None]]]:
    """Attach ``color`` and ``distance_vector`` to every node; return both."""
    order, distances = anchor_distances(graph, anchors)
    if not order:
        log.warning("no anchor is present in the graph: every node will be grey")
        for node in graph:
            graph.nodes[node]["color"] = config.UNREACHABLE_COLOR
            graph.nodes[node]["distance_vector"] = []
            graph.nodes[node]["field"] = None
        return order, distances

    for node, vector in distances.items():
        graph.nodes[node]["distance_vector"] = vector
        graph.nodes[node]["color"] = blend_color(vector, palette=palette, k=k)
        # Nearest anchor, so the exports can group/label nodes meaningfully.
        graph.nodes[node]["field"] = dominant_anchor(vector, order)

    colored = sum(1 for node in graph if graph.nodes[node]["color"] != config.UNREACHABLE_COLOR)
    log.info(
        "colouring: %d/%d nodes coloured, %d unreachable from every anchor (%s)",
        colored,
        graph.number_of_nodes(),
        graph.number_of_nodes() - colored,
        ", ".join(order),
    )
    return order, distances


def dominant_anchor(
    distance_vector: list[int | None], order: list[str]
) -> str | None:
    """Label a node with its nearest anchor (used in exports and tests)."""
    best: tuple[int, int] | None = None
    for index, distance in enumerate(distance_vector[: len(order)]):
        if distance is None:
            continue
        if best is None or (distance, index) < best:
            best = (distance, index)
    return order[best[1]] if best else None
