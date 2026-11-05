"""Graph analytics exposed to the MCP: centralities and community detection.

Kept separate from the viewer so the expensive calls are one import away and
the results can be attached as node attributes that the spec system already
understands (``node.size=…pagerank…``, ``color_by.column=betweenness``).
"""

from __future__ import annotations

from typing import Any

import networkx as nx

CENTRALITIES = ("betweenness", "closeness", "eigenvector", "pagerank")


def centrality(graph: nx.Graph, kind: str, k: int = 500, **_: Any) -> dict[str, float]:
    """One centrality per node. ``k`` samples the expensive betweenness run.

    Returns ``{node: float}``. Values are 0..1 for every metric; eigenvector
    falls back to PageRank (then degree) when the matrix iteration fails, which
    happens on the directed DAG's dangling nodes.
    """
    kind = (kind or "").strip().lower()
    if kind not in CENTRALITIES:
        raise ValueError(f"unknown centrality {kind!r}; expected one of {CENTRALITIES}")

    if kind == "betweenness":
        return nx.betweenness_centrality(graph, k=k, seed=0)
    if kind == "closeness":
        return nx.closeness_centrality(graph)
    if kind == "pagerank":
        return nx.pagerank(graph, max_iter=200, tol=1e-6)

    # eigenvector: the directed DAG makes the power iteration fragile, so fall
    # back through progressively more tolerant options rather than failing.
    try:
        return nx.eigenvector_centrality_numpy(graph)
    except Exception:
        pass
    try:
        return nx.eigenvector_centrality(graph, max_iter=1000)
    except Exception:
        pass
    return nx.pagerank(graph, max_iter=200, tol=1e-6)


def communities(
    graph: nx.Graph, resolution: float = 1.0, seed: int = 0, **_: Any
) -> tuple[dict[str, int], float]:
    """Louvain partition + modularity. Returns ``({node: int}, modularity)``."""
    g = graph.to_undirected()
    try:
        parts = nx.algorithms.community.louvain_communities(
            g, resolution=float(resolution), seed=int(seed)
        )
        modularity = nx.algorithms.community.modularity(g, parts)
    except Exception:
        # Any partition is better than none for colouring; fall back to a single
        # community so the tool still returns a usable result.
        return {node: 0 for node in g.nodes}, float("nan")

    partition: dict[str, int] = {}
    for community_id, members in enumerate(parts):
        for node in members:
            partition[node] = community_id
    return partition, modularity


def top_k(values: dict[str, float], n: int = 10) -> list[list[Any]]:
    """``[[node, value], ...]`` descending, for a compact tool reply."""
    ordered = sorted(values.items(), key=lambda pair: (-pair[1], str(pair[0])))
    return [list(pair) for pair in ordered[:n]]
