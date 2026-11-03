"""The prerequisite DAG: orientation, cycles, layering and the query views.

This is the graph the project was always about -- "what do I need before X" and
"what does X let me do" -- as opposed to the associative link graph.

**Orientation.** Every edge is stored as ``prerequisite -> dependent``, meaning
"you need the source before the target". Extraction relations have to be
flipped to get there, and the flip is uniform, which is worth stating because it
looks wrong at first:

    A depends_on B            "A uses B"                  -> B -> A
    A is a special case of B  "B is the general case"      -> B -> A
    A is a generalisation of B "B is the concrete case"    -> B -> A
    A is a branch of B        "B contains A"               -> B -> A

In all four, the *target* is the thing you meet first. Concrete before general
is how a curriculum actually runs, and it is also what a proof does (you reach
for the special case you already know).

Wikidata ``P279`` (subclass of) is deliberately **excluded** by default: it is
classification, and with 1,844 such edges it would swamp the 1,762 genuine ones
and turn the dependency graph back into a taxonomy.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

from . import config
from .titles import normalise_title

log = logging.getLogger(__name__)

#: relation -> True if (source, target) must be flipped to get prerequisite order.
FLIP = {
    "depends_on": True,
    "generalisation": True,
    "special_case": True,
    "part_of": True,
    "equivalent": False,   # symmetric: not a prerequisite relation
    "influenced_by": True,
}

#: Relations excluded from the prerequisite DAG by default.
SYMMETRIC_RELATIONS = {"equivalent"}


def load_edges(path: Path = config.RAW_DIR / "edges.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).assign(
        confidence=lambda frame: pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.5)
    )


def build_dag(
    edges: pd.DataFrame,
    titles: set[str] | None = None,
    min_confidence: float = 0.0,
    include_classification: bool = False,
) -> nx.DiGraph:
    """Build the ``prerequisite -> dependent`` DAG from extracted edges."""
    graph = nx.DiGraph()

    for _, row in edges.iterrows():
        relation = row["relation"]
        if relation in SYMMETRIC_RELATIONS:
            continue
        if not include_classification and (
            row["extractor"] == "wikidata" and relation == "special_case"
        ):
            continue
        confidence = float(row["confidence"])
        if confidence < min_confidence:
            continue

        source = normalise_title(row["source"])
        target = normalise_title(row["target"])
        if not source or not target or source == target:
            continue
        if titles is not None and (source not in titles or target not in titles):
            continue

        before, after = (target, source) if FLIP[relation] else (source, target)
        # Keep the strongest evidence when an edge is asserted twice.
        if graph.has_edge(before, after):
            existing = graph[before][after]
            if confidence > existing["confidence"]:
                graph[before][after].update(
                    confidence=confidence,
                    relation=relation,
                    extractor=row["extractor"],
                    evidence=row["evidence"],
                )
            continue
        graph.add_edge(
            before,
            after,
            confidence=confidence,
            relation=relation,
            extractor=row["extractor"],
            evidence=row["evidence"],
        )

    log.info(
        "dag: %d nodes, %d prerequisite edges (min_confidence=%.2f, classification=%s)",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        min_confidence,
        include_classification,
    )
    return graph


def cycle_report(graph: nx.DiGraph) -> list[list[str]]:
    """Non-trivial strongly connected components = the cycles.

    ``nx.simple_cycles`` is exponential; SCCs are linear and report the same
    thing in a usable form, since every cycle lives inside one SCC.
    """
    cycles = [
        sorted(component)
        for component in nx.strongly_connected_components(graph)
        if len(component) > 1
    ]
    cycles.sort(key=len, reverse=True)
    log.info(
        "cycles: %d non-trivial component(s), largest %d nodes",
        len(cycles),
        len(cycles[0]) if cycles else 0,
    )
    return cycles


def break_cycles(graph: nx.DiGraph) -> tuple[nx.DiGraph, int]:
    """Drop the weakest edge inside each SCC until the graph is a DAG.

    Layered layout needs acyclic input, and a cycle here is nearly always bad
    extraction rather than a real loop in mathematics. Dropping the
    lowest-confidence edge keeps the best-evidenced direction.
    """
    working = graph.copy()
    dropped = 0
    while True:
        components = [
            component
            for component in nx.strongly_connected_components(working)
            if len(component) > 1
        ]
        if not components:
            break
        for component in components:
            inner = [
                (u, v, working[u][v]["confidence"])
                for u, v in working.subgraph(component).edges()
            ]
            if not inner:
                continue
            weakest = min(inner, key=lambda item: item[2])
            working.remove_edge(weakest[0], weakest[1])
            dropped += 1
    if dropped:
        log.info("cycles: dropped %d weak edge(s) to make the graph acyclic", dropped)
    return working, dropped


def layered_positions(
    graph: nx.DiGraph, x_spacing: float = 1.0, y_spacing: float = 1.0
) -> dict[str, tuple[float, float]]:
    """Column = longest prerequisite depth, rows ordered alphabetically."""
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("layered_positions needs a DAG; call break_cycles first")

    depth: dict[str, int] = {}
    for node in nx.topological_sort(graph):
        predecessors = list(graph.predecessors(node))
        depth[node] = 0 if not predecessors else max(depth[p] + 1 for p in predecessors)

    columns: dict[int, list[str]] = defaultdict(list)
    for node, level in depth.items():
        columns[level].append(node)

    positions: dict[str, tuple[float, float]] = {}
    for level, nodes in columns.items():
        nodes.sort()
        for index, node in enumerate(nodes):
            offset = index - (len(nodes) - 1) / 2
            positions[node] = (level * x_spacing, -offset * y_spacing)
    return positions


def prerequisites_of(graph: nx.DiGraph, node: str) -> dict[str, int]:
    """Everything you need before ``node``, with hop distance."""
    return nx.single_source_shortest_path_length(graph.reverse(copy=False), node)


def enables(graph: nx.DiGraph, node: str) -> dict[str, int]:
    """Everything ``node`` unlocks, with hop distance."""
    return nx.single_source_shortest_path_length(graph, node)


def critical_path(graph: nx.DiGraph) -> list[str]:
    """Longest prerequisite chain -- reads as the spine of the graph."""
    if graph.number_of_edges() == 0:
        return []
    return nx.dag_longest_path(graph)


def summary(graph: nx.DiGraph) -> dict:
    components = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "sources": sum(1 for _, degree in graph.in_degree() if degree == 0),
        "sinks": sum(1 for _, degree in graph.out_degree() if degree == 0),
        "components": len(components),
        "largest_component": len(components[0]) if components else 0,
    }


def ego_graph(graph: nx.DiGraph, node: str, radius: int = 2) -> nx.DiGraph:
    """The portion of the DAG around one notion, both directions."""
    if node not in graph:
        return nx.DiGraph()
    forward = nx.ego_graph(graph, node, radius=radius, undirected=True)
    return graph.subgraph(set(forward.nodes) | {node}).copy()
