"""The curriculum DAG that used to live inside ``testgraph.py``.

It is a hand-written prerequisite model of a first analysis course
(29 notions, 39 "A is needed for B" edges). It shares no vocabulary or code
with the Wikidata pipeline, so it now lives on its own and is rendered with a
**layered DAG layout** -- deterministic, and it actually reads as a curriculum
instead of a spring-layout blob.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx

from . import config

log = logging.getLogger(__name__)

NODE_TYPES = {
    "Logique des propositions": "notion",
    "Ensembles et inclusion": "notion",
    "Applications entre ensembles": "definition",
    "Nombres reels R": "definition",
    "Ordre sur R et intervalles": "definition",
    "Supremum infimum": "definition",
    "Completude de R": "theorem",
    "Suite reelle": "definition",
    "Suite convergente et limite": "definition",
    "Suites monotones": "definition",
    "Suites de Cauchy": "definition",
    "Fonction reelle": "definition",
    "Limite de fonction en un point": "definition",
    "Limite de fonction a l infini": "definition",
    "Continuite en un point": "definition",
    "Continuite sur un intervalle": "definition",
    "Theoreme valeur intermediaire": "theorem",
    "Theoreme bornes atteintes": "theorem",
    "Theoreme de la bijection": "theorem",
    "Derivabilite en un point": "definition",
    "Derivee": "definition",
    "Regles de derivation": "theorem",
    "Theoreme de Rolle": "theorem",
    "Theoreme des accroissements finis": "theorem",
    "Developpement de Taylor": "theorem",
    "Fonction continue par morceaux": "definition",
    "Integrale de Riemann": "definition",
    "Proprietes de l integrale": "theorem",
    "Theoreme fondamental de l analyse": "theorem",
}

# (prerequisite, dependent)
PREREQUISITES = [
    # Fondations
    ("Logique des propositions", "Ensembles et inclusion"),
    ("Logique des propositions", "Nombres reels R"),
    ("Ensembles et inclusion", "Applications entre ensembles"),
    ("Ensembles et inclusion", "Nombres reels R"),
    ("Nombres reels R", "Ordre sur R et intervalles"),
    ("Ordre sur R et intervalles", "Supremum infimum"),
    ("Supremum infimum", "Completude de R"),
    # Suites
    ("Ensembles et inclusion", "Suite reelle"),
    ("Nombres reels R", "Suite reelle"),
    ("Suite reelle", "Suite convergente et limite"),
    ("Suite reelle", "Suites monotones"),
    ("Suite reelle", "Suites de Cauchy"),
    ("Completude de R", "Suites de Cauchy"),
    # Fonctions & limites
    ("Applications entre ensembles", "Fonction reelle"),
    ("Nombres reels R", "Fonction reelle"),
    ("Fonction reelle", "Limite de fonction en un point"),
    ("Limite de fonction en un point", "Limite de fonction a l infini"),
    ("Limite de fonction en un point", "Continuite en un point"),
    ("Continuite en un point", "Continuite sur un intervalle"),
    # Theoremes de continuite
    ("Continuite sur un intervalle", "Theoreme valeur intermediaire"),
    ("Continuite sur un intervalle", "Theoreme bornes atteintes"),
    ("Continuite sur un intervalle", "Theoreme de la bijection"),
    # Derivees
    ("Limite de fonction en un point", "Derivabilite en un point"),
    ("Derivabilite en un point", "Derivee"),
    ("Derivee", "Regles de derivation"),
    ("Continuite sur un intervalle", "Theoreme de Rolle"),
    ("Derivabilite en un point", "Theoreme de Rolle"),
    ("Theoreme de Rolle", "Theoreme des accroissements finis"),
    ("Regles de derivation", "Developpement de Taylor"),
    ("Theoreme des accroissements finis", "Developpement de Taylor"),
    # Integrale
    ("Fonction reelle", "Fonction continue par morceaux"),
    ("Nombres reels R", "Fonction continue par morceaux"),
    ("Fonction continue par morceaux", "Integrale de Riemann"),
    ("Suite convergente et limite", "Integrale de Riemann"),
    ("Integrale de Riemann", "Proprietes de l integrale"),
    ("Continuite sur un intervalle", "Integrale de Riemann"),
    ("Integrale de Riemann", "Theoreme fondamental de l analyse"),
    ("Derivee", "Theoreme fondamental de l analyse"),
    ("Continuite sur un intervalle", "Theoreme fondamental de l analyse"),
]

TYPE_COLORS = {
    "notion": "#4E79A7",
    "definition": "#59A14F",
    "theorem": "#E15759",
}


def build_curriculum_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for name, kind in NODE_TYPES.items():
        graph.add_node(name, label=name, type=kind, color=TYPE_COLORS[kind], size=18.0)
    graph.add_edges_from(PREREQUISITES)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("the curriculum prerequisites contain a cycle")
    return graph


def layered_layout(
    graph: nx.DiGraph, x_spacing: float = 320.0, y_spacing: float = 90.0
) -> dict[str, tuple[float, float]]:
    """Deterministic layout: column per prerequisite depth, ordered rows.

    Replaces ``nx.spring_layout`` for this graph. Spring layout is
    non-deterministic without a seed, ignores the fact that the graph is
    acyclic, and gives a picture you cannot compare between two runs.
    """
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("layered_layout needs a DAG")

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
            positions[node] = (level * x_spacing, offset * y_spacing)
    return positions


def longest_path(graph: nx.DiGraph) -> list[str]:
    """The critical path -- reads as "the spine" of the course."""
    return nx.dag_longest_path(graph)


def run(out_dir: Path = config.ARTIFACT_DIR) -> dict[str, Path]:
    from .export import write_gexf, write_json, write_pyvis

    graph = build_curriculum_graph()
    positions = layered_layout(graph)
    for node, (x, y) in positions.items():
        graph.nodes[node]["x"] = float(x)
        graph.nodes[node]["y"] = float(y)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "json": write_json(graph, out_dir / "curriculum.json"),
        "gexf": write_gexf(graph, out_dir / "curriculum.gexf"),
        "html": write_pyvis(graph, out_dir / "curriculum.html"),
    }
    log.info(
        "curriculum: %d nodes, %d edges, longest prerequisite chain of %d",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(longest_path(graph)),
    )
    return written
