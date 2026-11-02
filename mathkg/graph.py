"""Graph construction: normalised tables -> NetworkX ``DiGraph``.

Two documented defects are fixed here rather than downstream:

* **Node identity.** Nodes were keyed on ``en_url`` while edges came from HTML
  hrefs, so 586 articles existed twice. Both sides are now keyed on the
  canonical title produced by :mod:`mathkg.titles`.
* **Silent filtering.** The original ``define_max_degree`` rebound the
  DataFrame to a slice and then assigned through it, so the cap could silently
  do nothing (the shipped CSV still had a node of degree 2,016). The cap is now
  explicit, iterative, and reports what it dropped.

The graph is *directed* (``A`` links to ``B``); the undirected projection is
derived only when colouring needs it.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import networkx as nx
import pandas as pd

from . import config
from .titles import normalise_title

log = logging.getLogger(__name__)


def load_nodes(path: Path = config.NODES_CSV) -> pd.DataFrame:
    nodes = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    if "title" not in nodes.columns:
        raise ValueError(f"{path} has no 'title' column; run the fetch/import stage first")
    nodes["title"] = nodes["title"].map(normalise_title)
    nodes = nodes[nodes["title"] != ""]
    nodes["sitelinks"] = pd.to_numeric(
        nodes.get("sitelinks", 0), errors="coerce"
    ).fillna(0).astype(int)
    # One row per article, even if the source had several Wikidata items.
    nodes = nodes.sort_values("sitelinks", ascending=False).drop_duplicates(
        subset="title", keep="first"
    )
    return nodes.reset_index(drop=True)


def load_links(path: Path = config.LINKS_CSV) -> pd.DataFrame:
    links = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    for column in ("source", "target"):
        if column not in links.columns:
            raise ValueError(f"{path} has no '{column}' column")
        links[column] = links[column].map(normalise_title)
    links = links[(links["source"] != "") & (links["target"] != "")]
    return links.drop_duplicates().reset_index(drop=True)


def node_size(sitelinks: int) -> float:
    """Rendering size from the sitelink count (log scale, as before)."""
    return 5.0 + 5.0 * math.log10(1.0 + max(0, int(sitelinks)))


def build_graph(
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    anchors: set[str] | None = None,
    max_degree: int | None = None,
) -> nx.DiGraph:
    """Build the directed article graph, induced on the node set."""
    anchors = anchors or set()
    graph = nx.DiGraph()

    for _, row in nodes.iterrows():
        graph.add_node(
            row["title"],
            label=str(row.get("label") or row["title"]),
            qid=str(row.get("qid", "")),
            sitelinks=int(row["sitelinks"]),
            size=node_size(row["sitelinks"]),
            anchor=row["title"] in anchors,
            url=str(row.get("en_url", "")),
        )

    kept = 0
    for source, target in zip(links["source"], links["target"], strict=True):
        if source == target:
            continue  # self-links carry no structural information
        if source in graph and target in graph:
            graph.add_edge(source, target)
            kept += 1

    log.info(
        "graph: %d nodes, %d edges (%d link rows dropped as self/out-of-scope)",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(links) - kept,
    )

    if max_degree is not None:
        graph = cap_degree(graph, max_degree, protect=anchors)

    return graph


def cap_degree(
    graph: nx.DiGraph, max_degree: int, protect: set[str] | None = None
) -> nx.DiGraph:
    """Drop nodes whose total degree exceeds ``max_degree``, until stable.

    Iterative on purpose: removing one hub lowers its neighbours' degree, which
    can bring them back under the cap. Reported, never silent.

    ``protect`` nodes are never dropped. The anchors *are* the highest-degree
    nodes of this graph, so an unprotected cap deletes all of them and leaves a
    perfectly grey picture -- which is what happens with ``--max-degree 20``
    without this.
    """
    if max_degree <= 0:
        raise ValueError("max_degree must be positive")

    protected = protect or set()
    working = graph.copy()
    dropped_total = 0
    while True:
        degree = dict(working.degree())
        too_big = [
            node
            for node, deg in degree.items()
            if deg > max_degree and node not in protected
        ]
        if not too_big:
            break
        working.remove_nodes_from(too_big)
        dropped_total += len(too_big)
        log.info("max_degree=%d: dropped %d node(s)", max_degree, len(too_big))

    log.info(
        "max_degree=%d: %d nodes -> %d nodes, %d edges (%d dropped, %d protected)",
        max_degree,
        graph.number_of_nodes(),
        working.number_of_nodes(),
        working.number_of_edges(),
        dropped_total,
        len(protected & set(graph.nodes)),
    )
    return working


def anchor_order(fields: pd.DataFrame | None) -> list[str]:
    """Anchor titles in *file* order.

    Order matters: the palette is assigned positionally, so a field keeps its
    colour across runs -- and stays put if another field is added or removed --
    only if the order comes from ``fields_of_math.csv`` rather than from
    ``sorted()``.

    The URL is the authoritative source of a title's *casing*: Wikidata labels
    are lower-case ("mathematical analysis") while the Wikipedia page title is
    "Mathematical analysis", and the graph is keyed on the latter.
    """
    if fields is None or fields.empty:
        return []
    from .titles import title_from_url

    ordered: list[str] = []
    for _, row in fields.iterrows():
        # One title per anchor: the stored title, else the URL, else the label.
        for candidate in (
            row.get("title", ""),
            title_from_url(str(row.get("en_url", ""))) or "",
            row.get("label", ""),
        ):
            title = normalise_title(candidate)
            if title:
                if title not in ordered:
                    ordered.append(title)
                break
    return ordered


def anchor_titles(fields: pd.DataFrame | None) -> set[str]:
    """Membership set of the anchors (see :func:`anchor_order`)."""
    return set(anchor_order(fields))


def load_fields(path: Path = config.FIELDS_CSV) -> pd.DataFrame:
    from .titles import title_from_url

    fields = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    if "title" not in fields.columns:
        fields["title"] = [
            normalise_title(
                title_from_url(row.get("en_url", "")) or row.get("label", "")
            )
            for _, row in fields.iterrows()
        ]
    return fields


def summary(graph: nx.Graph) -> dict[str, float | int]:
    """Numbers worth printing next to every export, to catch drift."""
    undirected = graph.to_undirected()
    components = list(nx.connected_components(undirected))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "self_loops": nx.number_of_selfloops(graph),
        "isolated": sum(1 for _, deg in graph.degree() if deg == 0),
        "components": len(components),
        "largest_component": max((len(c) for c in components), default=0),
    }
