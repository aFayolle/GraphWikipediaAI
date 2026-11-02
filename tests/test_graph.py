"""Graph construction: induced subgraph, self-loops, and a cap that actually caps."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from mathkg.graph import build_graph, cap_degree, node_size, shape_metrics, summary


def nodes_df(rows):
    return pd.DataFrame(
        [
            {"qid": f"Q{i}", "label": label, "title": title,
             "sitelinks": sitelinks, "en_url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"}
            for i, (title, label, sitelinks) in enumerate(rows)
        ]
    )


def links_df(pairs):
    return pd.DataFrame(pairs, columns=["source", "target"])


@pytest.fixture()
def triangle():
    nodes = nodes_df([("A", "a", 10), ("B", "b", 100), ("C", "c", 5)])
    links = links_df([("A", "B"), ("B", "C"), ("C", "A")])
    return build_graph(nodes, links)


def test_basic_build(triangle):
    assert triangle.number_of_nodes() == 3
    assert triangle.number_of_edges() == 3
    assert triangle.is_directed()
    assert triangle.nodes["B"]["label"] == "b"


def test_node_size_is_log_scaled():
    assert node_size(0) == pytest.approx(5.0)
    assert node_size(9) == pytest.approx(10.0)
    assert node_size(100) > node_size(10)


def test_self_loops_are_dropped():
    nodes = nodes_df([("A", "a", 1), ("B", "b", 1)])
    links = links_df([("A", "A"), ("A", "B")])
    graph = build_graph(nodes, links)

    assert nx.number_of_selfloops(graph) == 0
    assert graph.number_of_edges() == 1


def test_edges_outside_the_node_set_are_dropped():
    nodes = nodes_df([("A", "a", 1)])
    links = links_df([("A", "Outside"), ("A", "A")])
    graph = build_graph(nodes, links)

    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 0


def test_anchors_are_flagged():
    nodes = nodes_df([("A", "a", 1), ("B", "b", 1)])
    graph = build_graph(nodes, links_df([("A", "B")]), anchors={"B"})

    assert graph.nodes["B"]["anchor"] is True
    assert graph.nodes["A"]["anchor"] is False


def test_cap_degree_is_iterative():
    """Removing the hub must push its neighbours back under the cap.

    This is the regression test for `define_max_degree`, whose effect was
    invisible in the shipped CSV (a node of degree 2,016 survived a cap of 20).
    """
    graph = nx.DiGraph()
    graph.add_edge("hub", "a")
    graph.add_edge("hub", "b")
    graph.add_edge("hub", "c")
    graph.add_edge("a", "b")

    capped = cap_degree(graph, max_degree=2)

    assert "hub" not in capped
    assert set(capped.nodes) == {"a", "b", "c"}
    assert max(dict(capped.degree()).values()) <= 2


def test_cap_degree_is_noop_when_under():
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    assert set(cap_degree(graph, 10).nodes) == {"a", "b"}


def test_cap_degree_protects_anchors():
    """The anchors are the hubs: an unprotected cap greys out the whole graph."""
    graph = nx.DiGraph()
    graph.add_edge("anchor", "a")
    graph.add_edge("anchor", "b")
    graph.add_edge("anchor", "c")
    graph.add_edge("hub", "d")
    graph.add_edge("hub", "e")
    graph.add_edge("hub", "f")

    capped = cap_degree(graph, max_degree=2, protect={"anchor"})

    assert "anchor" in capped          # kept despite degree 3
    assert "hub" not in capped         # ordinary hub still removed


def test_cap_degree_rejects_nonpositive():
    with pytest.raises(ValueError):
        cap_degree(nx.DiGraph(), 0)


def test_build_graph_with_max_degree():
    graph = build_graph(
        nodes_df([("A", "a", 1), ("B", "b", 1), ("C", "c", 1)]),
        links_df([("A", "B"), ("B", "C"), ("C", "A")]),
        max_degree=1,
    )
    assert graph.number_of_edges() <= 1


def test_summary_counts(triangle):
    triangle.add_node("orphan", label="o", size=1.0)
    stats = summary(triangle)

    assert stats["nodes"] == 4
    assert stats["edges"] == 3
    assert stats["self_loops"] == 0
    assert stats["isolated"] == 1
    assert stats["largest_component"] == 3


def test_shape_metrics_reports_small_world_resolution():
    """The colouring has as many levels as the graph has distances."""
    chain = nx.path_graph(6)  # a - b - c - d - e - f
    metrics = shape_metrics(chain, sample_sources=6)

    assert metrics["largest_component"] == 6
    assert metrics["max_eccentricity"] == 5
    assert metrics["distance_histogram"][1] > 0
    assert 0.0 <= metrics["share_within_4_hops"] <= 1.0


def test_shape_metrics_is_deterministic():
    graph = nx.gnp_random_graph(40, 0.15, seed=1)
    assert shape_metrics(graph, sample_sources=10) == shape_metrics(graph, sample_sources=10)


def test_shape_metrics_handles_empty_graph():
    assert shape_metrics(nx.DiGraph()) == {"largest_component": 0}
