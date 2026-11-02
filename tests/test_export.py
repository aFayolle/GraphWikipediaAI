"""Exports: the JSON, the GEXF and the in-memory graph must agree.

This is the regression test for the defect that prompted the refactor: the
shipped artifacts described 6,623 nodes (GEXF) and 7,444 nodes (JSON), so no
figure could be traced back to a run.
"""

from __future__ import annotations

import json

import networkx as nx

from mathkg.color import color_graph
from mathkg.export import export_all, write_distance_csv


def sample_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for name, sitelinks in [("Algebra", 195), ("Geometry", 213), ("Ring", 40)]:
        graph.add_node(
            name,
            label=name,
            sitelinks=sitelinks,
            size=5.0,
            color="#4E79A7",
            anchor=name in {"Algebra", "Geometry"},
            distance_vector=[0, 1],
        )
    graph.add_edge("Algebra", "Ring")
    graph.add_edge("Geometry", "Ring")
    graph.add_edge("Ring", "Algebra")
    # An isolated node: no path to any anchor, so `field` ends up None. The GEXF
    # writer refuses NoneType attributes, so this keeps that path honest.
    graph.add_node("Island", label="Island", sitelinks=1, size=5.0, anchor=False)
    return graph


def test_all_artifacts_agree(tmp_path):
    graph = sample_graph()
    order, distances = color_graph(graph, ["Algebra", "Geometry"])
    written = export_all(
        graph, distances=distances, order=order, out_dir=tmp_path, stem="g"
    )

    assert written["json"].exists()
    assert written["gexf"].exists()
    assert written["graphml"].exists()

    payload = json.loads(written["json"].read_text(encoding="utf-8"))
    assert len(payload["nodes"]) == graph.number_of_nodes()
    assert len(payload["links"]) == graph.number_of_edges()

    reloaded = nx.read_gexf(written["gexf"])
    assert reloaded.number_of_nodes() == graph.number_of_nodes()
    assert reloaded.number_of_edges() == graph.number_of_edges()


def test_json_has_no_coordinates_by_default(tmp_path):
    """The viewer computes its own layout; the old spring_layout was dead weight."""
    written = export_all(sample_graph(), out_dir=tmp_path, stem="g", pyvis=False)
    payload = json.loads(written["json"].read_text(encoding="utf-8"))

    assert all("x" not in node and "y" not in node for node in payload["nodes"])


def test_gexf_carries_viz_colours(tmp_path):
    written = export_all(sample_graph(), out_dir=tmp_path, stem="g", pyvis=False)
    text = written["gexf"].read_text(encoding="utf-8")

    # The old colored .gexf used palette values produced by a code path that no
    # longer existed; now the colour travels with the graph.
    assert "viz:color" in text
    assert 'r="78"' in text  # #4E79A7


def test_distance_vector_survives_export(tmp_path):
    graph = sample_graph()
    order, distances = color_graph(graph, ["Algebra", "Geometry"])
    written = export_all(
        graph, distances=distances, order=order, out_dir=tmp_path, stem="g", pyvis=False
    )

    reloaded = nx.read_gexf(written["gexf"])
    assert reloaded.nodes["Ring"]["distance_vector"] == "1,1"

    rows = written["distances"].read_text(encoding="utf-8").splitlines()
    assert rows[0] == "title,Algebra,Geometry"
    assert len(rows) == graph.number_of_nodes() + 1


def test_distance_csv_handles_unreachable(tmp_path):
    path = write_distance_csv(
        {"island": [None, 2]}, ["Algebra", "Geometry"], path=tmp_path / "d.csv"
    )
    content = path.read_text(encoding="utf-8")
    assert ",2" in content.splitlines()[1]


def test_pyvis_html_is_written(tmp_path):
    written = export_all(sample_graph(), out_dir=tmp_path, stem="g", pyvis=True)
    html = written["html"].read_text(encoding="utf-8")

    assert "Algebra" in html
    # Self-contained: no lib/ folder copied into the working directory, and no
    # CDN request, so the page still works offline.
    assert not (tmp_path / "lib").exists()
    assert "cdn.jsdelivr.net" not in html
    assert "vis-network" in html.lower()
