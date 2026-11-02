"""Colouring: nearest anchors, not furthest; no crash on unreachable anchors."""

from __future__ import annotations

import networkx as nx
import pytest

from mathkg import config
from mathkg.color import (
    anchor_distances,
    blend_color,
    color_graph,
    dominant_anchor,
    hex_to_rgb,
    rgb_to_hex,
)

PALETTE = ["#FF0000", "#00FF00", "#0000FF", "#FFFFFF"]


def test_hex_rgb_roundtrip():
    assert rgb_to_hex(hex_to_rgb("#4E79A7")) == "#4E79A7"


def test_rgb_output_is_clipped():
    assert rgb_to_hex([-5.0, 0.5, 99.0]) == "#0080FF"


def test_unreachable_node_is_grey():
    assert blend_color([None, None, None], palette=PALETTE) == config.UNREACHABLE_COLOR


def test_uses_nearest_anchor_not_furthest():
    # 0 is nearest, 3 is furthest. The old code used [-3], i.e. the furthest.
    assert blend_color([0, 5, 9, 40], palette=PALETTE, k=1) == "#FF0000"
    assert blend_color([40, 5, 9, 0], palette=PALETTE, k=1) == "#FFFFFF"


def test_blend_of_two_equidistant_anchors_is_midpoint():
    # red + green at equal distance -> yellow-ish, and in range.
    colour = blend_color([1, 1, None, None], palette=PALETTE, k=2)
    red, green, blue = hex_to_rgb(colour)
    assert red > 0.49 and green > 0.49 and blue < 0.01


def test_unreachable_anchors_are_ignored_not_fatal():
    # Previously `sorted([1, None])` raised TypeError.
    colour = blend_color([None, 0, None, None], palette=PALETTE, k=3)
    assert colour == "#00FF00"


def test_blend_is_deterministic():
    vectors = [[2, 1, 3, None], [0, 0, 0, 0], [None, 9, None, 2]]
    first = [blend_color(v, palette=PALETTE) for v in vectors]
    second = [blend_color(v, palette=PALETTE) for v in vectors]
    assert first == second


def test_ties_break_on_anchor_order():
    assert blend_color([2, 2, None, None], palette=PALETTE, k=1) == "#FF0000"


@pytest.fixture()
def chain():
    """a -- b -- c, with `a` as the only anchor."""
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("island", "itself")
    return graph


def test_anchor_distances_directed_is_treated_as_undirected(chain):
    order, distances = anchor_distances(chain, ["a"])
    assert order == ["a"]
    assert distances["c"] == [2]
    assert distances["island"] == [None]


def test_color_graph_marks_unreachable(chain):
    order, distances = color_graph(chain, ["a"], palette=PALETTE)
    assert order == ["a"]
    assert chain.nodes["a"]["color"] == "#FF0000"          # distance 0
    assert chain.nodes["island"]["color"] == config.UNREACHABLE_COLOR
    assert chain.nodes["c"]["distance_vector"] == [2]


def test_color_graph_without_anchors_is_grey(chain):
    order, _ = color_graph(chain, ["absent"], palette=PALETTE)
    assert order == []
    assert all(
        chain.nodes[node]["color"] == config.UNREACHABLE_COLOR for node in chain
    )


def test_dominant_anchor(chain):
    order, distances = anchor_distances(chain, ["a"])
    assert dominant_anchor(distances["a"], order) == "a"
    assert dominant_anchor(distances["island"], order) is None


def test_color_graph_records_the_nearest_field(chain):
    """`field` is what the JSON export groups nodes by."""
    color_graph(chain, ["a"], palette=PALETTE)

    assert chain.nodes["c"]["field"] == "a"
    assert chain.nodes["island"]["field"] is None
