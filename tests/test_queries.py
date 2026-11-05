"""Query tools: ego, prerequisite path, describe.

The path tests encode the honest behaviour that matters: a *directed*
prerequisite chain is a strong claim, and when only an undirected path exists
the caller must be told, because the direction is then mixed.
"""

from __future__ import annotations

import json

import networkx as nx

from mathkg.dependency import describe, ego_graph, path_between, shortest_path


def chain_graph() -> nx.DiGraph:
    """a -> b -> c -> d, plus an isolated pair e <-> f and a lone node g."""
    graph = nx.DiGraph()
    graph.add_edge("a", "b", confidence=0.9)
    graph.add_edge("b", "c", confidence=0.9)
    graph.add_edge("c", "d", confidence=0.9)
    graph.add_edge("e", "f", confidence=0.5)
    graph.add_node("g", sitelinks=1)
    return graph


# --- shortest_path / path_between ------------------------------------------- #

def test_directed_path_is_found():
    assert shortest_path(chain_graph(), "a", "d") == ["a", "b", "c", "d"]


def test_path_between_reports_direction():
    chain, how = path_between(chain_graph(), "a", "d")
    assert chain == ["a", "b", "c", "d"]
    assert how == "directed"


def test_reverse_direction_is_only_undirected():
    """d -> a has no directed path: the honest answer says so."""
    chain, how = path_between(chain_graph(), "d", "a")
    assert chain[0] == "d" and chain[-1] == "a"
    assert how == "undirected"


def test_no_path_between_components():
    chain, how = path_between(chain_graph(), "a", "e")
    assert chain is None
    assert how == "none"


def test_missing_node_returns_none():
    assert shortest_path(chain_graph(), "a", "nope") is None
    chain, how = path_between(chain_graph(), "nope", "a")
    assert chain is None and how == "none"


# --- describe --------------------------------------------------------------- #

def test_describe_counts_both_directions():
    info = describe(chain_graph(), "b")
    assert info["in_degree"] == 1
    assert info["out_degree"] == 1
    assert info["prerequisites"] == 1          # just a
    assert info["enables"] == 2                # c and d


def test_describe_lists_immediate_neighbours():
    info = describe(chain_graph(), "c")
    assert info["immediate_prerequisites"] == ["b"]
    assert info["immediate_enables"] == ["d"]


def test_describe_isolated_node():
    info = describe(chain_graph(), "g")
    assert info["present"] is True
    assert info["prerequisites"] == 0
    assert info["enables"] == 0


def test_describe_absent_node():
    assert describe(chain_graph(), "nope")["present"] is False


def test_describe_reports_a_longest_chain_through_the_node():
    info = describe(chain_graph(), "b")
    assert info["longest_chain_through"][0] == "a"
    assert info["longest_chain_through"][-1] == "d"


# --- ego -------------------------------------------------------------------- #

def test_ego_respects_the_radius():
    graph = chain_graph()
    one_hop = ego_graph(graph, "b", radius=1)
    assert set(one_hop.nodes) == {"b", "a", "c"}

    two_hops = ego_graph(graph, "b", radius=2)
    assert "d" in two_hops


def test_ego_of_absent_node_is_empty():
    assert ego_graph(chain_graph(), "nope").number_of_nodes() == 0


# --- the MCP front end must proxy, not reinvent --------------------------------- #

def test_mcp_handshake_and_tool_list():
    from mathkg.mcp_server import MathKGClient, handle

    client = MathKGClient()
    init = handle(client, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["protocolVersion"]
    assert init["result"]["serverInfo"]["name"] == "mathkg"

    listed = handle(client, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert names == {
        "get_state", "set_spec", "filter", "layout", "color_by", "ego", "path",
        "describe", "reset", "centrality", "community", "select",
        "render_png", "reload_graph",
    }


def _fake_post(record):
    def post(base, path, payload):
        record.append((path, payload))
        return {"ok": True, "path": path}
    return post


def test_mcp_describe_routes_to_query_endpoint():
    """describe must proxy to the viewer's /query, not reimplement it."""
    from mathkg.mcp_server import MathKGClient, handle

    calls = []
    client = MathKGClient(post=_fake_post(calls))
    handle(client, {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "describe", "arguments": {"node": "b"}},
    })
    assert calls == [("/query", {"kind": "describe", "node": "b"})]


def test_mcp_set_spec_forwards_patch():
    """set_spec forwards the raw patches; validation lives in the viewer now."""
    from mathkg.mcp_server import MathKGClient, handle

    calls = []
    client = MathKGClient(post=_fake_post(calls))
    handle(client, {
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "set_spec", "arguments": {"set": ["node.size=6 + degree"]}},
    })
    assert calls == [("/spec", {"set": ["node.size=6 + degree"]})]


def test_mcp_ego_routes_to_query_endpoint():
    from mathkg.mcp_server import MathKGClient, handle

    calls = []
    client = MathKGClient(post=_fake_post(calls))
    handle(client, {
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "ego", "arguments": {"node": "b", "radius": 1}},
    })
    assert calls == [("/query", {"kind": "ego", "node": "b", "radius": 1})]
