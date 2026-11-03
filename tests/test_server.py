"""Live viewer server: state, payloads, and the HTTP surface.

These tests run a real server on port 0 (an ephemeral port) in a thread, so the
HTTP surface is exercised rather than mocked.
"""

from __future__ import annotations

import json
import threading
import urllib.request

import networkx as nx
import pytest

from mathkg.server import ViewerState, make_server
from mathkg.view import ExpressionError, load_spec


def sample_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for name, sitelinks in [("Algebra", 195), ("Geometry", 213), ("Ring", 40), ("Ideal", 12)]:
        graph.add_node(name, label=name, sitelinks=sitelinks, kind="concept")
    graph.add_edge("Algebra", "Ring", confidence=0.8, evidence="test")
    graph.add_edge("Geometry", "Ring", confidence=0.7, evidence="test")
    graph.add_edge("Ring", "Ideal", confidence=0.6, evidence="test")
    return graph


# --- ViewerState ------------------------------------------------------------ #

def test_patch_spec_bumps_version_and_broadcasts():
    state = ViewerState(sample_graph(), load_spec())
    channel = state.subscribe()

    message = state.patch_spec(["node.size=6 + 3 * degree"])

    assert state.version == 1
    assert message["version"] == 1
    assert channel.get_nowait()["type"] == "style"
    assert state.spec["node"]["size"] == "6 + 3 * degree"


def test_patch_spec_rejects_a_bad_expression_without_changing_state():
    state = ViewerState(sample_graph(), load_spec())
    with pytest.raises(ExpressionError):
        state.patch_spec(["node.size=__import__('os').system('x')"])
    assert state.version == 0


def test_style_payload_covers_every_node():
    state = ViewerState(sample_graph(), load_spec())
    payload = state.style_payload()
    assert len(payload["nodes"]) == state.graph.number_of_nodes()
    assert all(node["size"] > 0 for node in payload["nodes"])


def test_style_payload_omits_unchanged_edges():
    state = ViewerState(sample_graph(), load_spec())
    first = state.style_payload()
    second = state.style_payload()          # same spec: nothing about edges changed
    assert first["edges"] and second["edges"] == []


def test_graph_payload_has_positions_for_layered_layout():
    state = ViewerState(sample_graph(), load_spec())
    payload = state.graph_payload()

    assert payload["physics"] is False
    assert len(payload["nodes"]) == 4
    assert all(node["x"] is not None and node["y"] is not None for node in payload["nodes"])
    assert len(payload["edges"]) == 3


def test_node_size_reflects_the_spec():
    state = ViewerState(sample_graph(), load_spec())
    small = {n["id"]: n["size"] for n in state.style_payload()["nodes"]}

    state.patch_spec(["node.size=6 + 20 * degree"])
    large = {n["id"]: n["size"] for n in state.style_payload()["nodes"]}

    assert large["Ring"] > small["Ring"]


def test_unsubscribed_channels_stop_receiving():
    state = ViewerState(sample_graph(), load_spec())
    channel = state.subscribe()
    state.unsubscribe(channel)
    state.patch_spec(["label.size=10"])
    assert channel.empty()


# --- HTTP surface ----------------------------------------------------------- #

@pytest.fixture()
def live_server():
    server = make_server(sample_graph(), load_spec(), None, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def test_index_serves_the_viewer_page(live_server):
    with urllib.request.urlopen(f"{live_server}/", timeout=5) as response:
        body = response.read().decode("utf-8")
    assert "vis-network" in body
    assert "EventSource" in body


def test_graph_endpoint_returns_nodes_and_edges(live_server):
    with urllib.request.urlopen(f"{live_server}/graph", timeout=5) as response:
        payload = json.loads(response.read())
    assert len(payload["nodes"]) == 4
    assert len(payload["edges"]) == 3


def test_spec_endpoint_patches_and_reports(live_server):
    request = urllib.request.Request(
        f"{live_server}/spec",
        data=json.dumps({"set": ["node.size=degree * 2"]}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        reply = json.loads(response.read())

    assert reply["ok"] is True
    assert reply["version"] == 1


def test_spec_endpoint_rejects_hostile_input(live_server):
    request = urllib.request.Request(
        f"{live_server}/spec",
        data=json.dumps({"set": ["node.size=__import__('os')"]}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    assert caught.value.code == 400


def test_unknown_route_is_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(f"{live_server}/nope", timeout=5)
    assert caught.value.code == 404
