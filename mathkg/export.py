"""Single export module: JSON, GEXF, GraphML, pyvis HTML.

The original project had **two** copies of ``export_graph_json`` (one in
``enrichment.py``, one in ``testgraph.py``) that had already drifted apart, and
both computed a full ``nx.spring_layout`` on the whole graph which pyvis then
threw away when it re-ran ForceAtlas2. There is now one implementation, and
layout is opt-in.

``viz``-style colours are written into the GEXF so Gephi opens the file already
coloured -- this is what let the original PNG and the ``.gexf`` disagree.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import networkx as nx
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def _rgb_from_hex(value: str) -> dict[str, int]:
    value = value.lstrip("#")
    return {
        "r": int(value[0:2], 16),
        "g": int(value[2:4], 16),
        "b": int(value[4:6], 16),
    }


def _export_copy(graph: nx.Graph, include_viz: bool = True) -> nx.Graph:
    """Graph with attributes coerced to things GEXF/GraphML/vis.js can store.

    ``distance_vector`` is a list of ``int|None``, which neither GEXF nor the
    pyvis JSON template can serialise; it is stored as a compact string. The
    ``viz`` dict is GEXF-only -- NetworkX's GraphML writer rejects dict values
    outright, which is why it is opt-out. Any remaining ``None`` (e.g. ``field``
    for a node with no reachable anchor) becomes an empty string, because the
    GEXF writer refuses ``NoneType`` attributes.
    """
    copy = graph.copy()
    for _, data in copy.nodes(data=True):
        vector = data.get("distance_vector")
        if isinstance(vector, list):
            data["distance_vector"] = ",".join(
                "inf" if distance is None else str(distance) for distance in vector
            )
        for key, value in list(data.items()):
            if value is None:
                data[key] = ""
        colour = data.get("color")
        if include_viz and isinstance(colour, str) and colour.startswith("#"):
            data["viz"] = {
                "color": {**_rgb_from_hex(colour), "a": 1.0},
                "size": float(data.get("size", 10.0)),
            }
    return copy


def write_json(graph: nx.Graph, path: Path = config.ARTIFACT_DIR / "graph.json") -> Path:
    """Node/link JSON for the web viewer.

    No coordinates by default: the viewer computes its own layout (which the
    original code's own comment acknowledged while still paying for a
    ``spring_layout`` nobody read).
    """
    payload = {
        "nodes": [
            {
                "id": node,
                "label": data.get("label", node),
                "size": float(data.get("size", 10.0)),
                "color": data.get("color", config.UNREACHABLE_COLOR),
                "group": data.get("field") or ("anchor" if data.get("anchor") else ""),
                "degree": int(graph.degree(node)),
                "sitelinks": int(data.get("sitelinks", 0)),
            }
            for node, data in graph.nodes(data=True)
        ],
        "links": [
            {"source": source, "target": target}
            for source, target in graph.edges()
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    log.info("json  : %d nodes, %d links -> %s", len(payload["nodes"]), len(payload["links"]), path)
    return path


def write_gexf(graph: nx.Graph, path: Path = config.ARTIFACT_DIR / "graph.gexf") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(_export_copy(graph), path, encoding="utf-8", prettyprint=True)
    log.info("gexf  : %d nodes, %d edges -> %s", graph.number_of_nodes(), graph.number_of_edges(), path)
    return path


def write_graphml(graph: nx.Graph, path: Path = config.ARTIFACT_DIR / "graph.graphml") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(_export_copy(graph, include_viz=False), path, encoding="utf-8")
    log.info("graphml: %d nodes -> %s", graph.number_of_nodes(), path)
    return path


def write_pyvis(
    graph: nx.Graph,
    path: Path = config.ARTIFACT_DIR / "graph.html",
    directed: bool | None = None,
) -> Path:
    """Interactive HTML, self-contained so it works offline.

    ``cdn_resources="in_line"`` embeds the vis.js runtime in the file. The
    alternative, ``"local"``, makes pyvis copy its ``lib/`` folder into
    ``os.getcwd()`` -- not next to the HTML -- which silently breaks the page
    when the file is opened from anywhere else. That is where the stray empty
    ``lib/`` skeleton in this repo came from.

    pyvis still hard-codes a Bootstrap CDN stylesheet in its template, so the
    generated file is post-processed to drop it: Bootstrap only styles the
    settings panel, and a "share one file" artifact should not phone home.
    """
    from pyvis.network import Network

    path.parent.mkdir(parents=True, exist_ok=True)
    network = Network(
        height="900px",
        width="100%",
        directed=graph.is_directed() if directed is None else directed,
        notebook=False,
        bgcolor="#101018",
        font_color="#f0f0f0",
        cdn_resources="in_line",
    )
    network.from_nx(_export_copy(graph))
    network.set_options(
        """
        {
          "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springStrength": 0.01,
              "damping": 0.4
            },
            "stabilization": {"iterations": 300}
          },
          "interaction": {"hover": true, "tooltipDelay": 120, "hideEdgesOnDrag": true},
          "edges": {"color": {"opacity": 0.25}, "width": 0.5}
        }
        """
    )
    network.write_html(str(path), notebook=False, open_browser=False)
    _strip_external_assets(path)
    log.info("pyvis : %d nodes -> %s", graph.number_of_nodes(), path)
    return path


_CDN_TAG = re.compile(
    r'<(?:link|script)\b[^>]*(?:href|src)="[^"]*cdn\.jsdelivr\.net[^"]*"[^>]*>'
    r'(?:\s*</script>)?',
    re.IGNORECASE,
)


def _strip_external_assets(path: Path) -> None:
    """Remove the leftover CDN ``<link>``/``<script>`` tags pyvis bakes in."""
    html = path.read_text(encoding="utf-8")
    cleaned = _CDN_TAG.sub("", html)
    if cleaned != html:
        path.write_text(cleaned, encoding="utf-8")


def write_distance_csv(
    distances: dict[str, list[int | None]],
    order: list[str],
    path: Path = config.ARTIFACT_DIR / "distances.csv",
) -> Path:
    """One row per node: its distance to each anchor. Makes the figure auditable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Built as strings: a pandas int column holding None becomes float, which
    # wrote distances as "1.0" and made the file look computed to two decimals.
    frame = pd.DataFrame(
        {
            "title": list(distances),
            **{
                anchor: [
                    "" if index >= len(row) or row[index] is None else str(row[index])
                    for row in distances.values()
                ]
                for index, anchor in enumerate(order)
            },
        },
        columns=["title", *order],
    )
    frame.to_csv(path, index=False, encoding="utf-8")
    log.info("distances -> %s", path)
    return path


def export_all(
    graph: nx.Graph,
    distances: dict[str, list[int | None]] | None = None,
    order: list[str] | None = None,
    out_dir: Path = config.ARTIFACT_DIR,
    stem: str = "graph",
    pyvis: bool = True,
) -> dict[str, Path]:
    """Write every artifact from one graph, so they cannot drift apart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "json": write_json(graph, out_dir / f"{stem}.json"),
        "gexf": write_gexf(graph, out_dir / f"{stem}.gexf"),
        "graphml": write_graphml(graph, out_dir / f"{stem}.graphml"),
    }
    if pyvis:
        written["html"] = write_pyvis(graph, out_dir / f"{stem}.html")
    if distances is not None and order is not None:
        written["distances"] = write_distance_csv(
            distances, order, out_dir / f"{stem}_distances.csv"
        )
    return written
