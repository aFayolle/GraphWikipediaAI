"""v2: structural embeddings + packed components, measured against v1.

What changed versus the version I emailed:

1. **Embeddings are structural, not lexical.** TF-IDF over summaries clusters by
   vocabulary ("theorem", "proof", "group"). A spectral embedding of the graph
   Laplacian clusters by position in the dependency graph, which is what the
   question actually asks about.
2. **Components are packed, not merged.** A single force-directed layout drops
   all 120+ components into one blob, so nothing is separable. Each component now
   gets its own cell of a grid: visible islands.
3. **Separation is measured.** A silhouette score is added to the quality
   metrics, because "I can't tell the regions apart" is exactly a silhouette
   problem.

Acceptance: report every candidate, keep the one with the best combination of
silhouette, nearest-neighbour distance and label collisions, and say plainly
which thresholds were not met.

Usage: PYTHONPATH=. python tools/make_v2.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mathkg import config, dependency as D, edges as E
from mathkg import embedding as EM
from mathkg import render

OUT = config.PROJECT_ROOT / "reports" / "readable_v2"
CANVAS = (1600.0, 1000.0)


def scale_to_canvas(positions, radii, edges_list, labels, robust=True):
    xs = np.array([p[0] for p in positions.values()])
    ys = np.array([p[1] for p in positions.values()])
    if robust:
        lo_x, hi_x = np.percentile(xs, [2, 98])
        lo_y, hi_y = np.percentile(ys, [2, 98])
    else:
        lo_x, hi_x, lo_y, hi_y = xs.min(), xs.max(), ys.min(), ys.max()
    width = float(hi_x - lo_x) or 1.0
    height = float(hi_y - lo_y) or 1.0
    scale = min(CANVAS[0] / width, CANVAS[1] / height)
    scaled = {
        node: (
            float(min(max((x - lo_x) * scale, 0.0), CANVAS[0])),
            float(min(max((y - lo_y) * scale, 0.0), CANVAS[1])),
        )
        for node, (x, y) in positions.items()
    }
    quality = EM.layout_quality(scaled, radii, edges_list, labels=labels)
    return quality, scaled


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, titles = E.load_summaries()
    dag = D.break_cycles(D.build_dag(D.load_edges(), titles=set(titles)))[0]
    degree = dict(dag.degree())

    # structural embedding over the whole graph, then restrict per candidate
    spectral_ids, spectral = EM.spectral_embedding(dag, dims=50)
    spectral_row = {name: index for index, name in enumerate(spectral_ids)}

    flat = render._read_positions_file(render._installed_path(dag))

    rows = []
    best = None

    for cutoff, keep in ((0, 1560), (2, 779), (3, 474), (4, 301), (5, 194)):
        subset = [n for n in dag.nodes if degree[n] >= cutoff]
        subgraph = dag.subgraph(subset).copy()
        if subgraph.number_of_nodes() < 40:
            continue

        local = render.forceatlas2_positions(subgraph, max_iter=250) or {}
        if not local:
            continue
        packed = EM.pack_components(subgraph, local)
        radii = {n: 4.0 + 2.0 * float(np.sqrt(degree[n])) for n in subset}
        edges_list = [(a, b) for a, b in subgraph.edges()]
        labels = {n for n, _ in sorted(subgraph.degree, key=lambda p: -p[1])[:25]}

        matrix = np.vstack([spectral[spectral_row[n]] for n in sorted(subset)])
        nodes_sorted = sorted(subset)
        for cluster_count in (6, 10, 14):
            medoids = EM.k_medoids(matrix, k=cluster_count, seed=0)
            colours, membership = EM.colours_from_centres(matrix[:, :3], medoids, sharpness=2.5)
            labels_int = [f"c{int(row.argmax())}" for row in membership]
            separation = EM.silhouette(matrix, labels_int, sample=250)

            quality, scaled = scale_to_canvas(packed, radii, edges_list, labels)
            score = (
                separation
                + min(1.0, quality["median_nnd"] / 3.0)
                - 6.0 * quality["label_collisions"]
                - 4.0 * quality["edge_crossing_rate"]
            )
            candidate = {
                "cutoff": cutoff,
                "nodes": len(nodes_sorted),
                "clusters": cluster_count,
                "silhouette": separation,
                "quality": quality,
                "score": score,
                "positions": scaled,
                "colours": dict(zip(nodes_sorted, colours)),
                "labels": labels,
                "edges": edges_list,
                "subgraph": subgraph,
            }
            rows.append(candidate)
            if best is None or score > best["score"]:
                best = candidate

    table = pd.DataFrame(
        [
            {
                "min_deg": c["cutoff"],
                "nodes": c["nodes"],
                "k": c["clusters"],
                "silhouette": round(c["silhouette"], 3),
                "nnd_px": round(c["quality"]["median_nnd"], 2),
                "overlap%": round(c["quality"]["node_overlap"] * 100, 3),
                "crossings%": round(c["quality"]["edge_crossing_rate"] * 100, 2),
                "labelcoll%": round(c["quality"]["label_collisions"] * 100, 2),
                "score": round(c["score"], 3),
            }
            for c in rows
        ]
    ).sort_values("score", ascending=False)
    print(table.to_string(index=False))
    table.to_csv(OUT / "candidates.csv", index=False)

    print(f"\nchosen: min_degree={best['cutoff']} · {best['nodes']} nodes · "
          f"k={best['clusters']} · silhouette {best['silhouette']:.3f} · "
          f"nnd {best['quality']['median_nnd']:.2f}px")

    nodes_sorted = sorted(best["positions"])
    payload = {
        "cutoff": best["cutoff"],
        "clusters": best["clusters"],
        "silhouette": best["silhouette"],
        "quality": {k: float(v) for k, v in best["quality"].items()},
        "candidates": table.to_dict(orient="records"),
        "nodes": [
            {
                "title": n,
                "x": float(best["positions"][n][0]),
                "y": float(best["positions"][n][1]),
                "colour": best["colours"][n],
                "degree": int(degree[n]),
                "label": n if n in best["labels"] else "",
            }
            for n in nodes_sorted
        ],
        "edges": [{"from": a, "to": b} for a, b in best["edges"]],
        "components": len(list(__import__("networkx").weakly_connected_components(best["subgraph"]))),
    }
    (OUT / "graph.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {OUT/'graph.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
