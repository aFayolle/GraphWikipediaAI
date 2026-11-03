"""Build the readable, embedding-coloured view, and measure it in screen pixels.

Previous rounds failed because "readable" was judged by eye on data-unit
coordinates. On a 1600x1000 canvas, 1,560 nodes have about 1 px^2 each: no
layout can be read at that density, whatever the algorithm. So every candidate
here is converted to *screen pixels* before it is scored.

Acceptance criteria for "readable", fixed in advance:
  * node_overlap      < 0.002   (fewer than 2 in 1000 node pairs touching)
  * median_nnd        > 3.0 px  (nearest neighbour at least 3 px away)
  * label_collisions  < 0.02    (under 2% of labelled pairs overlapping)

Usage: python tools/make_readable.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mathkg import config, dependency as D, edges as E
from mathkg import embedding as EM
from mathkg import render

CANVAS = (1600.0, 1000.0)
OUT = config.PROJECT_ROOT / "reports" / "readable"

THRESHOLDS = {"node_overlap": 0.002, "median_nnd": 3.0, "label_collisions": 0.02}


def screen_quality(positions, radii, edges_list, labels, robust: bool = False):
    """Scale to the canvas, then measure. This is what the eye actually sees.

    ``robust`` fits to the 5-95 percentile box instead of the full extent. A
    ForceAtlas2 run leaves a handful of far-flung outliers that dominate the
    bounding box, so fitting to everything squeezes the readable core into a
    few pixels: measured as a nearest-neighbour distance of 1.6 px with only
    125 nodes on screen.
    """
    xs = np.array([p[0] for p in positions.values()])
    ys = np.array([p[1] for p in positions.values()])
    if robust:
        lo_x, hi_x = np.percentile(xs, [5, 95])
        lo_y, hi_y = np.percentile(ys, [5, 95])
    else:
        lo_x, hi_x, lo_y, hi_y = xs.min(), xs.max(), ys.min(), ys.max()
    width = float(hi_x - lo_x) or 1.0
    height = float(hi_y - lo_y) or 1.0
    scale = min(CANVAS[0] / width, CANVAS[1] / height)

    def clamp(value: float, high: float) -> float:
        return float(min(max(value, 0.0), high))

    scaled_positions = {
        node: (
            clamp((x - float(lo_x)) * scale, CANVAS[0]),
            clamp((y - float(lo_y)) * scale, CANVAS[1]),
        )
        for node, (x, y) in positions.items()
    }
    # Radii are already in pixels. Scaling them down too made every node ~1 px,
    # which made "no overlap" trivially true and the metric meaningless.
    scaled_radii = {node: float(radius) for node, radius in radii.items()}
    quality = EM.layout_quality(scaled_positions, scaled_radii, edges_list, labels=labels)
    quality["scale"] = scale
    return quality, scaled_positions, scaled_radii


def is_readable(quality: dict) -> bool:
    return (
        quality["node_overlap"] < THRESHOLDS["node_overlap"]
        and quality["median_nnd"] > THRESHOLDS["median_nnd"]
        and quality["label_collisions"] < THRESHOLDS["label_collisions"]
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, titles = E.load_summaries()
    raw = pd.read_csv(
        config.LEGACY_ENRICHED_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    raw["title"] = raw["wiki_title"].where(raw["wiki_title"] != "", raw["label"])
    summaries = dict(zip(raw["title"], raw["wiki_summary"], strict=False))

    dag = D.break_cycles(D.build_dag(D.load_edges(), titles=set(titles)))[0]

    # Embeddings for every node, computed once: they give both the colours and a
    # candidate *position* (the graph layout has no regions to show; a semantic
    # projection may).
    all_docs = {n: f"{n} {summaries.get(n, '')}" for n in dag.nodes}
    all_ids, all_matrix = EM.build_embeddings(all_docs, dims=50)
    row_of = {name: index for index, name in enumerate(all_ids)}
    semantic = {
        name: (float(all_matrix[i, 0]), float(all_matrix[i, 1]))
        for name, i in row_of.items()
    }
    full_positions = render._read_positions_file(render._installed_path(dag))
    degree = dict(dag.degree())

    candidates = []
    for min_degree in (0, 1, 2, 3, 4, 5, 6):
        subset = [n for n in dag.nodes if degree[n] >= min_degree]
        positions = {n: full_positions[n] for n in subset if n in full_positions}
        if len(positions) < 40:
            continue
        radii = {n: 4.0 + 2.0 * float(np.sqrt(degree[n])) for n in positions}
        edges_list = [(a, b) for a, b in dag.edges() if a in positions and b in positions]
        labels = {n for n, _ in sorted(dag.degree, key=lambda p: -p[1])[:25] if n in positions}
        # Layout A: keep the coordinates imported from the full graph.
        # Layout B: recompute ForceAtlas2 on the subset only, which gives a tight
        # bounding box. Inherited coordinates keep the full graph's scale, so a
        # subset that is dense in the middle still zooms out to nothing --
        # measured as a nearest-neighbour distance of 0 px at 125 nodes.
        recomputed = render.forceatlas2_positions(
            dag.subgraph(list(positions)).copy(), max_iter=250
        )
        variants = [("inherited", positions)]
        if recomputed:
            variants.append(("recomputed", recomputed))
        semantic_subset = {n: semantic[n] for n in positions if n in semantic}
        if len(semantic_subset) == len(positions):
            variants.append(("semantic", semantic_subset))

        combinations = [(variant, layout_positions, robust)
                        for variant, layout_positions in variants
                        for robust in (False, True)]
        for variant, layout_positions, robust in combinations:
            quality, scaled_positions, scaled_radii = screen_quality(
                layout_positions, radii, edges_list, labels, robust=robust
            )
            variant = variant if not robust else variant + "*"
            candidates.append(
                {
                    "min_degree": min_degree,
                    "variant": variant,
                    "positions": layout_positions,
                    "scaled_positions": scaled_positions,
                    "scaled_radii": scaled_radii,
                    "edges": edges_list,
                    "labels": labels,
                    "quality": quality,
                }
            )

    print(f"{'min_deg':>7s} {'layout':>11s} {'nodes':>6s} {'overlap':>9s} {'nnd_px':>7s} "
          f"{'cross':>7s} {'labelcoll':>10s} {'verdict':>9s}")
    print("-" * 76)
    for item in candidates:
        q = item["quality"]
        verdict = "OK" if is_readable(q) else "crowded"
        print(f"{item['min_degree']:>7d} {item['variant']:>11s} {len(item['positions']):>6d} "
              f"{q['node_overlap']:>9.5f} {q['median_nnd']:>7.2f} "
              f"{q['edge_crossing_rate']:>7.4f} {q['label_collisions']:>10.4f} {verdict:>9s}")

    passing = [c for c in candidates if is_readable(c["quality"])]
    if passing:
        # Most nodes that still passes: keep as much of the graph as reads well.
        best = max(passing, key=lambda c: len(c["positions"]))
        why = "largest subset meeting all three thresholds"
    else:
        best = max(candidates, key=lambda c: c["quality"]["median_nnd"])
        why = "no subset met every threshold; took the best nearest-neighbour distance"

    print(f"\nchosen: min_degree={best['min_degree']}, layout {best['variant']} "
          f"({len(best['positions'])} nodes) — {why}")
    print(f"  overlap {best['quality']['node_overlap']:.5f} · nnd {best['quality']['median_nnd']:.2f}px"
          f" · label collisions {best['quality']['label_collisions']:.4f}")

    # --- embeddings + cluster-centre colours on the chosen subset ----------- #
    nodes = sorted(best["positions"])
    ids = nodes
    matrix = np.vstack([all_matrix[row_of[n]] for n in nodes])
    cluster_count = max(4, min(12, len(nodes) // 40))
    medoids = EM.k_medoids(matrix, k=cluster_count, seed=0)

    pca3 = matrix[:, :3]
    colours, membership = EM.colours_from_centres(pca3, medoids, sharpness=2.5)
    rgb_colours = EM.rgb_from_pca(matrix)

    pd.DataFrame(
        {
            "title": nodes,
            "colour_cluster": colours,
            "colour_rgb_pca": rgb_colours,
            "x": [best["scaled_positions"][n][0] for n in nodes],
            "y": [best["scaled_positions"][n][1] for n in nodes],
            "degree": [degree[n] for n in nodes],
            "dominant_centre": [nodes[int(row.argmax())] for row in membership],
        }
    ).to_csv(OUT / "nodes.csv", index=False)

    centres_rows = []
    for index, medoid_index in enumerate(medoids):
        dominant = int((membership.argmax(axis=1) == index).sum())
        centres_rows.append(
            {
                "medoid": nodes[medoid_index],
                "colour": EM.CENTRE_PALETTE[index % len(EM.CENTRE_PALETTE)],
                "dominant_members": dominant,
                "x": float(best["scaled_positions"][nodes[medoid_index]][0]),
                "y": float(best["scaled_positions"][nodes[medoid_index]][1]),
            }
        )
    centres = pd.DataFrame(centres_rows).sort_values("dominant_members", ascending=False)
    centres.to_csv(OUT / "centres.csv", index=False)
    print("\ncluster centres (dominant members):")
    for _, row in centres.iterrows():
        print(f"  {row['colour']}  {row['dominant_members']:4d}  {row['medoid']}")

    payload = {
        "min_degree": best["min_degree"],
        "layout": best["variant"],
        "quality": best["quality"],
        "thresholds": THRESHOLDS,
        "candidates": [
            {"min_degree": c["min_degree"], "layout": c["variant"], "nodes": len(c["positions"]),
             "quality": {k: float(v) for k, v in c["quality"].items()},
             "readable": bool(is_readable(c["quality"]))}
            for c in candidates
        ],
        "nodes": [
            {
                "title": n,
                "x": float(best["scaled_positions"][n][0]),
                "y": float(best["scaled_positions"][n][1]),
                "radius": float(best["scaled_radii"][n]),
                "colour": colours[i],
                "rgb_pca": rgb_colours[i],
                "degree": degree[n],
                "centre": nodes[int(membership[i].argmax())],
                "label": n if n in best["labels"] or degree[n] >= 8 else "",
            }
            for i, n in enumerate(nodes)
        ],
        "edges": [{"from": a, "to": b} for a, b in best["edges"]],
        "centres": centres_rows,
    }
    (OUT / "graph.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {OUT/'graph.json'} ({len(payload['nodes'])} nodes, {len(payload['edges'])} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
