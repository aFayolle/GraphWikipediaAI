"""Build a playable HTML animation of ForceAtlas2 opening the graph.

The point is to *watch* the layout work, not to receive a finished picture. So:

* Gephi's Toolkit writes a coordinate snapshot every N iterations.
* Every frame is drawn in ONE fixed coordinate system, taken from the final
  frame. If each frame were normalised to the canvas the expansion would be
  invisible -- the graph would look static while actually growing.
* Overlap is measured per frame, so "the repulsion opens it up" becomes a
  number that goes down rather than a claim.

Usage: PYTHONPATH=. python tools/make_animation.py [iterations] [stride]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from mathkg import config, dependency as D, edges as E
from mathkg import embedding as EM
from mathkg import render

OUT = config.PROJECT_ROOT / "reports" / "animation"
JAR = config.DATA_DIR / "tools" / "gephi-toolkit-all.jar"
JAVA = config.PROJECT_ROOT / "tools" / "GephiToolkitLayout.java"
CANVAS = (1600.0, 1000.0)


def run_gephi(gexf: Path, prefix: Path, iterations: int, stride: int) -> list[Path]:
    command = [
        "java", "-cp", str(JAR), str(JAVA), str(gexf), str(prefix),
        str(iterations), "true", "true", str(stride),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    for line in (result.stdout or "").splitlines():
        if line.startswith(("Gephi", "randomised", "wrote")):
            print(" ", line)
    if result.returncode != 0:
        raise SystemExit(result.stderr[-1500:])
    frames = sorted(
        prefix.parent.glob(prefix.name + ".*.tsv"),
        key=lambda p: int(re.search(r"\.(\d+)\.tsv$", p.name).group(1)),
    )
    return frames + [prefix]


def read_frame(path: Path) -> dict[str, tuple[float, float]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            out[parts[0]] = (float(parts[1]), float(parts[2]))
    return out


def main() -> int:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    min_degree = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    OUT.mkdir(parents=True, exist_ok=True)

    _, titles = E.load_summaries()
    dag = D.break_cycles(D.build_dag(D.load_edges(), titles=set(titles)))[0]

    # A size that can actually be watched: the full 1,560 is a stutter festival.
    keep = [n for n in dag.nodes if dag.degree(n) >= min_degree]
    subgraph = dag.subgraph(keep).copy()
    print(f"animating {subgraph.number_of_nodes()} nodes / {subgraph.number_of_edges()} edges")

    gexf = config.ARTIFACT_DIR / "gephi" / "anim_input.gexf"
    export_positions = render.forceatlas2_positions(subgraph, max_iter=50)
    from mathkg import export

    export.write_gexf(subgraph, gexf, positions=export_positions)

    frames_paths = run_gephi(gexf, OUT / "frame.tsv", iterations, stride)
    raw_frames = [read_frame(path) for path in frames_paths]
    nodes_sorted = sorted(subgraph.nodes)
    degree = dict(subgraph.degree())

    # one fixed coordinate system across every frame, from the final frame
    final = raw_frames[-1]
    xs = np.array([final[n][0] for n in nodes_sorted])
    ys = np.array([final[n][1] for n in nodes_sorted])
    lo_x, hi_x = np.percentile(xs, [1, 99])
    lo_y, hi_y = np.percentile(ys, [1, 99])
    scale = min((CANVAS[0] - 80) / ((hi_x - lo_x) or 1), (CANVAS[1] - 80) / ((hi_y - lo_y) or 1))

    def to_canvas(point):
        return (
            40 + (point[0] - lo_x) * scale,
            40 + (point[1] - lo_y) * scale,
        )

    radii = {n: 3.5 + 1.6 * float(np.sqrt(degree[n])) for n in nodes_sorted}
    index = {n: i for i, n in enumerate(nodes_sorted)}
    edges_ij = [[index[a], index[b]] for a, b in subgraph.edges()]

    # colours from the structural embedding, computed once
    spectral_ids, spectral = EM.spectral_embedding(subgraph, dims=30)
    row = {name: i for i, name in enumerate(spectral_ids)}
    matrix = np.vstack([spectral[row[n]] for n in nodes_sorted])
    medoids = EM.k_medoids(matrix, k=10, seed=0)
    colours, _ = EM.colours_from_centres(matrix[:, :3], medoids, sharpness=2.5)
    centres = [nodes_sorted[m] for m in medoids]

    labels = {n for n, _ in sorted(subgraph.degree, key=lambda p: -p[1])[:20]}

    payload_frames, overlap_series = [], []
    for path, frame in zip(frames_paths, raw_frames):
        iteration = int(re.search(r"\.(\d+)\.tsv$", path.name).group(1)) if ".tsv." in path.name else iterations
        canvas_positions = {n: to_canvas(frame[n]) for n in nodes_sorted if n in frame}
        quality = EM.layout_quality(canvas_positions, radii, list(subgraph.edges()), labels=labels)
        overlap_series.append(
            {"iteration": iteration,
             "overlap": round(quality["node_overlap"] * 100, 3),
             "nnd": round(quality["median_nnd"], 2)}
        )
        payload_frames.append(
            [[round(canvas_positions[n][0]), round(canvas_positions[n][1])] for n in nodes_sorted]
        )

    print("\n  iteration   overlap%   nnd(px)")
    for row_ in overlap_series:
        print(f"  {row_['iteration']:>9d} {row_['overlap']:>10.3f} {row_['nnd']:>9.2f}")
    first, last = overlap_series[0], overlap_series[-1]
    print(f"\nopening: overlap {first['overlap']:.2f}% -> {last['overlap']:.2f}%, "
          f"nnd {first['nnd']:.2f} -> {last['nnd']:.2f} px")

    payload = {
        "nodes": [
            {"r": round(radii[n], 1), "c": colours[i], "l": n if n in labels else "",
             "d": int(degree[n])}
            for i, n in enumerate(nodes_sorted)
        ],
        "edges": edges_ij,
        "frames": payload_frames,
        "series": overlap_series,
        "centres": centres,
        "iterations": iterations,
        "stride": stride,
    }
    (OUT / "animation.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {OUT/'animation.json'} ({len(payload_frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
