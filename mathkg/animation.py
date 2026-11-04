"""ForceAtlas2 frames, produced by Gephi's Toolkit and scaled for the browser.

Used by the live server, so the physics the user watches *is* the same pipeline
as the tab: one graph, one renderer, one set of coordinates.

The scaling ratio is exposed because it is the direct lever on repulsion: a
higher value spreads the settled graph, which is the answer to a crowded centre.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import numpy as np

from . import config, export, render

log = logging.getLogger(__name__)

JAR = config.DATA_DIR / "tools" / "gephi-toolkit-all.jar"
JAVA = config.PROJECT_ROOT / "tools" / "GephiToolkitLayout.java"
WORK = config.ARTIFACT_DIR / "animation"
CANVAS = (1600.0, 1000.0)


def _read_frame(path: Path) -> dict[str, tuple[float, float]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                out[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue
    return out


def build_frames(
    graph,
    iterations: int = 400,
    stride: int = 40,
    scaling: float = 10.0,
) -> dict:
    """Run Gephi's FA2 on ``graph`` and return frames scaled to the canvas."""
    if not JAR.exists():
        raise FileNotFoundError(f"missing {JAR}")
    WORK.mkdir(parents=True, exist_ok=True)
    gexf = WORK / "input.gexf"
    prefix = WORK / "frame.tsv"
    for stale in WORK.glob("frame.tsv*"):
        stale.unlink()

    seed_positions = render.forceatlas2_positions(graph, max_iter=40)
    export.write_gexf(graph, gexf, positions=seed_positions)

    command = ["java", "-cp", str(JAR), str(JAVA), str(gexf), str(prefix),
               str(iterations), "true", "true", str(stride), str(scaling)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    for line in (result.stdout or "").splitlines():
        if line.startswith(("Gephi", "randomised")):
            log.info("%s", line)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1200:])

    paths = sorted(
        WORK.glob("frame.tsv.*.tsv"),
        key=lambda p: int(re.search(r"\.(\d+)\.tsv$", p.name).group(1)),
    ) + [prefix]
    raw = [_read_frame(path) for path in paths if path.exists()]
    if not raw:
        raise RuntimeError("Gephi produced no frames")

    nodes = sorted(node for node in graph.nodes if node in raw[-1])
    final = raw[-1]
    xs = np.array([final[n][0] for n in nodes])
    ys = np.array([final[n][1] for n in nodes])
    lo_x, hi_x = np.percentile(xs, [1, 99])
    lo_y, hi_y = np.percentile(ys, [1, 99])
    scale = min((CANVAS[0] - 80) / ((hi_x - lo_x) or 1), (CANVAS[1] - 80) / ((hi_y - lo_y) or 1))

    frames = []
    for path, snapshot in zip(paths, raw):
        match = re.search(r"\.(\d+)\.tsv$", path.name)
        iteration = int(match.group(1)) if match else iterations
        frames.append(
            {
                "iteration": iteration,
                "points": [
                    [round(40 + (snapshot[n][0] - lo_x) * scale),
                     round(40 + (snapshot[n][1] - lo_y) * scale)]
                    for n in nodes
                    if n in snapshot
                ],
            }
        )
    log.info("animation: %d frames for %d nodes (scaling %.1f)", len(frames), len(nodes), scaling)
    return {"nodes": nodes, "frames": frames, "scaling": scaling, "iterations": iterations}
