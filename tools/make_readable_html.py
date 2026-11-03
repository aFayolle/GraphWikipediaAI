"""Render reports/readable/graph.json into one self-contained HTML.

SVG built in Python, no JS libraries: the file has to survive email. A small
script switches between the two colouring strategies the brief asked for
(cluster-centre blend vs PCA->RGB), so they can be compared in place.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from mathkg import config

SOURCE = config.PROJECT_ROOT / "reports" / "readable" / "graph.json"
TARGET = config.PROJECT_ROOT / "reports" / "readable" / "mathkg_readable.html"

WIDTH, HEIGHT = 1600, 1000
PAD = 40


def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    nodes = payload["nodes"]
    edges = payload["edges"]
    centres = payload["centres"]
    quality = payload["quality"]
    thresholds = payload["thresholds"]

    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = (max_x - min_x) or 1.0
    span_y = (max_y - min_y) or 1.0
    scale = min((WIDTH - 2 * PAD) / span_x, (HEIGHT - 2 * PAD) / span_y)

    def place(node):
        # Coordinates arrive already scaled to the canvas and clamped, so the
        # second scaling must not shrink them again; it only fits the padding.
        return (
            PAD + (node["x"] - min_x) * scale,
            PAD + (node["y"] - min_y) * scale,
        )

    by_title = {n["title"]: n for n in nodes}

    edge_parts = []
    for edge in edges:
        a, b = by_title.get(edge["from"]), by_title.get(edge["to"])
        if not a or not b:
            continue
        x1, y1 = place(a)
        x2, y2 = place(b)
        edge_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" />'
        )

    node_parts, label_parts = [], []
    for node in nodes:
        x, y = place(node)
        radius = max(3.0, min(22.0, node["radius"] * 0.55))
        node_parts.append(
            f'<circle class="n" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
            f'data-cluster="{node["colour"]}" data-pca="{node["rgb_pca"]}" '
            f'fill="{node["colour"]}"><title>{html.escape(node["title"])} '
            f'— degree {node["degree"]}</title></circle>'
        )
        if node["label"]:
            label_parts.append(
                f'<text x="{x:.1f}" y="{y - radius - 4:.1f}">{html.escape(node["label"])}</text>'
            )

    centre_parts = []
    for index, centre in enumerate(centres, start=1):
        node = by_title.get(centre["medoid"])
        if not node:
            continue
        x, y = place(node)
        centre_parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="none" '
            f'stroke="{centre["colour"]}" stroke-width="2.5" />'
            f'<text class="cn" x="{x:.1f}" y="{y - 16:.1f}">{index}</text>'
        )

    legend = "\n".join(
        f'<li><span class="sw" style="background:{c["colour"]}"></span>'
        f'<b>{index}.</b> {html.escape(c["medoid"])} '
        f'<span class="dim">({c["dominant_members"]} nodes)</span></li>'
        for index, c in enumerate(centres, start=1)
    )

    candidate_rows = "\n".join(
        f'<tr class="{"ok" if c["readable"] else ""}">'
        f'<td>{c["min_degree"]}</td><td>{c["layout"]}</td><td>{c["nodes"]}</td>'
        f'<td>{c["quality"]["node_overlap"] * 100:.3f}%</td>'
        f'<td>{c["quality"]["median_nnd"]:.2f}</td>'
        f'<td>{c["quality"]["edge_crossing_rate"] * 100:.2f}%</td>'
        f'<td>{c["quality"]["label_collisions"] * 100:.2f}%</td>'
        f'<td>{"readable" if c["readable"] else "crowded"}</td></tr>'
        for c in payload["candidates"]
    )

    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<title>MathKG — a readable dependency graph</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; background: #05070c; color: #e8eef8;
         font-family: ui-sans-serif, system-ui, sans-serif; }}
  .wrap {{ max-width: 1680px; margin: 0 auto; padding: 28px 32px 60px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  h2 {{ font-size: 15px; margin: 34px 0 10px; color: #9fb6d8; font-weight: 600;
        text-transform: uppercase; letter-spacing: .08em; }}
  p, li {{ font-size: 13.5px; line-height: 1.65; color: #c6d4e8; }}
  .dim {{ color: #7f93b0; }}
  .stage {{ position: relative; border-radius: 14px; overflow: hidden;
            background: radial-gradient(circle at 50% 42%, #16294a 0%, #0b1425 40%, #05070c 80%);
            border: 1px solid rgba(120,170,255,.12); }}
  svg {{ display: block; width: 100%; height: auto; }}
  .stage {{ max-height: 78vh; overflow: auto; }}   /* pan by scrolling */
  .zoom {{ position: absolute; right: 14px; top: 14px; display: flex; gap: 6px; }}
  .zoom button {{ padding: 4px 10px; font-size: 14px; }}
  svg line {{ stroke: #6f8bb5; stroke-opacity: .16; stroke-width: .7; }}
  svg circle.n {{ stroke: #05070c; stroke-width: .8; }}
  svg text {{ fill: #eaf2ff; font-size: 11px; text-anchor: middle;
              paint-order: stroke; stroke: #05070c; stroke-width: 3px; }}
  svg text.cn {{ font-size: 12px; font-weight: 700; stroke-width: 3px; }}
  table {{ border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ padding: 5px 12px; text-align: right;
            border-bottom: 1px solid rgba(120,170,255,.10); }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  tr.ok td {{ color: #8ac926; }}
  .legend {{ list-style: none; padding: 0; display: grid;
             grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 6px 22px; }}
  .legend li {{ font-size: 13px; }}
  .sw {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px;
         margin-right: 7px; vertical-align: -1px; }}
  .controls {{ margin: 12px 0 18px; }}
  button {{ background: #16233a; color: #dbe7f8; border: 1px solid rgba(120,170,255,.25);
            border-radius: 8px; padding: 7px 14px; font-size: 13px; cursor: pointer; }}
  button.active {{ background: #2a9d8f; border-color: #2a9d8f; color: #04121a; font-weight: 600; }}
  .note {{ border-left: 2px solid #e76f51; padding-left: 14px; margin: 18px 0; }}
  code {{ background: #101a2b; padding: 1px 6px; border-radius: 5px; font-size: 12.5px; }}
</style></head>
<body><div class="wrap">

<h1>MathKG — a readable dependency graph</h1>
<p class="dim">{len(nodes)} notions · {len(edges)} prerequisite edges ·
coloured from a TF-IDF/SVD embedding of each article's title and lead summary ·
positions measured for readability, not chosen by eye.</p>

<h2>Why the previous picture was unreadable — measured</h2>
<p>On a 1600×1000 canvas the full graph puts the <b>median nearest neighbour at
1.13 px</b>: nodes are closer than their own radius, so no colouring or styling
can rescue it. The fix is density, not decoration. Every candidate below was
scaled to the canvas and scored on four counts:</p>
<table>
<tr><th>min degree</th><th>layout</th><th>nodes</th><th>node overlap</th>
<th>median nnd (px)</th><th>edge crossings</th><th>label collisions</th><th>verdict</th></tr>
{candidate_rows}
</table>
<p class="dim">Thresholds for "readable": overlap &lt; {thresholds['node_overlap'] * 100:.1f}% ·
nnd &gt; {thresholds['median_nnd']:.1f} px · label collisions &lt;
{thresholds['label_collisions'] * 100:.0f}%.</p>

<h2>The chosen view</h2>
<p>min degree <b>{payload['min_degree']}</b> · {payload['layout']} coordinates ·
node overlap <b>{quality['node_overlap'] * 100:.3f}%</b> ·
median nearest neighbour <b>{quality['median_nnd']:.2f} px</b> ·
edge crossings <b>{quality['edge_crossing_rate'] * 100:.2f}%</b> ·
label collisions <b>{quality['label_collisions'] * 100:.2f}%</b>.</p>

<div class="controls">
  <button id="b-cluster" class="active">colour: distance to cluster centres</button>
  <button id="b-pca">colour: PCA → RGB</button>
  <button id="b-toggle-labels">toggle labels</button>
</div>

<div class="stage">
<svg viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <g>{"".join(edge_parts)}</g>
  <g>{"".join(node_parts)}</g>
  <g id="labels">{"".join(label_parts)}</g>
  <g>{"".join(centre_parts)}</g>
</svg>
</div>

<h2>Cluster centres (k-medoids on the 50-D embedding)</h2>
<ul class="legend">{legend}</ul>
<p class="dim">Circled numbers on the graph mark the medoid of each cluster.
Colour is a soft blend of the three nearest centre colours, weighted by
exp(−d²/σ), so a node between two mathematical areas gets a between colour
instead of being forced into one.</p>

<h2>How the colour was built</h2>
<ol>
  <li><b>Embed</b>: TF-IDF over title + lead summary (sublinear tf, idf, L2
      normalised), then a truncated SVD to 50 dimensions — LSA. Offline, no model
      download.</li>
  <li><b>Centres</b>: k-medoids (PAM-lite, 25 iterations) on those 50 dimensions,
      so a centre is an actual article rather than an average.</li>
  <li><b>Colour</b>: for each node, distances to all centres →
      exp(−d²/σ) weights → keep the 3 strongest → blend their palette colours.</li>
  <li><b>Compare</b>: the button above also offers the plain PCA→RGB projection
      from the brief, so the two strategies can be judged side by side.</li>
</ol>

<div class="note">
<p><b>Honest caveats.</b> The embedding is bag-of-words over short texts: it
captures vocabulary, not mathematical structure, so two unrelated notions that
share boilerplate ("theorem", "proof", "group") will sit close. The clusters are
therefore <i>lexical</i> families, not a classification of mathematics. And the
positions are inherited from ForceAtlas2 on the full graph — the measurement
chose the node subset, not a new layout.</p>
</div>

<h2>Reproduce</h2>
<p><code>PYTHONPATH=. python tools/make_readable.py</code> regenerates the
metrics and <code>graph.json</code>; this page is written by
<code>tools/make_readable_html.py</code>.</p>

</div>
<script>
const nodes = Array.from(document.querySelectorAll("circle.n"));
const labels = document.getElementById("labels");
const bCluster = document.getElementById("b-cluster");
const bPca = document.getElementById("b-pca");
const bLabels = document.getElementById("b-toggle-labels");

bCluster.onclick = () => {{
  nodes.forEach(n => n.setAttribute("fill", n.dataset.cluster));
  bCluster.classList.add("active"); bPca.classList.remove("active");
}};
bPca.onclick = () => {{
  nodes.forEach(n => n.setAttribute("fill", n.dataset.pca));
  bPca.classList.add("active"); bCluster.classList.remove("active");
}};
bLabels.onclick = () => {{
  labels.style.display = labels.style.display === "none" ? "" : "none";
}};
</script>
</body></html>
"""
    TARGET.write_text(document, encoding="utf-8")
    print(f"{TARGET} ({TARGET.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
