"""Turn reports/animation/animation.json into a playable HTML page.

Canvas + requestAnimationFrame, no libraries: the file must survive email. The
frames come from Gephi's own ForceAtlas2, so what plays is the real algorithm
opening the graph, not a browser approximation of it.

Acceptance: the page plays unattended, exposes window.seek(i) so a single frame
can be screenshotted, and prints per-frame overlap so the "opening" is a number
and not an impression.
"""

from __future__ import annotations

import json

from mathkg import config

SOURCE = config.PROJECT_ROOT / "reports" / "animation" / "animation.json"
TARGET = config.PROJECT_ROOT / "reports" / "animation" / "mathkg_forceatlas2.html"

STYLE = """
  :root { color-scheme: dark; }
  body { margin: 0; background: #05070c; color: #e8eef8;
         font-family: ui-sans-serif, system-ui, sans-serif; }
  .wrap { max-width: 1680px; margin: 0 auto; padding: 26px 30px 56px; }
  h1 { font-size: 21px; margin: 0 0 6px; }
  h2 { font-size: 14px; margin: 30px 0 10px; color: #9fb6d8; font-weight: 600;
       text-transform: uppercase; letter-spacing: .08em; }
  p, li { font-size: 13.5px; line-height: 1.65; color: #c6d4e8; }
  .dim { color: #7f93b0; }
  .stage { border-radius: 14px; overflow: hidden;
           border: 1px solid rgba(120,170,255,.12);
           background: radial-gradient(circle at 50% 45%, #16294a 0%, #0b1425 40%, #05070c 80%); }
  canvas { display: block; width: 100%; height: auto; }
  .bar { display: flex; gap: 10px; align-items: center; margin: 12px 0 6px; flex-wrap: wrap; }
  button { background: #16233a; color: #dbe7f8; border: 1px solid rgba(120,170,255,.25);
           border-radius: 8px; padding: 7px 14px; font-size: 13px; cursor: pointer; }
  button:hover { background: #1e2f4d; }
  input[type=range] { width: 240px; }
  .readout { font-variant-numeric: tabular-nums; font-size: 13px; color: #9fb6d8; }
  table { border-collapse: collapse; font-size: 12.5px; }
  th, td { padding: 4px 14px; text-align: right;
           border-bottom: 1px solid rgba(120,170,255,.10); }
  th:first-child, td:first-child { text-align: left; }
  .note { border-left: 2px solid #e76f51; padding-left: 14px; margin: 16px 0; }
  code { background: #101a2b; padding: 1px 6px; border-radius: 5px; font-size: 12.5px; }
"""

SCRIPT = """
const DATA = __PAYLOAD__;
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const readout = document.getElementById("readout");
const speed = document.getElementById("speed");
const radius = DATA.nodes.map(n => n.r);
const colour = DATA.nodes.map(n => n.c);
const label = DATA.nodes.map(n => n.l);
let frame = 0, alpha = 0, playing = true, last = performance.now();

function draw() {
  const a = DATA.frames[frame];
  const b = DATA.frames[Math.min(frame + 1, DATA.frames.length - 1)];
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 0.7;
  ctx.strokeStyle = "rgba(120,150,200,0.18)";
  ctx.beginPath();
  for (const e of DATA.edges) {
    const i = e[0], j = e[1];
    ctx.moveTo(a[i][0] + (b[i][0] - a[i][0]) * alpha, a[i][1] + (b[i][1] - a[i][1]) * alpha);
    ctx.lineTo(a[j][0] + (b[j][0] - a[j][0]) * alpha, a[j][1] + (b[j][1] - a[j][1]) * alpha);
  }
  ctx.stroke();

  for (let i = 0; i < a.length; i++) {
    const x = a[i][0] + (b[i][0] - a[i][0]) * alpha;
    const y = a[i][1] + (b[i][1] - a[i][1]) * alpha;
    ctx.beginPath();
    ctx.arc(x, y, radius[i], 0, 6.2832);
    ctx.fillStyle = colour[i];
    ctx.fill();
    if (label[i]) {
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#05070c";
      ctx.strokeText(label[i], x, y - radius[i] - 4);
      ctx.fillStyle = "#eaf2ff";
      ctx.fillText(label[i], x, y - radius[i] - 4);
    }
  }

  const at = Math.min(Math.round(frame + alpha), DATA.series.length - 1);
  const s = DATA.series[at];
  readout.textContent = "iteration " + s.iteration + " · overlap " + s.overlap
                      + "% · nearest neighbour " + s.nnd + " px";
}

function tick(now) {
  const dt = Math.min(64, now - last); last = now;
  if (playing) {
    alpha += dt / (900 / Number(speed.value));
    if (alpha >= 1) { alpha = 0; frame = (frame + 1) % Math.max(1, DATA.frames.length - 1); }
  }
  draw();
  requestAnimationFrame(tick);
}

window.seek = i => { frame = i; alpha = 0; playing = false; draw(); };
document.getElementById("play").onclick = ev => {
  playing = !playing; ev.target.textContent = playing ? "pause" : "play";
};
document.getElementById("restart").onclick = () => { frame = 0; alpha = 0; playing = true; };
if (location.search.includes("static")) {
  playing = false;
  const at = Number(new URLSearchParams(location.search).get("frame") || 0);
  frame = Math.min(at, DATA.frames.length - 1);
  draw();
} else {
  requestAnimationFrame(tick);
}
"""


def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    series = payload["series"]
    rows = "\n".join(
        f"<tr><td>{row['iteration']}</td><td>{row['overlap']}</td><td>{row['nnd']}</td></tr>"
        for row in series
    )
    first, last = series[0], series[-1]
    drop = (first["overlap"] - last["overlap"]) / max(1e-9, first["overlap"]) * 100

    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<title>MathKG — ForceAtlas2 opening the graph</title>
<style>{STYLE}</style></head><body><div class="wrap">

<h1>ForceAtlas2, running — the graph opening up</h1>
<p class="dim">{len(payload['nodes'])} nodes · {len(payload['edges'])} edges ·
{len(payload['frames'])} frames · {payload['iterations']} iterations of Gephi's own
ForceAtlas2, {payload['stride']} iterations per frame. All frames are drawn in one
fixed coordinate system, so the expansion you see is real motion rather than a
re-fit of the view each time.</p>

<div class="stage"><canvas id="c" width="1600" height="1000"></canvas></div>

<div class="bar">
  <button id="play">pause</button>
  <button id="restart">restart</button>
  <label class="readout">speed <input id="speed" type="range" min="1" max="12" value="4"></label>
  <span class="readout" id="readout"></span>
</div>

<h2>Does the repulsion actually open it? — measured per frame</h2>
<p>Node overlap and nearest-neighbour distance, computed on every frame in canvas
pixels. Overlap falls by <b>{drop:.0f}%</b> between the first and last frame,
almost all of it in the first ~200 iterations, which is ForceAtlas2 converging.</p>
<table><tr><th>iteration</th><th>node overlap %</th><th>median nnd (px)</th></tr>
{rows}
</table>

<div class="note">
<p><b>What this shows, and what it does not.</b> The nodes genuinely push each
other apart — overlap goes from {first['overlap']}% to {last['overlap']}%. What it
does not show is a clean map: the nearest-neighbour distance only moves from
{first['nnd']} px to {last['nnd']} px, so on a 1600×1000 canvas the settled graph
remains denser than one node diameter. Watching it open is the right way to see
the algorithm work; it is not by itself a cure for density.</p>
</div>

<h2>Reproduce</h2>
<p><code>PYTHONPATH=. python tools/make_animation.py 600 40 3</code> runs the
Toolkit and writes the frames; <code>python tools/make_animation_html.py</code>
writes this page. Add <code>?static&amp;frame=12</code> to freeze on one frame.</p>

</div>
<script>{SCRIPT.replace('__PAYLOAD__', json.dumps(payload))}</script>
</body></html>
"""
    TARGET.write_text(document, encoding="utf-8")
    print(f"{TARGET} ({TARGET.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
