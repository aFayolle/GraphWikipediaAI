"""Snapshot the LIVE server, so what I show you is what you see.

Why this exists: the live tab, the emailed report and the emailed animation were
three different pipelines with their own node subsets, layouts and palettes. No
wonder they looked unrelated. This tool takes the running server as the single
source of truth, and records it verbatim: the JSON it serves, the spec it holds,
readability measured on those exact coordinates, and a Playwright capture of the
page itself.
"""
from __future__ import annotations

import base64, json, subprocess, sys, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from mathkg import config, embedding as EM

OUT = config.PROJECT_ROOT / "reports" / "live_snapshot"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


def fetch(path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=60) as response:
        return json.loads(response.read())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    state = fetch("/state")
    graph = fetch("/graph")

    positions, radii = {}, {}
    for node in graph["nodes"]:
        if node["x"] is None:
            continue
        positions[node["id"]] = (node["x"], node["y"])
        radii[node["id"]] = max(1.0, node["size"])
    edges = [(e["from"], e["to"]) for e in graph["edges"]]
    labels = {n["id"] for n in graph["nodes"] if n["label"]}

    quality = EM.layout_quality(positions, radii, edges, labels=labels) if len(positions) > 2 else {}

    # capture the page exactly as the browser shows it
    png = OUT / "live.png"
    subprocess.run(
        [sys.executable, "tools/shot.py", f"http://127.0.0.1:{PORT}/?static",
         str(png), "10"],
        check=False, timeout=300,
    )
    encoded = base64.b64encode(png.read_bytes()).decode("ascii") if png.exists() else ""

    spec = state["spec"]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = "".join(
        f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"
        for key, value in (quality or {}).items() if isinstance(value, float)
    )
    document = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8" />
<title>MathKG — live server snapshot</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ margin:0; background:#05070c; color:#e8eef8;
        font-family: ui-sans-serif, system-ui, sans-serif; }}
 .wrap {{ max-width:1680px; margin:0 auto; padding:26px 30px 50px; }}
 h1 {{ font-size:20px; margin:0 0 4px; }}
 h2 {{ font-size:13px; margin:26px 0 8px; color:#9fb6d8; text-transform:uppercase;
       letter-spacing:.08em; }}
 p {{ font-size:13.5px; line-height:1.6; color:#c6d4e8; }}
 .dim {{ color:#7f93b0; }}
 img {{ width:100%; display:block; border-radius:12px;
        border:1px solid rgba(120,170,255,.12); }}
 table {{ border-collapse:collapse; font-size:13px; }}
 td {{ padding:4px 16px; border-bottom:1px solid rgba(120,170,255,.10); }}
 code {{ background:#101a2b; padding:1px 6px; border-radius:5px; }}
 pre {{ background:#0b1425; padding:12px 14px; border-radius:8px; font-size:12.5px;
        overflow-x:auto; color:#c6d4e8; }}
</style></head><body><div class="wrap">
<h1>Live server snapshot — {stamp}</h1>
<p class="dim">This is not a re-render. It is the state of the running viewer on
port {PORT}: the same JSON it served, the same spec it held, measured on those
coordinates, plus a capture of the page itself.</p>
<h2>The page, exactly as the browser draws it</h2>
<img src="data:image/png;base64,{encoded}" alt="live server capture" />
<h2>Spec the server was holding</h2>
<pre>{json.dumps(spec, indent=2)}</pre>
<h2>Readability of those coordinates, measured</h2>
<p class="dim">{len(positions)} nodes with positions · {len(edges)} edges ·
{len(labels)} labelled.</p>
<table>{rows}</table>
<h2>Why this differed from the emails</h2>
<p>The live tab, the emailed report and the emailed animation were <b>three
separate pipelines</b>: different node subsets (779 / 194 / 474), different
layouts (imported / inherited / Toolkit frames) and different colour algorithms
(degree quintile / TF-IDF clusters / spectral clusters). Nothing was shared, so
nothing matched. This snapshot removes the divergence: it records the server as
the single source of truth.</p>
</div></body></html>"""
    target = OUT / "live.html"
    target.write_text(document, encoding="utf-8")
    print(f"{target} ({target.stat().st_size/1024:.0f} KB)")
    if quality:
        print(f"live nodes {len(positions)} · overlap {quality['node_overlap']*100:.3f}%"
              f" · nnd {quality['median_nnd']:.2f}px"
              f" · label collisions {quality['label_collisions']*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
