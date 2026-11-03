"""Prove the viewer is animating by sampling node positions twice."""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"

SAMPLE = """
() => {
  const { nodes, network } = window.__state();
  const ids = nodes.get().slice(0, 60).map(n => n.id);
  const out = {};
  for (const [id, p] of Object.entries(network.getPositions(ids))) {
    out[id] = [+p.x.toFixed(3), +p.y.toFixed(3)];
  }
  return { ticks: window.__motionTicks, animated: window.__motionNodes, positions: out };
}
"""

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(url, wait_until="load", timeout=60_000)
    page.wait_for_timeout(4000)
    first = page.evaluate(SAMPLE)
    page.wait_for_timeout(1500)
    second = page.evaluate(SAMPLE)
    browser.close()

moved, total, biggest = 0, 0, 0.0
print(f"animation ticks: {first['ticks']} -> {second['ticks']} (nodes animated: {second['animated']})")
for node, (x, y) in first["positions"].items():
    if node not in second["positions"]:
        continue
    nx_, ny_ = second["positions"][node]
    d = ((nx_ - x) ** 2 + (ny_ - y) ** 2) ** 0.5
    total += 1
    biggest = max(biggest, d)
    if d > 0.5:
        moved += 1

print(f"sampled {total} nodes over 1.5 s")
print(f"nodes that moved (>0.5 px): {moved} ({moved / max(total, 1):.1%})")
print(f"largest displacement: {biggest:.1f} px")
print("-> the graph is alive" if moved else "-> static")
