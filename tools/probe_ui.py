"""Focused probe: style events, edges during drag, physics motion."""
from __future__ import annotations
import json, sys
from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    page = b.new_page(viewport={"width": 1400, "height": 900})
    errs: list[str] = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.goto(url, wait_until="load", timeout=60_000)
    page.wait_for_timeout(7000)

    print("physics object present:", page.evaluate("() => Boolean(window.__state().network.physics)"))

    # physics motion: positions must drift on their own
    a = page.evaluate("() => Object.fromEntries(window.__state().nodes.get().slice(0,30).map(n => { const p = window.__state().network.getPositions([n.id])[n.id]; return [n.id, [p.x, p.y]]; }))")
    page.wait_for_timeout(2000)
    c = page.evaluate("() => Object.fromEntries(window.__state().nodes.get().slice(0,30).map(n => { const p = window.__state().network.getPositions([n.id])[n.id]; return [n.id, [p.x, p.y]]; }))")
    moved = sum(1 for k in a if abs(c[k][0]-a[k][0]) + abs(c[k][1]-a[k][1]) > 1)
    print(f"physics moving: {moved}/30 nodes drifted in 2 s")

    # style patch
    before = page.evaluate("() => Object.fromEntries(window.__state().nodes.get().slice(0,30).map(n => [n.id, n.size]))")
    page.evaluate("""async () => fetch('/spec', {method: 'POST', headers: {'Content-Type': 'application/json'},
                       body: JSON.stringify({set: ['node.size=70 + 8 * degree']})})""")
    page.wait_for_timeout(1200)
    after = page.evaluate("() => Object.fromEntries(window.__state().nodes.get().slice(0,30).map(n => [n.id, n.size]))")
    changed = sum(1 for k in before if abs(after.get(k, 0) - before[k]) > 0.5)
    print(f"style events received: {page.evaluate('() => window.__styleEvents')}")
    print(f"sizes changed: {changed}/30")
    if changed == 0:
        payload = page.evaluate("() => window.__lastStyle && window.__lastStyle.nodes ? window.__lastStyle.nodes.slice(0,3) : null")
        print("last style payload sample:", json.dumps(payload))
    print("errors:", errs[:3])
    b.close()
