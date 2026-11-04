"""Trigger the FA2 animation in a real browser and capture the settled result."""
from __future__ import annotations
import json, sys
from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
out = sys.argv[2] if len(sys.argv) > 2 else "reports/live_snapshot/after_anim.png"
scaling = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{url}/?static", wait_until="load", timeout=60_000)
    page.wait_for_timeout(6000)
    reply = page.evaluate(
        """async (scaling) => {
             const r = await fetch('/animate', {method: 'POST',
               headers: {'Content-Type': 'application/json'},
               body: JSON.stringify({iterations: 400, stride: 40, scaling})});
             return await r.json();
           }""",
        scaling,
    )
    print("server:", json.dumps(reply))
    page.wait_for_timeout(6000)          # playback is ~11 frames x 220 ms
    page.screenshot(path=out, timeout=90_000, animations="disabled")
    print("captured", out, "errors:", errors[:3])
    page.close(); browser.close()
