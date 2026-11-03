"""Diagnose the live page: JS errors, console output, and whether motion runs."""
from __future__ import annotations
import sys
from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
errors: list[str] = []
console: list[str] = []

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    page = browser.new_page(viewport={"width": 1200, "height": 800})
    page.on("pageerror", lambda e: errors.append(f"PAGEERROR: {e}"))
    page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
    page.goto(url, wait_until="load", timeout=60_000)
    page.wait_for_timeout(6000)
    t1 = page.evaluate("window.__motionTicks")
    n1 = page.evaluate("window.__motionNodes")
    page.wait_for_timeout(1500)
    t2 = page.evaluate("window.__motionTicks")
    page.close()
    browser.close()

print(f"ticks: {t1} -> {t2}   | nodes animated: {n1}")
print(f"motion {'RUNNING' if (t2 or 0) > (t1 or 0) else 'NOT RUNNING'}")
for e in errors[:6]:
    print(e)
for c in console[:6]:
    print(c)
