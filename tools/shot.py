"""Screenshot the live viewer with Playwright's bundled Chromium.

This is the channel that was missing: the snap `firefox` cannot run here and
there is no system screenshot tool, but Playwright ships its own headless
Chromium which needs neither a display nor root. So the picture I look at is the
same page the user is looking at -- vis-network, physics, arrows, fonts -- rather
than a matplotlib re-rendering of the same data.

Usage::

    python tools/shot.py [url] [out.png] [settle_seconds] [freeze]

A continuously animating page cannot be screenshotted reliably (Playwright
times out waiting for a stable frame), so ``freeze`` calls the page's
``stopMotion()`` first. To *prove* motion, read positions twice instead --
see ``tools/motion_probe.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/artifacts/browser.png")
    settle = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    freeze = len(sys.argv) > 4 and sys.argv[4].lower() in {"freeze", "1", "true"}

    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            # --no-sandbox is required in this container; there is no user namespace.
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(url, wait_until="load", timeout=60_000)
        # Let vis-network's physics settle before capturing.
        page.wait_for_timeout(int(settle * 1000))
        if freeze:
            page.evaluate("window.stopMotion && window.stopMotion()")
            page.wait_for_timeout(400)
        page.screenshot(path=str(out), timeout=90_000, animations="disabled")
        browser.close()

    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
