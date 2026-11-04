"""UI interaction tests against the live viewer, run in a real browser.

Written because the user reported that dragging nodes "works very badly" and I
had only ever taken screenshots. Screenshots cannot feel a drag; a mouse can.

Each test drives actual input events with Playwright and asserts on the state
vis-network reports back, so a broken interaction fails here instead of in the
user's hands.

Usage: PYTHONPATH=. python tools/ui_test.py [url]
"""

from __future__ import annotations

import json
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"

POSITIONS = """
() => {
  const { nodes, network } = window.__state();
  const out = {};
  for (const node of nodes.get()) {
    const p = network.getPositions([node.id])[node.id];
    if (p) out[node.id] = [p.x, p.y];
  }
  return out;
}
"""

DOM_OF = """
(id) => {
  const { network } = window.__state();
  const p = network.getPositions([id])[id];
  if (!p) return null;
  const d = network.canvasToDOM({ x: p.x, y: p.y });
  return [d.x, d.y];
}
"""

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))

        page.goto(URL, wait_until="load", timeout=60_000)
        page.wait_for_timeout(7000)

        # --- 1. the page actually rendered something ---------------------- #
        payload = page.evaluate(POSITIONS)
        check("page renders the graph", len(payload) > 50, f"{len(payload)} nodes with positions")
        check("no JavaScript errors on load", not errors, "; ".join(errors[:2]))

        # --- 2. clicking a node selects it -------------------------------- #
        target = max(payload, key=lambda node: 0)
        dom = page.evaluate(DOM_OF, target)
        if dom:
            page.mouse.click(dom[0], dom[1])
            page.wait_for_timeout(300)
            selected = page.evaluate("() => window.__state().network.getSelectedNodes()")
            check("click selects a node", bool(selected), f"selected {selected[:1]}")
        else:
            check("click selects a node", False, "could not locate the node in DOM")

        # --- 3. DRAGGING: does the node stay where it is put? ------------- #
        before = page.evaluate(POSITIONS)
        # Pick a node that is actually on screen and big enough to hit: taking the
        # alphabetically first id grabbed an off-canvas node, so the synthetic
        # mouse press landed on empty canvas ("moved 0 px").
        target, dom = page.evaluate("""() => {
          const { nodes, network } = window.__state();
          const ranked = nodes.get().slice().sort((a, b) => (b.degree || 0) - (a.degree || 0));
          for (const n of ranked) {
            const p = network.getPositions([n.id])[n.id];
            if (!p) continue;
            const d = network.canvasToDOM({ x: p.x, y: p.y });
            const underHud = d.x < 430 && d.y < 190;
            if (d.x > 70 && d.x < 1330 && d.y > 70 && d.y < 830 && !underHud) {
              return [n.id, [d.x, d.y]];
            }
          }
          return [null, null];
        }""")
        if dom:
            page.mouse.move(dom[0], dom[1])
            page.mouse.down()
            for step in range(1, 13):
                page.mouse.move(dom[0] + step * 15, dom[1] + step * 8)
                page.wait_for_timeout(16)
            page.mouse.up()
            page.wait_for_timeout(250)
            just_after = page.evaluate(POSITIONS)
            moved = abs(just_after.get(target, [0, 0])[0] - before[target][0])
            check("drag moves the node", moved > 20, f"moved {moved:.0f} px")

            # The defect was the node teleporting back to where it started.
            # With physics on a released node legitimately settles, so assert on
            # the real failure: it must not return to its original position.
            page.wait_for_timeout(1500)
            settled = page.evaluate(POSITIONS)
            start_x = before[target][0]
            now_x = settled.get(target, [start_x, 0])[0]
            dropped_x = just_after.get(target, [start_x, 0])[0]
            travelled = abs(dropped_x - start_x) or 1.0
            kept = abs(now_x - start_x) / travelled
            check(
                "dragged node is not teleported back",
                kept > 0.3,
                f"kept {kept:.0%} of the drag after settling ({travelled:.0f} px dragged)",
            )
        else:
            check("drag moves the node", False, "could not locate the node in DOM")
            check("dragged node stays put (no snap-back)", False, "skipped")

        # --- 4. zoom responds to the wheel -------------------------------- #
        scale_before = page.evaluate("() => window.__state().network.getScale()")
        page.mouse.move(700, 450)
        page.mouse.wheel(0, -240)
        page.wait_for_timeout(400)
        scale_after = page.evaluate("() => window.__state().network.getScale()")
        check("wheel zooms", abs(scale_after - scale_before) > 0.01,
              f"scale {scale_before:.2f} -> {scale_after:.2f}")

        # --- 5. a live spec patch changes node sizes ---------------------- #
        # Unmistakable value: a random size could coincide with the spec already
        # in place, which produced a false "0/40 changed" failure.
        sizes_before = page.evaluate(
            "() => Object.fromEntries(window.__state().nodes.get().slice(0, 40).map(n => [n.id, n.size]))"
        )
        page.evaluate(
            """async () => fetch('/spec', {method: 'POST',
                 headers: {'Content-Type': 'application/json'},
                 body: JSON.stringify({set: ['node.size=200 + 0 * degree']})})"""
        )
        sizes_after, events = {}, 0
        for _ in range(15):
            page.wait_for_timeout(400)
            sizes_after = page.evaluate(
                "() => Object.fromEntries(window.__state().nodes.get().slice(0, 40).map(n => [n.id, n.size]))"
            )
            events = page.evaluate("() => window.__styleEvents")
            if sizes_after and all(abs(v - 200) < 0.5 for v in sizes_after.values()):
                break
        patched = sum(1 for v in sizes_after.values() if abs(v - 200) < 0.5)
        check("live patch resizes nodes", patched > 10,
              f"{patched}/{len(sizes_after)} nodes now 200 px, {events} style event(s)")
        # put the real expression back
        page.evaluate(
            """async () => fetch('/spec', {method: 'POST',
                 headers: {'Content-Type': 'application/json'},
                 body: JSON.stringify({set: ['node.size=9 + 11 * sqrt(degree)']})})"""
        )
        page.wait_for_timeout(600)

        # --- 6. animation playback moves nodes ---------------------------- #
        snapshot_a = page.evaluate(POSITIONS)
        page.evaluate(
            """async () => fetch('/animate', {method: 'POST',
                 headers: {'Content-Type': 'application/json'},
                 body: JSON.stringify({iterations: 200, stride: 40, scaling: 10})})"""
        )
        page.wait_for_timeout(3500)
        snapshot_b = page.evaluate(POSITIONS)
        moved_nodes = sum(
            1 for node, xy in snapshot_a.items()
            if node in snapshot_b and abs(snapshot_b[node][0] - xy[0]) + abs(snapshot_b[node][1] - xy[1]) > 5
        )
        check("animation moves nodes", moved_nodes > 20, f"{moved_nodes} nodes moved")

        check("no JavaScript errors during interaction", not errors, "; ".join(errors[:2]))
        page.close()
        browser.close()

    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} UI checks passed")
    if failed:
        print("failing:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
