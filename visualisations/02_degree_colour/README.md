# 02 — Colour by degree quintile, size ∝ log(sitelinks)

**Strategy.** Stop encoding "importance" once and start encoding two different
things: **degree** (how embedded a notion is in the dependency web) as colour,
**sitelinks** (how encyclopaedically established it is) as size.

**What we try to show.** That these two measures disagree. A notion can be
heavily depended upon while being obscure, and famous while peripheral.

**What we do to the data and the graph.**
- `node.color_by = {column: degree, palette: 5 steps dark→orange}`, binned by
  equal-count quantiles (robust to the degree-1 majority)
- `node.size = 4 + 30 * log(1 + sitelinks)`
- all 1,945 edges kept; edges thin and faint so the nodes carry the reading

**Results.** The layout separates the two axes visibly: high-degree nodes (yellow
/ orange) sit in a dense core around `Mathematical induction`, `Structure theorem
for finitely generated…`, `Donaldson's theorem`; the periphery is a ring of
dark-teal degree-1 nodes. But **the ring is not "unimportant maths" — it is
degree 1**, and 781 of 1,560 nodes (50%) are leaves.

**Analysis.** The interesting outliers are the nodes that are *big but pale*:
large sitelinks, low degree. Those are the encyclopaedically famous notions that
this dependency graph barely touches — a direct measure of the graph's coverage
gap, not of the notion's importance. The reverse (small and orange) is the more
useful signal: heavily depended upon, barely known. Those are the load-bearing
technical lemmas.
