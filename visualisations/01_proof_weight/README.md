# 01 — Edges weighted by proof mentions

**Strategy.** Keep only dependencies that a proof actually leans on. An edge is
kept when the target is mentioned *inside a proof/derivation section* of the
source article, weighted by how often it appears there.

**What we try to show.** Which dependencies are load-bearing, and which are just
"See also" proximity. If the prerequisite graph is built from citations, a large
share of edges should turn out to be incidental.

**What we do to the data and the graph.**
- `edge_weight.formula = occurrences * in_proof` (from `data/raw/mentions.csv`)
- `edge_weight.minimum = 0.5`, so an edge needs at least one in-proof mention
- everything else untouched: 1,560 nodes, 1,945 prerequisite edges, imported layout

**Results.** 1,155 of 1,945 edges survive — **790 edges (41%) are dependencies
that are never cited inside a proof.** Width is proportional to the mention count,
so a thin line is a weak claim even when it survives.

**Analysis.** 41% is a lot, and it matches what the earlier audits found: the
phrase extractor contributes 141 edges of which some are generic ("follows from
mathematics"), and Wikidata's `part_of` edges are classification rather than
prerequisite. What remains is the more defensible core — `Hahn–Banach ← Zorn`,
`FTC ← mean value theorem` and similar survive on multiple in-proof mentions,
while taxonomic edges drop out. So this is a cheap and useful *filter*, not a
precision guarantee: an edge that lives in a proof can still be wrong (the
`quadratic reciprocity → Pythagorean theorem` inversion survives it).

**Watch it live.** `run.py --spec visualisations/01_proof_weight/spec.yaml serve`
