# 03 — The spine: nodes with degree ≥ 4

**Strategy.** Strip the leaves and look only at the connected backbone:
`filter.min_degree = 4`, colour by degree, width by in-proof mentions.

**What we try to show.** Whether the prerequisite graph has a readable structure
at all once the 50% of leaf nodes are removed — and whether that structure is a
"map" or just a dense hairball.

**What we do to the data and the graph.**
- `filter.min_degree = 4`: 1,560 → **301 nodes**, 1,945 → **535 edges**
- `node.size = 8 + 3 * degree`, colour = degree quintiles
- `edge_weight.minimum = 0.0` (nothing dropped here; the filter already did the work)

**Results.** A 301-node graph that can be read without zooming. The core is
dominated by `Mathematical induction`, `Brouwer fixed-point theorem`,
`Fundamental theorem of algebra`, `Banach–Alaoglu theorem` — analysis-flavoured
hubs rather than the classical foundations.

**Analysis.** That last point is the finding, not a detail. The spine is not
"the foundations of mathematics" — it is **whatever the Wikipedia proof sections
cite most often**, which biases hard towards analysis, topology and named
theorems. Foundations (set theory, logic, category theory) barely appear. This is
the same bias the critical-path measurement showed, and it is a property of the
extraction source, not of mathematics. Any claim of the form "this is the tree of
mathematics" made from this graph would be measuring Wikipedia's citation habits.
