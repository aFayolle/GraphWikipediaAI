# Scoping report — what counts as a node

Corpus: **5,968 items**. Produced by `run.py scope`; every figure is
reproducible from the `P31`/`P279` data cached under `data/cache/wdqs_scope/`.

## 1. Type coverage

| kind | items | share |
|---|---:|---:|
| concept | 3,009 | 50.4% |
| object | 2,460 | 41.2% |
| (no P31) | 1,950 | 32.7% |
| theorem | 1,453 | 24.3% |
| (P31 but no rule matched) | 693 | 11.6% |
| expression | 433 | 7.3% |
| problem | 183 | 3.1% |
| structure | 176 | 2.9% |
| field of study | 111 | 1.9% |
| academic discipline | 102 | 1.7% |
| property | 98 | 1.6% |
| number | 98 | 1.6% |
| lemma | 85 | 1.4% |
| field | 82 | 1.4% |
| algorithm | 72 | 1.2% |
| conjecture | 70 | 1.2% |
| constant | 26 | 0.4% |
| publication | 4 | 0.1% |
| software | 1 | 0.0% |

**1,950 items (33%) carry no `P31` at all**, and
**693 more (12%) have a `P31` whose ancestors match
no rule in `TYPE_ROOTS`**. Together that is 44% of the corpus that this typology cannot
place. The second group is *this module's* blind spot, not a property of the data:
examples are `(−2,3,7) pretzel knot`, `1 + 2 + 3 + 4 + · · ·`, `1 − 2 + 4 − 8 + …`, `⅕`, `110-vertex Iofinova–Ivanov graph`, `16-cell`, `2 + 2 = 5`, `257-gon`.

So the assumption that a type filter could cleanly define the corpus does not hold,
and the coverage figure must be stated whenever a type-based rule is used.

Items with no `P31` have median degree **12**, against **8**
for the items this typology does classify: the untyped third is *better* connected
than the classified two thirds, so it is not the noisy third and must not be
dropped wholesale. Untyped examples: `'t Hooft–Polyakov monopole`, `*-algebra`, `1-planar graph`, `Three-torus`, `4-manifold`, `4-polytope`, `65537-gon`, `AND gate`.

## 2. A `P279`-based "is this mathematics?" test fails

Requiring descent from `mathematical concept` / `mathematical object` / `theorem`
rejects 1,009 items — and those items are `subobject classifier`,
`biclique-free graph`, `Tietze's graph`, `character table`, `Romberg's method`,
`elongated square cupola`. They are mathematics; they live in sibling branches
(`mathematical structure`, `geometric object`, `set`, `property`).

Wikidata's class tree is not a taxonomy of mathematics, so no global
"is it maths" predicate is available from it.

## 3. What can be decided

Statement-like types are reliable and are the natural backbone of a dependency
graph:

| kind | items |
|---|---:|
| theorem | 1,453 |
| lemma | 85 |
| conjecture | 70 |
| problem | 183 |
| structure | 176 |

Statements (theorem + lemma + conjecture): **1,503** — `Gauss–Lucas theorem`, `Courant–Friedrichs–Lewy condition`, `Pedoe's inequality`, `Noether's second theorem`, `Pólya enumeration theorem`, `Lamé's theorem`, `identity theorem`, `Szemerédi's theorem`.

**Definitions are not typeable.** Wikidata has no usable `definition` class
(`Q101072` matches exactly 1 item), so the definition layer must be discovered
from edge extraction — from what theorems actually cite — not from typing.

## 4. What can be excluded

Explicit noise classes (`publication`, `software`): **5 items** — `Mereology`, `Urania Propitia`, `Last Year at Marienbad`, `Linear Algebra Done Right`, `TITAN2D`.

That is a small, safe removal — not the mass clean-up the first render seemed to
demand. The visible junk (bare numerals, `bowling average`, `redshift`) is mostly
*legitimate but out of scope* material: items about specific integers, statistics
techniques, physics concepts. That is a **scope** decision to make explicitly,
not a data-quality bug to fix.

## 5. Decision

Core = statements + notions, minus explicit noise: **3,292 items**,
written to `data/raw/core_nodes.csv`. Untyped items are kept as *context* (they
are well connected) rather than silently deleted.

Consequences for the next phases:

- the dependency graph is seeded on the 1,503 statements;
- definitions must be discovered as targets of statement edges, not by typing;
## 6. Folding (same notion, two titles)

Redirects are already merged by `mathkg/fetch.py`. A crude title normalisation
(drop a trailing parenthetical, lower-case, strip non-alphanumerics, drop a plural
`s`) groups the corpus into **5,945 keys**, of which **22 hold more than one item** — **45 items**, i.e. up to **23 nodes (0.4%)** are redundant under that rule.

These are candidates, not verdicts: some are genuinely distinct (`angle` vs
`directed angle`), others are the real thing (`principal component analysis`,
which has two Wikidata items). Folding is therefore a node-level review task with
a short override list, not a category rule.
