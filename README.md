# MathKG — Wikidata + Wikipedia → coloured graph

A reproducible pipeline that turns the Wikidata items maintained by
**WikiProject Mathematics** into a link graph of English Wikipedia articles,
coloured by their distance to twelve "fields of mathematics".

The colouring is the point: instead of opaque Louvain community ids, every node
is coloured from its BFS distance to the 12 anchor fields, so a colour reads as
*"2 hops from number theory, 3 from topology"*.

### How the colour works

1. BFS from each of the 12 anchor articles over the **undirected** projection of
   the graph (`directed` only matters for the link direction, not for proximity).
2. Take the **3 nearest reachable** anchors — ties broken by anchor order, so
   the figure is reproducible.
3. Blend their palette colours with inverse-**square** weights (`1/(1+d)²`).
   The original used `1/(1+d)`, which is too flat: most nodes are one hop from
   several fields, so everything averaged into the same mud.
4. A node at distance 0 from an anchor *is* an anchor, and gets that field's
   pure colour — the twelve reference points stay identifiable.
5. No reachable anchor at all → grey (`#B0B0B0`), never a crash.

Palette colours are assigned positionally in `fields_of_math.csv` order, so a
field keeps its colour when another field is added. `data/artifacts/graph_distances.csv`
holds the distance vector behind every single node, so any colour in the figure
can be recomputed by hand.

---

## Quickstart

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Wikimedia requires a real contact address in the User-Agent:
export MATHKG_CONTACT="you@example.org"

# Rebuild everything from the scrape that is already in the repo (no network):
.venv/bin/python run.py all-offline

# …or fetch fresh data (network: Wikidata SPARQL + the Wikipedia action API):
.venv/bin/python run.py all
```

Artifacts land in `data/artifacts/`:

| File | What it is |
|---|---|
| `graph.json` | node/link JSON for the web viewer (no coordinates — the viewer lays it out) |
| `graph.gexf` | coloured graph, opens pre-coloured in Gephi |
| `graph.graphml` | same graph for other tools |
| `graph.html` | self-contained interactive page (vis.js inlined, no CDN) |
| `graph_distances.csv` | one row per node: distance to each anchor — makes the figure auditable |

## Stages

Each stage reads only what the previous one wrote, so a run is reproducible end
to end.

| Command | In | Out |
|---|---|---|
| `run.py wikidata` | Wikidata SPARQL | `data/raw/nodes.csv`, `data/raw/fields.csv` |
| `run.py fetch` | MediaWiki action API | `data/raw/nodes.csv`, `data/raw/links.csv` |
| `run.py import-legacy` | `df_filtered.csv` | `data/raw/nodes.csv`, `data/raw/links.csv` |
| `run.py build` | the two raw files | `data/artifacts/*` |
| `run.py all` | — | wikidata → fetch → build |
| `run.py all-offline` | — | import-legacy → build |
| `run.py curriculum` | — | the 29-node prerequisite DAG |

`build` **verifies its own output**: it re-reads the JSON and the GEXF and
refuses to succeed if either disagrees with the in-memory graph. The shipped
artifacts in this repo did not agree (6,623 nodes in the GEXF vs 7,444 in the
JSON), which is what made the figure untraceable.

Options: `--max-degree N` (anchors are protected, never dropped), `--no-html`
(skip the pyvis render), `--verbose`.

## Tests

```bash
.venv/bin/python -m pytest tests -q     # 54 tests, no network
```

---

## What changed, and why

The original code was three scripts (`download_raw_data.py`, `enrichment.py`,
`testgraph.py`, 1,184 lines). Fixes, by original defect:

**Data integrity**

- **Articles existed twice.** Nodes were keyed on `en_url` while edges came from
  HTML hrefs, so 586 articles (≈9%) appeared twice — `Euclidean plane` *and*
  `Plane`, with near-identical adjacency and no edge between them. Everything
  now goes through `mathkg/titles.py`: one canonical title per article, URL
  decoding and redirect resolution included. 6,867 scraped rows collapse to
  5,968 distinct articles.
- **SPARQL paging was unstable.** `download_raw_data.py` paged with
  `LIMIT/OFFSET` on a query with **no `ORDER BY`**, so SPARQL was free to
  duplicate or skip rows between pages. Both queries are now totally ordered,
  and pages are written individually so a crash loses one page, not the run.
- **`eval()` on CSV cells.** `eval(i)` on the `neighbour` column executed
  whatever was in the file. Now `ast.literal_eval`; there is a test with a
  hostile cell asserting it stays inert.
- **The pipeline read its own output.** `nodes_csv="df_filtered.csv"` and
  `save_filtered_csv("df_filtered.csv")` were the same path, so every run
  re-ingested the previous run's enriched file. `data/raw/` is now a pure
  function of the previous stage.

**Anything that could silently fail**

- **`define_max_degree` did not work.** It rebound the DataFrame to a slice and
  assigned through it, so the cap could quietly do nothing — the shipped CSV
  still contained `mathematics` at degree 2,016 under a cap of 20. The cap is
  now explicit, iterative, reports what it dropped, and **protects the
  anchors** (they are the hubs; an unprotected cap deletes all 12 and greys out
  the entire picture).
- **A network error killed a multi-hour run.** The `except` block touched
  `resp`, which does not exist when `requests.get` itself raised, so an
  intermittent connection error produced `UnboundLocalError` *inside the error
  handler* and killed everything after ~6,000 pages. `mathkg/http.py` retries
  connection errors, timeouts, 429 and 5xx, honouring `Retry-After`.
- **One HTTP identity, and an honest one.** The scraper impersonated Chrome
  (`Mozilla/5.0 … Chrome/142`) over 16 unthrottled threads while two other
  modules used bot UAs with placeholder addresses. There is now one UA, built
  from `MATHKG_CONTACT`, one request at a time with a minimum interval.
- **Dead code that could not run.** `nx.read_gpickle` was removed in NetworkX
  3.0 (this project's own GEXF says NetworkX 3.2.1); `explore_with_pyvis` was
  defined inside a class without `self`; `color_graph` still contained a
  `breakpoint()`; `distance()` returned nothing. All removed.

**The colouring**

- `sorted(dist)[-3]` selected the three **furthest** anchors, contradicting its
  own comment. Now the nearest `k`.
- A single unreachable anchor made `sorted()` raise `TypeError`. Unreachable
  anchors are skipped; nodes with no reachable anchor are grey rather than a
  crash.
- `(1 - poids) / norm(poids)` divided by the norm of the raw integer vector and
  produced negative, out-of-range RGB. Weights are positive, normalised, and
  clipped to `[0, 1]`.

**Packaging**

- **Two copies of `export_graph_json`** (one in each module) had already
  drifted apart. One implementation now, in `mathkg/export.py`.
- **A full `spring_layout` on 6,623 nodes** was computed for the JSON and then
  thrown away when pyvis re-ran ForceAtlas2. Layout is opt-in; coordinates are
  no longer emitted.
- **pyvis wrote its `lib/` folder into `os.getcwd()`**, not next to the HTML —
  the origin of the stray empty `lib/` in this repo. The HTML is now
  self-contained (`cdn_resources="in_line"`, plus the Bootstrap CDN tags pyvis
  hard-codes are stripped), so one file is shareable.
- `testgraph.py`'s hard-coded 29-node course model became
  `mathkg/curriculum.py`, with a deterministic **layered DAG layout** instead of
  a spring layout (29 nodes, 39 edges, longest prerequisite chain = 10).

---

## Known limitations — read before publishing a figure

1. **The colouring has about three usable levels, and that is a property of the
   graph, not of the code.** Measured on the current data (`run.py build` prints
   this as `graph shape`): 95.8% of node pairs are within 4 hops, 51.3% are
   exactly 3 hops. So "distance to the 12 anchors" collapses to `d ∈ {1,2,3,4}`
   and **most nodes tie between several anchors**. Ties are broken
   deterministically by anchor order, which is why `Algebra` (first row of
   `fields_of_math.csv`) ends up as the nearest field for 2,305 of 5,968 nodes.
   A distance-based colouring is honest but coarse on a small-world graph; if
   you want graded separation, the next step is a continuous proximity measure
   (**personalised PageRank** from each anchor, or distance-weighted path
   counts) rather than BFS hops. That is a method change, not a bug fix, so it
   is not applied here.
2. **Two different edge definitions exist, and only one of them was ever
   documented.** The original scraper extracted links from
   `re.search('From Wikipedia…(<p>.*?<h2)')`, i.e. **only the lead section and
   infobox**, and any layout change produced a silent empty node after three
   3-second retries. `run.py fetch` uses the action API and gets **all**
   namespace-0 links. These are different graphs:
   - `import-legacy` → lead-section co-mention (what `df_filtered.csv` holds);
   - `fetch` → real article links.

   Pick one deliberately and say which in the caption.
3. **`P6104` means "maintained by WikiProject Mathematics"**, not "is
   mathematics". The dataset legitimately contains `triskaidekaphobia`,
   `Go (game)` and `magnet`. The old keyword filter that tried to fix this
   (`filtrer()`) only ever existed in commented-out code and has not been
   reintroduced — a `filter` stage is the natural next step.
4. **The degree-cap question is open.** On this graph a cap of 20 drops 1,150
   nodes and shatters the graph into ~1,600 components. The default is
   therefore "keep everything".
4. **No temporal dimension.** The notes in `Historique de pensée.txt` aim at
   *evolution* (how mathematical ideas developed); a static link graph cannot
   show that. That needs a different data source (revision history / Wikidata
   inception dates), not a different layout.
5. `Historique de pensée.txt` also contains unrelated personal notes (train
   times, addresses). Scrub it before making this repository public.

## Layout of the repository

```
run.py                  one entry point, all stages
mathkg/
  config.py             paths, HTTP identity, palette, tunables
  queries.py            the SPARQL queries (ordered!)
  http.py               polite retry session (one UA, backoff, Retry-After)
  titles.py             title <-> URL normalisation, namespace filtering
  wikidata.py           SPARQL stage (resumable, atomic writes)
  fetch.py              MediaWiki API stage + legacy-CSV import
  graph.py              node/edge construction, degree cap, summary
  color.py              anchor distances and colour blending
  export.py             JSON / GEXF / GraphML / pyvis, one implementation
  curriculum.py         the 29-node prerequisite DAG + layered layout
tests/                  54 offline tests
data/                   generated (gitignored): raw/, cache/, artifacts/
```

`df_filtered.csv`, `wikiproject_math_nodes.csv` and `fields_of_math.csv` are
kept at the repository root as the original inputs; the pipeline reads them but
never overwrites them.

---

## The live loop

Styling is a **document**, not code: a view spec holds expressions over node and
edge columns, and a change costs one message to an open browser tab.

```bash
# 1. start the viewer once (holds the graph in memory)
.venv/bin/python run.py serve                 # http://127.0.0.1:8765

# 2. open it, leave it open
xdg-open http://127.0.0.1:8765

# 3. change the view from another terminal — the tab updates in place
.venv/bin/python run.py set --set 'node.size=6 + 2.5 * degree'
.venv/bin/python run.py set --set 'label.size=9' --set 'label.count=60'
.venv/bin/python run.py set --set 'edge_weight.formula=occurrences * in_proof'
```

Each `set` prints the round trip the server measured (`POST /spec`: **57–62 ms**
for 1,560 nodes) and the tab shows the apply latency it measured itself.

### What can be expressed

| column | available on |
|---|---|
| `degree`, `sitelinks`, `kind`, `is_anchor` | nodes |
| `confidence`, `popularity`, `occurrences`, `relative_position`, `in_lead`, `in_proof` | edges |

Expressions support `+ - * / **`, comparisons, `if/else`, and
`min max abs sqrt log log10 exp floor ceil round pow`. No `eval`: unknown names,
attribute access, subscripts and comprehensions are rejected — 12 hostile inputs
are covered by tests, and the server answers 400 rather than storing a bad spec.

The weighting from the brief, written out — occurrences, weighted by how early the
mention appears, and *negatively* by the target's popularity:

```yaml
edge_weight:
  formula: "occurrences * (1 - relative_position) / log(1 + popularity)"
  minimum: 0.15
```

### Canned views

```bash
for v in views/*.yaml; do .venv/bin/python run.py --spec "$v" --out "data/artifacts/views/$(basename "$v" .yaml).png" view; done
```

| view | what it does |
|---|---|
| `degree_size` | node size ∝ degree |
| `proof_weighted` | edges kept only if cited *inside a proof* (2026 → 1155 edges) |
| `mention_weighted` | the formula above (2026 → 1649 edges) |
| `hubs_only` | degree ≥ 4 (1,560 → 339 nodes) |

### Queries

```bash
.venv/bin/python run.py ego --node "Topology" --radius 2        # 10 at 1 hop, 81 at 2
.venv/bin/python run.py describe --node "Topology"              # counts + longest chain
.venv/bin/python run.py path --from "Foundations of mathematics" \
                           --to "Pythagorean theorem"
```

`path` reports **how** it found the chain: `directed` (a real prerequisite chain),
`undirected` (direction mixed — a weaker claim, warned about), or `none`. For the
foundations → Pythagoras case it honestly returns an undirected 4-step path,
because no directed chain exists in the current edge set. That is a gap in the
edges, not in the query.

### MCP

The same core is exposed over MCP for any client that speaks it:

```bash
.venv/bin/python -m mathkg.mcp_server            # JSON-RPC on stdin/stdout
```

Tools: `get_state`, `set_spec`, `filter`, `ego`, `path`, `describe`,
`render_png`, `reload_graph`. Verified with a handshake, `tools/list`, a real
`describe` call, and a hostile `set_spec` that is rejected.

### Reverting a change

Nothing is hidden in a database. A view lives in `views/<name>.yaml` or in a
command you can re-run, so `git diff` shows exactly what changed and
`git checkout views/` undoes it.
