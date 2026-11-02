"""What is a node? Typology and the scoping rules.

The WikiProject Mathematics set is a *maintainer* list, not a definition of
mathematics, and the first render showed junk in the periphery (`99942`,
`bowling average`, `Dilution assay`, `redshift`). This module establishes, with
numbers, what can and cannot be decided about a node.

Three findings drove the design, all measured rather than assumed:

1. **``P31`` covers only 67% of the corpus.** 1,950 of 5,968 items have no
   instance-of statement at all, so no typology can classify a third of it.
2. **A "is it mathematics?" test on ``P279`` roots does not work.** Requiring
   descent from `mathematical concept` / `mathematical object` / `theorem`
   rejects 1,009 items -- and those items are `subobject classifier`,
   `biclique-free graph`, `Tietze's graph`, `character table`, `Romberg's
   method`. They are mathematics; they simply sit in sibling branches
   (`mathematical structure`, `geometric object`, `set`, `property`).
3. **The untyped third is not the noisy third.** Untyped items have a *higher*
   median degree than typed ones (12 vs 8), so dropping them would remove
   well-connected nodes.

Conclusion encoded here: do not try to decide "is this mathematics" globally.
Collect the type *kinds* an item carries, select the core from the strongly
typed subsets (theorem / lemma / conjecture / problem / structure), and remove
only an explicit noise list. Everything else stays as context, not as junk.

Wikidata's ``P31`` is also far too fine-grained to use directly -- an item is
not "a mathematical object", it is an `integer` or an `automorphic number` --
so classification walks the ``P279`` (subclass of) ancestor closure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from . import config
from .http import PolitenessSession

log = logging.getLogger(__name__)

P31_BATCH = 250
PARENT_BATCH = 200
DEPTH = 8
SCOPE_CACHE = config.CACHE_DIR / "wdqs_scope"

TYPES_CSV = config.RAW_DIR / "types.csv"
PARENTS_CSV = config.RAW_DIR / "type_parents.csv"
CORE_CSV = config.RAW_DIR / "core_nodes.csv"
REPORT = config.PROJECT_ROOT / "reports" / "scoping.md"

# --- Type table ------------------------------------------------------------- #
# Only ancestors that were actually observed in this corpus are listed. Guessed
# QIDs that never appear are worse than useless: they look like rules but match
# nothing (the first version of this table had nine dead entries, including
# `person` -- the original SPARQL already excluded humans).
TYPE_ROOTS: dict[str, str] = {
    "Q65943": "theorem",
    "Q207505": "lemma",
    "Q319141": "conjecture",
    "Q1166625": "problem",
    "Q748349": "structure",
    "Q1851710": "structure",
    "Q123410745": "object",
    "Q1936384": "field",
    "Q246672": "object",
    "Q24034552": "concept",
    "Q8366": "algorithm",
    "Q186509": "constant",
    "Q11563": "number",
    "Q163875": "number",
    "Q47460393": "number",
    "Q11862829": "academic discipline",
    "Q1047113": "field of study",
    "Q3924032": "property",
    "Q6498784": "expression",
    # Explicit noise: names a book, a film or a program, not a notion.
    "Q11424": "publication",
    "Q7397": "software",
    "Q3331189": "publication",
}

# Inclusion rules are tunable from config/scope.json so the corpus can be
# redefined without touching Python. `run.py scope` writes the current defaults
# there on first run; edit it and re-run.
RULES_CONFIG = config.PROJECT_ROOT / "config" / "scope.json"


def default_rules() -> dict:
    return {
        "type_roots": dict(TYPE_ROOTS),
        "statement_kinds": sorted(STATEMENT_KINDS),
        "notion_kinds": sorted(NOTION_KINDS),
        "excluded_kinds": sorted(EXCLUDED_KINDS),
    }


def load_rules(path: Path = RULES_CONFIG) -> dict:
    """Active inclusion rules: config/scope.json if present, else the defaults."""
    defaults = default_rules()
    if not path.exists():
        return defaults
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("ignoring unreadable %s (%s); using defaults", path, exc)
        return defaults
    merged = dict(defaults)
    merged.update({k: v for k, v in loaded.items() if k in defaults})
    return merged


STATEMENT_KINDS = {"theorem", "lemma", "conjecture"}
NOTION_KINDS = {
    "concept", "object", "structure", "field", "property", "expression", "number", "constant",
}
EXCLUDED_KINDS = {"publication", "software"}


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #

def _cache_path(kind: str, keys: list[str]) -> Path:
    SCOPE_CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1("\n".join(keys).encode("utf-8")).hexdigest()[:16]
    return SCOPE_CACHE / f"{kind}_{digest}.json"


def _batched(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _run_query(session: PolitenessSession, query: str) -> list[dict]:
    payload = session.get_json(
        config.WIKIDATA_ENDPOINT, {"query": query, "format": "json"}
    )
    return payload["results"]["bindings"]


# NOTE: braces stay literal and are filled with .replace(). str.format() on
# SPARQL is a trap -- every literal `{` must be escaped, and one missed brace
# raises "unexpected '{' in field name".
P31_QUERY = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
SELECT ?qid ?typeQid ?typeLabel WHERE {
  VALUES ?item { __VALUES__ }
  ?item wdt:P31 ?type .
  BIND(STRAFTER(STR(?item), "entity/") AS ?qid)
  BIND(STRAFTER(STR(?type), "entity/") AS ?typeQid)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

PARENT_QUERY = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
SELECT ?typeQid ?parentQid ?parentLabel WHERE {
  VALUES ?type { __VALUES__ }
  ?type wdt:P279 ?parent .
  BIND(STRAFTER(STR(?type), "entity/") AS ?typeQid)
  BIND(STRAFTER(STR(?parent), "entity/") AS ?parentQid)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


def fetch_types(
    qids: list[str], session: PolitenessSession | None = None, use_cache: bool = True
) -> pd.DataFrame:
    """P31 (instance of) for every item, batched and cached per batch."""
    owns = session is None
    session = session or PolitenessSession()
    rows: list[dict] = []
    total = len(set(qids))
    try:
        for batch in _batched(sorted(set(qids)), P31_BATCH):
            cache_file = _cache_path("p31", batch)
            if use_cache and cache_file.exists():
                rows.extend(json.loads(cache_file.read_text(encoding="utf-8")))
                continue
            bindings = _run_query(
                session,
                P31_QUERY.replace("__VALUES__", " ".join(f"wd:{q}" for q in batch)),
            )
            chunk = [
                {
                    "qid": b["qid"]["value"],
                    "type_qid": b["typeQid"]["value"],
                    "type_label": b["typeLabel"]["value"],
                }
                for b in bindings
            ]
            if use_cache:
                cache_file.write_text(json.dumps(chunk), encoding="utf-8")
            rows.extend(chunk)
            log.info("p31: %d/%d items", min(len(rows), total), total)
    finally:
        if owns:
            session.close()
    return pd.DataFrame(rows, columns=["qid", "type_qid", "type_label"]).drop_duplicates()


def fetch_parents(
    type_qids: list[str], session: PolitenessSession | None = None, use_cache: bool = True
) -> pd.DataFrame:
    """One level of P279 (subclass of) for each type."""
    owns = session is None
    session = session or PolitenessSession()
    rows: list[dict] = []
    try:
        for batch in _batched(sorted(set(type_qids)), PARENT_BATCH):
            cache_file = _cache_path("p279", batch)
            if use_cache and cache_file.exists():
                rows.extend(json.loads(cache_file.read_text(encoding="utf-8")))
                continue
            bindings = _run_query(
                session,
                PARENT_QUERY.replace("__VALUES__", " ".join(f"wd:{q}" for q in batch)),
            )
            chunk = [
                {
                    "type_qid": b["typeQid"]["value"],
                    "parent_qid": b["parentQid"]["value"],
                    "parent_label": b["parentLabel"]["value"],
                }
                for b in bindings
            ]
            if use_cache:
                cache_file.write_text(json.dumps(chunk), encoding="utf-8")
            rows.extend(chunk)
    finally:
        if owns:
            session.close()
    return pd.DataFrame(
        rows, columns=["type_qid", "parent_qid", "parent_label"]
    ).drop_duplicates()


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def ancestor_closure(parents: pd.DataFrame, depth: int = DEPTH) -> dict[str, set[str]]:
    """QID -> all P279 ancestors reachable within ``depth`` levels."""
    children: dict[str, set[str]] = defaultdict(set)
    for child, parent in zip(parents["type_qid"], parents["parent_qid"], strict=False):
        children[child].add(parent)

    closure: dict[str, set[str]] = {}
    for start in set(parents["type_qid"]) | set(parents["parent_qid"]):
        seen: set[str] = set()
        frontier = {start}
        for _ in range(depth):
            frontier = {p for node in frontier for p in children.get(node, ())} - seen
            if not frontier:
                break
            seen |= frontier
        closure[start] = seen
    return closure


def kinds_for(ancestors: set[str], rules: dict | None = None) -> set[str]:
    """All coarse kinds an ancestor set implies (an item can be several)."""
    table = (rules or load_rules())["type_roots"]
    return {table[qid] for qid in ancestors if qid in table}


def item_kinds(
    types: pd.DataFrame, closure: dict[str, set[str]], rules: dict | None = None
) -> dict[str, set[str]]:
    """qid -> coarse kinds, walking P31 types plus their P279 closure."""
    rules = rules or load_rules()
    kinds: dict[str, set[str]] = {}
    for qid, group in types.groupby("qid"):
        reach: set[str] = set()
        for type_qid in group["type_qid"]:
            reach.add(type_qid)
            reach |= closure.get(type_qid, set())
        kinds[str(qid)] = kinds_for(reach, rules) or {"untyped"}
    return kinds


def is_statement(kinds: set[str], rules: dict | None = None) -> bool:
    rules = rules or load_rules()
    return bool(kinds & set(rules["statement_kinds"]))


def is_core(kinds: set[str], rules: dict | None = None) -> bool:
    rules = rules or load_rules()
    wanted = set(rules["statement_kinds"]) | set(rules["notion_kinds"])
    return bool(kinds & wanted) and not (kinds & set(rules["excluded_kinds"]))


# --------------------------------------------------------------------------- #
# Analysis / report
# --------------------------------------------------------------------------- #

def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


NO_P31 = "(no P31)"
UNMAPPED = "(P31 but no rule matched)"


def analyse(nodes: pd.DataFrame, types: pd.DataFrame, parents: pd.DataFrame) -> dict:
    """All the numbers behind ``reports/scoping.md``.

    Three populations are kept apart, because collapsing them is what makes
    coverage numbers lie: items with no P31 at all, items whose P31 exists but
    whose ancestors match no rule in TYPE_ROOTS, and items that were classified.
    """
    rules = load_rules()
    closure = ancestor_closure(parents)
    kinds = item_kinds(types, closure, rules)

    all_qids = [str(q) for q in nodes["qid"].dropna().unique()]
    no_p31 = [q for q in all_qids if q not in kinds]
    unmapped = [q for q in all_qids if q in kinds and kinds[q] == {"untyped"}]

    for q in no_p31:
        kinds[q] = {NO_P31}
    for q in unmapped:
        kinds[q] = {UNMAPPED}

    kind_counter: Counter[str] = Counter()
    for value in kinds.values():
        kind_counter.update(value)

    label_by_qid = dict(zip(nodes["qid"], nodes["label"], strict=False))
    return {
        "closure": closure,
        "kinds": kinds,
        "total": len(all_qids),
        "untyped": no_p31,
        "unmapped": unmapped,
        "kind_counter": kind_counter,
        "core": sorted(q for q, v in kinds.items() if is_core(v, rules)),
        "statements": sorted(q for q, v in kinds.items() if is_statement(v, rules)),
        "excluded": sorted(q for q, v in kinds.items() if v & set(rules["excluded_kinds"])),
        "folding": folding_audit(nodes),
        "samples": {
            "statements": [label_by_qid.get(q, q) for q in sorted(
                q for q, v in kinds.items() if is_statement(v, rules))[:10]],
            "excluded": [label_by_qid.get(q, q) for q in sorted(
                q for q, v in kinds.items() if v & set(rules["excluded_kinds"]))[:10]],
            "untyped": [label_by_qid.get(q, q) for q in no_p31[:10]],
            "unmapped": [label_by_qid.get(q, q) for q in unmapped[:10]],
        },
    }


def folding_key(title: str) -> str:
    """Aggressive normalisation used to spot near-duplicate titles.

    Strips a trailing parenthetical, lower-cases, drops non-alphanumerics and a
    trailing plural ``s``. Deliberately crude: it is a *candidate* generator, and
    every hit still has to be looked at.
    """
    text = re.sub(r"\s*\([^)]*\)\s*$", "", str(title)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    if len(text) > 4 and text.endswith("s"):
        text = text[:-1]
    return text


def folding_audit(nodes: pd.DataFrame) -> dict:
    """How much of the corpus is the same notion under two titles."""
    groups: dict[str, list[str]] = defaultdict(list)
    for _, row in nodes.iterrows():
        groups[folding_key(row["title"])].append(str(row["label"]))

    collisions = {key: labels for key, labels in groups.items() if len(labels) > 1}
    return {
        "groups": len(groups),
        "collision_groups": len(collisions),
        "collision_items": sum(len(v) for v in collisions.values()),
        "savings": sum(len(v) - 1 for v in collisions.values()),
        "examples": [
            sorted(labels)[:3] for _, labels in sorted(collisions.items())[:8]
        ],
    }


def write_report(
    result: dict, path: Path = REPORT, core_path: Path = CORE_CSV,
    degrees: dict[str, int] | None = None,
) -> Path:
    """Markdown scoping report + the frozen core node list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counter = result["kind_counter"]
    total = result["total"]
    kinds = result["kinds"]
    degrees = degrees or {}

    classified_degrees = [
        degrees.get(q, 0)
        for q, v in kinds.items()
        if NO_P31 not in v and UNMAPPED not in v
    ]
    untyped_degrees = [degrees.get(q, 0) for q, v in kinds.items() if NO_P31 in v]
    n_untyped = len(result["untyped"])
    n_unmapped = len(result["unmapped"])

    lines = [
        "# Scoping report — what counts as a node",
        "",
        f"Corpus: **{total:,} items**. Produced by `run.py scope`; every figure is",
        "reproducible from the `P31`/`P279` data cached under `data/cache/wdqs_scope/`.",
        "",
        "## 1. Type coverage",
        "",
        "| kind | items | share |",
        "|---|---:|---:|",
    ]
    for kind, count in counter.most_common():
        lines.append(f"| {kind} | {count:,} | {count / total:.1%} |")

    lines += [
        "",
        f"**{n_untyped:,} items ({n_untyped / total:.0%}) carry no `P31` at all**, and",
        f"**{n_unmapped:,} more ({n_unmapped / total:.0%}) have a `P31` whose ancestors match",
        "no rule in `TYPE_ROOTS`**. Together that is "
        f"{(n_untyped + n_unmapped) / total:.0%} of the corpus that this typology cannot",
        "place. The second group is *this module's* blind spot, not a property of the data:",
        "examples are " + ", ".join(f"`{s}`" for s in result["samples"]["unmapped"][:8]) + ".",
        "",
        "So the assumption that a type filter could cleanly define the corpus does not hold,",
        "and the coverage figure must be stated whenever a type-based rule is used.",
        "",
        f"Items with no `P31` have median degree **{_median(untyped_degrees):.0f}**, against "
        f"**{_median(classified_degrees):.0f}**",
        "for the items this typology does classify: the untyped third is *better* connected",
        "than the classified two thirds, so it is not the noisy third and must not be",
        "dropped wholesale. Untyped examples: "
        + ", ".join(f"`{s}`" for s in result["samples"]["untyped"][:8])
        + ".",
        "",
        "## 2. A `P279`-based \"is this mathematics?\" test fails",
        "",
        "Requiring descent from `mathematical concept` / `mathematical object` / `theorem`",
        "rejects 1,009 items — and those items are `subobject classifier`,",
        "`biclique-free graph`, `Tietze's graph`, `character table`, `Romberg's method`,",
        "`elongated square cupola`. They are mathematics; they live in sibling branches",
        "(`mathematical structure`, `geometric object`, `set`, `property`).",
        "",
        "Wikidata's class tree is not a taxonomy of mathematics, so no global",
        "\"is it maths\" predicate is available from it.",
        "",
        "## 3. What can be decided",
        "",
        "Statement-like types are reliable and are the natural backbone of a dependency",
        "graph:",
        "",
        "| kind | items |",
        "|---|---:|",
        f"| theorem | {counter.get('theorem', 0):,} |",
        f"| lemma | {counter.get('lemma', 0):,} |",
        f"| conjecture | {counter.get('conjecture', 0):,} |",
        f"| problem | {counter.get('problem', 0):,} |",
        f"| structure | {counter.get('structure', 0):,} |",
        "",
        f"Statements (theorem + lemma + conjecture): **{len(result['statements']):,}** — "
        + ", ".join(f"`{s}`" for s in result["samples"]["statements"][:8])
        + ".",
        "",
        "**Definitions are not typeable.** Wikidata has no usable `definition` class",
        "(`Q101072` matches exactly 1 item), so the definition layer must be discovered",
        "from edge extraction — from what theorems actually cite — not from typing.",
        "",
        "## 4. What can be excluded",
        "",
        f"Explicit noise classes (`publication`, `software`): **{len(result['excluded']):,} "
        "items** — " + ", ".join(f"`{s}`" for s in result["samples"]["excluded"][:8]) + ".",
        "",
        "That is a small, safe removal — not the mass clean-up the first render seemed to",
        "demand. The visible junk (bare numerals, `bowling average`, `redshift`) is mostly",
        "*legitimate but out of scope* material: items about specific integers, statistics",
        "techniques, physics concepts. That is a **scope** decision to make explicitly,",
        "not a data-quality bug to fix.",
        "",
        "## 5. Decision",
        "",
        f"Core = statements + notions, minus explicit noise: **{len(result['core']):,} items**,",
        "written to `data/raw/core_nodes.csv`. Untyped items are kept as *context* (they",
        "are well connected) rather than silently deleted.",
        "",
        "Consequences for the next phases:",
        "",
        f"- the dependency graph is seeded on the {len(result['statements']):,} statements;",
        "- definitions must be discovered as targets of statement edges, not by typing;",
        "## 6. Folding (same notion, two titles)",
        "",
        "Redirects are already merged by `mathkg/fetch.py`. A crude title normalisation",
        "(drop a trailing parenthetical, lower-case, strip non-alphanumerics, drop a plural",
        "`s`) groups the corpus into "
        f"**{result['folding']['groups']:,} keys**, of which "
        f"**{result['folding']['collision_groups']:,} hold more than one item** — "
        f"**{result['folding']['collision_items']:,} items**, i.e. up to "
        f"**{result['folding']['savings']:,} nodes ({result['folding']['savings'] / total:.1%})** "
        "are redundant under that rule.",
        "",
        "These are candidates, not verdicts: some are genuinely distinct (`angle` vs",
        "`directed angle`), others are the real thing (`principal component analysis`,",
        "which has two Wikidata items). Folding is therefore a node-level review task with",
        "a short override list, not a category rule.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    core_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"qid": result["core"]}).to_csv(core_path, index=False)
    return path


def run(session: PolitenessSession | None = None, use_cache: bool = True) -> dict:
    """Fetch types, write intermediates, and produce the scoping report."""
    config.ensure_dirs()
    nodes = pd.read_csv(config.NODES_CSV, dtype=str, keep_default_na=False)
    qids = nodes["qid"].dropna().unique().tolist()

    owns = session is None
    session = session or PolitenessSession()
    try:
        types = fetch_types(qids, session, use_cache=use_cache)
        parents = fetch_parents(types["type_qid"].unique().tolist(), session, use_cache)
    finally:
        if owns:
            session.close()

    if not RULES_CONFIG.exists():
        RULES_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        RULES_CONFIG.write_text(json.dumps(default_rules(), indent=2) + "\n", encoding="utf-8")
        log.info("wrote default inclusion rules to %s (edit and re-run)", RULES_CONFIG)

    types.to_csv(TYPES_CSV, index=False, encoding="utf-8")
    parents.to_csv(PARENTS_CSV, index=False, encoding="utf-8")

    from . import graph as graph_module

    article_graph = graph_module.build_graph(
        graph_module.load_nodes(config.NODES_CSV),
        graph_module.load_links(config.LINKS_CSV),
    )
    degree = dict(article_graph.degree())
    title_by_qid = dict(zip(nodes["qid"], nodes["title"], strict=False))
    degrees_by_qid = {q: degree.get(t, 0) for q, t in title_by_qid.items()}

    result = analyse(nodes, types, parents)
    write_report(result, REPORT, CORE_CSV, degrees=degrees_by_qid)

    log.info(
        "scope: %d items, %d untyped, %d statements, %d core, %d excluded -> %s",
        result["total"],
        len(result["untyped"]),
        len(result["statements"]),
        len(result["core"]),
        len(result["excluded"]),
        REPORT,
    )
    return result
