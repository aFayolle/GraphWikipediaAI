"""The dependency relation: pluggable extractors, each edge auditable.

The funnel strategy decides how this module is used: extraction runs **wide**
(every item, every signal), and the *views* narrow down later. Nothing is
filtered here on the grounds of being unimportant.

Two extractors, deliberately kept separate so their precision can be measured
independently:

* :func:`wikidata_edges` -- structured properties. Precise, reproducible, but
  sparse for mathematics.
* :func:`phrase_edges` -- relations stated in the article's lead sentence
  ("is a generalization of", "is a special case of", "follows from", ...).
  Cheap here because the lead text is already cached in ``df_filtered.csv``, so
  no network and no re-scraping.

Every edge carries the property or the sentence that produced it. An edge whose
evidence cannot be shown is not an edge, it is an opinion.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from . import config
from .http import PolitenessSession

log = logging.getLogger(__name__)

EDGE_BATCH = 200
EDGE_CACHE = config.CACHE_DIR / "wdqs_edges"
EDGES_CSV = config.RAW_DIR / "edges.csv"
OVERRIDES_CSV = config.PROJECT_ROOT / "overrides" / "edges.csv"

# Relation vocabulary. Kept small and meaningful: these are the distinctions a
# reader of a dependency graph actually needs.
RELATIONS = (
    "generalisation",   # A generalises B  (A is more abstract than B)
    "special_case",     # A is a special case of B
    "depends_on",       # A uses / follows from B
    "equivalent",       # A and B are equivalent formulations
    "part_of",          # structural containment
    "influenced_by",    # historical influence (Wikidata P737)
)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str
    extractor: str
    confidence: float
    evidence: str

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Wikidata extractor
# --------------------------------------------------------------------------- #

# (property, relation, confidence): structured statements are reliable about
# *what* they say, less so about it meaning a dependency.
WIKIDATA_PROPERTIES = [
    ("P279", "special_case", 0.8),   # subclass of
    ("P361", "part_of", 0.6),        # part of
    ("P527", "part_of", 0.6),        # has part(s)
    ("P155", "depends_on", 0.5),     # follows
    ("P156", "depends_on", 0.5),     # followed by
    ("P737", "influenced_by", 0.5),  # influenced by
]

# One query per property: simpler and much faster than a VALUES over props.
EDGE_QUERY = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
SELECT ?qid ?targetQid ?targetLabel WHERE {
  VALUES ?item { __VALUES__ }
  ?item wdt:__PROP__ ?target .
  BIND(STRAFTER(STR(?item), "entity/") AS ?qid)
  BIND(STRAFTER(STR(?target), "entity/") AS ?targetQid)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


def _cache_path(kind: str, keys: list[str]) -> Path:
    EDGE_CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1("\n".join(keys).encode("utf-8")).hexdigest()[:16]
    return EDGE_CACHE / f"{kind}_{digest}.json"


def _batched(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def wikidata_edges(
    qids: list[str],
    title_by_qid: dict[str, str],
    session: PolitenessSession | None = None,
    use_cache: bool = True,
    title_lookup: dict[str, str] | None = None,
) -> list[Edge]:
    """Structured dependency-ish statements from Wikidata.

    Wikidata returns *labels* ("regular polygon"), while the graph is keyed on
    article *titles* ("Regular polygon"). Targets are therefore resolved through
    a lower-cased title lookup, and the ones that do not correspond to a node in
    the corpus are dropped and counted -- an edge to a node that does not exist
    cannot be drawn or reasoned about.
    """
    dropped = 0
    owns = session is None
    session = session or PolitenessSession()
    edges: list[Edge] = []
    try:
        for prop, relation, confidence in WIKIDATA_PROPERTIES:
            query = EDGE_QUERY.replace("__PROP__", prop)
            for batch in _batched(sorted(set(qids)), EDGE_BATCH):
                cache_file = _cache_path(f"{prop}", batch)
                if use_cache and cache_file.exists():
                    rows = json.loads(cache_file.read_text(encoding="utf-8"))
                else:
                    payload = session.get_json(
                        config.WIKIDATA_ENDPOINT,
                        {"query": query.replace("__VALUES__", " ".join(f"wd:{q}" for q in batch)),
                         "format": "json"},
                    )
                    rows = [
                        {"qid": b["qid"]["value"], "target": b["targetLabel"]["value"]}
                        for b in payload["results"]["bindings"]
                    ]
                    if use_cache:
                        cache_file.write_text(json.dumps(rows), encoding="utf-8")

                for row in rows:
                    source = title_by_qid.get(row["qid"])
                    target = (
                        title_lookup.get(row["target"].lower()) if title_lookup else row["target"]
                    )
                    if not source:
                        continue
                    if not target or target == source:
                        dropped += 1
                        continue
                    edges.append(
                        Edge(
                            source=source,
                            target=target,
                            relation=relation,
                            extractor="wikidata",
                            confidence=confidence,
                            evidence=f"Wikidata {prop}",
                        )
                    )
            log.info("wikidata edges: %s -> %d so far (%d unresolved dropped)",
                     prop, len(edges), dropped)
    finally:
        if owns:
            session.close()
    return edges


# --------------------------------------------------------------------------- #
# Phrase extractor
# --------------------------------------------------------------------------- #

# Ordered: the first pattern that matches a sentence wins, so the most specific
# relations are listed first.
PHRASE_PATTERNS: list[tuple[str, str, float, str]] = [
    ("special_case", r"is a (?:special|particular|specific) case of ([^.;]{2,80})", 0.85,
     "special case of"),
    ("generalisation", r"is a (?:generalization|generalisation|generalized version|"
                       r"generalised version|generalisation) of ([^.;]{2,80})", 0.85,
     "generalization of"),
    ("generalisation", r"(?:generalizes|generalises) ([^.;]{2,80})", 0.8, "generalizes"),
    ("equivalent", r"is equivalent to ([^.;]{2,80})", 0.8, "is equivalent to"),
    ("depends_on", r"(?:follows from|is a consequence of|is a corollary of|"
                   r"can be deduced from|is proved (?:using|by)) ([^.;]{2,80})", 0.8,
     "follows from"),
    ("depends_on", r"(?:makes use of|uses the (?:notion|concept|theory) of) ([^.;]{2,80})",
     0.6, "uses"),
    ("part_of", r"is a (?:branch|subfield|subfield|part|subarea) of ([^.;]{2,80})", 0.8,
     "branch of"),
]

COMPILED = [(relation, re.compile(pattern, re.IGNORECASE), confidence, label)
            for relation, pattern, confidence, label in PHRASE_PATTERNS]


def _lookup_table(titles: list[str]) -> dict[str, str]:
    return {title.lower(): title for title in titles}


LEADING_FILLER = {"a", "an", "the", "some", "any", "both", "all", "one", "its", "of"}

# Meta-vocabulary that is technically an article title but never a meaningful
# dependency target. Without this, "follows from the theorem" yields an edge to
# `Theorem`, and "follows from the mathematics of quantum mechanics" an edge to
# `Mathematics` -- true statements that carry no information.
GENERIC_TARGETS = {
    "theorem", "lemma", "mathematics", "mathematical proof", "axiom", "definition",
    "number", "angle", "function", "formula", "equation", "set", "space", "point",
    "line", "curve", "area", "volume", "geometry", "algebra", "calculus", "logic",
    "quantity", "value", "property", "statement", "proposition", "theory", "method",
    "algorithm", "problem", "result", "example", "case", "concept", "structure",
}


def _resolve(phrase: str, lookup: dict[str, str]) -> str | None:
    """Resolve the object of the relation to a known article title.

    The object starts the captured phrase ("special case of **a parallelogram**
    and a kite"), so a match is only accepted at the *first content word*.
    Free-text search was tried first and produced confident nonsense: "special
    case of the n-gon interior angle sum formula" resolved to `Angle`, and
    "generalization of the special number field sieve" to `Number`. Insisting
    on the opening position trades recall for not inventing edges.
    """
    words = re.findall(r"[A-Za-z0-9'\u2019\-]+", phrase.lower())
    while words and words[0] in LEADING_FILLER:
        words.pop(0)
    for size in range(min(6, len(words)), 0, -1):
        candidate = " ".join(words[:size])
        found = lookup.get(candidate)
        if found:
            return None if candidate in GENERIC_TARGETS else found
    return None


def phrase_edges(
    summaries: dict[str, str],
    titles: list[str],
    max_sentences: int = 40,
) -> list[Edge]:
    """Relations stated in the lead text ("is a generalization of", ...).

    The lead text is already cached in ``df_filtered.csv``, so this costs no
    network and no re-scraping -- which is why the funnel can afford to run it
    over every item instead of a sample.
    """
    lookup = _lookup_table(titles)
    edges: list[Edge] = []

    for source, text in summaries.items():
        if not text or source not in lookup.values():
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)[:max_sentences]
        for sentence in sentences:
            for relation, pattern, confidence, label in COMPILED:
                match = pattern.search(sentence)
                if not match:
                    continue
                target = _resolve(match.group(1), lookup)
                if not target or target == source:
                    continue
                edges.append(
                    Edge(
                        source=source,
                        target=target,
                        relation=relation,
                        extractor="phrase",
                        confidence=confidence,
                        evidence=f"...{label} {match.group(1).strip()[:60]}",
                    )
                )
                break  # one relation per sentence, the most specific one
    return edges


# --------------------------------------------------------------------------- #
# Manual overrides
# --------------------------------------------------------------------------- #

def apply_overrides(
    edges: list[Edge], path: Path = OVERRIDES_CSV, known_titles: set[str] | None = None
) -> tuple[list[Edge], dict]:
    """Apply the versioned add/remove/reweight file.

    Hand corrections are the only honest way to fix the residual extraction
    errors (a direction inversion, a target resolved one word too early). Keeping
    them in a CSV under git means the correction is reviewable and survives
    re-runs -- unlike editing generated output.

    Rows that reference a title outside the corpus are reported and ignored:
    a typo in an override file would otherwise silently do nothing.
    """
    if not path.exists():
        return edges, {"add": 0, "remove": 0, "reweight": 0, "unknown": []}

    frame = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
    if frame.empty:
        return edges, {"add": 0, "remove": 0, "reweight": 0, "unknown": []}

    added: list[Edge] = []
    removed: set[tuple[str, str]] = set()
    reweighted: dict[tuple[str, str, str], float] = {}
    unknown: list[str] = []

    for _, row in frame.iterrows():
        action = str(row.get("action", "")).strip().lower()
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        relation = str(row.get("relation", "")).strip() or "depends_on"
        reason = str(row.get("reason", "")).strip()

        if known_titles is not None:
            for title in (source, target):
                if title and title not in known_titles:
                    unknown.append(f"{action}:{title}")

        if action == "add":
            added.append(Edge(source, target, relation, "override", 1.0, reason or "manual"))
        elif action == "remove":
            removed.add((source, target))
        elif action == "reweight":
            try:
                reweighted[(source, target, relation)] = float(row.get("confidence", "") or 1.0)
            except ValueError:
                log.warning("override: bad confidence for %s -> %s", source, target)
        else:
            log.warning("override: unknown action %r", action)

    kept: list[Edge] = []
    for edge in edges:
        if (edge.source, edge.target) in removed:
            continue
        key = (edge.source, edge.target, edge.relation)
        if key in reweighted:
            edge = Edge(edge.source, edge.target, edge.relation, edge.extractor,
                        reweighted[key], edge.evidence)
        kept.append(edge)

    kept.extend(added)
    stats = {"add": len(added), "remove": len(removed), "reweight": len(reweighted),
             "unknown": unknown}
    if unknown:
        log.warning("overrides reference %d unknown title(s): %s", len(unknown), unknown[:5])
    log.info("overrides: +%d -%d ~%d", stats["add"], stats["remove"], stats["reweight"])
    return kept, stats


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def to_frame(edges: list[Edge]) -> pd.DataFrame:
    if not edges:
        return pd.DataFrame(
            columns=["source", "target", "relation", "extractor", "confidence", "evidence"]
        )
    frame = pd.DataFrame([edge.as_dict() for edge in edges])
    return (
        frame.drop_duplicates(subset=["source", "target", "relation"])
        .sort_values(["source", "target"])
        .reset_index(drop=True)
    )


def write_edges(edges: list[Edge], path: Path = EDGES_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = to_frame(edges)
    frame.to_csv(path, index=False, encoding="utf-8")
    log.info("edges: %d rows -> %s", len(frame), path)
    return path


def load_summaries(path: Path = config.LEGACY_ENRICHED_CSV) -> tuple[dict[str, str], list[str]]:
    """Lead text + canonical titles, from the already-scraped CSV."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame["title"] = frame["wiki_title"].where(frame["wiki_title"] != "", frame["label"])
    summaries = dict(zip(frame["title"], frame["wiki_summary"], strict=False))
    return summaries, sorted({t for t in frame["title"] if t})


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #

def run(use_wikidata: bool = True, use_cache: bool = True) -> pd.DataFrame:
    """Extract every edge the funnel can afford, then write data/raw/edges.csv.

    Extraction is deliberately wide; narrowing happens in the views. The funnel
    only works if this stage is allowed to over-produce.
    """
    config.ensure_dirs()
    summaries, titles = load_summaries()

    nodes = pd.read_csv(config.NODES_CSV, dtype=str, keep_default_na=False)
    nodes["title"] = nodes["title"].where(nodes.get("title", "").astype(str) != "", nodes["label"])
    # Wikidata labels are lower-case; edges arrive as labels, nodes as titles.
    title_by_qid = dict(zip(nodes["qid"], nodes["title"], strict=False))

    out = phrase_edges(summaries, titles)
    log.info("phrase edges: %d", len(out))

    if use_wikidata:
        qids = [q for q in nodes["qid"].dropna().unique()]
        title_lookup = {t.lower(): t for t in nodes["title"] if t}
        with PolitenessSession() as session:
            out += wikidata_edges(
                qids, title_by_qid, session, use_cache=use_cache, title_lookup=title_lookup
            )

    known = set(titles)
    out, override_stats = apply_overrides(out, known_titles=known)

    frame = to_frame(out)
    write_edges(out)
    log.info("edges total: %d (overrides %s)", len(frame), override_stats)
    return frame
