"""Mention-level edge statistics.

`edges.csv` says *that* A links to B. It cannot say where in A, how often, or
whether the mention sits in the lead, a statement section or a proof. All of
that is needed for a weighted view like "edge weight from position in the page,
number of occurrences, and negatively from the target's popularity" -- so this
module derives it from the wikitext that `mathkg/edges.py` already cached.

Two fidelity levels are recorded explicitly:

* ``fidelity="wikitext"`` -- real counts, offsets and section names.
* ``fidelity="summary"``  -- the article was never fetched, so the row is a
  degenerate guess (one mention, position 0, no section). It exists so that
  *every* edge has a row, but a view that trusts it should say so.

Rows are per ordered ``(source, target)`` pair, which is exactly the granularity
of `edges.csv`.
"""

from __future__ import annotations

import bisect
import logging
import re

import pandas as pd

from . import config
from . import edges as edges_module
from .titles import normalise_title

log = logging.getLogger(__name__)

MENTIONS_CSV = config.RAW_DIR / "mentions.csv"

COLUMNS = [
    "source",
    "target",
    "occurrences",
    "first_offset",
    "page_length",
    "relative_position",
    "section_index",
    "section_name",
    "in_lead",
    "in_proof",
    "fidelity",
]

LINK = re.compile(r"\[\[([^\]|]+)")


def _sections_with_offsets(wikitext: str) -> list[tuple[int, str]]:
    """``[(start_offset, heading), ...]`` for level 2-3 headings."""
    starts: list[tuple[int, str]] = [(0, "(lead)")]
    for match in re.finditer(r"\n(=={1,3})\s*(.+?)\s*\1\s*\n", wikitext):
        starts.append((match.end(), match.group(2)))
    return starts


def _section_at(starts: list[tuple[int, str]], offset: int) -> tuple[int, str]:
    """Which section contains ``offset`` (binary search over section starts)."""
    positions = [start for start, _ in starts]
    index = bisect.bisect_right(positions, offset) - 1
    index = max(0, min(index, len(starts) - 1))
    return index, starts[index][1]


def stats_for_text(
    source: str, wikitext: str, lookup: dict[str, str]
) -> list[dict[str, object]]:
    """One row per distinct target mentioned in ``source``."""
    if not wikitext:
        return []

    starts = _sections_with_offsets(wikitext)
    length = max(1, len(wikitext))
    aggregated: dict[str, dict[str, object]] = {}

    for match in LINK.finditer(wikitext):
        target = lookup.get(normalise_title(match.group(1)).lower())
        if not target or target == source:
            continue
        offset = match.start()
        section_index, section_name = _section_at(starts, offset)
        record = aggregated.get(target)
        if record is None:
            aggregated[target] = {
                "source": source,
                "target": target,
                "occurrences": 1,
                "first_offset": offset,
                "page_length": length,
                "relative_position": offset / length,
                "section_index": section_index,
                "section_name": section_name,
                "in_lead": section_index == 0,
                "in_proof": bool(edges_module.PROOF_HEADING.search(section_name)),
                "fidelity": "wikitext",
            }
        else:
            record["occurrences"] = int(record["occurrences"]) + 1
            if offset < int(record["first_offset"]):
                record.update(
                    first_offset=offset,
                    relative_position=offset / length,
                    section_index=section_index,
                    section_name=section_name,
                    in_lead=section_index == 0,
                    in_proof=bool(edges_module.PROOF_HEADING.search(section_name)),
                )
    return list(aggregated.values())


def fallback_rows(edges: pd.DataFrame, covered: set[tuple[str, str]]) -> list[dict[str, object]]:
    """Degenerate rows so every edge has a mention record, clearly labelled."""
    rows = []
    for _, row in edges.iterrows():
        key = (row["source"], row["target"])
        if key in covered:
            continue
        rows.append(
            {
                "source": row["source"],
                "target": row["target"],
                "occurrences": 1,
                "first_offset": 0,
                "page_length": 0,
                "relative_position": 0.0,
                "section_index": -1,
                "section_name": "",
                "in_lead": True,
                "in_proof": False,
                "fidelity": "summary",
            }
        )
    return rows


def build(
    wikitexts: dict[str, str],
    titles: list[str],
    edge_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Mention statistics for every edge, with fallbacks for unfetched pages."""
    lookup = {title.lower(): title for title in titles}
    rows: list[dict[str, object]] = []

    for source, wikitext in wikitexts.items():
        if source not in lookup.values():
            continue
        rows.extend(stats_for_text(source, wikitext, lookup))

    log.info("mentions: %d (source, target) pairs from wikitext", len(rows))

    if edge_frame is not None:
        covered = {(row["source"], row["target"]) for row in rows}
        missing = fallback_rows(edge_frame, covered)
        if missing:
            log.info("mentions: %d fallback rows for unfetched pages", len(missing))
        rows.extend(missing)

    frame = pd.DataFrame(rows, columns=COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(["source", "target"]).reset_index(drop=True)


def run(use_cache: bool = True) -> pd.DataFrame:
    """Fetch/read wikitext, compute stats, write ``data/raw/mentions.csv``."""
    config.ensure_dirs()
    _, titles = edges_module.load_summaries()
    edge_frame = pd.read_csv(config.RAW_DIR / "edges.csv", dtype=str, keep_default_na=False)
    targets = edges_module.core_titles()

    from .http import PolitenessSession

    with PolitenessSession() as session:
        wikitexts = edges_module.fetch_wikitext(targets, session, use_cache=use_cache)

    frame = build(wikitexts, titles, edge_frame)
    frame.to_csv(MENTIONS_CSV, index=False, encoding="utf-8")
    log.info("mentions: %d rows -> %s", len(frame), MENTIONS_CSV)
    return frame
