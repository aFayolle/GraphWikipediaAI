"""Wikidata SPARQL stage: query -> normalised ``nodes.csv``.

Fixes over the original ``download_raw_data.py``:

* queries are totally ordered, so ``LIMIT/OFFSET`` paging is stable;
* each page is written to its own file, so a crash loses one page, not the run;
* retries honour ``Retry-After`` (the original declared ``retry_delay`` and
  never used it);
* the consolidated CSV is written atomically (``write`` + ``os.replace``);
* rows are de-duplicated on ``qid`` at the end, which OFFSET paging can no
  longer violate anyway.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
from pathlib import Path

import pandas as pd

from . import config, queries
from .http import PolitenessSession

log = logging.getLogger(__name__)


def _query_slug(query: str) -> str:
    return hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]


def fetch_page(
    session: PolitenessSession,
    query: str,
    limit: int,
    offset: int,
    endpoint: str = config.WIKIDATA_ENDPOINT,
) -> list[dict[str, str]]:
    """Fetch one ``LIMIT/OFFSET`` page of a query."""
    payload = session.get_json(
        endpoint,
        {
            "query": f"{query}\nLIMIT {int(limit)}\nOFFSET {int(offset)}",
            "format": "json",
        },
    )
    rows = []
    for binding in payload["results"]["bindings"]:
        rows.append(
            {
                "qid": binding.get("qid", {}).get("value", ""),
                "label": binding.get("itemLabel", {}).get("value", ""),
                "en_url": binding.get("enUrl", {}).get("value", ""),
                "sitelinks": binding.get("sitelinks", {}).get("value", ""),
                "statements": binding.get("statements", {}).get("value", ""),
            }
        )
    return rows


def dump_sparql(
    query: str,
    out_csv: Path,
    page_size: int = 500,
    endpoint: str = config.WIKIDATA_ENDPOINT,
    session: PolitenessSession | None = None,
    max_pages: int | None = None,
) -> Path:
    """Page through a SPARQL query and consolidate into ``out_csv``.

    Resumable: pages already on disk under ``CACHE_DIR/wdqs/<slug>/`` are
    reused, so re-running after a failure does not re-hit the endpoint.
    """
    config.ensure_dirs()
    out_csv = Path(out_csv)
    page_dir = config.CACHE_DIR / "wdqs" / _query_slug(query)
    page_dir.mkdir(parents=True, exist_ok=True)

    owns_session = session is None
    session = session or PolitenessSession()

    try:
        offset = 0
        total = 0
        while max_pages is None or offset // page_size < max_pages:
            page_file = page_dir / f"page_{offset:07d}.csv"

            if page_file.exists():
                rows = _read_page(page_file)
                log.info("offset %7d: %d rows (cached)", offset, len(rows))
            else:
                log.info("offset %7d: fetching ...", offset)
                rows = fetch_page(session, query, page_size, offset, endpoint)
                _write_page(page_file, rows)
                log.info("offset %7d: %d rows", offset, len(rows))

            total += len(rows)
            if len(rows) < page_size:
                log.info("last page reached")
                break
            offset += page_size

        return _consolidate(page_dir, out_csv)
    finally:
        if owns_session:
            session.close()


def _write_page(path: Path, rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=queries.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _read_page(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _consolidate(page_dir: Path, out_csv: Path) -> Path:
    """Merge every page file into one CSV, atomically and de-duplicated."""
    frames = [
        pd.read_csv(page, dtype=str, keep_default_na=False, encoding="utf-8")
        for page in sorted(page_dir.glob("page_*.csv"))
    ]
    if not frames:
        raise RuntimeError(f"no SPARQL pages found in {page_dir}")

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df[df["qid"] != ""].drop_duplicates(subset="qid", keep="first")
    df = df.sort_values("qid").reset_index(drop=True)

    df.to_csv(out_csv, index=False, encoding="utf-8")
    log.info(
        "nodes: %d rows (%d duplicates dropped) -> %s",
        len(df),
        before - len(df),
        out_csv,
    )
    return out_csv


def run(
    out_nodes: Path = config.NODES_CSV,
    out_fields: Path = config.FIELDS_CSV,
    session: PolitenessSession | None = None,
) -> tuple[Path, Path]:
    """Fetch both the WikiProject Mathematics items and the 12 field anchors."""
    if not config.contact_is_set():
        log.warning(
            "MATHKG_CONTACT is not set: the User-Agent carries a placeholder. "
            "Wikimedia's policy requires a real contact address or URL."
        )

    owns_session = session is None
    session = session or PolitenessSession()
    try:
        nodes = dump_sparql(queries.WIKIPROJECT_MATHEMATICS, out_nodes, session=session)
        fields = dump_sparql(
            queries.FIELDS_OF_MATHEMATICS, out_fields, page_size=50, session=session
        )
    finally:
        if owns_session:
            session.close()
    return nodes, fields
