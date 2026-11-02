"""Wikipedia stage: titles -> ``pages.csv`` + ``links.csv``, via the action API.

Replaces the HTML-scraping loop in the original ``enrichment.py``. What was
wrong there, and what this does instead:

* **Browser impersonation.** The scraper sent a spoofed Chrome User-Agent over
  16 unthrottled threads. This uses one honest bot UA, one request at a time,
  with a minimum interval and ``maxlag``.
* **Fragile scraping.** Edges came from
  ``re.search('From Wikipedia...(<p>.*?<h2)')``, i.e. only the lead section,
  and any layout change produced a silent empty node after three 3-second
  retries. The action API returns structured links and never needs a regex.
* **No redirect handling.** ``redirects=1`` resolves every title to its
  canonical page, which removes the duplicated-article problem at the source.
* **No caching.** Every run re-fetched 6,867 pages. Responses are now cached
  on disk and never re-fetched.

Link scope note: the API gives *all* namespace-0 links of a page. That is
deliberately different from the original lead-section-only extraction -- the
lead-only graph measures "co-mentioned in the introduction", the full one
measures "linked from the article". Both are defensible, but only one of them
was ever documented. :func:`from_legacy_csv` preserves the old lead-only data
so the two can be compared.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config
from .http import PolitenessSession
from .titles import article_titles, normalise_title, title_from_url

log = logging.getLogger(__name__)


@dataclass
class Page:
    """Everything one Wikipedia article contributes to the graph."""

    title: str
    exists: bool
    ns: int = 0
    summary: str = ""
    links: list[str] = field(default_factory=list)


class MediaWikiClient:
    """Batched, cached, polite client for the MediaWiki action API."""

    def __init__(
        self,
        lang: str = config.WIKI_LANG,
        session: PolitenessSession | None = None,
        titles_per_request: int = config.TITLES_PER_REQUEST,
        use_cache: bool = True,
    ) -> None:
        self.lang = lang
        self.api_url = config.WIKI_API_URL.format(lang=lang)
        self.session = session or PolitenessSession()
        self.titles_per_request = titles_per_request
        self.use_cache = use_cache
        self._cache_dir = config.CACHE_DIR / "wikipedia" / lang

    # -- cache -------------------------------------------------------------- #

    def _cache_path(self, titles: list[str]) -> Path:
        key = hashlib.sha1("\n".join(titles).encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.json"

    # -- one API call ------------------------------------------------------- #

    def _query(self, titles: list[str]) -> dict:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
            "prop": "info|extracts|links",
            "inprop": "url",
            "exintro": "1",
            "explaintext": "1",
            "exlimit": str(self.titles_per_request),
            "plnamespace": "0",
            "pllimit": "max",
            "maxlag": "5",
            "titles": "|".join(titles),
        }

        payload = self.session.get_json(self.api_url, params)

        # `maxlag` and read-only errors are transient by design: retry them.
        while "error" in payload and payload["error"].get("code") in {
            "maxlag",
            "readonly",
            "ratelimited",
        }:
            log.warning("API asked us to back off (%s)", payload["error"]["code"])
            self.session._sleep(5.0)
            payload = self.session.get_json(self.api_url, params)

        if "error" in payload:
            raise RuntimeError(f"API error: {payload['error']}")

        # Continue fetching link pages for the batch.
        links_by_title: dict[str, list[str]] = {}
        while True:
            query = payload.get("query", {})
            for page in query.get("pages", []):
                if page.get("missing") or page.get("ns") != 0:
                    continue
                links_by_title.setdefault(page["title"], []).extend(
                    link["title"] for link in page.get("links", [])
                )
            if "continue" not in payload:
                break
            follow = dict(params)
            follow.update(payload["continue"])
            payload = self.session.get_json(self.api_url, follow)

        return payload

    # -- public ------------------------------------------------------------- #

    def fetch(self, titles: list[str]) -> dict[str, Page]:
        """Fetch many titles, returning ``{input_title: Page}``."""
        wanted = article_titles(titles)
        pages: dict[str, Page] = {}

        for start in range(0, len(wanted), self.titles_per_request):
            batch = wanted[start:start + self.titles_per_request]
            pages.update(self._fetch_batch(batch))
            log.info(
                "wikipedia: %d/%d titles", min(start + len(batch), len(wanted)), len(wanted)
            )
        return pages

    def _fetch_batch(self, batch: list[str]) -> dict[str, Page]:
        cache_path = self._cache_path(batch)
        if self.use_cache and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = self._query(batch)
            if self.use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload), encoding="utf-8")

        return self._parse(batch, payload)

    @staticmethod
    def _parse(batch: list[str], payload: dict) -> dict[str, Page]:
        query = payload.get("query", {})

        # Follow the normalisation + redirect chain back to the input title.
        alias: dict[str, str] = {}
        for step in query.get("normalized", []) + query.get("redirects", []):
            alias[normalise_title(step["from"])] = normalise_title(step["to"])

        def canonical(title: str) -> str:
            seen = set()
            current = normalise_title(title)
            while current in alias and current not in seen:
                seen.add(current)
                current = alias[current]
            return current

        by_title = {
            normalise_title(page["title"]): page
            for page in query.get("pages", [])
        }

        result: dict[str, Page] = {}
        for original in batch:
            target = canonical(original)
            page = by_title.get(target)
            if page is None or page.get("missing"):
                result[original] = Page(title=target, exists=False)
                continue

            result[original] = Page(
                title=normalise_title(page["title"]),
                exists=True,
                ns=page.get("ns", 0),
                summary=(page.get("extract") or "").strip(),
                links=article_titles([link["title"] for link in page.get("links", [])]),
            )
        return result


# --------------------------------------------------------------------------- #
# Offline import of the legacy scrape
# --------------------------------------------------------------------------- #

def from_legacy_csv(path: Path = config.LEGACY_ENRICHED_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild ``nodes.csv``/``links.csv`` from the existing ``df_filtered.csv``.

    This preserves the lead-section-only edges that were already scraped, so
    the pipeline can run end to end with no network access and without throwing
    away hours of scraping. It also *fixes* the two data defects on the way in:
    duplicate ``en_url`` rows and redirect aliases are merged onto one canonical
    title per article.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"qid", "label", "en_url", "sitelinks", "neighbour"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    # Optional enrichment columns: present in the real file, not in fixtures.
    for optional in ("wiki_title", "wiki_fullurl", "wiki_summary"):
        if optional not in df.columns:
            df[optional] = ""

    # Canonical title: the resolved page title if we have it, else the URL.
    titles = []
    for _, row in df.iterrows():
        title = normalise_title(row.get("wiki_title", "")) or title_from_url(row["en_url"])
        titles.append(title or normalise_title(row["label"]))
    df["title"] = titles
    df = df[df["title"] != ""]

    # URL -> canonical title, so neighbour URLs land on the same keys as nodes.
    url_to_title: dict[str, str] = {}
    for _, row in df.iterrows():
        for column in ("en_url", "wiki_fullurl"):
            raw = str(row.get(column, "")).strip()
            if raw and raw.lower() != "nan":
                url_to_title[raw] = row["title"]

    df["sitelinks"] = pd.to_numeric(df["sitelinks"], errors="coerce").fillna(0).astype(int)

    # One node per canonical article: keep the most-linked representative and
    # remember every qid that maps onto it.
    nodes = (
        df.sort_values("sitelinks", ascending=False)
        .groupby("title", as_index=False)
        .agg(
            qid=("qid", "first"),
            label=("label", "first"),
            en_url=("en_url", "first"),
            sitelinks=("sitelinks", "max"),
            n_qids=("qid", "nunique"),
            summary=("wiki_summary", "first"),
        )
    )

    edge_rows = []
    for _, row in df.iterrows():
        try:
            neighbours = ast.literal_eval(row["neighbour"])
        except (ValueError, SyntaxError):
            neighbours = []
        if not isinstance(neighbours, list):
            continue
        for url in neighbours:
            target = url_to_title.get(str(url).strip()) or title_from_url(str(url))
            target = normalise_title(target)
            if target and target != row["title"]:
                edge_rows.append({"source": row["title"], "target": target})

    links = pd.DataFrame(edge_rows, columns=["source", "target"]).drop_duplicates()
    log.info(
        "legacy import: %d rows -> %d nodes, %d edges (lead-section scope)",
        len(df),
        len(nodes),
        len(links),
    )
    return nodes, links


def write_raw(nodes: pd.DataFrame, links: pd.DataFrame,
              out_nodes: Path = config.NODES_CSV,
              out_links: Path = config.LINKS_CSV) -> tuple[Path, Path]:
    """Persist the normalised intermediates."""
    config.ensure_dirs()
    nodes.to_csv(out_nodes, index=False, encoding="utf-8")
    links.to_csv(out_links, index=False, encoding="utf-8")
    log.info("wrote %s (%d nodes) and %s (%d edges)", out_nodes, len(nodes), out_links, len(links))
    return out_nodes, out_links


def run(
    nodes_csv: Path = config.NODES_CSV,
    out_nodes: Path = config.NODES_CSV,
    out_links: Path = config.LINKS_CSV,
) -> tuple[Path, Path]:
    """Fetch every node title through the action API and write intermediates."""
    nodes = pd.read_csv(nodes_csv, dtype=str, keep_default_na=False)
    if "title" not in nodes.columns:
        nodes["title"] = ""

    titles = [
        normalise_title(t) or title_from_url(url) or normalise_title(label)
        for t, url, label in zip(
            nodes["title"], nodes["en_url"], nodes["label"], strict=False
        )
    ]

    client = MediaWikiClient()
    try:
        pages = client.fetch([t for t in titles if t])
    finally:
        client.session.close()

    rows_nodes, rows_links = [], []
    for (_, row), title in zip(nodes.iterrows(), titles, strict=False):
        page = pages.get(title)
        canonical = page.title if page and page.exists else title
        rows_nodes.append(
            {
                "qid": row["qid"],
                "label": row["label"],
                "en_url": row["en_url"],
                "sitelinks": row["sitelinks"],
                "title": canonical,
                "summary": page.summary if page else "",
                "exists": bool(page and page.exists),
            }
        )
        if page and page.exists:
            for target in page.links:
                if target != canonical:
                    rows_links.append({"source": canonical, "target": target})

    nodes_out = pd.DataFrame(rows_nodes)
    links_out = pd.DataFrame(rows_links, columns=["source", "target"]).drop_duplicates()
    return write_raw(nodes_out, links_out, out_nodes, out_links)
