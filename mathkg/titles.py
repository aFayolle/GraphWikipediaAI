"""Title <-> URL normalisation.

The graph is keyed on Wikipedia titles, but the original pipeline keyed nodes
on ``en_url`` while the scraper emitted URLs straight out of an HTML href.
Those two never matched exactly (percent-encoding, ``_`` vs space, ``#frag``,
redirects), which is why 586 articles existed twice in the graph under two
different keys. Everything now goes through :func:`normalise_title`.
"""

from __future__ import annotations

from urllib.parse import unquote, urlparse

WIKI_PREFIXES = (
    "wikipedia.org/wiki/",
    "wikipedia.org/w/index.php",
)


def title_from_url(url: str) -> str | None:
    """Extract the article title from a Wikipedia URL, or ``None``."""
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    if not url or url.lower() in {"nan", "none"}:
        return None

    if url.startswith("/wiki/"):
        raw = url[len("/wiki/"):]
    else:
        parsed = urlparse(url)
        if any(prefix in f"{parsed.netloc}{parsed.path}" for prefix in WIKI_PREFIXES) or (
            parsed.netloc.endswith("wikipedia.org") and parsed.path.startswith("/wiki/")
        ):
            raw = parsed.path[len("/wiki/"):] if parsed.path.startswith("/wiki/") else ""
        elif "://" not in url:
            raw = url  # already a bare title
        else:
            return None

    return normalise_title(raw) or None


def normalise_title(title: str | None) -> str:
    """Canonical, comparable form of an article title.

    Decodes percent-escapes, drops any ``#fragment``, turns underscores into
    spaces and collapses whitespace. Deliberately does *not* touch letter case:
    the MediaWiki API already returns canonical casing, and guessing here would
    merge distinct titles.
    """
    if title is None:
        return ""
    text = str(title)
    if "#" in text:
        text = text.split("#", 1)[0]
    text = unquote(text)
    text = text.replace("_", " ")
    text = " ".join(text.split())
    return text.strip()


def url_from_title(title: str, lang: str = "en") -> str:
    """Inverse of :func:`title_from_url` (for display and legacy matching)."""
    from urllib.parse import quote

    return f"https://{lang}.wikipedia.org/wiki/{quote(normalise_title(title).replace(' ', '_'))}"


def is_article_title(title: str) -> bool:
    """Reject namespace prefixes (``File:``, ``Category:``, ``Help:`` ...).

    The original scraper used a bare ``href="/wiki/..."`` regex and relied on a
    set-membership check to drop these, so non-article links consumed budget and
    only disappeared by accident.
    """
    normalised = normalise_title(title)
    if not normalised:
        return False
    _, separator, _ = normalised.partition(":")
    return not separator


def article_titles(titles: list[str]) -> list[str]:
    """Filter to namespace-0 article titles, de-duplicated, order preserved."""
    seen: set[str] = set()
    out: list[str] = []
    for title in titles:
        normalised = normalise_title(title)
        if not normalised or normalised in seen:
            continue
        if not is_article_title(normalised):
            continue
        seen.add(normalised)
        out.append(normalised)
    return out
