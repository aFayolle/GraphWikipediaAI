"""Central configuration: paths, HTTP identity, palette, tunables.

Everything that used to be scattered (user agents, output paths, magic
numbers, duplicated palette) lives here so the pipeline has one source of
truth.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Everything generated goes under DATA_DIR; override with MATHKG_DATA.
DATA_DIR = Path(os.environ.get("MATHKG_DATA") or (PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"          # normalised intermediates (nodes/links/pages)
CACHE_DIR = DATA_DIR / "cache"      # raw HTTP responses, never re-fetched
ARTIFACT_DIR = DATA_DIR / "artifacts"  # graphs, json, html

# The inputs that already existed in the repo. Kept in place, read-only.
LEGACY_ENRICHED_CSV = PROJECT_ROOT / "df_filtered.csv"
LEGACY_FIELDS_CSV = PROJECT_ROOT / "fields_of_math.csv"
LEGACY_NODES_CSV = PROJECT_ROOT / "wikiproject_math_nodes.csv"

NODES_CSV = RAW_DIR / "nodes.csv"    # one row per Wikidata item
LINKS_CSV = RAW_DIR / "links.csv"    # one row per (source, target) link
FIELDS_CSV = RAW_DIR / "fields.csv"  # the 12 anchor "fields of mathematics"

# --------------------------------------------------------------------------- #
# HTTP identity
# --------------------------------------------------------------------------- #
# Wikimedia's User-Agent policy requires a *real* contact address or URL, and
# forbids browser impersonation. Set it once:
#
#     export MATHKG_CONTACT="you@example.org"
#
WIKI_CONTACT = os.environ.get("MATHKG_CONTACT", "").strip()

WIKI_LANG = "en"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WIKI_API_URL = "https://{lang}.wikipedia.org/w/api.php"

# Politeness: one request at a time, with a floor between requests.
MIN_REQUEST_INTERVAL = 0.5   # seconds between two API calls
MAX_RETRIES = 5
TITLES_PER_REQUEST = 20      # extracts are limited to 20 titles per query

# --------------------------------------------------------------------------- #
# Graph / rendering tunables
# --------------------------------------------------------------------------- #

DEFAULT_MAX_DEGREE: int | None = None  # None = keep everything
ANCHOR_NEIGHBOURS = 3                  # how many anchors a node is coloured from
UNREACHABLE_COLOR = "#B0B0B0"          # grey: no path to any anchor

# Anchor palette, assigned positionally in `fields_of_math.csv` order, so a
# field keeps its colour when another field is added or removed.
# Tableau-10, but *without* its grey (#BAB0AC): a grey anchor is
# indistinguishable from UNREACHABLE_COLOR and reads as "no data".
ANCHOR_PALETTE = [
    "#4E79A7",  # bleu
    "#F28E2B",  # orange
    "#E15759",  # rouge
    "#76B7B2",  # bleu-vert
    "#59A14F",  # vert
    "#EDC948",  # jaune
    "#B07AA1",  # violet
    "#FF9DA7",  # rose
    "#9C755F",  # brun
    "#17BECF",  # cyan
    "#1F77B4",  # bleu profond
    "#2CA02C",  # vert profond
]


def user_agent() -> str:
    """Return the User-Agent string sent to Wikimedia endpoints.

    The contact is deliberately *not* hard-coded: a placeholder address in a
    User-Agent is exactly what Wikimedia's policy asks you not to do.
    """
    contact = WIKI_CONTACT or "MATHKG_CONTACT-NOT-SET (set MATHKG_CONTACT)"
    return f"MathKG/1.0 ({contact}) python-requests"


def contact_is_set() -> bool:
    return bool(WIKI_CONTACT)


def ensure_dirs() -> None:
    for d in (RAW_DIR, CACHE_DIR, ARTIFACT_DIR):
        d.mkdir(parents=True, exist_ok=True)
