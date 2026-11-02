"""Importing the existing df_filtered.csv, without eval() and without duplicates.

Two regressions covered here:

* ``eval()`` ran whatever was inside the ``neighbour`` cell. A CSV is data.
* Articles that exist twice (Wikidata sitelink vs redirect target) used to
  become two nodes with near-identical adjacency and no edge between them.
"""

from __future__ import annotations

import pandas as pd

from mathkg.fetch import from_legacy_csv

COLUMNS = [
    "qid", "label", "en_url", "sitelinks", "statements",
    "wiki_title", "wiki_fullurl", "wiki_summary", "neighbour",
]


def write_legacy(tmp_path, rows):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    path = tmp_path / "df_filtered.csv"
    # Same encoding as the real file: it has a BOM.
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_redirect_alias_is_merged_into_one_node(tmp_path):
    """`Euclidean plane` -> `Plane` must not create two nodes."""
    path = write_legacy(
        tmp_path,
        [
            {
                "qid": "Q1", "label": "Euclidean plane",
                "en_url": "https://en.wikipedia.org/wiki/Euclidean_plane",
                "sitelinks": 30, "statements": 10,
                "wiki_title": "Plane",
                "wiki_fullurl": "https://en.wikipedia.org/wiki/Plane",
                "wiki_summary": "In mathematics, a plane is flat.",
                "neighbour": "['https://en.wikipedia.org/wiki/Line_(geometry)']",
            },
            {
                "qid": "Q2", "label": "Plane",
                "en_url": "https://en.wikipedia.org/wiki/Plane",
                "sitelinks": 90, "statements": 40,
                "wiki_title": "Plane",
                "wiki_fullurl": "https://en.wikipedia.org/wiki/Plane",
                "wiki_summary": "In mathematics, a plane is flat.",
                "neighbour": "['https://en.wikipedia.org/wiki/Line_(geometry)']",
            },
            {
                "qid": "Q3", "label": "Line",
                "en_url": "https://en.wikipedia.org/wiki/Line_(geometry)",
                "sitelinks": 70, "statements": 20,
                "wiki_title": "Line (geometry)",
                "wiki_fullurl": "https://en.wikipedia.org/wiki/Line_(geometry)",
                "wiki_summary": "A line is straight.",
                "neighbour": "[]",
            },
        ],
    )

    nodes, links = from_legacy_csv(path)

    assert sorted(nodes["title"]) == ["Line (geometry)", "Plane"]
    plane = nodes.set_index("title").loc["Plane"]
    assert plane["n_qids"] == 2          # both Wikidata items are accounted for
    assert plane["sitelinks"] == 90      # most-linked representative wins

    # Both aliases produced the same edge; it must appear once, once canonical.
    pairs = set(zip(links["source"], links["target"], strict=False))
    assert pairs == {("Plane", "Line (geometry)")}


def test_no_eval_of_csv_content(tmp_path, capsys):
    """A hostile cell must be inert."""
    malicious = "__import__('builtins').print('PWNED')"
    path = write_legacy(
        tmp_path,
        [
            {
                "qid": "Q1", "label": "A", "en_url": "https://en.wikipedia.org/wiki/A",
                "sitelinks": 1, "statements": 1, "wiki_title": "A",
                "wiki_fullurl": "https://en.wikipedia.org/wiki/A",
                "wiki_summary": "", "neighbour": malicious,
            }
        ],
    )

    nodes, links = from_legacy_csv(path)

    assert "PWNED" not in capsys.readouterr().out
    assert nodes.shape[0] == 1
    assert links.empty


def test_duplicate_qid_rows_collapse(tmp_path):
    row = {
        "qid": "Q7754", "label": "mathematical analysis",
        "en_url": "https://en.wikipedia.org/wiki/Mathematical_analysis",
        "sitelinks": 136, "statements": 78, "wiki_title": "Mathematical analysis",
        "wiki_fullurl": "https://en.wikipedia.org/wiki/Mathematical_analysis",
        "wiki_summary": "", "neighbour": "[]",
    }
    path = write_legacy(tmp_path, [row, dict(row)])

    nodes, _ = from_legacy_csv(path)
    assert len(nodes) == 1


def test_self_links_are_not_emitted(tmp_path):
    path = write_legacy(
        tmp_path,
        [
            {
                "qid": "Q1", "label": "A", "en_url": "https://en.wikipedia.org/wiki/A",
                "sitelinks": 1, "statements": 1, "wiki_title": "A",
                "wiki_fullurl": "https://en.wikipedia.org/wiki/A",
                "wiki_summary": "",
                "neighbour": "['https://en.wikipedia.org/wiki/A']",
            }
        ],
    )

    _, links = from_legacy_csv(path)
    assert links.empty
