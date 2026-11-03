"""Mention statistics: counts, positions, section attribution.

The case-insensitivity check is a regression test for a mistake I made while
verifying by hand: the article links `[[well-ordering theorem]]` three times
lower-case and once capitalised, so a naive case-sensitive count reports 1
instead of 4. The extractor resolves titles case-insensitively, and that is
deliberate.
"""

from __future__ import annotations

import pandas as pd

from mathkg.mentions import fallback_rows, stats_for_text

WIKITEXT = """{{For|the film|Zorns Lemma}} [[File:x.svg|thumb|caption]]
Zorn's lemma is stated [[well-ordering theorem|here]] and again [[Well-ordering theorem]].
It uses the [[maximal ideal]] notion. See also [[well-ordering theorem]].

== Statement ==
Every [[partially ordered set]] with [[maximal ideal]]s has a [[upper bound]].

== Proof ==
Apply [[Zorn's lemma|the lemma]] using the [[axiom of choice]] and one more
[[well-ordering theorem]], plus [[Tychonoff's theorem]].
"""

LOOKUP = {
    "well-ordering theorem": "Well-ordering theorem",
    "maximal ideal": "Maximal ideal",
    "partially ordered set": "Partially ordered set",
    "upper bound": "Upper bound",
    "axiom of choice": "Axiom of choice",
    "tychonoff's theorem": "Tychonoff's theorem",
    "zorn's lemma": "Zorn's lemma",
}


def rows_by_target(rows):
    return {row["target"]: row for row in rows}


def test_counts_case_variants_together():
    rows = rows_by_target(stats_for_text("Zorn's lemma", WIKITEXT, LOOKUP))
    # 2 lower-case + 1 capitalised in the lead, + 1 lower-case inside Proof
    assert rows["Well-ordering theorem"]["occurrences"] == 4


def test_occurrences_within_one_section_accumulate():
    rows = rows_by_target(stats_for_text("Zorn's lemma", WIKITEXT, LOOKUP))
    assert rows["Maximal ideal"]["occurrences"] == 2


def test_self_links_are_ignored():
    rows = rows_by_target(stats_for_text("Zorn's lemma", WIKITEXT, LOOKUP))
    assert "Zorn's lemma" not in rows


def test_unknown_targets_are_ignored():
    rows = rows_by_target(stats_for_text("Zorn's lemma", WIKITEXT, LOOKUP))
    assert "x.svg" not in rows


def test_first_mention_controls_position_and_section():
    rows = rows_by_target(stats_for_text("Zorn's lemma", WIKITEXT, LOOKUP))
    well_ordering = rows["Well-ordering theorem"]
    # first mention is in the lead, before the "Statement" heading
    assert well_ordering["in_lead"] is True
    assert well_ordering["section_name"] == "(lead)"
    assert 0.0 <= well_ordering["relative_position"] <= 1.0


def test_proof_flag_marks_mentions_first_seen_in_a_proof():
    rows = rows_by_target(stats_for_text("Zorn's lemma", WIKITEXT, LOOKUP))
    # Axiom of choice appears only inside "== Proof =="
    assert rows["Axiom of choice"]["in_proof"] is True
    assert rows["Tychonoff's theorem"]["in_proof"] is True
    assert rows["Well-ordering theorem"]["in_proof"] is False


def test_fidelity_is_recorded():
    rows = stats_for_text("Zorn's lemma", WIKITEXT, LOOKUP)
    assert all(row["fidelity"] == "wikitext" for row in rows)


def test_empty_wikitext_returns_nothing():
    assert stats_for_text("X", "", LOOKUP) == []


def test_fallback_covers_every_edge_not_already_present():
    edges = pd.DataFrame(
        {
            "source": ["A", "A", "B"],
            "target": ["B", "C", "C"],
        }
    )
    covered = {("A", "B")}
    rows = fallback_rows(edges, covered)

    pairs = {(row["source"], row["target"]) for row in rows}
    assert pairs == {("A", "C"), ("B", "C")}
    assert all(row["fidelity"] == "summary" for row in rows)
