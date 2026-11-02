"""Title normalisation: the fix for the 586 duplicated articles."""

from __future__ import annotations

import pytest

from mathkg.titles import (
    article_titles,
    is_article_title,
    normalise_title,
    title_from_url,
    url_from_title,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Algebraic_geometry", "Algebraic geometry"),
        ("Displacement_(geometry)#Cartesian", "Displacement (geometry)"),
        ("  spaced   out  ", "spaced out"),
        ("C%2B%2B", "C++"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalise_title(raw, expected):
    assert normalise_title(raw) == expected


def test_title_from_url_handles_full_and_relative_urls():
    assert title_from_url("https://en.wikipedia.org/wiki/Angle") == "Angle"
    assert title_from_url("/wiki/Angle") == "Angle"
    assert title_from_url("https://en.wikipedia.org/wiki/Fibonacci_number") == "Fibonacci number"
    assert title_from_url("/wiki/Displacement_(geometry)") == "Displacement (geometry)"
    assert title_from_url("Angle") == "Angle"
    assert title_from_url("https://example.org/not-wikipedia") is None
    assert title_from_url("nan") is None


def test_roundtrip():
    title = "Displacement (geometry)"
    assert title_from_url(url_from_title(title)) == title


@pytest.mark.parametrize(
    ("title", "is_article"),
    [
        ("Algebra", True),
        ("File:Diagram.svg", False),
        ("Category:Algebra", False),
        ("Help:Contents", False),
        ("", False),
    ],
)
def test_namespace_filter(title, is_article):
    assert is_article_title(title) is is_article


def test_article_titles_dedupes_preserving_order():
    raw = ["B", "A", "B", "File:X.png", "A", "C"]
    assert article_titles(raw) == ["B", "A", "C"]
