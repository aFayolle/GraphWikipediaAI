"""MediaWiki API client: continuation merging, redirects, missing pages.

The first test is the regression test for a bug found while writing this
refactor: ``_query`` accumulated links across continuation responses and then
returned the *last* payload, so every article except the last one in a batch
came back with zero links.
"""

from __future__ import annotations

from mathkg.fetch import MediaWikiClient


class StubSession:
    """Replays scripted API payloads, no network."""

    max_retries = 3

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def get_json(self, url, params):
        self.calls.append(dict(params))
        return self.payloads.pop(0)

    @staticmethod
    def _sleep(_seconds):
        pass

    def close(self):
        pass


def make_client(payloads):
    return MediaWikiClient(session=StubSession(payloads), use_cache=False)


def page(title, links=(), ns=0, missing=False, extract=""):
    entry = {"title": title, "ns": ns, "extract": extract}
    if missing:
        entry["missing"] = True
    else:
        entry["links"] = [{"title": name} for name in links]
    return entry


def test_continuation_pages_are_merged():
    """`prop=links` returns 500 links at a time; all chunks must be kept."""
    client = make_client(
        [
            {
                "query": {"pages": [page("Mathematics", ["Algebra"]), page("Topology")]},
                "continue": {"plcontinue": "tok", "continue": "||"},
            },
            {
                "query": {"pages": [page("Mathematics", ["Geometry", "Number theory"])]},
            },
        ]
    )

    result = client._query(["Mathematics", "Topology"])

    maths = next(p for p in result["query"]["pages"] if p["title"] == "Mathematics")
    assert [link["title"] for link in maths["links"]] == [
        "Algebra",
        "Geometry",
        "Number theory",
    ]
    assert len(client.session.calls) == 2
    assert client.session.calls[1]["plcontinue"] == "tok"


def test_redirects_resolve_to_one_page():
    client = make_client(
        [
            {
                "query": {
                    "normalized": [{"from": "euclidean plane", "to": "Euclidean plane"}],
                    "redirects": [{"from": "Euclidean plane", "to": "Plane"}],
                    "pages": [page("Plane", ["Line (geometry)"])],
                }
            }
        ]
    )

    result = client.fetch(["euclidean plane"])

    assert result["euclidean plane"].title == "Plane"
    assert result["euclidean plane"].exists is True
    assert result["euclidean plane"].links == ["Line (geometry)"]


def test_missing_page_is_reported_not_raised():
    client = make_client([{"query": {"pages": [page("Nonexistent", missing=True)]}}])

    result = client.fetch(["Nonexistent"])

    assert result["Nonexistent"].exists is False
    assert result["Nonexistent"].links == []


def test_non_article_namespaces_are_dropped():
    """A `Category:` title is filtered out before it is ever requested."""
    client = make_client([])

    result = client.fetch(["Category:Algebra"])

    assert result == {}
    assert client.session.calls == []


def test_extract_becomes_summary():
    client = make_client(
        [{"query": {"pages": [page("Algebra", extract="  Algebra is a branch.  ")]}}]
    )

    assert client.fetch(["Algebra"])["Algebra"].summary == "Algebra is a branch."


def test_transient_api_error_is_retried():
    client = make_client(
        [
            {"error": {"code": "maxlag", "info": "Waiting for a database server"}},
            {"query": {"pages": [page("Algebra", ["Ring"])]}},
        ]
    )

    result = client.fetch(["Algebra"])

    assert result["Algebra"].links == ["Ring"]
    assert len(client.session.calls) == 2
