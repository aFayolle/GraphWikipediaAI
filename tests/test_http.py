"""HTTP politeness and retry behaviour.

The first test is the regression test for the bug that could kill a multi-hour
run: the original code touched ``resp`` inside its ``except`` block, so a
connection error raised ``UnboundLocalError`` from inside the error handler.
"""

from __future__ import annotations

import pytest
import requests

from mathkg.http import PolitenessSession


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload or {"ok": True}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    """Stand-in for ``requests.Session`` that replays scripted outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.headers: dict[str, str] = {}

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        pass


def make_client(outcomes, max_retries=3):
    sleeps: list[float] = []
    client = PolitenessSession(
        min_interval=0, max_retries=max_retries, sleep=sleeps.append
    )
    client.session = FakeSession(outcomes)
    return client, sleeps


def test_connection_error_is_retried_not_unbound_local():
    client, sleeps = make_client(
        [requests.ConnectionError("boom"), FakeResponse(payload={"ok": "second"})]
    )

    assert client.get_json("https://example.org", {}) == {"ok": "second"}
    assert client.session.calls == 2
    assert len(sleeps) == 1  # backed off once, then succeeded


def test_timeout_is_retried():
    client, _ = make_client([requests.Timeout("slow"), FakeResponse()])
    assert client.get_json("https://example.org", {}) == {"ok": True}
    assert client.session.calls == 2


def test_429_honours_retry_after():
    client, sleeps = make_client(
        [FakeResponse(status_code=429, headers={"Retry-After": "7"}), FakeResponse()]
    )

    client.get_json("https://example.org", {})

    assert sleeps == [7.0]


def test_5xx_retried_then_succeeds():
    client, _ = make_client([FakeResponse(status_code=503), FakeResponse()])
    assert client.get_json("https://example.org", {}) == {"ok": True}
    assert client.session.calls == 2


def test_404_is_not_retried():
    client, _ = make_client([FakeResponse(status_code=404)])

    with pytest.raises(requests.HTTPError):
        client.get_json("https://example.org", {})

    assert client.session.calls == 1


def test_gives_up_after_max_retries():
    client, _ = make_client([FakeResponse(status_code=503)] * 3, max_retries=3)

    with pytest.raises(RuntimeError):
        client.get_json("https://example.org", {})

    assert client.session.calls == 3


def test_user_agent_is_sent():
    client = PolitenessSession(min_interval=0)
    assert "MathKG" in client.session.headers["User-Agent"]
    client.close()
