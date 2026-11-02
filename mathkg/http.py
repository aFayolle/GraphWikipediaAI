"""Shared HTTP behaviour for every Wikimedia endpoint.

The original code had three different identities (a bot UA with a placeholder
address, another one hard-coded inside a method, and a spoofed Chrome UA used
for thousands of scrapes) and no retry logic at all. This module gives both
stages one polite client:

* one honest User-Agent, with a real contact if ``MATHKG_CONTACT`` is set;
* a minimum interval between requests;
* retries on 429/5xx/connection errors, honouring ``Retry-After``;
* ``maxlag`` for the action API (Wikimedia asks bots to use it).
"""

from __future__ import annotations

import random
import time
from typing import Any

import requests

from . import config

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class PolitenessSession:
    """A ``requests`` session that behaves itself."""

    def __init__(
        self,
        user_agent: str | None = None,
        min_interval: float = config.MIN_REQUEST_INTERVAL,
        max_retries: int = config.MAX_RETRIES,
        timeout: float = 60.0,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent or config.user_agent()
        self.session.headers["Accept"] = "application/json"
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._sleep = sleep
        self._clock = clock
        self._last_request = 0.0

    # -- internals ---------------------------------------------------------- #

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (self._clock() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = self._clock()

    @staticmethod
    def _retry_after(response: requests.Response) -> float | None:
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    def _backoff(self, attempt: int, retry_after: float | None) -> None:
        if retry_after is not None:
            delay = retry_after
        else:
            delay = min(2 ** attempt, 60) * (0.5 + random.random() / 2)
        self._sleep(delay)

    # -- public API --------------------------------------------------------- #

    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET returning parsed JSON, with retries. Raises on final failure."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            self._throttle()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                # IMPORTANT: `response` does not exist here. The original code
                # touched it inside its except block and crashed with
                # UnboundLocalError, killing the whole multi-hour run.
                last_error = exc
                self._backoff(attempt, None)
                continue

            if response.status_code in RETRY_STATUS:
                last_error = requests.HTTPError(
                    f"{response.status_code} from {url}", response=response
                )
                self._backoff(attempt, self._retry_after(response))
                continue

            if response.status_code >= 400:
                raise requests.HTTPError(
                    f"{response.status_code} from {url}: {response.text[:300]}",
                    response=response,
                )

            return response.json()

        raise RuntimeError(
            f"giving up on {url} after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> PolitenessSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
