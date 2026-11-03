"""HTTP download with an offline guard, retries and exponential backoff.

Plain idea: a download either succeeds, fails for a reason worth retrying (a
timeout, a dropped connection, HTTP 429 or 5xx), or fails for good (HTTP 4xx
other than 429). Only the first kind is retried, with waits of
``backoff * 2**k`` seconds, ``k = 0, 1, ...``.

Network access is off by default. A call goes out only if the caller passes
``allow_network=True`` or the environment sets ``QP_DATA_ALLOW_NETWORK=1``.
Tests never do either; they pass recorded responses through ``opener``.

Example (no network: the opener is a stub)::

    >>> calls = []
    >>> def opener(url, timeout):
    ...     calls.append(url)
    ...     if len(calls) < 2:
    ...         raise TimeoutError("slow")
    ...     return b"ok"
    >>> download("https://example.org/x", allow_network=True, opener=opener,
    ...          sleep=lambda s: None)
    b'ok'
    >>> len(calls)
    2
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from .errors import FetchError, NetworkDisabled

__all__ = ["ENV_FLAG", "USER_AGENT", "Opener", "download", "network_allowed"]

ENV_FLAG = "QP_DATA_ALLOW_NETWORK"
#: Sent with every request. SEC EDGAR and some others refuse requests without one.
USER_AGENT = "quant-portfolio-research/3.0"

Opener = Callable[[str, float], bytes]


def network_allowed(allow_network: bool | None = None) -> bool:
    """True if the caller opted in, or (``allow_network is None``) the env flag is ``1``."""
    if allow_network is None:
        return os.environ.get(ENV_FLAG, "") == "1"
    return bool(allow_network)


def _urlopen(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: bytes = resp.read()
    return data


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError))


def download(
    url: str,
    *,
    allow_network: bool | None = None,
    retries: int = 3,
    backoff: float = 0.5,
    timeout: float = 30.0,
    opener: Opener | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Return the body of ``url``.

    ``retries`` is the number of extra attempts after the first, so at most
    ``retries + 1`` requests are made. ``opener(url, timeout)`` replaces the
    real HTTP call (tests pass recorded bytes this way); the offline guard
    applies only when no opener is given.

    Raises
    ------
    NetworkDisabled
        No opener was given and the caller did not opt into network access.
    FetchError
        A permanent HTTP error, or every attempt failed.
    """
    if retries < 0 or backoff < 0 or timeout <= 0:
        raise ValueError("retries and backoff must be >= 0 and timeout > 0")
    if opener is None:
        if not network_allowed(allow_network):
            raise NetworkDisabled(
                f"Network access is disabled (offline default). Pass allow_network=True "
                f"or set {ENV_FLAG}=1 to fetch {url}"
            )
        opener = _urlopen
    last: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return opener(url, timeout)
        except Exception as exc:
            last = exc
            if not _retryable(exc):
                raise FetchError(f"permanent failure fetching {url}: {exc}") from exc
            if attempt < retries:
                sleep(backoff * 2**attempt)
    raise FetchError(f"gave up on {url} after {retries + 1} attempts: {last}") from last
