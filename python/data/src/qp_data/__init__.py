"""qp_data: one data layer for the quant-portfolio projects (plan W9).

- :func:`fetch` downloads a series from a keyless source (retries with
  backoff, offline unless opted in), parses it, adds release dates and
  stores the bytes and the parsed vintage in a :class:`Store`.
- :func:`get` returns what was known about a series at a given time.
- :mod:`qp_data.release` holds the publication-lag rules and
  :func:`check_release`, the look-ahead check research code shares.

Series keys are ``<source>:<id>``, for example ``fred:DGS10``,
``ecb:EXR:USD``, ``nyfed:SOFR``, ``coinbase:BTC-USD:1d``. See
:mod:`qp_data.sources` for the full list.

FRED is read through the keyless ``fredgraph.csv`` download. There are no
ALFRED vintages (owner decision D6), so the value first published is not
recoverable; the release *date* is enforced from release calendars instead
and later revisions are captured by the store's own snapshots.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from . import release, sources
from .errors import (
    DataError,
    FetchError,
    NetworkDisabled,
    NoVintage,
    SchemaError,
    SourceNotPermitted,
    UnknownReleaseRule,
)
from .net import Opener, download, network_allowed
from .release import check_release, release_date
from .store import ManifestEntry, Store

__version__ = "3.0.0.dev0"

__all__ = [
    "DataError",
    "FetchError",
    "ManifestEntry",
    "NetworkDisabled",
    "NoVintage",
    "SchemaError",
    "SourceNotPermitted",
    "Store",
    "UnknownReleaseRule",
    "check_release",
    "download",
    "fetch",
    "fetch_frame",
    "get",
    "network_allowed",
    "release",
    "release_date",
    "sources",
]


def fetch_frame(
    key: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
    *,
    allow_network: bool | None = None,
    opener: Opener | None = None,
    retries: int = 3,
    timeout: float = 30.0,
) -> tuple[pd.DataFrame, bytes, str]:
    """Download and parse ``key`` without storing it: ``(frame, bytes, url)``."""
    url = sources.build_url(key, start, end)
    payload = download(url, allow_network=allow_network, opener=opener, retries=retries, timeout=timeout)
    return sources.parse(key, payload), payload, url


def fetch(
    key: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
    *,
    store: Store | None = None,
    allow_network: bool | None = None,
    opener: Opener | None = None,
    retrieved_at: str | dt.datetime | None = None,
) -> ManifestEntry:
    """Download ``key``, store it as a new vintage and return its manifest entry.

    Raises :class:`UnknownReleaseRule` before downloading if the series has no
    publication-lag rule, so nothing unlagged ever enters the store.
    """
    release.rule_for(key)
    st = store or Store()
    frame, payload, url = fetch_frame(key, start, end, allow_network=allow_network, opener=opener)
    return st.put(key, key.split(":", 1)[0], url, payload, frame, retrieved_at=retrieved_at)


def get(
    key: str,
    as_of: str | dt.datetime | dt.date | None = None,
    *,
    store: Store | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """What was known about ``key`` at ``as_of``; see :meth:`Store.get`."""
    return (store or Store()).get(key, as_of, strict=strict)
