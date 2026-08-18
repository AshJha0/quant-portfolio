"""Guarded live FX data loader (Frankfurter API).  NOT used by tests.

Network access is confined to this module per CONVENTIONS.md: importing it is
side-effect free, and the fetch function raises a clear ``RuntimeError`` when
offline.  The rest of the package (and the entire test suite) runs on the
deterministic generators in :mod:`fx_pairs.data.synthetic`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pandas as pd

__all__ = ["fetch_frankfurter_legs"]

_BASE_URL = "https://api.frankfurter.app"


def fetch_frankfurter_legs(
    currencies: tuple[str, ...] = ("EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"),
    start: str = "2018-01-01",
    end: str | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch daily ECB reference rates and return a CCYUSD USD-leg panel.

    Frankfurter serves ECB reference rates.  We request USD-based quotes
    (units of CCY per USD) and invert to USD legs (USD per 1 CCY), matching
    the convention of :mod:`fx_pairs.universe`.

    Raises
    ------
    RuntimeError
        If the network is unavailable or the API cannot be reached.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    symbols = ",".join(c.upper() for c in currencies)
    url = f"{_BASE_URL}/{start}..{end}?from=USD&to={symbols}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise RuntimeError(
            "could not reach the Frankfurter API (offline?); use "
            "fx_pairs.data.synthetic for deterministic offline data"
        ) from exc
    rates = payload.get("rates", {})
    if not rates:
        raise RuntimeError("Frankfurter returned no rates for the request")
    df = pd.DataFrame.from_dict(rates, orient="index").sort_index()
    df.index = pd.to_datetime(df.index)
    # payload is CCY per USD; invert to USD per CCY (USD legs)
    return (1.0 / df).rename_axis("date")
