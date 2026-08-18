"""Guarded live market-data loaders (Frankfurter FX, FRED rates).

Network access is **opt-in**: set the environment variable
``FX_RATES_ALLOW_NETWORK=1`` before calling any loader, otherwise a
``RuntimeError`` is raised immediately.  Nothing in the library or the test
suite imports these functions on a code path that executes them — tests are
fully offline (CONVENTIONS.md).

These loaders return *raw* observations; mapping them into the quote
structures expected by :mod:`fx_rates.bootstrap` (deposit/swap par quotes,
basis spreads) is left to the caller because free sources do not publish a
complete, consistent multi-currency quote set.
"""

from __future__ import annotations

import json
import os
import urllib.request

__all__ = ["network_allowed", "load_frankfurter_spot", "load_fred_series"]

_ENV_FLAG = "FX_RATES_ALLOW_NETWORK"


def network_allowed() -> bool:
    """True if the user has explicitly opted in to network access."""
    return os.environ.get(_ENV_FLAG, "") == "1"


def _require_network() -> None:
    if not network_allowed():
        raise RuntimeError(
            f"Network access is disabled. Set {_ENV_FLAG}=1 to enable live data "
            "loaders (tests never do this)."
        )


def load_frankfurter_spot(base: str = "EUR", quote: str = "USD",
                          timeout: float = 10.0) -> float:
    """Latest ECB reference spot for ``base``/``quote`` from frankfurter.app.

    Returns the spot as quote-currency units per 1 base unit.
    """
    _require_network()
    url = f"https://api.frankfurter.app/latest?from={base}&to={quote}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # pragma: no cover
        payload = json.loads(resp.read().decode("utf-8"))
    return float(payload["rates"][quote])  # pragma: no cover


def load_fred_series(series_id: str, api_key: str | None = None,
                     timeout: float = 10.0) -> list[tuple[str, float]]:
    """Recent observations of a FRED series (e.g. ``DGS5``, ``SOFR``).

    Requires a FRED API key (``FRED_API_KEY`` env var or ``api_key``).
    Returns ``[(date_iso, value), ...]`` with missing values dropped.
    """
    _require_network()
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED API key required: pass api_key or set FRED_API_KEY")
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={key}&file_type=json&limit=100"
        "&sort_order=desc"
    )
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # pragma: no cover
        payload = json.loads(resp.read().decode("utf-8"))
    out = []  # pragma: no cover
    for obs in payload.get("observations", []):  # pragma: no cover
        if obs.get("value") not in (".", "", None):
            out.append((obs["date"], float(obs["value"])))
    return out  # pragma: no cover
