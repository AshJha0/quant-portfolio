"""Guarded live FX data loader (Frankfurter / ECB reference rates).

Network access is **opt-in only**: nothing in this module runs at import
time, no test imports it with the guard open, and every call raises unless
``allow_network=True`` is passed explicitly or the environment variable
``FX_VAR_ALLOW_NETWORK=1`` is set.  Library code paths used by tests rely
exclusively on :mod:`fx_var.data.synthetic` (see CONVENTIONS.md).

Frankfurter serves ECB reference rates - daily fixings only, quoted as
QUOTE units per 1 unit of the requested base.  For the engine's USD-factor
convention we request base=USD and invert: CCYUSD = 1 / (CCY per USD).
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pandas as pd

from ..common import fx_factor

__all__ = ["load_frankfurter", "frankfurter_factor_returns"]

_API = "https://api.frankfurter.app"


def _check_guard(allow_network: bool | None) -> None:
    if allow_network is None:
        allow_network = os.environ.get("FX_VAR_ALLOW_NETWORK", "") == "1"
    if not allow_network:
        raise RuntimeError(
            "Network access is disabled by default (offline test policy). "
            "Pass allow_network=True or set FX_VAR_ALLOW_NETWORK=1 to fetch "
            "live ECB reference rates from Frankfurter."
        )


def load_frankfurter(
    symbols: list[str],
    start: str = "2023-01-01",
    end: str | None = None,
    timeout: float = 15.0,
    allow_network: bool | None = None,
) -> pd.DataFrame:
    """Fetch daily CCYUSD spot levels from Frankfurter (ECB fixings).

    Parameters
    ----------
    symbols : list of str
        Currencies, e.g. ``["EUR", "JPY", "GBP"]`` (USD excluded).
    start, end : str
        ISO dates; ``end=None`` means latest.
    allow_network : bool, optional
        Explicit opt-in; falls back to ``FX_VAR_ALLOW_NETWORK=1``.

    Returns
    -------
    pandas.DataFrame
        Business-day indexed USD price of 1 unit of each currency
        (columns = currency codes).
    """
    _check_guard(allow_network)
    symbols = [s.upper() for s in symbols if s.upper() != "USD"]
    if not symbols:
        raise ValueError("symbols must contain at least one non-USD currency")
    span = f"{start}..{end}" if end else f"{start}.."
    url = f"{_API}/{span}?from=USD&to={','.join(symbols)}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode("utf-8"))
    rates = payload.get("rates", {})
    if not rates:
        raise RuntimeError(f"Frankfurter returned no rates for {url}")
    frame = pd.DataFrame(rates).T.sort_index()
    frame.index = pd.to_datetime(frame.index)
    # Frankfurter gives CCY per USD; invert to USD per CCY (engine convention)
    return (1.0 / frame[symbols]).astype(float)


def frankfurter_factor_returns(
    symbols: list[str],
    start: str = "2023-01-01",
    end: str | None = None,
    allow_network: bool | None = None,
) -> pd.DataFrame:
    """Daily ``FX:CCY`` log returns from Frankfurter fixings.

    Convenience wrapper producing columns named per the engine's factor
    convention, ready for :func:`fx_var.historical_var.historical_var`.
    Note ECB fixings are once-a-day snapshots (14:15 CET) - fine for a demo,
    not a substitute for a desk's official closing rates (see
    docs/DESK_GUIDE.md on rate sourcing across time zones).
    """
    spots = load_frankfurter(symbols, start, end, allow_network=allow_network)
    rets = np.log(spots).diff().dropna()
    rets.columns = [fx_factor(c) for c in rets.columns]
    return rets
