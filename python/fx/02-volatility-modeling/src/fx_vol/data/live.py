"""Optional live FX data loader (Frankfurter / ECB reference rates).

Import-guarded per CONVENTIONS.md: nothing in the test suite touches this
module's network path. The loader needs the optional ``requests`` dependency
and internet access; both failures raise informative ``RuntimeError`` rather
than crashing at import time.

ECB reference rates are once-a-day (~16:00 CET) fixings, EUR-based; they are
fine for daily realized-vol studies but are NOT tradable prices and carry no
intraday information (no highs/lows -- range estimators cannot be used).
"""

from __future__ import annotations

import pandas as pd

from ..returns import pair_currencies

__all__ = ["load_frankfurter"]

_API = "https://api.frankfurter.dev/v1"


def load_frankfurter(
    pair: str = "EURUSD",
    start: str = "2015-01-04",
    end: str | None = None,
    timeout: float = 30.0,
) -> pd.Series:
    """Load daily BASE/QUOTE reference rates from the Frankfurter API (ECB data).

    Parameters
    ----------
    pair : str
        6-letter pair, e.g. 'EURUSD' (= USD per EUR). Either leg may be EUR
        or any ECB-covered currency; crosses are computed by Frankfurter
        via EUR triangulation.
    start, end : str
        ISO dates; ``end=None`` means latest available.
    timeout : float
        HTTP timeout in seconds.

    Returns
    -------
    pandas.Series
        Daily rates indexed by date, named after the pair.

    Raises
    ------
    RuntimeError
        If ``requests`` is not installed or the network call fails.
    """
    base, quote = pair_currencies(pair)
    try:
        import requests  # deferred: optional dependency
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "load_frankfurter requires the optional 'requests' package: pip install requests"
        ) from exc

    url = f"{_API}/{start}..{end or ''}"
    try:  # pragma: no cover - network path, never exercised in tests
        resp = requests.get(url, params={"base": base, "symbols": quote}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Frankfurter request failed for {pair}: {exc}") from exc

    rates = {pd.Timestamp(d): v[quote] for d, v in payload["rates"].items() if quote in v}
    if not rates:  # pragma: no cover
        raise RuntimeError(f"Frankfurter returned no {pair} data between {start} and {end}")
    out = pd.Series(rates, name=pair).sort_index()
    return out
