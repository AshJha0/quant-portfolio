"""Live FX data loader (Frankfurter API). NOT used by tests.

Deprecated: use ``qp_data.fetch("frankfurter:USD:EUR,GBP,...", start, end)``
and invert. :func:`fetch_frankfurter_legs` now delegates to qp_data
(retries, backoff, one parser) and is removed in v4.0.0.

Importing this module is side-effect free, and the fetch function raises a
clear ``RuntimeError`` when offline. The rest of the package (and the entire
test suite) runs on the deterministic generators in
:mod:`fx_pairs.data.synthetic`.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["fetch_frankfurter_legs"]


def fetch_frankfurter_legs(
    currencies: tuple[str, ...] = ("EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"),
    start: str = "2018-01-01",
    end: str | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch daily ECB reference rates and return a CCYUSD USD-leg panel.

    Frankfurter serves ECB reference rates. We request USD-based quotes
    (units of CCY per USD) and invert to USD legs (USD per 1 CCY), matching
    the convention of :mod:`fx_pairs.universe`.

    Raises
    ------
    RuntimeError
        If qp_data is missing, the network is unavailable or the API cannot
        be reached.
    """
    try:
        from qp_data import legacy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("live data now comes from qp_data: pip install -e python/data") from exc
    legacy.deprecated("fx_pairs.data.live.fetch_frankfurter_legs", 'qp_data.fetch("frankfurter:USD:...")')
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        df = legacy.frankfurter("USD", [c.upper() for c in currencies], start, end, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(
            "could not reach the Frankfurter API (offline?); use "
            "fx_pairs.data.synthetic for deterministic offline data"
        ) from exc
    # payload is CCY per USD; invert to USD per CCY (USD legs)
    return (1.0 / df).rename_axis("date")
