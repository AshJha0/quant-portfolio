"""Optional live FX data loader (Frankfurter / ECB reference rates).

Deprecated: use ``qp_data.fetch("frankfurter:<BASE>:<QUOTE>", start, end)``
and ``qp_data.get``. :func:`load_frankfurter` now delegates to qp_data
(retries, backoff, one parser) and is removed in v4.0.0. It no longer needs
the optional ``requests`` package.

Nothing in the test suite touches this module's network path.

ECB reference rates are once-a-day (14:15 CET) fixings, EUR-based; they are
fine for daily realized-vol studies but are NOT tradable prices and carry no
intraday information (no highs/lows: range estimators cannot be used).
"""

from __future__ import annotations

import pandas as pd

from ..returns import pair_currencies

__all__ = ["load_frankfurter"]


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
        If qp_data is not installed or the download fails.
    """
    base, quote = pair_currencies(pair)
    try:
        from qp_data import legacy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("live data now comes from qp_data: pip install -e python/data") from exc
    legacy.deprecated("fx_vol.data.live.load_frankfurter", 'qp_data.fetch("frankfurter:...")')
    try:  # pragma: no cover - network path, never exercised in tests
        frame = legacy.frankfurter(base, [quote], start, end, timeout=timeout)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Frankfurter request failed for {pair}: {exc}") from exc
    if quote not in frame.columns or frame[quote].dropna().empty:  # pragma: no cover
        raise RuntimeError(f"Frankfurter returned no {pair} data between {start} and {end}")
    out = frame[quote].dropna().rename(pair)  # pragma: no cover
    out.index.name = None  # pragma: no cover
    return out  # pragma: no cover
