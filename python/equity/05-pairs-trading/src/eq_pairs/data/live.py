"""Optional live market-data loader (yfinance). Import-guarded; never used in tests.

The test suite and examples run entirely on the deterministic generators in
:mod:`eq_pairs.data.synthetic`. This module exists so the same pipeline can be
pointed at real tickers when a network connection and ``yfinance`` are
available: ``pip install eq-pairs[live]``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

__all__ = ["load_prices"]

try:  # pragma: no cover - optional dependency
    import yfinance as _yf

    _HAS_YF = True
except ImportError:  # pragma: no cover
    _yf = None
    _HAS_YF = False


def load_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download daily adjusted close prices for ``tickers`` via yfinance.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance tickers, e.g. ``["XOM", "CVX"]``.
    start, end : str
        ISO dates bounding the sample.

    Returns
    -------
    pandas.DataFrame
        Adjusted close prices, one column per ticker, business-day index.

    Raises
    ------
    ImportError
        If ``yfinance`` is not installed.
    """
    if not _HAS_YF:  # pragma: no cover
        raise ImportError(
            "yfinance is required for live data: pip install eq-pairs[live]"
        )
    data = _yf.download(  # pragma: no cover
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    closes = data["Close"]  # pragma: no cover
    if isinstance(closes, pd.Series):  # pragma: no cover
        closes = closes.to_frame(tickers[0])
    return closes.dropna(how="all")  # pragma: no cover
