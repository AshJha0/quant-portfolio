"""Optional live price loader (yfinance) -- import-guarded, never used by tests.

Importing this module never requires yfinance or a network connection: the
dependency is imported inside :func:`load_prices` only. Install the extra
with ``pip install -e ".[live]"``.
"""

from __future__ import annotations

import importlib.util

import pandas as pd

__all__ = ["load_prices"]

#: True when the optional ``yfinance`` extra is importable. Determined by
#: a spec lookup, so checking it never imports yfinance itself.
_HAS_YF: bool = importlib.util.find_spec("yfinance") is not None


def load_prices(ticker: str, start: str | None = "2015-01-01") -> pd.Series:
    """Download daily adjusted closes for ``ticker`` as a price Series.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol, e.g. ``"SPY"``.
    start : str, optional
        ISO start date; ``None`` fetches maximum available history.

    Returns
    -------
    pandas.Series
        Close prices indexed by date, ascending -- the same shape the rest
        of the package expects (and that ``data.synthetic.generate``
        produces after ``set_index("Date")["Adj Close"]``).

    Raises
    ------
    ImportError
        If yfinance is not installed, with an actionable message.
    RuntimeError
        If the download returns no rows (offline, or an unknown ticker).
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "yfinance is required for live data: pip install 'eq-signal-backtest[live]'"
        ) from exc
    data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if data is None or len(data) == 0:  # pragma: no cover - network dependent
        raise RuntimeError(f"yfinance returned no data for {ticker!r} (offline or bad ticker)")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # multi-ticker column shape
        close = close.iloc[:, 0]
    return close.astype(float).rename(ticker)
