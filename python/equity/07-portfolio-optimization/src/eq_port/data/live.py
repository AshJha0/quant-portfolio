"""Optional live market-data loader (yfinance). Import-guarded; never used
by tests, which rely exclusively on :mod:`eq_port.data.synthetic`."""

from __future__ import annotations

import pandas as pd

__all__ = ["load_prices", "load_returns"]


def load_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    Requires the optional dependency ``yfinance`` (``pip install eq-port[live]``)
    and network access. Raises ``ImportError`` with instructions otherwise.

    Parameters
    ----------
    tickers : list[str]
        Yahoo tickers, e.g. ``["SPY", "TLT", "GLD"]``.
    start, end : str
        ISO dates.

    Returns
    -------
    pd.DataFrame
        (T, N) adjusted close prices, columns = tickers.
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - network/optional
        raise ImportError(
            "yfinance is required for live data: pip install eq-port[live]. "
            "Offline workflows should use eq_port.data.synthetic instead."
        ) from exc
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    px = data["Close"]
    if isinstance(px, pd.Series):  # single ticker
        px = px.to_frame(tickers[0])
    return px.dropna(how="all")


def load_returns(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Simple daily returns from :func:`load_prices` (rows with any NaN dropped)."""
    px = load_prices(tickers, start, end)
    return px.pct_change().dropna(how="any")
