"""Optional live market-data loader (yfinance). Never used by the tests.

Import-guarded per the portfolio conventions: importing this module is
safe without yfinance installed; calling :func:`load_option_chain` without
it raises ``ImportError`` with install instructions. All library code used
in tests relies exclusively on :mod:`eq_options.data.synthetic`.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["load_option_chain", "HAS_YFINANCE"]

try:  # pragma: no cover - depends on optional extra
    import yfinance as _yf

    HAS_YFINANCE = True
except ImportError:  # pragma: no cover
    _yf = None
    HAS_YFINANCE = False


def load_option_chain(ticker: str, expiry: str | None = None) -> pd.DataFrame:
    """Download an option chain from Yahoo Finance (requires ``eq-options[live]``).

    Parameters
    ----------
    ticker : str
        Underlying symbol, e.g. ``"SPY"``.
    expiry : str, optional
        Expiry date ``YYYY-MM-DD``; defaults to the nearest listed expiry.

    Returns
    -------
    pandas.DataFrame
        Columns: ``strike``, ``type`` ('call'/'put'), ``last_price``,
        ``bid``, ``ask``, ``implied_vol`` (Yahoo's), ``expiry``.

    Raises
    ------
    ImportError
        If yfinance is not installed (``pip install eq-options[live]``).
    ValueError
        If the ticker lists no option expiries.
    """
    if not HAS_YFINANCE:  # pragma: no cover
        raise ImportError(
            "yfinance is required for live data: pip install 'eq-options[live]'"
        )
    tk = _yf.Ticker(ticker)  # pragma: no cover
    expiries = tk.options  # pragma: no cover
    if not expiries:  # pragma: no cover
        raise ValueError(f"no listed option expiries for {ticker!r}")
    exp = expiry or expiries[0]  # pragma: no cover
    chain = tk.option_chain(exp)  # pragma: no cover
    frames = []  # pragma: no cover
    for opt_type, df in (("call", chain.calls), ("put", chain.puts)):  # pragma: no cover
        frames.append(pd.DataFrame({
            "strike": df["strike"],
            "type": opt_type,
            "last_price": df["lastPrice"],
            "bid": df["bid"],
            "ask": df["ask"],
            "implied_vol": df["impliedVolatility"],
            "expiry": exp,
        }))
    return pd.concat(frames, ignore_index=True)  # pragma: no cover
