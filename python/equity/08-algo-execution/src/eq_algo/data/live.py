"""Optional live market-data loader (network access; NOT used by tests).

Import-guarded per portfolio conventions: the library and test suite never
touch the network.  Install with ``pip install eq-algo[live]`` to enable.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["load_prices_yahoo"]

try:  # pragma: no cover - optional dependency
    import yfinance as _yf
except ImportError:  # pragma: no cover
    _yf = None


def load_prices_yahoo(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices (dates x tickers) from Yahoo Finance.

    Raises ``ImportError`` if ``yfinance`` is not installed.  Not used by the
    test suite; synthetic generators in :mod:`eq_algo.data.synthetic` are the
    deterministic substitute.
    """
    if _yf is None:  # pragma: no cover
        raise ImportError("yfinance is required: pip install eq-algo[live]")
    data = _yf.download(tickers, start=start, end=end, progress=False,
                        auto_adjust=True)  # pragma: no cover
    return data["Close"]  # pragma: no cover
