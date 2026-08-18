"""Optional live market-data loader (import-guarded; never used by tests).

Per CONVENTIONS.md, library code paths exercised by tests must be offline.
This module is the *only* place that touches the network and degrades
gracefully when ``yfinance`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import yfinance as _yf

    HAS_YFINANCE = True
except ImportError:  # pragma: no cover
    _yf = None
    HAS_YFINANCE = False


def load_returns(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.Series:
    """Download adjusted closes via yfinance and return daily log-returns.

    Raises
    ------
    ImportError
        If ``yfinance`` is not installed (install extra: ``pip install
        eq-vol[live]``).
    """
    if not HAS_YFINANCE:  # pragma: no cover
        raise ImportError(
            "yfinance is required for live data: pip install 'eq-vol[live]'. "
            "All models and tests run on synthetic data (eq_vol.data.synthetic)."
        )
    px = _yf.download(ticker, start=start, end=end, progress=False)["Close"]  # pragma: no cover
    if isinstance(px, pd.DataFrame):  # pragma: no cover
        px = px.iloc[:, 0]
    return np.log(px).diff().dropna().rename(f"{ticker}_logret")  # pragma: no cover
