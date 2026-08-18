"""Optional live market-data loader (yfinance) — import-guarded, never used
by tests or library code paths.

Install with ``pip install eq-var[live]``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["load_returns_yfinance"]


def load_returns_yfinance(
    tickers: list[str], period: str = "2y", interval: str = "1d"
) -> "np.ndarray":
    """Download adjusted-close simple returns for ``tickers`` via yfinance.

    Returns a (T, n) array aligned on common dates.  Raises ``ImportError``
    with an actionable message when yfinance is not installed, and
    ``RuntimeError`` when the download comes back empty (offline).
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "yfinance is required for live data: pip install eq-var[live]"
        ) from exc
    data = yf.download(tickers, period=period, interval=interval, progress=False)
    if data is None or len(data) == 0:  # pragma: no cover - network dependent
        raise RuntimeError("yfinance returned no data (offline or bad tickers)")
    close = data["Close"] if "Close" in data else data
    returns = close.pct_change().dropna()
    return returns.to_numpy(dtype=float)
