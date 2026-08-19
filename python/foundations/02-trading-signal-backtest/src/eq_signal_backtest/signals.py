"""Signal generation for the long/flat moving-average crossover strategy.

A single design decision lives here: the signal is defined purely on
*closing* prices, using simple (not exponential) moving averages, and it
is binary (1.0 = bullish / long, 0.0 = everything else / flat). It says
nothing about *when* that signal is tradeable -- that is the execution
lag applied in :mod:`eq_signal_backtest.engine`, deliberately kept in a
different function so the no-look-ahead property is visible and testable
in one place.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["ma_crossover_signal"]


def ma_crossover_signal(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    """1.0 when the fast moving average is above the slow one, else 0.0.

    Computed on close prices only (no intraday information). This is the
    *signal*, not the *position* -- see
    :func:`eq_signal_backtest.engine.run_backtest` for the execution lag
    that turns a same-day signal into a next-day position.

    Parameters
    ----------
    prices : pandas.Series
        Close prices, float, ascending date index.
    fast, slow : int
        Lookback windows in trading days for the fast/slow simple moving
        averages. ``fast`` must be strictly shorter than ``slow``.

    Returns
    -------
    pandas.Series
        Same index as ``prices``. 1.0 where ``fast_ma > slow_ma``, 0.0
        otherwise -- including the first ``slow - 1`` observations, where
        the slow moving average is undefined (``NaN``) and the comparison
        ``NaN > x`` is ``False`` by construction, so the warm-up period is
        always flat rather than raising or fabricating a signal.

    Raises
    ------
    ValueError
        If ``fast >= slow``.
    """
    if fast >= slow:
        raise ValueError(
            f"fast window must be shorter than slow window, got fast={fast}, slow={slow}"
        )
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)
