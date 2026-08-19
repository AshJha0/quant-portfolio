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

import numpy as np
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
        If ``fast >= slow``, if either window is not an integer >= 1, or
        if ``prices`` contains non-finite values (a ``NaN`` close silently
        blanks ``slow`` days of the moving average and therefore ``slow``
        days of signal, which looks identical to a genuine flat period).
    """
    for name, window in (("fast", fast), ("slow", slow)):
        if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
            raise ValueError(f"{name} window must be an int, got {window!r}")
        if window < 1:
            raise ValueError(f"{name} window must be >= 1, got {window}")
    if fast >= slow:
        raise ValueError(
            f"fast window must be shorter than slow window, got fast={fast}, slow={slow}"
        )
    if len(prices) and not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError(
            "ma_crossover_signal: prices contains non-finite values (NaN/inf); "
            "a missing close blanks the moving averages for `slow` days and "
            "produces a flat signal that is indistinguishable from a real one"
        )
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)
