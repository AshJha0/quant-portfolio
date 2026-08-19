"""Parameter-sensitivity map for the moving-average crossover signal.

Used to visualise how sensitive the backtest result is to the choice of
(fast, slow) windows. A strategy whose Sharpe collapses one grid cell
away from the chosen parameters is curve-fit, not robust; a broad
plateau of similar Sharpe values is what robustness looks like.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .engine import run_backtest
from .signals import ma_crossover_signal

__all__ = ["parameter_grid"]


def parameter_grid(
    prices: pd.Series,
    fast_range: Iterable[int],
    slow_range: Iterable[int],
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Sharpe ratio for every valid (fast, slow) pair.

    Parameters
    ----------
    prices : pandas.Series
        Close prices to backtest on. Callers are responsible for passing
        only in-sample data when this grid will be used for parameter
        selection (see :func:`eq_signal_backtest.split.select_best_params`).
    fast_range, slow_range : iterable of int
        Candidate lookback windows. Combinations with ``fast >= slow``
        are skipped (see :func:`eq_signal_backtest.signals.ma_crossover_signal`).
    cost_bps : float, default 5.0
        One-way transaction cost in basis points, passed through to
        :func:`eq_signal_backtest.engine.run_backtest`.

    Returns
    -------
    pandas.DataFrame
        Pivoted so the index is ``fast``, the columns are ``slow``, and
        the values are the strategy's annualised Sharpe ratio. Cells for
        skipped (``fast >= slow``) combinations are absent (``NaN`` after
        the pivot fills the rectangular grid).
    """
    rows = []
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            sig = ma_crossover_signal(prices, fast, slow)
            res = run_backtest(prices, sig, cost_bps)
            rows.append({"fast": fast, "slow": slow, "sharpe": res.stats["sharpe"]})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot(index="fast", columns="slow", values="sharpe")
