"""Turn per-asset studies into portfolio results before P&L (W24c).

Plain idea: a study should report what a book built from its signals would
have earned, not the return of one signal. ``portfolio_backtest`` re-weights
on a schedule from history only, holds the weights over the next periods,
and charges a linear cost on traded weight.

Timing (the repository convention, ``qp_core.execution``): weights decided
from returns up to and including row ``t`` earn the return of row ``t+1``.
The cost of moving from the held weights to the new ones is booked on row
``t+1``. Weights drift with returns between rebalances is *not* modelled:
the book is re-set to its target weights each period (constant-mix), which
is stated so results are not mistaken for buy-and-hold.

Identity (tested): ``net = gross - cost`` row by row, ``cost >= 0``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["PortfolioResult", "portfolio_backtest"]


@dataclass(frozen=True)
class PortfolioResult:
    weights: pd.DataFrame  # weights held during each row
    gross: pd.Series
    cost: pd.Series
    net: pd.Series
    turnover: pd.Series


def portfolio_backtest(
    returns: pd.DataFrame,
    weight_fn: Callable[[pd.DataFrame, np.ndarray], np.ndarray],
    est_window: int = 60,
    rebalance_every: int = 1,
    cost_bps: float = 0.0,
) -> PortfolioResult:
    """Walk-forward portfolio P&L.

    ``weight_fn(history, w_prev)`` returns target weights from the returns
    up to the decision row and the currently held weights (so turnover
    penalties can see them). First decision on row ``est_window - 1``.
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("returns contain NaN or infinite values")
    if est_window < 2 or rebalance_every < 1 or cost_bps < 0:
        raise ValueError("need est_window >= 2, rebalance_every >= 1, cost_bps >= 0")
    t_n, n = returns.shape
    if t_n <= est_window:
        raise ValueError("sample shorter than est_window + 1")
    r = returns.to_numpy(dtype=float)
    held = np.zeros((t_n, n))
    cost = np.zeros(t_n)
    w = np.zeros(n)
    for t in range(est_window - 1, t_n - 1):
        if (t - (est_window - 1)) % rebalance_every == 0:
            new = np.asarray(weight_fn(returns.iloc[: t + 1], w.copy()), dtype=float)
            if new.shape != (n,) or not np.isfinite(new).all():
                raise ValueError("weight_fn must return N finite weights")
            cost[t + 1] = np.abs(new - w).sum() * cost_bps * 1e-4
            w = new
        held[t + 1] = w
    gross = (held * r).sum(axis=1)
    idx = returns.index
    turnover = np.zeros(t_n)
    turnover[1:] = np.abs(np.diff(held, axis=0)).sum(axis=1)
    return PortfolioResult(
        weights=pd.DataFrame(held, index=idx, columns=returns.columns),
        gross=pd.Series(gross, index=idx, name="gross"),
        cost=pd.Series(cost, index=idx, name="cost"),
        net=pd.Series(gross - cost, index=idx, name="net"),
        turnover=pd.Series(turnover, index=idx, name="turnover"),
    )
