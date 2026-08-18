"""Walk-forward backtester with strict no-lookahead estimation, drifted
weights between rebalances, and exact turnover / transaction-cost
accounting, plus the standard strategy zoo (equal weight, min-variance,
tangency raw/shrunk, ERC, static 60/40-style benchmark).

Mechanics (per day, oldest first; simple returns):

* Day t is a REBALANCE day if ``(t - first) % rebalance_every == 0`` with
  ``first = window`` (the first day with a full estimation history).
* On a rebalance day the strategy sees ``returns.iloc[t - window : t]`` —
  the window ends at day t-1, STRICTLY before the day whose return the
  new weights earn. No estimate ever uses information from day t onward.
* Turnover at a rebalance is ``sum_i |w_target_i - w_drift_i|`` where
  ``w_drift`` are the previous target weights drifted by realised returns
  (zero before the first trade, so the initial buy-in counts as turnover
  of ``sum |w|``). Cost = ``cost_bps / 1e4 * turnover``, deducted from
  that day's net return.
* Between rebalances weights drift:
  ``w_i <- w_i (1 + r_i,t) / (1 + w' r_t)`` (fully-invested convention;
  a weight sum != 1 is treated as the remainder held in zero-rate cash).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .covariance import ledoit_wolf_cc, psd_repair, sample_cov
from .mvo import max_sharpe_constrained, min_variance_constrained
from .returns_est import james_stein_mean, sample_mean
from .risk_parity import erc_weights

__all__ = [
    "BacktestResult",
    "run_backtest",
    "run_race",
    "strategy_equal_weight",
    "make_min_variance_strategy",
    "make_tangency_strategy",
    "make_erc_strategy",
    "make_static_strategy",
]

Strategy = Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class BacktestResult:
    """Walk-forward backtest output.

    Attributes
    ----------
    net_returns : pd.Series
        Daily net-of-cost portfolio returns over the backtest span.
    gross_returns : pd.Series
        Daily gross portfolio returns (before costs).
    weights : pd.DataFrame
        Target weights at each rebalance date (rows = rebalance dates).
    turnover : pd.Series
        Two-sided turnover ``sum |w_new - w_drift|`` at each rebalance.
    costs : pd.Series
        Cost drag (return units) charged at each rebalance date.
    """

    net_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series

    @property
    def total_cost(self) -> float:
        """Sum of per-rebalance cost drags (return units)."""
        return float(self.costs.sum())


def run_backtest(
    returns: pd.DataFrame,
    strategy: Strategy,
    window: int,
    rebalance_every: int = 21,
    cost_bps: float = 0.0,
) -> BacktestResult:
    """Run a single-strategy walk-forward backtest.

    Parameters
    ----------
    returns : (T, N) pd.DataFrame
        Simple per-period returns, oldest first.
    strategy : callable
        Maps the (window, N) estimation DataFrame (strictly past data) to
        an (N,) weight vector. Weights may be levered (sum != 1); the
        remainder is zero-rate cash.
    window : int
        Estimation window length in periods (>= 1); the backtest starts
        at row index ``window``.
    rebalance_every : int
        Rebalance every k-th backtest day (1 = daily).
    cost_bps : float
        One-way proportional cost in basis points applied to two-sided
        turnover (e.g. 10 => 10bp of each dollar traded).

    Returns
    -------
    BacktestResult
    """
    if not isinstance(returns, pd.DataFrame):
        returns = pd.DataFrame(np.asarray(returns, dtype=float))
    t_total, n = returns.shape
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if rebalance_every < 1:
        raise ValueError(f"rebalance_every must be >= 1, got {rebalance_every}")
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    if t_total <= window:
        raise ValueError(
            f"need more than window={window} rows, got {t_total}: nothing to backtest"
        )

    r = returns.to_numpy(dtype=float)
    idx = returns.index
    cost_rate = cost_bps / 1e4

    net = np.empty(t_total - window)
    gross = np.empty(t_total - window)
    reb_dates: list = []
    reb_weights: list[np.ndarray] = []
    reb_turnover: list[float] = []
    reb_costs: list[float] = []

    w = np.zeros(n)  # holdings before the first trade
    for k, t in enumerate(range(window, t_total)):
        cost_today = 0.0
        if (t - window) % rebalance_every == 0:
            est = returns.iloc[t - window : t]  # ends at t-1: strictly past
            w_new = np.asarray(strategy(est), dtype=float).ravel()
            if w_new.shape[0] != n:
                raise ValueError(
                    f"strategy returned {w_new.shape[0]} weights for {n} assets"
                )
            if not np.all(np.isfinite(w_new)):
                raise ValueError("strategy returned non-finite weights")
            turnover = float(np.abs(w_new - w).sum())
            cost_today = cost_rate * turnover
            w = w_new.copy()
            reb_dates.append(idx[t])
            reb_weights.append(w.copy())
            reb_turnover.append(turnover)
            reb_costs.append(cost_today)
        g = float(w @ r[t])
        gross[k] = g
        net[k] = g - cost_today
        # drift weights with realised returns (cash residual has zero return)
        w = w * (1.0 + r[t]) / (1.0 + g)

    span = idx[window:]
    reb_idx = pd.Index(reb_dates)
    return BacktestResult(
        net_returns=pd.Series(net, index=span, name="net"),
        gross_returns=pd.Series(gross, index=span, name="gross"),
        weights=pd.DataFrame(reb_weights, index=reb_idx, columns=returns.columns),
        turnover=pd.Series(reb_turnover, index=reb_idx, name="turnover"),
        costs=pd.Series(reb_costs, index=reb_idx, name="cost"),
    )


def run_race(
    returns: pd.DataFrame,
    strategies: dict[str, Strategy],
    window: int,
    rebalance_every: int = 21,
    cost_bps: float = 0.0,
) -> dict[str, BacktestResult]:
    """Run several strategies through :func:`run_backtest` on identical
    data/windows/costs; returns ``{name: BacktestResult}``."""
    return {
        name: run_backtest(returns, fn, window, rebalance_every, cost_bps)
        for name, fn in strategies.items()
    }


# --------------------------------------------------------------------------
# strategy zoo
# --------------------------------------------------------------------------

def strategy_equal_weight(est: pd.DataFrame) -> np.ndarray:
    """1/N weights (estimation-free benchmark; DeMiguel et al. 2009)."""
    n = est.shape[1]
    return np.full(n, 1.0 / n)


def _cov_estimate(est: pd.DataFrame, use_lw: bool) -> np.ndarray:
    if use_lw:
        sigma = ledoit_wolf_cc(est).cov
    else:
        sigma = sample_cov(est)
    return psd_repair(sigma, eps=1e-10)


def make_min_variance_strategy(use_lw: bool = True) -> Strategy:
    """Long-only minimum-variance strategy (Ledoit-Wolf covariance by
    default; ``use_lw=False`` for the raw sample covariance)."""

    def _strategy(est: pd.DataFrame) -> np.ndarray:
        return min_variance_constrained(_cov_estimate(est, use_lw), bounds=(0.0, 1.0))

    return _strategy


def make_tangency_strategy(
    shrink_mean: bool = False,
    use_lw: bool = True,
    rf: float = 0.0,
) -> Strategy:
    """Long-only maximum-Sharpe strategy.

    ``shrink_mean=False`` uses the raw sample mean (the classic error
    maximizer); ``shrink_mean=True`` uses the James-Stein shrunk mean.
    """

    def _strategy(est: pd.DataFrame) -> np.ndarray:
        mu = james_stein_mean(est).mean if shrink_mean else sample_mean(est)
        sigma = _cov_estimate(est, use_lw)
        try:
            return max_sharpe_constrained(mu, sigma, rf=rf, bounds=(0.0, 1.0))
        except ValueError:
            # all estimated excess returns <= rf: park in min-variance
            return min_variance_constrained(sigma, bounds=(0.0, 1.0))

    return _strategy


def make_erc_strategy(use_lw: bool = True) -> Strategy:
    """Equal-risk-contribution strategy (long-only, unlevered)."""

    def _strategy(est: pd.DataFrame) -> np.ndarray:
        return erc_weights(_cov_estimate(est, use_lw))

    return _strategy


def make_static_strategy(weights: np.ndarray) -> Strategy:
    """Static benchmark (e.g. 60/40-style): the same fixed target weights
    at every rebalance (still drifts between rebalances and pays costs)."""
    w = np.asarray(weights, dtype=float).ravel()

    def _strategy(est: pd.DataFrame) -> np.ndarray:
        if est.shape[1] != w.shape[0]:
            raise ValueError(
                f"static weights have {w.shape[0]} entries for {est.shape[1]} assets"
            )
        return w.copy()

    return _strategy
