"""Performance and risk metrics: Sharpe with Lo (2002) SE, Sortino, MDD, turnover.

Conventions: daily P&L in units of account; annualisation factor 252 business
days; Sharpe assumes zero funding benchmark (carry is already inside the P&L,
which is precisely the point of the carry decomposition).
"""

from __future__ import annotations

import numpy as np
from qp_core.metrics import downside_deviation, max_drawdown_from_pnl
import pandas as pd

from . import research_stats as _rs

__all__ = [
    "sharpe_ratio",
    "sharpe_se_lo",
    "sortino_ratio",
    "max_drawdown",
    "hit_rate",
    "turnover",
    "trade_pnls",
    "summarize",
]


def _clean(returns: pd.Series | np.ndarray) -> np.ndarray:
    """Drop non-finite entries.

    Metrics *tolerate* NaN here (unlike the pricing and
    signal layers, which reject it): a warmup window legitimately produces
    NaN P&L before the first position exists, and a metric over "the days
    that actually traded" is the desk's intent.  Inf is dropped for the same
    reason it must never be averaged: one infinite bar would make every
    downstream statistic infinite.
    """
    r = np.asarray(returns, dtype=float)
    return r[np.isfinite(r)]


def _check_ann_factor(ann_factor: float) -> float:
    """Validate the annualisation factor (``ann_factor <= 0`` is False for NaN)."""
    if not np.isfinite(ann_factor) or ann_factor <= 0.0:
        raise ValueError(
            f"ann_factor must be positive and finite, got {ann_factor!r}"
        )
    return float(ann_factor)


def sharpe_ratio(returns: pd.Series | np.ndarray, ann_factor: float = 252.0) -> float:
    """Annualised Sharpe ratio ``mean/std * sqrt(ann_factor)``; NaN if std=0."""
    _check_ann_factor(ann_factor)
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0.0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(ann_factor))


def sharpe_se_lo(
    returns: pd.Series | np.ndarray,
    q: int | None = None,
    ann_factor: float = 252.0,
) -> tuple[float, float]:
    """Sharpe ratio with the Lo (2002) / Newey-West standard error.

    Delegates to :func:`fx_pairs.research_stats.lo_sharpe_se` (GMM delta
    method, Bartlett-kernel long-run variance truncated at ``q`` lags).

    Parameters
    ----------
    q : int, optional
        Truncation lag; default ``floor(4 (T/100)^(2/9))`` (Newey-West rule).

    Returns
    -------
    (sharpe_annualised, se_annualised); ``(nan, nan)`` for fewer than 10
    observations or zero variance.
    """
    _check_ann_factor(ann_factor)
    r = _clean(returns)
    T = len(r)
    if T < 10 or r.std(ddof=1) == 0.0:
        return float("nan"), float("nan")
    sr = r.mean() / r.std(ddof=1)
    lag = None if q is None else max(min(int(q), T - 2), 0)
    se = _rs.lo_sharpe_se(r, periods_per_year=ann_factor, lag=lag)
    return float(sr * np.sqrt(ann_factor)), float(se)


def sortino_ratio(
    returns: pd.Series | np.ndarray, ann_factor: float = 252.0
) -> float:
    """Annualised Sortino: mean over downside deviation (returns below 0)."""
    _check_ann_factor(ann_factor)
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    dd = downside_deviation(r)
    if dd == 0.0:
        return float("inf") if r.mean() > 0 else float("nan")
    return float(r.mean() / dd * np.sqrt(ann_factor))


def max_drawdown(pnl: pd.Series | np.ndarray) -> float:
    """Maximum drawdown of the cumulative-P&L equity curve (>= 0).

    Parameters
    ----------
    pnl : array-like
        Per-period P&L (not the equity curve); the curve is its cumsum.
    """
    return max_drawdown_from_pnl(_clean(pnl))


def hit_rate(trade_pnls: list[float] | np.ndarray) -> float:
    """Fraction of trades with strictly positive P&L; NaN with no trades."""
    p = np.asarray(list(trade_pnls), dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(np.mean(p > 0))


def turnover(positions: pd.Series | np.ndarray, ann_factor: float = 252.0) -> float:
    """Annualised one-leg turnover: mean |Δposition| per bar * ann_factor.

    Position units are spread units; a round trip of a 1-unit position
    contributes 2 to the |Δposition| sum.
    """
    _check_ann_factor(ann_factor)
    pos = np.asarray(positions, dtype=float)
    if not np.isfinite(pos).all():
        raise ValueError("positions contain NaN or infinite values")
    if len(pos) < 2:
        return 0.0
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return float(dpos.mean() * ann_factor)


def trade_pnls(result) -> list[float]:
    """Per-trade total P&L from a BacktestResult's trade list.

    Trade P&L is the sum of total P&L booked over ``(entry, exit]``: the bars
    on which the trade's position was earning returns: plus the entry cost
    booked at the entry bar.
    """
    total = result.total_pnl.to_numpy()
    cost = result.cost_pnl.to_numpy()
    out = []
    for tr in result.trades:
        pnl = float(total[tr.entry + 1 : tr.exit + 1].sum()) + float(cost[tr.entry])
        out.append(pnl)
    return out


def summarize(result, ann_factor: float = 252.0) -> dict[str, float]:
    """One-line summary of a BacktestResult (used by the pipeline and docs).

    Includes the carry-vs-spot decomposition: the number that tells you
    whether the 'mean reversion' P&L was really carry in disguise.
    """
    r = result.total_pnl
    sr, se = sharpe_se_lo(r, ann_factor=ann_factor)
    tps = trade_pnls(result)
    dec = result.decomposition()
    return {
        "total_pnl": dec["total"],
        "spot_pnl": dec["spot"],
        "carry_pnl": dec["carry"],
        "cost_pnl": dec["costs"],
        "sharpe": sr,
        "sharpe_se_lo": se,
        "sortino": sortino_ratio(r, ann_factor),
        "max_drawdown": max_drawdown(r),
        "hit_rate": hit_rate(tps),
        "n_trades": float(len(tps)),
        "turnover": turnover(result.positions, ann_factor),
    }
