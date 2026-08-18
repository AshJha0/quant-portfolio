"""Performance and risk metrics: Sharpe with Lo (2002) SE, Sortino, MDD, turnover.

Conventions: daily P&L in units of account; annualisation factor 252 business
days; Sharpe assumes zero funding benchmark (carry is already inside the P&L,
which is precisely the point of the carry decomposition).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    r = np.asarray(returns, dtype=float)
    return r[np.isfinite(r)]


def sharpe_ratio(returns: pd.Series | np.ndarray, ann_factor: float = 252.0) -> float:
    """Annualised Sharpe ratio ``mean/std * sqrt(ann_factor)``; NaN if std=0."""
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
    """Sharpe ratio with a Lo (2002)-style autocorrelation-robust standard error.

    Under iid returns ``Var(SR_hat) ~ (1 + SR^2/2) / T`` (per-period SR).
    With autocorrelated returns the variance of the sample mean inflates by
    the Newey-West factor ``g = 1 + 2 * sum_{k=1..q} (1 - k/(q+1)) rho_k``;
    we report ``SE = sqrt((g + SR^2/2) / T) * sqrt(ann_factor)``.  For iid
    data ``g ~ 1`` and this collapses to the classical Lo SE.

    Parameters
    ----------
    q : int, optional
        Number of autocorrelation lags; default ``floor(T^(1/3))``.

    Returns
    -------
    (sharpe_annualised, se_annualised)
    """
    r = _clean(returns)
    T = len(r)
    if T < 10:
        return float("nan"), float("nan")
    sd = r.std(ddof=1)
    if sd == 0.0:
        return float("nan"), float("nan")
    sr = r.mean() / sd
    if q is None:
        q = int(np.floor(T ** (1.0 / 3.0)))
    q = max(min(q, T - 2), 0)
    x = r - r.mean()
    denom = float(x @ x)
    g = 1.0
    for k in range(1, q + 1):
        rho_k = float(x[k:] @ x[:-k]) / denom
        g += 2.0 * (1.0 - k / (q + 1.0)) * rho_k
    g = max(g, 1e-6)
    se = np.sqrt((g + 0.5 * sr**2) / T)
    return float(sr * np.sqrt(ann_factor)), float(se * np.sqrt(ann_factor))


def sortino_ratio(
    returns: pd.Series | np.ndarray, ann_factor: float = 252.0
) -> float:
    """Annualised Sortino: mean over downside deviation (returns below 0)."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    downside = np.minimum(r, 0.0)
    dd = np.sqrt(np.mean(downside**2))
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
    r = _clean(pnl)
    if len(r) == 0:
        return 0.0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.maximum(eq, 0.0))
    return float(np.max(peak - eq, initial=0.0))


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
    pos = np.asarray(positions, dtype=float)
    if len(pos) < 2:
        return 0.0
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return float(dpos.mean() * ann_factor)


def trade_pnls(result) -> list[float]:
    """Per-trade total P&L from a BacktestResult's trade list.

    Trade P&L is the sum of total P&L booked over ``(entry, exit]`` — the bars
    on which the trade's position was earning returns — plus the entry cost
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

    Includes the carry-vs-spot decomposition — the number that tells you
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
