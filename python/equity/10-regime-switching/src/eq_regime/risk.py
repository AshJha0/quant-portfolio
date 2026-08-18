"""Regime-conditional risk analytics.

Reports:

* per-regime realized return / vol / Sharpe / max drawdown of the strategy
  and its benchmark — regime day counts PARTITION the sample exactly
  (test-enforced);
* regime-transition P&L attribution — the per-regime P&L contributions sum
  to the total P&L identically (test-enforced);
* worst-case ("flip-aftermath") analysis — cumulative strategy P&L in the
  K days after each detected regime flip, quantifying how much a late
  detection costs.

Conventions: daily simple net returns; vol and Sharpe annualised with
sqrt(252); max drawdown positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252

__all__ = [
    "per_regime_stats",
    "transition_attribution",
    "flip_aftermath",
    "regime_runs",
]


def _mdd(net: np.ndarray) -> float:
    """Max drawdown of the compounded curve of a return subseries."""
    if len(net) == 0:
        return np.nan
    eq = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(eq)
    return float((1.0 - eq / peak).max())


def per_regime_stats(
    net_returns: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Per-regime performance table.

    Parameters
    ----------
    net_returns : pd.Series
        Daily simple net returns.
    regimes : pd.Series
        Regime label per day, SAME index as ``net_returns`` (each day
        belongs to exactly one regime — the counts partition the sample).

    Returns
    -------
    pd.DataFrame
        One row per regime plus a ``TOTAL`` row.  Columns: ``days``,
        ``ann_return`` (annualised mean), ``ann_vol``, ``sharpe``,
        ``max_drawdown``, ``total_pnl`` (sum of net returns).
    """
    if not net_returns.index.equals(regimes.index):
        raise ValueError("net_returns and regimes must share the same index")
    rows: dict[str, dict[str, float]] = {}
    for regime in pd.unique(regimes):
        r = net_returns[regimes == regime].to_numpy()
        std = r.std(ddof=1) if len(r) > 1 else np.nan
        rows[str(regime)] = {
            "days": float(len(r)),
            "ann_return": float(r.mean() * TRADING_DAYS),
            "ann_vol": float(std * np.sqrt(TRADING_DAYS)) if std == std else np.nan,
            "sharpe": float(r.mean() / std * np.sqrt(TRADING_DAYS)) if std and std > 0 else np.nan,
            "max_drawdown": _mdd(r),
            "total_pnl": float(r.sum()),
        }
    all_r = net_returns.to_numpy()
    std = all_r.std(ddof=1)
    rows["TOTAL"] = {
        "days": float(len(all_r)),
        "ann_return": float(all_r.mean() * TRADING_DAYS),
        "ann_vol": float(std * np.sqrt(TRADING_DAYS)),
        "sharpe": float(all_r.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else np.nan,
        "max_drawdown": _mdd(all_r),
        "total_pnl": float(all_r.sum()),
    }
    return pd.DataFrame(rows).T


def transition_attribution(net_returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Attribute total P&L to regimes and to transition days.

    Each day is attributed to its regime; days on which the regime CHANGED
    from the previous day are additionally broken out as ``flip_days`` so
    the cost of switching is visible.  The identity
    ``sum(regime P&L) == total P&L`` holds exactly (test-enforced) because
    the regime rows partition the sample.

    Returns
    -------
    pd.DataFrame
        Rows: one per regime, plus ``TOTAL``.  Columns ``pnl``, ``days``,
        ``flip_days``, ``flip_pnl`` (P&L of that regime's entry days).
    """
    if not net_returns.index.equals(regimes.index):
        raise ValueError("net_returns and regimes must share the same index")
    flips = regimes.ne(regimes.shift(1)).to_numpy().copy()
    flips[0] = False
    rows: dict[str, dict[str, float]] = {}
    for regime in pd.unique(regimes):
        mask = (regimes == regime).to_numpy()
        rows[str(regime)] = {
            "pnl": float(net_returns.to_numpy()[mask].sum()),
            "days": float(mask.sum()),
            "flip_days": float((mask & flips).sum()),
            "flip_pnl": float(net_returns.to_numpy()[mask & flips].sum()),
        }
    rows["TOTAL"] = {
        "pnl": float(net_returns.sum()),
        "days": float(len(net_returns)),
        "flip_days": float(flips.sum()),
        "flip_pnl": float(net_returns.to_numpy()[flips].sum()),
    }
    return pd.DataFrame(rows).T


def regime_runs(regimes: pd.Series) -> pd.DataFrame:
    """Contiguous regime runs: label, start, end, length in days."""
    labels = regimes.to_numpy()
    idx = regimes.index
    starts = [0] + [t for t in range(1, len(labels)) if labels[t] != labels[t - 1]]
    rows = []
    for i, s in enumerate(starts):
        e = starts[i + 1] - 1 if i + 1 < len(starts) else len(labels) - 1
        rows.append(
            {"regime": labels[s], "start": idx[s], "end": idx[e], "days": e - s + 1}
        )
    return pd.DataFrame(rows)


def flip_aftermath(
    net_returns: pd.Series,
    regimes: pd.Series,
    k: int = 10,
) -> pd.DataFrame:
    """Cumulative P&L in the K days after each regime flip (worst-case view).

    A regime model detects turns LATE by construction (it needs evidence to
    accumulate), so the days immediately after a flip are where mis-timing
    hurts.  For each flip date this reports the flip, the new regime, and
    the strategy's cumulative net return over days ``[t, t+K)``.

    Returns
    -------
    pd.DataFrame
        One row per flip: ``date``, ``from_regime``, ``to_regime``,
        ``pnl_next_{k}d`` (window truncated at the sample end).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not net_returns.index.equals(regimes.index):
        raise ValueError("net_returns and regimes must share the same index")
    labels = regimes.to_numpy()
    net = net_returns.to_numpy()
    rows = []
    for t in range(1, len(labels)):
        if labels[t] != labels[t - 1]:
            window = net[t : t + k]
            rows.append(
                {
                    "date": net_returns.index[t],
                    "from_regime": labels[t - 1],
                    "to_regime": labels[t],
                    f"pnl_next_{k}d": float(np.prod(1.0 + window) - 1.0),
                }
            )
    return pd.DataFrame(rows)
