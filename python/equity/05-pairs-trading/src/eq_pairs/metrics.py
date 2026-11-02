"""Performance and risk metrics for pairs backtests.

Conventions: daily data, 252 trading days/year, returns are simple daily
returns on allocated capital (net P&L / capital). Sharpe and Sortino are
annualised with sqrt(252). The Sharpe standard error is reported both under
the iid assumption and with the Lo (2002) GMM correction for serial
correlation (Newey-West weights, truncated lag window). Mean-reverting
strategies produce autocorrelated daily P&L, where the iid SE is too small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qp_core.metrics import downside_deviation, max_drawdown_from_equity
from qp_core.metrics import drawdown_series as _drawdown_series

from .research_stats import lo_sharpe_se

__all__ = [
    "avg_holding_period",
    "cost_drag",
    "drawdown_series",
    "hit_rate",
    "max_drawdown",
    "sharpe_ratio",
    "sharpe_se",
    "sortino_ratio",
    "summary",
    "turnover",
]

ArrayLike = np.ndarray | pd.Series | list

_COST_COLUMNS = ("commission", "slippage", "impact", "borrow", "financing", "auction")


def _to_array(returns: ArrayLike, min_len: int = 2) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1:
        raise ValueError("returns must be 1-D")
    if len(r) < min_len:
        raise ValueError(f"need at least {min_len} observations, got {len(r)}")
    if np.any(~np.isfinite(r)):
        raise ValueError("returns contain NaN/inf")
    return r


def sharpe_ratio(returns: ArrayLike, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio mean/std * sqrt(periods) (rf = 0).

    Returns NaN when the return standard deviation is zero (all-cash or
    constant P&L stream: a Sharpe is meaningless, not infinite).
    """
    r = _to_array(returns)
    if np.all(r == r[0]):  # constant stream: std is 0 up to rounding
        return np.nan
    sd = r.std(ddof=1)
    if sd == 0.0:
        return np.nan
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sharpe_se(
    returns: ArrayLike,
    periods_per_year: int = 252,
    lo_adjust: bool = False,
    q: int | None = None,
) -> float:
    """Standard error of the annualised Sharpe ratio.

    iid case (lo_adjust=False): SE = sqrt((1 + SR_p^2 / 2) / T) * sqrt(P)
    with SR_p the per-period Sharpe (Lo 2002, eq. 12).

    lo_adjust=True: Lo (2002) GMM delta method with a Newey-West (Bartlett)
    long-run variance truncated at ``q`` lags (default
    ``floor(4 (T/100)^(2/9))``). Delegates to
    :func:`eq_pairs.research_stats.lo_sharpe_se`. With positively
    autocorrelated P&L (typical of mean-reversion books) it exceeds the iid SE.

    Returns NaN for a constant stream (the Sharpe ratio is undefined).
    """
    r = _to_array(returns, min_len=3)
    t = len(r)
    if np.all(r == r[0]):
        return np.nan
    sd = r.std(ddof=1)
    if sd == 0.0:
        return np.nan
    if lo_adjust:
        if q is not None and q >= t:
            raise ValueError(f"q must be < T, got q={q}, T={t}")
        return lo_sharpe_se(r, periods_per_year, lag=q)
    sr_p = r.mean() / sd
    return float(np.sqrt((1.0 + 0.5 * sr_p**2) / t) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: ArrayLike, periods_per_year: int = 252, mar: float = 0.0
) -> float:
    """Annualised Sortino ratio: (mean - mar) / downside deviation * sqrt(P).

    Downside deviation = sqrt(mean(min(r - mar, 0)^2)) over ALL observations
    (full-sample convention). NaN when there are no observations below the
    MAR (undefined, not infinite).
    """
    r = _to_array(returns)
    dd = downside_deviation(r, mar)
    if dd == 0.0:
        return np.nan
    return float((r.mean() - mar) / dd * np.sqrt(periods_per_year))


def drawdown_series(equity: ArrayLike) -> np.ndarray:
    """Drawdown from running peak, in the units of ``equity`` (<= 0)."""
    return _drawdown_series(_to_array(equity, min_len=1))


def max_drawdown(equity: ArrayLike) -> float:
    """Maximum drawdown depth (non-negative number, units of ``equity``).

    Pass an equity/cumulative-P&L curve, not returns. A monotonically rising
    curve returns ``0.0`` (normalised: negating a drawdown series whose
    minimum is ``0.0`` would otherwise yield ``-0.0``, which prints as a
    negative drawdown in reports).
    """
    return max_drawdown_from_equity(_to_array(equity, min_len=1))


def hit_rate(trade_pnls: ArrayLike) -> float:
    """Fraction of round trips with strictly positive net P&L.

    NaN for an empty trade list (no information, not 0%).
    """
    p = np.asarray(trade_pnls, dtype=float)
    if p.size == 0:
        return np.nan
    return float(np.mean(p > 0.0))


def avg_holding_period(bars_held: ArrayLike) -> float:
    """Average round-trip holding period in bars; NaN if no trades."""
    b = np.asarray(bars_held, dtype=float)
    if b.size == 0:
        return np.nan
    return float(b.mean())


def turnover(
    total_traded_notional: float, capital: float, n_days: int, days_per_year: int = 252
) -> float:
    """Annualised turnover: traded notional per year / capital.

    E.g. 12.0 = the book trades 12x its capital per year (both sides
    counted, per-leg notional).
    """
    if capital <= 0:
        raise ValueError(f"capital must be positive, got {capital}")
    if n_days <= 0:
        raise ValueError(f"n_days must be positive, got {n_days}")
    years = n_days / days_per_year
    return float(total_traded_notional / capital / years)


def cost_drag(
    total_costs: float, capital: float, n_days: int, days_per_year: int = 252
) -> float:
    """Annualised cost drag as a fraction of capital (e.g. 0.012 = 1.2%/yr)."""
    if capital <= 0:
        raise ValueError(f"capital must be positive, got {capital}")
    if n_days <= 0:
        raise ValueError(f"n_days must be positive, got {n_days}")
    years = n_days / days_per_year
    return float(total_costs / capital / years)


def summary(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    ledger: pd.DataFrame,
    capital: float,
    days_per_year: int = 252,
) -> dict[str, float]:
    """One-stop metrics table for a PairResult / PortfolioResult.

    Parameters
    ----------
    daily : DataFrame
        Must contain net_pnl, gross_pnl, commission, slippage, borrow.
    trades : DataFrame
        Must contain pnl and bars_held (may be empty).
    ledger : DataFrame
        Must contain notional (may be empty).
    capital : float
        Capital base for returns/turnover (dollars).

    Returns
    -------
    dict of metric name -> value.
    """
    if capital <= 0:
        raise ValueError(f"capital must be positive, got {capital}")
    ret = daily["net_pnl"].to_numpy(dtype=float) / capital
    equity = capital + daily["net_pnl"].cumsum().to_numpy(dtype=float)
    n = len(ret)
    cost_cols = [c for c in _COST_COLUMNS if c in daily.columns]
    costs_total = float(daily[cost_cols].to_numpy().sum())
    traded = float(ledger["notional"].sum()) if len(ledger) else 0.0
    all_cash = bool(np.all(ret == 0.0))
    return {
        "n_days": n,
        "total_net_pnl": float(daily["net_pnl"].sum()),
        "total_gross_pnl": float(daily["gross_pnl"].sum()),
        "total_costs": costs_total,
        "ann_return": float(ret.mean() * days_per_year),
        "ann_vol": float(ret.std(ddof=1) * np.sqrt(days_per_year)) if n > 1 else np.nan,
        "sharpe": np.nan if all_cash else sharpe_ratio(ret, days_per_year),
        "sharpe_se_iid": np.nan if all_cash else sharpe_se(ret, days_per_year),
        "sharpe_se_lo": np.nan
        if all_cash
        else sharpe_se(ret, days_per_year, lo_adjust=True),
        "sortino": np.nan if all_cash else sortino_ratio(ret, days_per_year),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": hit_rate(trades["pnl"]) if len(trades) else np.nan,
        "avg_holding_days": avg_holding_period(trades["bars_held"])
        if len(trades)
        else np.nan,
        "n_trades": len(trades),
        "turnover": turnover(traded, capital, n, days_per_year) if n > 0 else np.nan,
        "cost_drag": cost_drag(costs_total, capital, n, days_per_year)
        if n > 0
        else np.nan,
    }
