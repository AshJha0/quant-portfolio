"""Performance and downside-risk metrics with FX-desk conventions.

All inputs are per-period (daily unless stated) log or simple returns; the
metrics do not care which as long as usage is consistent.  Annualisation uses
252 business days.  Downside metrics (skew, CVaR, drawdown) are first-class:
for carry books the tail statistics ARE the risk report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cvar_opt import empirical_cvar, empirical_var

TRADING_DAYS: int = 252


def _arr(returns: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(returns, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("empty return series")
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "return series contains NaN/Inf; drop or fill missing observations "
            "before computing metrics (a NaN would silently poison every stat)"
        )
    return x


def _degenerate_scale(x: np.ndarray) -> float:
    """Absolute tolerance below which a spread counts as numerically zero.

    An exactly-constant series does NOT produce ``std == 0`` in floating
    point: ``np.full(50, 4e-4).std(ddof=1)`` is ~5e-20, not 0.  An exact
    ``== 0`` guard therefore lets a constant series through and returns a
    Sharpe of ~1e17.  We compare against the scale of the data instead;
    any real FX return series has a standard deviation many orders of
    magnitude above this threshold.
    """
    return 1e-14 * max(1.0, float(np.abs(x).max()))


def annualized_return(
    returns: pd.Series | np.ndarray, periods: int = TRADING_DAYS
) -> float:
    """Arithmetic annualised mean: ``mean * periods``."""
    return float(_arr(returns).mean() * periods)


def annualized_vol(
    returns: pd.Series | np.ndarray, periods: int = TRADING_DAYS
) -> float:
    """Annualised volatility ``std(ddof=1) * sqrt(periods)``."""
    x = _arr(returns)
    if x.size < 2:
        raise ValueError("need >= 2 observations for volatility")
    return float(x.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    rf: float = 0.0,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ``(mean - rf_per_period) / std * sqrt(periods)``.

    Parameters
    ----------
    rf : float
        Per-period risk-free rate to subtract (0 for already-excess returns —
        FX long-short P&L is self-financing so excess by construction).
    """
    x = _arr(returns) - rf
    sd = x.std(ddof=1)
    if sd <= _degenerate_scale(x):
        raise ValueError("zero volatility: Sharpe undefined")
    return float(x.mean() / sd * np.sqrt(periods))


def sharpe_se_lo(
    returns: pd.Series | np.ndarray, periods: int = TRADING_DAYS
) -> float:
    """Standard error of the ANNUALISED Sharpe ratio, Lo (2002), iid case.

    For the per-period Sharpe ``SR``, ``se(SR_hat) = sqrt((1 + SR^2/2)/T)``;
    the annualised SE scales by ``sqrt(periods)``.  Lo's autocorrelation
    correction is a further multiplier not applied here; for FX styles at a
    monthly rebalance the iid SE is the standard desk t-stat denominator.
    """
    x = _arr(returns)
    if x.size < 2:
        raise ValueError("need >= 2 observations")
    sd = x.std(ddof=1)
    if sd <= _degenerate_scale(x):
        raise ValueError("zero volatility: Sharpe standard error undefined")
    sr = x.mean() / sd
    return float(np.sqrt((1.0 + 0.5 * sr**2) / x.size) * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series | np.ndarray,
    mar: float = 0.0,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualised Sortino: mean excess over MAR divided by downside deviation.

    Downside deviation uses the FULL-sample root mean square of
    ``min(r - mar, 0)`` (not just loss days), the standard convention.
    """
    x = _arr(returns) - mar
    downside = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    if downside <= _degenerate_scale(x):
        raise ValueError("no downside observations: Sortino undefined")
    return float(x.mean() / downside * np.sqrt(periods))


def max_drawdown(returns: pd.Series | np.ndarray, log_returns: bool = True) -> float:
    """Maximum peak-to-trough drawdown (positive number, fraction of peak).

    Parameters
    ----------
    log_returns : bool
        If True (default) compound via ``exp(cumsum)``; else via
        ``cumprod(1+r)``.
    """
    x = _arr(returns)
    curve = np.exp(np.cumsum(x)) if log_returns else np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(curve)
    return float((1.0 - curve / peak).max())


def skewness(returns: pd.Series | np.ndarray) -> float:
    """Sample skewness (biased, moment definition — matches scipy default)."""
    x = _arr(returns)
    xc = x - x.mean()
    m2 = np.mean(xc**2)
    if m2 <= _degenerate_scale(x) ** 2:
        raise ValueError("zero variance: skewness undefined")
    return float(np.mean(xc**3) / m2**1.5)


def excess_kurtosis(returns: pd.Series | np.ndarray) -> float:
    """Sample excess kurtosis (biased, Fisher definition — scipy default)."""
    x = _arr(returns)
    xc = x - x.mean()
    m2 = np.mean(xc**2)
    if m2 <= _degenerate_scale(x) ** 2:
        raise ValueError("zero variance: kurtosis undefined")
    return float(np.mean(xc**4) / m2**2 - 3.0)


def summary(
    returns: pd.Series | np.ndarray,
    alpha: float = 0.95,
    periods: int = TRADING_DAYS,
) -> dict[str, float]:
    """One-stop stat block: return/vol/Sharpe(+SE)/Sortino/MDD/skew/kurt/VaR/CVaR.

    VaR/CVaR are per-period loss fractions at level ``alpha``.
    """
    return {
        "ann_return": annualized_return(returns, periods),
        "ann_vol": annualized_vol(returns, periods),
        "sharpe": sharpe_ratio(returns, periods=periods),
        "sharpe_se": sharpe_se_lo(returns, periods),
        "sortino": sortino_ratio(returns, periods=periods),
        "max_drawdown": max_drawdown(returns),
        "skew": skewness(returns),
        "excess_kurtosis": excess_kurtosis(returns),
        "var": empirical_var(returns, alpha),
        "cvar": empirical_cvar(returns, alpha),
    }


def style_attribution(
    style_returns: pd.DataFrame, weights: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    """Per-style P&L attribution of a multi-style book.

    Parameters
    ----------
    style_returns : pd.DataFrame
        Daily returns per style sleeve.
    weights : pd.Series or pd.DataFrame
        Static weights (Series) or daily weight panel (DataFrame, same index).

    Returns
    -------
    pd.DataFrame
        Daily contribution per style plus a ``total`` column equal to their
        sum (exact additivity — tested).
    """
    if isinstance(weights, pd.Series):
        contrib = style_returns.mul(weights, axis=1)
    else:
        if not weights.index.equals(style_returns.index):
            raise ValueError("weights panel must share the returns index")
        contrib = style_returns * weights
    contrib = contrib.copy()
    contrib["total"] = contrib.sum(axis=1)
    return contrib
