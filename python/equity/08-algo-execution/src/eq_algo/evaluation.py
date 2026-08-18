"""Performance & signal evaluation: IC stats, Newey-West, deflated Sharpe,
capacity curves.

Conventions: returns are daily simple returns; Sharpe/Sortino are annualised
with 252; drawdowns on the compounded wealth curve.  The deflated Sharpe
ratio follows Bailey & Lopez de Prado (2014) and is the project's guard
against backtest overfitting: when N signal variants were tried, the best
in-sample Sharpe must be benchmarked against the expected maximum Sharpe of
N pure-noise strategies, not against zero (see docs/METHODOLOGY.md).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "newey_west_se",
    "newey_west_tstat",
    "ic_summary",
    "quantile_monotonicity",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "perf_summary",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "capacity_curve",
]

_EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# IC statistics
# ---------------------------------------------------------------------------

def newey_west_se(x: pd.Series | np.ndarray, lags: int = 5) -> float:
    """Newey-West (Bartlett-kernel) standard error of the sample mean.

    ``Var_NW(mean) = (g0 + 2 * sum_l (1 - l/(L+1)) * g_l) / n`` with sample
    autocovariances ``g_l``.  Robust to autocorrelation in the series (daily
    IC series are autocorrelated; overlapping multi-day ICs even more so).
    """
    arr = pd.Series(x).dropna().to_numpy(dtype=float)
    n = arr.size
    if n < 2:
        raise ValueError("need at least 2 observations")
    if lags < 0:
        raise ValueError("lags must be >= 0")
    d = arr - arr.mean()
    g0 = float(d @ d) / n
    s = g0
    for l in range(1, min(lags, n - 1) + 1):
        gl = float(d[l:] @ d[:-l]) / n
        s += 2.0 * (1.0 - l / (lags + 1.0)) * gl
    s = max(s, 0.0)
    return float(np.sqrt(s / n))


def newey_west_tstat(x: pd.Series | np.ndarray, lags: int = 5) -> float:
    """t-statistic of the mean using the Newey-West standard error."""
    arr = pd.Series(x).dropna()
    se = newey_west_se(arr, lags)
    if se == 0.0:
        raise ValueError("Newey-West SE is zero; t-stat undefined")
    return float(arr.mean() / se)


def ic_summary(ic: pd.Series, lags: int = 5) -> dict[str, float]:
    """Summary of a daily IC series: mean, std, naive and NW t-stats, ICIR."""
    x = ic.dropna()
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 IC observations")
    mean, std = float(x.mean()), float(x.std(ddof=1))
    return {
        "mean_ic": mean,
        "ic_std": std,
        "n_obs": float(n),
        "tstat_naive": mean / (std / np.sqrt(n)),
        "tstat_nw": newey_west_tstat(x, lags),
        "icir_annual": mean / std * np.sqrt(252.0),
    }


def quantile_monotonicity(decile_means: pd.Series | Sequence[float]) -> float:
    """Spearman correlation between quantile index and mean forward return.

    +1 means perfectly monotone increasing across quantiles (what a healthy
    alpha should show); input is the time-averaged return per quantile,
    lowest first.
    """
    y = np.asarray(pd.Series(decile_means).dropna(), dtype=float)
    if y.size < 3:
        raise ValueError("need at least 3 quantiles")
    rho, _ = stats.spearmanr(np.arange(y.size), y)
    return float(rho)


# ---------------------------------------------------------------------------
# Return-based performance
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio of periodic returns (rf = 0)."""
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    if r.size < 2:
        raise ValueError("need at least 2 returns")
    sd = r.std(ddof=1)
    if sd == 0.0:
        raise ValueError("zero volatility; Sharpe undefined")
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sortino ratio: mean over downside deviation (rf = 0)."""
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    if r.size < 2:
        raise ValueError("need at least 2 returns")
    downside = np.minimum(r, 0.0)
    dd = np.sqrt((downside**2).mean())
    if dd == 0.0:
        raise ValueError("no downside observations; Sortino undefined")
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    """Maximum drawdown of the compounded wealth curve, as a positive fraction."""
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    if r.size == 0:
        raise ValueError("empty return series")
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    return float(np.max(1.0 - wealth / peak))


def perf_summary(returns: pd.Series, turnover: pd.Series | None = None,
                 periods_per_year: int = 252) -> dict[str, float]:
    """Sharpe / Sortino / MDD / annualised return & vol / mean turnover."""
    r = returns.dropna()
    out = {
        "ann_return": float(r.mean() * periods_per_year),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(periods_per_year)),
        "sharpe": sharpe_ratio(r, periods_per_year),
        "sortino": sortino_ratio(r, periods_per_year),
        "max_drawdown": max_drawdown(r),
    }
    if turnover is not None:
        out["mean_daily_turnover"] = float(turnover.dropna().mean())
    return out


# ---------------------------------------------------------------------------
# Deflated Sharpe ratio (Bailey & Lopez de Prado 2014)
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: int,
                               skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR: probability the true Sharpe exceeds ``sr_benchmark``.

    ``PSR = Phi( (SR - SR*) * sqrt(n-1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) )``

    All Sharpe ratios here are **per-period** (not annualised); ``kurt`` is
    the raw (non-excess) kurtosis, 3.0 for a Gaussian.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom_sq <= 0:
        raise ValueError("invalid skew/kurtosis combination (variance of SR <= 0)")
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / np.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """Expected maximum Sharpe of ``n_trials`` zero-skill strategies.

    Bailey-Lopez de Prado approximation to E[max of N iid N(0, var_sharpe)]:

    ``SR* = sqrt(V) * ( (1-gamma) * z(1 - 1/N) + gamma * z(1 - 1/(N*e)) )``

    with Euler-Mascheroni ``gamma``.  By convention ``SR* = 0`` for a single
    trial (no selection took place), which makes DSR = PSR at N = 1.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if var_sharpe < 0:
        raise ValueError("var_sharpe must be >= 0")
    if n_trials == 1:
        return 0.0
    e = _EULER_GAMMA
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sharpe) * ((1.0 - e) * z1 + e * z2))


def deflated_sharpe_ratio(returns: pd.Series | np.ndarray, n_trials: int,
                          var_sharpe_trials: float | None = None) -> dict[str, float]:
    """Deflated Sharpe ratio of a strategy selected among ``n_trials`` variants.

    The observed (per-period) Sharpe is benchmarked against the expected max
    Sharpe of ``n_trials`` noise strategies with cross-trial Sharpe variance
    ``var_sharpe_trials`` (default: the estimator variance ``1/n_obs`` — a
    conservative floor when the true cross-trial dispersion is unknown).

    Returns a dict with ``sr`` (per period), ``sr_annual``, ``sr_benchmark``,
    ``psr0`` (PSR vs 0) and ``dsr``.
    """
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = r.size
    if n < 4:
        raise ValueError("need at least 4 returns")
    sd = r.std(ddof=1)
    if sd == 0:
        raise ValueError("zero volatility")
    sr = float(r.mean() / sd)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, bias=False, fisher=False))
    if var_sharpe_trials is None:
        var_sharpe_trials = 1.0 / n
    sr_star = expected_max_sharpe(n_trials, var_sharpe_trials)
    return {
        "sr": sr,
        "sr_annual": sr * np.sqrt(252.0),
        "skew": skew,
        "kurt": kurt,
        "n_obs": float(n),
        "sr_benchmark": sr_star,
        "psr0": probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurt),
        "dsr": probabilistic_sharpe_ratio(sr, sr_star, n, skew, kurt),
    }


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

def capacity_curve(gross_returns: pd.Series, mean_turnover: float,
                   aum_grid: Sequence[float], adv_dollars: float, n_names: int,
                   sigma_daily: float = 0.02, linear_cost_bps: float = 5.0,
                   impact_coef: float = 0.1) -> pd.DataFrame:
    """Cost-adjusted Sharpe as a function of AUM (capacity estimate).

    Assumes daily one-way turnover ``mean_turnover`` spread evenly over
    ``n_names`` names with average dollar ADV ``adv_dollars``; the per-day
    cost drag (fraction of NAV) at AUM ``A`` is

        drag(A) = T * [ lin + k * sigma * sqrt( A * T / (n * ADV$) ) ]

    (linear + square-root impact on the average trade).  Because the drag is
    a constant shift of the daily return, the net Sharpe is strictly
    decreasing in AUM — the capacity number is where it crosses a threshold
    (e.g. half the gross Sharpe).  This is a planning approximation: real
    capacity work re-runs the full backtest per AUM (see docs/DESK_GUIDE.md).
    """
    if mean_turnover < 0:
        raise ValueError("mean_turnover must be >= 0")
    if adv_dollars <= 0 or n_names <= 0:
        raise ValueError("adv_dollars and n_names must be > 0")
    r = gross_returns.dropna()
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    if sd == 0:
        raise ValueError("zero volatility in gross returns")
    rows = []
    for aum in aum_grid:
        if aum <= 0:
            raise ValueError("AUM values must be > 0")
        part = aum * mean_turnover / (n_names * adv_dollars)
        drag = mean_turnover * (linear_cost_bps * 1e-4
                                + impact_coef * sigma_daily * np.sqrt(part))
        rows.append({
            "aum": float(aum),
            "participation": float(part),
            "daily_cost_drag": float(drag),
            "net_sharpe": (mu - drag) / sd * np.sqrt(252.0),
        })
    return pd.DataFrame(rows).set_index("aum")
