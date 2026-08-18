"""Performance and risk metrics.

Conventions: inputs are simple per-period returns (default daily,
``periods_per_year=252``); annualised return is geometric; volatility is
the sample standard deviation (ddof=1) scaled by sqrt(periods_per_year);
Sharpe uses arithmetic mean excess return over per-period vol, annualised
by sqrt(periods_per_year) (industry convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "annualized_return",
    "annualized_vol",
    "sharpe_ratio",
    "sharpe_lo",
    "LoSharpeResult",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "diversification_ratio",
    "effective_n",
    "realized_risk_contributions",
    "summary_table",
]


def _as_vector(returns: np.ndarray | pd.Series) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 1:
        raise ValueError("returns must be non-empty")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns contain NaN or infinite values")
    return r


def annualized_return(returns: np.ndarray | pd.Series, periods_per_year: float = 252.0) -> float:
    """Geometric annualised return: (prod(1+r))^(ppy/T) - 1."""
    r = _as_vector(returns)
    growth = float(np.prod(1.0 + r))
    if growth <= 0.0:
        return -1.0  # wealth wiped out (or worse); cap at total loss
    return growth ** (periods_per_year / r.size) - 1.0


def annualized_vol(returns: np.ndarray | pd.Series, periods_per_year: float = 252.0) -> float:
    """Annualised volatility: std(r, ddof=1) * sqrt(ppy). Zero for T < 2."""
    r = _as_vector(returns)
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: np.ndarray | pd.Series,
    rf: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Annualised Sharpe: mean(r - rf) / std(r, ddof=1) * sqrt(ppy).

    ``rf`` is the PER-PERIOD risk-free rate. Returns 0.0 when vol is 0.
    """
    r = _as_vector(returns) - rf
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(r)) / sd * np.sqrt(periods_per_year)


@dataclass(frozen=True)
class LoSharpeResult:
    """Annualised Sharpe with Lo (2002) autocorrelation-adjusted inference.

    Attributes
    ----------
    sharpe : float
        Annualised Sharpe (iid sqrt-time scaling, for comparability).
    sharpe_lo : float
        Annualised Sharpe using Lo's autocorrelation-adjusted scaling
        eta(q) = ppy / sqrt(ppy + 2 * sum_{k<q} (q - k) rho_k); equals the
        iid value when returns are serially uncorrelated.
    se : float
        Standard error of the ANNUALISED Sharpe:
        sqrt((1 + SR_p^2 / 2) / T) * eta(q), where SR_p is the per-period
        Sharpe (Lo 2002, eq. for iid-normal sampling variance, scaled by
        the autocorrelation-adjusted annualisation factor).
    n_lags : int
        Number of autocorrelation lags q used.
    """

    sharpe: float
    sharpe_lo: float
    se: float
    n_lags: int


def sharpe_lo(
    returns: np.ndarray | pd.Series,
    rf: float = 0.0,
    periods_per_year: float = 252.0,
    n_lags: int | None = None,
) -> LoSharpeResult:
    """Annualised Sharpe with Lo (2002) autocorrelation-adjusted scaling
    and standard error.

    Serially correlated returns break the sqrt-time Sharpe annualisation:
    positive autocorrelation (e.g. from stale pricing / smoothing)
    overstates the annual Sharpe. Lo's factor replaces sqrt(q) with
    ``q / sqrt(q + 2 sum_{k=1}^{q-1} (q - k) rho_k)`` for q = ppy.

    Parameters
    ----------
    returns : array-like
        Simple per-period returns.
    rf : float
        Per-period risk-free rate.
    periods_per_year : float
        Aggregation horizon q (252 for daily -> annual).
    n_lags : int, optional
        Autocorrelation lags used (default min(ppy - 1, T // 4)).

    Returns
    -------
    LoSharpeResult
    """
    r = _as_vector(returns) - rf
    t = r.size
    if t < 3:
        raise ValueError(f"need at least 3 observations, got {t}")
    sd = float(np.std(r, ddof=1))
    if sd == 0.0:
        return LoSharpeResult(0.0, 0.0, float(np.sqrt(1.0 / t * periods_per_year)), 0)
    sr_p = float(np.mean(r)) / sd  # per-period Sharpe
    q = int(round(periods_per_year))
    k_max = min(q - 1, t // 4) if n_lags is None else int(n_lags)
    k_max = max(k_max, 0)
    rc = r - r.mean()
    denom = float(rc @ rc)
    adj = float(q)
    for k in range(1, k_max + 1):
        rho_k = float(rc[k:] @ rc[:-k]) / denom
        adj += 2.0 * (q - k) * rho_k
    adj = max(adj, 1e-12)  # guard against pathological negative sums
    eta = q / np.sqrt(adj)
    se_ann = float(np.sqrt((1.0 + 0.5 * sr_p**2) / t) * eta)
    return LoSharpeResult(
        sharpe=sr_p * np.sqrt(periods_per_year),
        sharpe_lo=sr_p * eta,
        se=se_ann,
        n_lags=k_max,
    )


def sortino_ratio(
    returns: np.ndarray | pd.Series,
    rf: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Annualised Sortino: mean excess return over downside deviation.

    Downside deviation = sqrt(mean(min(r - rf, 0)^2)) over ALL periods
    (full-sample convention). Returns inf when there are no down periods
    and the mean excess return is positive.
    """
    r = _as_vector(returns) - rf
    down = np.minimum(r, 0.0)
    dd = float(np.sqrt(np.mean(down**2)))
    mean = float(np.mean(r))
    if dd == 0.0:
        return float("inf") if mean > 0 else 0.0
    return mean / dd * np.sqrt(periods_per_year)


def max_drawdown(returns: np.ndarray | pd.Series) -> float:
    """Maximum peak-to-trough drawdown of compounded wealth, as a
    POSITIVE fraction (0.25 = -25% drawdown). Wealth starts at 1 and the
    running peak includes that starting point."""
    r = _as_vector(returns)
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(wealth)
    dd = 1.0 - wealth / peak
    return float(dd.max())


def calmar_ratio(returns: np.ndarray | pd.Series, periods_per_year: float = 252.0) -> float:
    """Annualised return / max drawdown; inf if drawdown is 0 and return > 0."""
    ar = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    if mdd == 0.0:
        return float("inf") if ar > 0 else 0.0
    return ar / mdd


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """Choueifaty-Coignard diversification ratio
    DR = (w' sigma_vec) / sqrt(w' Sigma w) for long-only w; DR >= 1 with
    equality iff the portfolio is a single asset or all correlations are 1.
    """
    w = np.asarray(weights, dtype=float).ravel()
    sigma = np.asarray(cov, dtype=float)
    if w.shape[0] != sigma.shape[0]:
        raise ValueError(
            f"dimension mismatch: weights {w.shape[0]}, cov {sigma.shape}"
        )
    if np.any(w < -1e-12):
        raise ValueError("diversification ratio is defined for long-only weights")
    num = float(w @ np.sqrt(np.diag(sigma)))
    den = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    if den == 0.0:
        raise ValueError("portfolio has zero volatility")
    return num / den


def effective_n(weights: np.ndarray) -> float:
    """Effective number of holdings 1 / sum(w_i^2) for weights summing
    to 1; equals N for equal weight and 1 for a single-asset portfolio."""
    w = np.asarray(weights, dtype=float).ravel()
    s = float(np.sum(w**2))
    if s == 0.0:
        raise ValueError("weights are all zero")
    return 1.0 / s


def realized_risk_contributions(
    weights: np.ndarray, returns: np.ndarray | pd.DataFrame
) -> np.ndarray:
    """Ex-post risk contributions using the realised sample covariance:
    RC_i = w_i (S w)_i / (w' S w), returned as FRACTIONS summing to 1.

    Parameters
    ----------
    weights : (N,) array-like
        Fixed weights over the evaluation window.
    returns : (T, N) array-like
        Realised per-period returns over the window (T >= 2).
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"returns must be (T, N), got shape {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("need at least 2 observations")
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != x.shape[1]:
        raise ValueError(
            f"dimension mismatch: weights {w.shape[0]}, returns {x.shape[1]} assets"
        )
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / (x.shape[0] - 1)
    rc = w * (s @ w)
    tot = rc.sum()
    if tot <= 0:
        raise ValueError("realised portfolio variance is zero")
    return rc / tot


def summary_table(
    returns_by_name: dict[str, np.ndarray | pd.Series],
    rf: float = 0.0,
    periods_per_year: float = 252.0,
) -> pd.DataFrame:
    """Metrics table (one row per strategy): AnnRet, AnnVol, Sharpe,
    Sharpe SE (Lo), Sortino, MaxDD, Calmar."""
    rows = {}
    for name, r in returns_by_name.items():
        lo = sharpe_lo(r, rf=rf, periods_per_year=periods_per_year)
        rows[name] = {
            "AnnRet": annualized_return(r, periods_per_year),
            "AnnVol": annualized_vol(r, periods_per_year),
            "Sharpe": sharpe_ratio(r, rf, periods_per_year),
            "SharpeSE_Lo": lo.se,
            "Sortino": sortino_ratio(r, rf, periods_per_year),
            "MaxDD": max_drawdown(r),
            "Calmar": calmar_ratio(r, periods_per_year),
        }
    return pd.DataFrame(rows).T
