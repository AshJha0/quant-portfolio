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
from qp_core.metrics import downside_deviation, max_drawdown_from_returns
from scipy.stats import norm

__all__ = [
    "LoSharpeResult",
    "SharpeDiffResult",
    "annualized_return",
    "annualized_vol",
    "calmar_ratio",
    "diversification_ratio",
    "effective_n",
    "max_drawdown",
    "realized_risk_contributions",
    "sharpe_difference_test",
    "sharpe_lo",
    "sharpe_ratio",
    "sortino_ratio",
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
    ``q / sqrt(q + 2 sum_{k=1}^{L} (q - k) b_k rho_k)`` for q = ppy.

    Summing all q - 1 sample autocorrelations (Lo's formula taken
    literally at q = 252) adds hundreds of noisy terms: on iid data the
    adjusted-to-raw ratio then spans roughly 0.76 to 1.54. The sum is
    therefore truncated at L lags with Bartlett weights
    b_k = 1 - k / (L + 1) (Newey and West 1987), which keeps the
    variance-ratio estimate non-negative and its noise small.

    Parameters
    ----------
    returns : array-like
        Simple per-period returns.
    rf : float
        Per-period risk-free rate.
    periods_per_year : float
        Aggregation horizon q (252 for daily -> annual).
    n_lags : int, optional
        Truncation lag L (default floor(T ** (1/3)), capped at q - 1).
        ``n_lags=0`` gives the iid scaling.

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
    if n_lags is None:
        k_max = int(np.floor(t ** (1.0 / 3.0)))
    else:
        if n_lags < 0:
            raise ValueError(f"n_lags must be >= 0, got {n_lags}")
        k_max = int(n_lags)
    k_max = min(k_max, q - 1, t - 1)
    rc = r - r.mean()
    denom = float(rc @ rc)
    adj = float(q)
    for k in range(1, k_max + 1):
        rho_k = float(rc[k:] @ rc[:-k]) / denom
        adj += 2.0 * (q - k) * (1.0 - k / (k_max + 1.0)) * rho_k
    adj = max(adj, 1e-12)  # guard against pathological negative sums
    eta = q / np.sqrt(adj)
    se_ann = float(np.sqrt((1.0 + 0.5 * sr_p**2) / t) * eta)
    return LoSharpeResult(
        sharpe=sr_p * np.sqrt(periods_per_year),
        sharpe_lo=sr_p * eta,
        se=se_ann,
        n_lags=k_max,
    )



@dataclass(frozen=True)
class SharpeDiffResult:
    """Jobson-Korkie test of equal Sharpe ratios, Memmel (2003) form.

    Attributes
    ----------
    diff : float
        Annualised Sharpe of series 1 minus series 2.
    se : float
        Annualised standard error of ``diff`` under iid normal returns.
    z : float
        Test statistic (per-period difference over its standard error).
    p_value : float
        Two-sided p-value from the standard normal.
    """

    diff: float
    se: float
    z: float
    p_value: float


def sharpe_difference_test(
    r1: np.ndarray | pd.Series,
    r2: np.ndarray | pd.Series,
    periods_per_year: float = 252.0,
) -> SharpeDiffResult:
    """Test H0: SR1 = SR2 for two return series observed on the same dates.

    Memmel's (2003) correction of the Jobson-Korkie (1981) statistic:

        Var(SR1 - SR2) = (1/T) [2 - 2 rho + (SR1^2 + SR2^2 - 2 SR1 SR2 rho^2) / 2]

    with per-period Sharpe ratios and rho the correlation of the two
    series. It assumes iid normal returns. Use a block bootstrap when the
    returns are serially correlated or fat-tailed.
    """
    a = _as_vector(r1)
    b = _as_vector(r2)
    if a.shape != b.shape:
        raise ValueError(f"series lengths differ: {a.size} vs {b.size}")
    t = a.size
    if t < 3:
        raise ValueError(f"need at least 3 observations, got {t}")
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    if sa == 0.0 or sb == 0.0:
        raise ValueError("a series has zero variance")
    s1, s2 = float(np.mean(a)) / sa, float(np.mean(b)) / sb
    rho = float(np.corrcoef(a, b)[0, 1])
    var = (2.0 - 2.0 * rho + 0.5 * (s1**2 + s2**2 - 2.0 * s1 * s2 * rho**2)) / t
    se = float(np.sqrt(max(var, 0.0)))
    if se == 0.0:
        z = 0.0 if s1 == s2 else float(np.sign(s1 - s2) * np.inf)
    else:
        z = (s1 - s2) / se
    p = float(2.0 * norm.sf(abs(z)))
    ann = float(np.sqrt(periods_per_year))
    return SharpeDiffResult((s1 - s2) * ann, se * ann, float(z), p)


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
    dd = downside_deviation(r)
    mean = float(np.mean(r))
    if dd == 0.0:
        return float("inf") if mean > 0 else 0.0
    return mean / dd * np.sqrt(periods_per_year)


def max_drawdown(returns: np.ndarray | pd.Series) -> float:
    """Maximum peak-to-trough drawdown of compounded wealth, as a
    POSITIVE fraction (0.25 = -25% drawdown). Wealth starts at 1 and the
    running peak includes that starting point."""
    return max_drawdown_from_returns(_as_vector(returns), "simple", include_start=True)


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
