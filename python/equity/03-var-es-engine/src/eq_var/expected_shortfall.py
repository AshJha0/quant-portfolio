"""Expected Shortfall (ES) for all three VaR families.

``ES_alpha = (1/alpha) * integral_0^alpha VaR_u du``: the average loss in the
worst ``alpha`` tail. ES >= VaR, ES is coherent, and FRTB uses 97.5 % ES
(docs/METHODOLOGY.md).

Conventions: ``alpha`` = tail probability, ES positive for losses. The
empirical estimator is the S2 estimator in ``eq_var._tail``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from . import _tail
from ._validation import finite_array, finite_scalar, integer, validate_alpha

__all__ = [
    "es_standard_error_bootstrap",
    "expected_shortfall",
    "normal_es",
    "parametric_es",
    "student_t_es",
]

MIN_OBS_ES = 10


def _sample(pnl: object, alpha: float) -> np.ndarray:
    arr = finite_array("pnl", pnl, ndim=1, min_size=0)
    if arr.size < MIN_OBS_ES:
        raise ValueError(f"need at least {MIN_OBS_ES} P&L observations for empirical ES, got {arr.size}")
    return arr


def expected_shortfall(pnl: np.ndarray, alpha: float = 0.01) -> float:
    """Empirical ES (spec S2).

    Losses sorted descending ``L(1) >= ... >= L(n)``, ``k = n * alpha``,
    ``m = ceil(k)``: ``ES = [sum_{i<m} L(i) + (k - (m-1)) L(m)] / k``. This is
    the exact integral of the empirical quantile function over the worst
    ``alpha`` of mass, so ``ES >= VaR = L(m)`` always.
    """
    alpha = validate_alpha(alpha)
    return _tail.var_es(_sample(pnl, alpha), alpha)[1]


def _sigma_mean(sigma: object, mean: object) -> tuple[float, float]:
    return finite_scalar("sigma", sigma, min_value=0.0), finite_scalar("mean", mean)


def _df(df: object) -> float:
    df = finite_scalar("df", df)
    if not df > 2.0:
        raise ValueError(f"Student-t df must be > 2 for finite variance, got {df}")
    return df


def normal_es(sigma: float, alpha: float = 0.01, mean: float = 0.0) -> float:
    """Closed-form ES for normal P&L: ``ES = sigma * phi(z_alpha)/alpha - mean``."""
    alpha = validate_alpha(alpha)
    sigma, mean = _sigma_mean(sigma, mean)
    z = norm.ppf(alpha)
    return float(sigma * norm.pdf(z) / alpha - mean)


def student_t_es(sigma: float, alpha: float = 0.01, df: float = 6.0, mean: float = 0.0) -> float:
    """Closed-form ES for variance-matched Student-t P&L.

    ``ES_std = f_nu(q) (nu + q^2) / ((nu - 1) alpha) * sqrt((nu-2)/nu)`` with
    ``q = t_nu^{-1}(alpha)``.
    """
    alpha = validate_alpha(alpha)
    sigma, mean = _sigma_mean(sigma, mean)
    df = _df(df)
    q = student_t.ppf(alpha, df)
    es_std = student_t.pdf(q, df) * (df + q**2) / ((df - 1.0) * alpha)
    return float(sigma * es_std * np.sqrt((df - 2.0) / df) - mean)


def parametric_es(
    exposures: np.ndarray,
    cov: np.ndarray,
    alpha: float = 0.01,
    dist: Literal["normal", "t"] = "normal",
    df: float = 6.0,
    mean: float = 0.0,
    horizon_days: int = 1,
) -> float:
    """Variance-covariance ES from dollar exposures and factor covariance."""
    from .parametric_var import portfolio_sigma  # local import avoids cycle

    alpha = validate_alpha(alpha)
    horizon_days = integer("horizon_days", horizon_days, min_value=1)
    mean = finite_scalar("mean", mean)
    if dist not in ("normal", "t"):
        raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")
    sigma = portfolio_sigma(exposures, cov) * np.sqrt(horizon_days)
    mu = mean * horizon_days
    if dist == "normal":
        return normal_es(sigma, alpha, mu)
    return student_t_es(sigma, alpha, df, mu)


def es_standard_error_bootstrap(
    pnl: np.ndarray,
    alpha: float = 0.01,
    n_boot: int = 500,
    seed: int | np.random.Generator | None = 0,
) -> float:
    """Bootstrap standard error of the empirical ES (sample std, ddof=1).

    ES averages the ``alpha`` tail, so its estimation error exceeds the VaR
    quantile's at the same alpha.
    """
    alpha = validate_alpha(alpha)
    n_boot = integer("n_boot", n_boot, min_value=2)
    arr = _sample(pnl, alpha)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    ests = np.empty(n_boot)
    for b in range(n_boot):
        ests[b] = _tail.var_es(arr[rng.integers(0, arr.size, size=arr.size)], alpha)[1]
    return float(np.std(ests, ddof=1))
