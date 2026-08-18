"""Expected Shortfall (ES / CVaR) for all three VaR families.

ES_alpha = -(1/alpha) * integral_0^alpha Q_u(pnl) du — the average loss in
the worst ``alpha`` tail.  ES >= VaR by construction, ES is coherent
(subadditive), and FRTB replaced 99 % VaR with 97.5 % ES as the market-risk
capital measure (docs/METHODOLOGY.md).

Conventions: ``alpha`` = tail probability, ES positive for losses.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import norm, t as student_t

__all__ = [
    "expected_shortfall",
    "normal_es",
    "student_t_es",
    "parametric_es",
    "es_standard_error_bootstrap",
]


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5) (tail probability), got {alpha}")


def expected_shortfall(pnl: np.ndarray, alpha: float = 0.01) -> float:
    """Empirical Expected Shortfall (exact tail integral of the step CDF).

    With sorted P&L ``x_(1) <= ... <= x_(n)`` and ``k = floor(alpha * n)``:

    ``ES = -(1/(alpha*n)) * [ sum_{i<=k} x_(i) + (alpha*n - k) * x_(k+1) ]``

    i.e. the exact integral of the empirical quantile function over
    ``(0, alpha]``, with a fractional weight on the boundary order statistic.
    This estimator is consistent, satisfies ES >= VaR (with the same
    order-statistic quantile) and is exact on known arrays (unit tested).
    """
    _validate_alpha(alpha)
    arr = np.sort(np.asarray(pnl, dtype=float).ravel())
    n = arr.size
    if n < 10:
        raise ValueError(f"need at least 10 P&L observations for empirical ES, got {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("pnl contains NaN or infinite values")
    an = alpha * n
    k = int(np.floor(an))
    tail_sum = float(arr[:k].sum())
    frac = an - k
    if frac > 0.0:
        tail_sum += frac * float(arr[k])
    return float(-tail_sum / an)


def normal_es(sigma: float, alpha: float = 0.01, mean: float = 0.0) -> float:
    """Closed-form ES for normal P&L: ``ES = sigma * phi(z_alpha)/alpha - mean``.

    The identity ``E[-X | X <= Q_alpha] = sigma * phi(z)/alpha - mu`` with
    ``z = Phi^{-1}(alpha)`` is unit-tested against numerical integration
    to 1e-10.
    """
    _validate_alpha(alpha)
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    z = norm.ppf(alpha)
    return float(sigma * norm.pdf(z) / alpha - mean)


def student_t_es(
    sigma: float, alpha: float = 0.01, df: float = 6.0, mean: float = 0.0
) -> float:
    """Closed-form ES for variance-matched Student-t P&L.

    For standardised t with df ``nu`` (unit variance after scaling by
    ``sqrt((nu-2)/nu)``):

    ``ES_std = f_nu(q) * (nu + q^2) / ((nu - 1) * alpha) * sqrt((nu-2)/nu)``

    where ``q = t_nu^{-1}(alpha)`` and ``f_nu`` is the t density.
    """
    _validate_alpha(alpha)
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if df <= 2:
        raise ValueError(f"Student-t df must be > 2 for finite variance, got {df}")
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

    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    sigma = portfolio_sigma(exposures, cov) * np.sqrt(horizon_days)
    mu = mean * horizon_days
    if dist == "normal":
        return normal_es(sigma, alpha, mu)
    if dist == "t":
        return student_t_es(sigma, alpha, df, mu)
    raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")


def es_standard_error_bootstrap(
    pnl: np.ndarray,
    alpha: float = 0.01,
    n_boot: int = 500,
    seed: int | np.random.Generator | None = 0,
) -> float:
    """Bootstrap standard error of the empirical ES estimate.

    ES averages the ``alpha`` tail, so its estimation error exceeds the VaR
    quantile's at the same alpha — an important caveat when quoting ES97.5 on
    a 250-day window (~6 tail points).  Documented in docs/VALIDATION.md.
    """
    _validate_alpha(alpha)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    arr = np.asarray(pnl, dtype=float).ravel()
    if arr.size < 10:
        raise ValueError(f"need at least 10 observations to bootstrap, got {arr.size}")
    ests = np.empty(n_boot)
    for b in range(n_boot):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        ests[b] = expected_shortfall(sample, alpha)
    return float(np.std(ests, ddof=1))
