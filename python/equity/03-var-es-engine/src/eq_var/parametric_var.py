"""Parametric (variance-covariance) VaR.

Portfolio sigma from dollar exposures ``w`` and factor-return covariance
``Sigma``: ``sigma_p = sqrt(w' Sigma w)``.  Quantiles from the normal,
Student-t (variance-matched) or Cornish-Fisher expansion.

Conventions: ``alpha`` = tail probability; VaR positive for losses; daily
covariance in factor-return units matching ``Portfolio.delta_exposures``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import norm, t as student_t

__all__ = [
    "sample_covariance",
    "ewma_covariance",
    "portfolio_sigma",
    "parametric_var",
    "cornish_fisher_z",
    "cornish_fisher_domain_ok",
    "cornish_fisher_var",
]


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5) (tail probability), got {alpha}")


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance of a (T, n) factor-return matrix."""
    arr = np.atleast_2d(np.asarray(returns, dtype=float))
    if arr.shape[0] < 2:
        raise ValueError(f"need at least 2 observations for a covariance, got {arr.shape[0]}")
    return np.cov(arr, rowvar=False, ddof=1).reshape(arr.shape[1], arr.shape[1])


def ewma_covariance(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA covariance forecast for the day after the sample.

    ``Sigma_{t+1} = lam * Sigma_t + (1 - lam) * r_t r_t'``, seeded with the
    sample covariance.  Zero-mean returns assumed (standard at daily horizon).
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"decay lam must be in (0, 1), got {lam}")
    arr = np.atleast_2d(np.asarray(returns, dtype=float))
    if arr.shape[0] < 2:
        raise ValueError(f"need at least 2 observations for a covariance, got {arr.shape[0]}")
    cov = sample_covariance(arr)
    for r in arr:
        cov = lam * cov + (1.0 - lam) * np.outer(r, r)
    return cov


def portfolio_sigma(exposures: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio P&L standard deviation ``sqrt(w' Sigma w)`` (currency units)."""
    w = np.asarray(exposures, dtype=float).ravel()
    sig = np.atleast_2d(np.asarray(cov, dtype=float))
    if sig.shape != (w.size, w.size):
        raise ValueError(f"covariance shape {sig.shape} does not match {w.size} exposures")
    var = float(w @ sig @ w)
    if var < -1e-10 * max(1.0, float(np.abs(w).max()) ** 2):
        raise ValueError("covariance matrix is not positive semi-definite (w'Sw < 0)")
    return float(np.sqrt(max(var, 0.0)))


def parametric_var(
    exposures: np.ndarray,
    cov: np.ndarray,
    alpha: float = 0.01,
    dist: Literal["normal", "t"] = "normal",
    df: float = 6.0,
    mean: float = 0.0,
    horizon_days: int = 1,
) -> float:
    """Variance-covariance VaR.

    Parameters
    ----------
    exposures : array (n,)
        Dollar exposures per factor (``Portfolio.delta_exposures``).
    cov : array (n, n)
        Daily factor-return covariance.
    alpha : float
        Tail probability.
    dist : {"normal", "t"}
        Tail model.  ``"t"`` uses the Student-t quantile rescaled to unit
        variance (``* sqrt((df-2)/df)``), so sigma is matched and only the
        tail shape changes.
    df : float
        Degrees of freedom for ``dist="t"`` (must be > 2 for finite variance).
    mean : float
        Expected daily P&L (usually 0 at daily horizon).
    horizon_days : int
        Square-root-of-time scaling of sigma (and linear scaling of mean).

    Returns
    -------
    float — VaR as a positive loss.
    """
    _validate_alpha(alpha)
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    sigma = portfolio_sigma(exposures, cov) * np.sqrt(horizon_days)
    mu = mean * horizon_days
    if dist == "normal":
        z = norm.ppf(alpha)
    elif dist == "t":
        if df <= 2:
            raise ValueError(f"Student-t df must be > 2 for finite variance, got {df}")
        z = student_t.ppf(alpha, df) * np.sqrt((df - 2.0) / df)
    else:
        raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")
    return float(-(mu + z * sigma))


# --------------------------------------------------------------------------- #
# Cornish-Fisher
# --------------------------------------------------------------------------- #
def cornish_fisher_z(z: np.ndarray | float, skew: float, excess_kurt: float) -> np.ndarray | float:
    """Cornish-Fisher adjusted quantile.

    ``z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36`` with skewness
    ``S`` and *excess* kurtosis ``K``.  Reduces to ``z`` when S = K = 0.
    """
    z = np.asarray(z, dtype=float)
    out = (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * excess_kurt / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )
    return float(out) if out.ndim == 0 else out


def cornish_fisher_domain_ok(
    skew: float, excess_kurt: float, z_range: float = 3.5, n_grid: int = 2001
) -> bool:
    """Check that the CF quantile map is monotone on ``[-z_range, z_range]``.

    The fourth-order Cornish-Fisher expansion is only a valid quantile
    function where ``dz_cf/dz > 0``; for large skew/kurtosis the cubic
    polynomial becomes non-monotone and the implied 'density' goes negative,
    producing nonsense VaR (e.g. the 99 % 'quantile' above the 95 % one).
    We check the analytic derivative

    ``dz_cf/dz = 1 + zS/3 + (3z^2-3)K/24 - (6z^2-5)S^2/36``

    on a dense grid covering the tail probabilities used in practice
    (|z| <= 3.5 covers alpha >= 0.02 %).

    Returns
    -------
    bool — True when the expansion is monotone (safe to use).
    """
    z = np.linspace(-z_range, z_range, n_grid)
    deriv = (
        1.0
        + z * skew / 3.0
        + (3.0 * z**2 - 3.0) * excess_kurt / 24.0
        - (6.0 * z**2 - 5.0) * skew**2 / 36.0
    )
    return bool(np.all(deriv > 0.0))


def cornish_fisher_var(
    sigma: float,
    alpha: float = 0.01,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
    mean: float = 0.0,
    check_domain: bool = True,
) -> float:
    """Cornish-Fisher VaR: moment-corrected parametric quantile.

    ``VaR = -(mean + z_cf(alpha) * sigma)`` with the CF-adjusted quantile.
    With ``check_domain=True`` (default) a ``ValueError`` is raised when
    (skew, excess_kurt) lie outside the monotonicity region — outside it the
    expansion is not a quantile function and the number is not a VaR.
    """
    _validate_alpha(alpha)
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if check_domain and not cornish_fisher_domain_ok(skew, excess_kurt):
        raise ValueError(
            f"Cornish-Fisher expansion is non-monotone for skew={skew}, "
            f"excess_kurt={excess_kurt}; outside its validity region the "
            "'quantile' is not a quantile. Use historical or MC VaR instead."
        )
    z = cornish_fisher_z(norm.ppf(alpha), skew, excess_kurt)
    return float(-(mean + z * sigma))
