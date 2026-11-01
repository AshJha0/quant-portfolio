"""Parametric (variance-covariance) VaR.

Portfolio sigma from dollar exposures ``w`` and factor-return covariance
``Sigma``: ``sigma_p = sqrt(w' Sigma w)``.  Quantiles from the normal,
Student-t (variance-matched) or Cornish-Fisher expansion.

Conventions: ``alpha`` = tail probability; VaR positive for losses; daily
covariance in factor-return units matching ``Portfolio.delta_exposures``.
"""

from __future__ import annotations

from typing import Literal

import warnings

import numpy as np
from scipy.stats import norm, t as student_t

from ._validation import finite_array, finite_scalar, integer, validate_alpha

__all__ = [
    "sample_covariance",
    "ewma_covariance",
    "portfolio_sigma",
    "parametric_var",
    "cornish_fisher_z",
    "cornish_fisher_valid",
    "cornish_fisher_domain_ok",
    "cornish_fisher_var",
]


def _validate_alpha(alpha: float) -> float:
    return validate_alpha(alpha)


def _validate_finite_array(name: str, arr: np.ndarray) -> None:
    """Reject NaN/inf anywhere in a market-data array.

    Every comparison with NaN is False, so sign/shape guards alone let a NaN
    covariance or exposure through and the caller receives a NaN "VaR"
    instead of an error.
    """
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def _validate_finite_scalar(name: str, value: float) -> None:
    finite_scalar(name, value)


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance of a (T, n) factor-return matrix."""
    arr = np.atleast_2d(np.asarray(returns, dtype=float))
    if arr.shape[0] < 2:
        raise ValueError(f"need at least 2 observations for a covariance, got {arr.shape[0]}")
    _validate_finite_array("returns", arr)
    return np.cov(arr, rowvar=False, ddof=1).reshape(arr.shape[1], arr.shape[1])


def ewma_covariance(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA covariance forecast for the day after the sample.

    ``Sigma_{t+1} = lam * Sigma_t + (1 - lam) * r_t r_t'``, seeded with the
    sample covariance.  Zero-mean returns assumed (standard at daily horizon).
    """
    lam = finite_scalar("lam", lam)
    if not 0.0 < lam < 1.0:
        raise ValueError(f"decay lam must be in (0, 1), got {lam}")
    arr = np.atleast_2d(np.asarray(returns, dtype=float))
    if arr.shape[0] < 2:
        raise ValueError(f"need at least 2 observations for a covariance, got {arr.shape[0]}")
    _validate_finite_array("returns", arr)
    cov = sample_covariance(arr)
    for r in arr:
        cov = lam * cov + (1.0 - lam) * np.outer(r, r)
    return cov


def portfolio_sigma(exposures: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio P&L standard deviation ``sqrt(w' Sigma w)`` (currency units)."""
    w = finite_array("exposures", exposures, ndim=1, min_size=1)
    sig = finite_array("covariance", cov, ndim=2, min_size=1)
    if sig.shape != (w.size, w.size):
        raise ValueError(f"covariance shape {sig.shape} does not match {w.size} exposures")
    _validate_finite_array("exposures", w)
    _validate_finite_array("covariance", sig)
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
    float, VaR as a positive loss.
    """
    alpha = _validate_alpha(alpha)
    horizon_days = integer("horizon_days", horizon_days, min_value=1)
    _validate_finite_scalar("mean", mean)
    _validate_finite_scalar("df", df)
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
    z = finite_array("z", z, min_size=1)
    skew = finite_scalar("skew", skew)
    excess_kurt = finite_scalar("excess_kurt", excess_kurt)
    out = (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * excess_kurt / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )
    return float(out) if out.ndim == 0 else out


def _cf_min_derivative(skew: float, excess_kurt: float, z_lo: float, z_hi: float) -> float:
    """Exact minimum of ``g'(z)`` on the closed interval ``[z_lo, z_hi]``.

    ``g'(z) = A z^2 + B z + C`` with ``A = K/8 - S^2/6``, ``B = S/3``,
    ``C = 1 - K/8 + 5 S^2/36``. The minimum of a quadratic on a closed
    interval is at an endpoint or, when ``A > 0``, at the vertex ``-B/(2A)``
    if the vertex is inside.
    """
    a = excess_kurt / 8.0 - skew**2 / 6.0
    b = skew / 3.0
    c = 1.0 - excess_kurt / 8.0 + 5.0 * skew**2 / 36.0

    def g(z: float) -> float:
        return a * z * z + b * z + c

    m = min(g(z_lo), g(z_hi))
    if a > 0.0:
        z_star = -b / (2.0 * a)
        if z_lo <= z_star <= z_hi:
            m = min(m, g(z_star))
    return m


def cornish_fisher_valid(skew: float, excess_kurt: float, alpha: float) -> bool:
    """True iff the CF map is increasing on ``[-z_a, z_a]``, ``z_a = Phi^{-1}(1 - alpha)`` (spec S6).

    The check is exact (closed-form minimum of the quadratic derivative), and
    it depends on alpha: a dip in ``g'`` beyond ``z_a`` does not affect the
    quantile at ``alpha``. Example: ``(S, K) = (0.122, -0.427)`` has its
    minimum derivative at ``z = -4`` (an endpoint of the old fixed interval),
    so it is valid at ``alpha = 0.01``. Non-finite inputs raise ``ValueError``.
    """
    alpha = validate_alpha(alpha)
    skew = finite_scalar("skew", skew)
    excess_kurt = finite_scalar("excess_kurt", excess_kurt)
    z_a = float(norm.ppf(1.0 - alpha))
    return bool(_cf_min_derivative(skew, excess_kurt, -z_a, z_a) > 0.0)


def cornish_fisher_domain_ok(
    skew: float, excess_kurt: float, z_range: float = 3.5, n_grid: int = 2001
) -> bool:
    """Deprecated: use ``cornish_fisher_valid(skew, excess_kurt, alpha)``.

    Kept for one release with its old meaning: monotonicity on the fixed
    interval ``[-z_range, z_range]`` regardless of alpha. ``n_grid`` is unused.
    """
    warnings.warn(
        "cornish_fisher_domain_ok is deprecated; use cornish_fisher_valid(skew, excess_kurt, alpha), "
        "which checks the interval implied by alpha",
        DeprecationWarning,
        stacklevel=2,
    )
    z_range = finite_scalar("z_range", z_range, min_value=0.0, strict=True)
    integer("n_grid", n_grid, min_value=2)
    skew = finite_scalar("skew", skew)
    excess_kurt = finite_scalar("excess_kurt", excess_kurt)
    return bool(_cf_min_derivative(skew, excess_kurt, -z_range, z_range) > 0.0)


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
    (skew, excess_kurt) lie outside the monotonicity region, outside it the
    expansion is not a quantile function and the number is not a VaR.
    """
    alpha = validate_alpha(alpha)
    _validate_finite_scalar("sigma", sigma)
    _validate_finite_scalar("mean", mean)
    _validate_finite_scalar("skew", skew)
    _validate_finite_scalar("excess_kurt", excess_kurt)
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if check_domain and not cornish_fisher_valid(skew, excess_kurt, alpha):
        raise ValueError(
            f"Cornish-Fisher expansion is non-monotone for skew={skew}, "
            f"excess_kurt={excess_kurt}; outside its validity region the "
            "'quantile' is not a quantile. Use historical or MC VaR instead."
        )
    z = cornish_fisher_z(norm.ppf(alpha), skew, excess_kurt)
    return float(-(mean + z * sigma))
