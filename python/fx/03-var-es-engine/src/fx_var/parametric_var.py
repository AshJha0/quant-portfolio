"""Parametric (variance-covariance) VaR: normal, Student-t, Cornish-Fisher.

The book is linearised into factor exposures ``w`` (finite-difference
deltas from :meth:`fx_var.book.Book.linear_exposures` - so options enter
via their GK delta/vega and forwards via their deposit legs), and the
portfolio P&L variance is ``w' Sigma w`` with ``Sigma`` a sample or EWMA
covariance of daily factor returns.

Distributional overlays on the same sigma:
* **normal**          - RiskMetrics classic; underestimates FX tails.
* **t**               - standardised Student-t (unit variance) captures the
                        fat tails of EM currency returns at equal sigma.
* **Cornish-Fisher**  - moment-corrected quantile using the portfolio's
                        empirical skew/kurtosis; only valid inside the
                        monotonicity domain of the expansion, which this
                        module checks explicitly (Maillard 2012 - outside
                        the domain the "quantile" is not a quantile).

Multi-day horizon: sigma scales by sqrt(h) (i.i.d. assumption; see
docs/VALIDATION.md F4 for when this breaks for carry books).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .book import Book, Market
import warnings as _w

from .common import (require_positions, resolve_alpha, validate_alpha,
                     validate_finite, validate_horizon, validate_returns,
                     warn_peg_factors)
from .expected_shortfall import normal_es, normal_var, t_es, t_var

__all__ = [
    "ParametricVaRResult",
    "sample_cov",
    "ewma_cov",
    "var_covar",
    "parametric_var",
    "cornish_fisher_z",
    "cornish_fisher_domain_ok",
    "cornish_fisher_valid",
    "cornish_fisher_var",
]


@dataclass(frozen=True)
class ParametricVaRResult:
    """Variance-covariance VaR result (positive base-ccy losses)."""

    var: float
    es: float
    alpha: float
    horizon_days: float
    dist: str
    sigma: float  # 1-day portfolio P&L std in base ccy
    exposures: pd.Series
    flagged_peg_factors: tuple[str, ...] = ()


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Unbiased sample covariance of daily factor returns."""
    if len(returns) < 2:
        raise ValueError("need at least 2 observations for a sample covariance")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("returns contain NaN or infinite values (NaN policy: refuse)")
    return returns.cov(ddof=1)


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94,
             seed_window: int = 30) -> pd.DataFrame:
    """RiskMetrics zero-mean EWMA covariance forecast after the last observation.

    ``S_t = lam S_{t-1} + (1-lam) r_{t-1} r_{t-1}'``.  The seed is the
    zero-mean second moment ``mean(r r')`` over the first
    ``min(seed_window, n)`` rows, the same convention as
    :func:`fx_var.historical_var.ewma_volatility`.  Returns the
    one-step-ahead forecast as a DataFrame.
    """
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lambda must be in (0, 1), got {lam}")
    if int(seed_window) != seed_window or seed_window < 1:
        raise ValueError(f"seed_window must be a positive integer, got {seed_window}")
    r = returns.to_numpy(dtype=float)
    if r.ndim != 2 or r.shape[0] < 1:
        raise ValueError("returns must be a non-empty 2-D frame")
    if not np.isfinite(r).all():
        raise ValueError("returns contain NaN or infinite values (NaN policy: refuse)")
    head = r[: min(int(seed_window), r.shape[0])]
    s = head.T @ head / head.shape[0]
    for row in r:
        s = lam * s + (1.0 - lam) * np.outer(row, row)
    return pd.DataFrame(s, index=returns.columns, columns=returns.columns)


def portfolio_sigma(exposures: pd.Series, cov: pd.DataFrame) -> float:
    """1-day portfolio P&L standard deviation ``sqrt(w' Sigma w)``."""
    w = exposures.to_numpy(dtype=float)
    sig = cov.loc[exposures.index, exposures.index].to_numpy(dtype=float)
    validate_finite(exposures=w, cov=sig)
    var = float(w @ sig @ w)
    # Relative tolerance: rounding in w' S w is of order eps * |w|'|S||w|,
    # so the threshold scales with the book and the covariance units.
    scale = float(np.abs(w) @ np.abs(sig) @ np.abs(w))
    if var < -1e-10 * scale:
        raise ValueError(
            f"covariance matrix is not positive semi-definite along the book: "
            f"w' S w = {var:.6g} (relative {var / scale:.3g})"
        )
    return float(np.sqrt(max(var, 0.0)))


def var_covar(
    exposures: pd.Series,
    cov: pd.DataFrame,
    alpha: float | None = None,
    horizon_days: float = 1.0,
    dist: str = "normal",
    df: float = 6.0,
    mean: float = 0.0,
    *,
    confidence: float | None = None,
) -> tuple[float, float]:
    """Closed-form (VaR, ES) for a linear book: pure function for testing.

    Parameters
    ----------
    exposures : pandas.Series
        Factor exposures ``w`` (base ccy per unit factor move).
    cov : pandas.DataFrame
        Daily factor covariance (must cover ``exposures.index``).
    dist : {"normal", "t"}
        Tail model on the portfolio sigma; ``df`` used for ``"t"``.
    mean : float
        Expected 1-day P&L (usually 0 at daily horizon).
    alpha : float
        Tail probability in (0, 0.5).  Default 0.01.
    confidence : float, optional
        Deprecated.  ``alpha = 1 - confidence``.
    """
    alpha = resolve_alpha(alpha, confidence)
    validate_finite(mean=mean)
    validate_horizon(horizon_days)
    sig1 = portfolio_sigma(exposures, cov)
    scale = np.sqrt(horizon_days)
    sig, mu = sig1 * scale, mean * horizon_days
    if dist == "normal":
        return normal_var(sig, alpha, mu), normal_es(sig, alpha, mu)
    if dist == "t":
        return t_var(sig, alpha, df, mu), t_es(sig, alpha, df, mu)
    raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")


def parametric_var(
    book: Book,
    market: Market,
    returns: pd.DataFrame,
    alpha: float | None = None,
    horizon_days: float = 1.0,
    dist: str = "normal",
    df: float = 6.0,
    cov_method: str = "sample",
    ewma_lambda: float = 0.94,
    min_obs: int = 60,
    warn_pegs: bool = True,
    *,
    confidence: float | None = None,
) -> ParametricVaRResult:
    """Variance-covariance VaR/ES of ``book`` from a factor-return history.

    ``cov_method`` selects ``"sample"`` or ``"ewma"`` covariance.  See
    :func:`var_covar` for the distributional overlays.  ``alpha`` is the
    tail probability (default 0.01).  An empty book raises ``ValueError``.
    """
    alpha = resolve_alpha(alpha, confidence)
    validate_horizon(horizon_days)
    require_positions(book)
    factors = book.factors(market)
    if not factors:
        raise ValueError("book has no risk factors (base-currency cash only)")
    rets = validate_returns(returns, factors, min_obs=min_obs)
    flagged = tuple(warn_peg_factors(rets, factors)) if warn_pegs else ()
    if cov_method == "sample":
        cov = sample_cov(rets)
    elif cov_method == "ewma":
        cov = ewma_cov(rets, ewma_lambda)
    else:
        raise ValueError(f"cov_method must be 'sample' or 'ewma', got {cov_method!r}")
    w = book.linear_exposures(market, factors)
    var, es = var_covar(w, cov, alpha, horizon_days, dist, df)
    sig1 = portfolio_sigma(w, cov)
    return ParametricVaRResult(var, es, alpha, horizon_days, dist, sig1, w, flagged)


# --------------------------------------------------------------------------
# Cornish-Fisher
# --------------------------------------------------------------------------
def cornish_fisher_z(z: np.ndarray | float, skew: float, excess_kurtosis: float):
    """Cornish-Fisher adjusted quantile ``z_cf(z; S, K)``.

    ``z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36``.
    """
    validate_finite(z=z, skew=skew, excess_kurtosis=excess_kurtosis)
    z = np.asarray(z, dtype=float)
    s, k = float(skew), float(excess_kurtosis)
    return (
        z
        + (z**2 - 1.0) * s / 6.0
        + (z**3 - 3.0 * z) * k / 24.0
        - (2.0 * z**3 - 5.0 * z) * s**2 / 36.0
    )


def _cf_min_derivative(s: float, k: float, lo: float, hi: float) -> float:
    """Minimum of ``g'(z) = A z^2 + B z + C`` on ``[lo, hi]`` (closed form)."""
    a = k / 8.0 - s**2 / 6.0
    b = s / 3.0
    c = 1.0 - k / 8.0 + 5.0 * s**2 / 36.0

    def g(z: float) -> float:
        return a * z * z + b * z + c

    m = min(g(lo), g(hi))
    if a > 0.0:
        z_star = -b / (2.0 * a)
        if lo <= z_star <= hi:
            m = min(m, g(z_star))
    return m


def cornish_fisher_valid(skew: float, ex_kurt: float, alpha: float) -> bool:
    """True if the Cornish-Fisher map is increasing on ``[-z_a, z_a]``.

    ``z_a = Phi^{-1}(1 - alpha)``.  The map
    ``g(z) = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36`` defines a
    quantile function only where it is monotone (Maillard 2012).  Its
    derivative is the quadratic ``A z^2 + B z + C`` with ``A = K/8 - S^2/6``,
    ``B = S/3``, ``C = 1 - K/8 + 5 S^2/36``, so the minimum on the interval
    is at an endpoint or at the vertex.  The check is exact.

    The interval depends on ``alpha``: a 99 % VaR only uses the map on
    ``|z| <= 2.326``.  Example: ``(S, K) = (0.122, -0.427)`` has its
    minimum derivative at the endpoint ``z = -4`` (about -9.2e-4), so it is
    invalid on ``|z| <= 4`` but valid at ``alpha = 0.01`` and ``0.001``.

    Raises
    ------
    ValueError
        If any input is not finite or ``alpha`` is outside (0, 0.5).
    """
    validate_finite(skew=skew, ex_kurt=ex_kurt, alpha=alpha)
    alpha = validate_alpha(alpha)
    z_a = float(norm.isf(alpha))
    return bool(_cf_min_derivative(float(skew), float(ex_kurt), -z_a, z_a) > 0.0)


def cornish_fisher_domain_ok(
    skew: float, excess_kurtosis: float, z_range: float = 4.0, n_grid: int = 801
) -> bool:
    """Deprecated: monotonicity of the CF map on the fixed ``[-z_range, z_range]``.

    Use :func:`cornish_fisher_valid`, which uses the interval implied by
    ``alpha``.  This predicate keeps its old behaviour for one release:
    the fixed interval, and False (not an error) for non-finite input.
    ``n_grid`` is validated but unused.
    """
    _w.warn(
        "cornish_fisher_domain_ok is deprecated; use "
        "cornish_fisher_valid(skew, ex_kurt, alpha)",
        DeprecationWarning,
        stacklevel=2,
    )
    z_range = float(z_range)
    if n_grid < 3:
        raise ValueError(f"n_grid must be >= 3, got {n_grid}")
    if not (z_range > 0.0) or not np.isfinite(z_range):
        raise ValueError(f"z_range must be finite and > 0, got {z_range}")
    s, k = float(skew), float(excess_kurtosis)
    if not np.isfinite(s) or not np.isfinite(k):
        return False
    return bool(_cf_min_derivative(s, k, -z_range, z_range) > 0.0)


def cornish_fisher_var(
    sigma: float,
    skew: float,
    excess_kurtosis: float,
    alpha: float | None = None,
    mean: float = 0.0,
    horizon_days: float = 1.0,
    check_domain: bool = True,
    *,
    confidence: float | None = None,
) -> float:
    """Cornish-Fisher VaR (positive loss) with an explicit domain check.

    Parameters
    ----------
    sigma, mean : float
        1-day P&L std and mean.
    skew, excess_kurtosis : float
        Sample skewness and excess kurtosis of daily P&L.
    alpha : float
        Tail probability in (0, 0.5).  Default 0.01.
    check_domain : bool
        If True (default), raise ``ValueError`` when
        :func:`cornish_fisher_valid` fails at this ``alpha``.
    confidence : float, optional
        Deprecated.  ``alpha = 1 - confidence``.

    Raises
    ------
    ValueError
        If the domain check fails and ``check_domain`` is True.
    """
    alpha = resolve_alpha(alpha, confidence)
    validate_horizon(horizon_days)
    validate_finite(sigma=sigma, skew=skew, excess_kurtosis=excess_kurtosis,
                    mean=mean)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if check_domain and not cornish_fisher_valid(skew, excess_kurtosis, alpha):
        raise ValueError(
            f"Cornish-Fisher expansion is non-monotone on |z| <= z_alpha for "
            f"skew={skew:.3f}, excess_kurtosis={excess_kurtosis:.3f}, "
            f"alpha={alpha}: outside the validity domain; use historical or "
            "t VaR (set check_domain=False to force)"
        )
    z = norm.ppf(alpha)  # lower tail of the P&L distribution
    zcf = float(cornish_fisher_z(z, skew, excess_kurtosis))
    scale = np.sqrt(horizon_days)
    return float(-(mean * horizon_days) - sigma * scale * zcf)
