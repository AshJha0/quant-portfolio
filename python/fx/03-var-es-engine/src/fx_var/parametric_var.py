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
from .common import (validate_alpha, validate_finite, validate_horizon,
                     validate_returns, warn_peg_factors)
from .expected_shortfall import normal_es, normal_var, t_es, t_var

__all__ = [
    "ParametricVaRResult",
    "sample_cov",
    "ewma_cov",
    "var_covar",
    "parametric_var",
    "cornish_fisher_z",
    "cornish_fisher_domain_ok",
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
    return returns.cov(ddof=1)


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics EWMA covariance forecast after the last observation.

    ``S_t = lam S_{t-1} + (1-lam) r_{t-1} r_{t-1}'``, seeded with the sample
    covariance.  Returns the one-step-ahead forecast as a DataFrame.
    """
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lambda must be in (0, 1), got {lam}")
    r = returns.to_numpy(dtype=float)
    s = np.cov(r, rowvar=False, ddof=1)
    s = np.atleast_2d(s)
    for row in r:
        s = lam * s + (1.0 - lam) * np.outer(row, row)
    return pd.DataFrame(s, index=returns.columns, columns=returns.columns)


def portfolio_sigma(exposures: pd.Series, cov: pd.DataFrame) -> float:
    """1-day portfolio P&L standard deviation ``sqrt(w' Sigma w)``."""
    w = exposures.to_numpy(dtype=float)
    sig = cov.loc[exposures.index, exposures.index].to_numpy(dtype=float)
    validate_finite(exposures=w, cov=sig)
    var = float(w @ sig @ w)
    if var < -1e-12:
        raise ValueError("covariance matrix is not positive semi-definite")
    return float(np.sqrt(max(var, 0.0)))


def var_covar(
    exposures: pd.Series,
    cov: pd.DataFrame,
    alpha: float = 0.99,
    horizon_days: float = 1.0,
    dist: str = "normal",
    df: float = 6.0,
    mean: float = 0.0,
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
    """
    validate_alpha(alpha)
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
    alpha: float = 0.99,
    horizon_days: float = 1.0,
    dist: str = "normal",
    df: float = 6.0,
    cov_method: str = "sample",
    ewma_lambda: float = 0.94,
    min_obs: int = 60,
    warn_pegs: bool = True,
) -> ParametricVaRResult:
    """Variance-covariance VaR/ES of ``book`` from a factor-return history.

    ``cov_method`` selects ``"sample"`` or ``"ewma"`` covariance.  See
    :func:`var_covar` for the distributional overlays.
    """
    validate_alpha(alpha)
    validate_horizon(horizon_days)
    factors = book.factors(market)
    if not factors:
        return ParametricVaRResult(0.0, 0.0, alpha, horizon_days, dist, 0.0,
                                   pd.Series(dtype=float))
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
    z = np.asarray(z, dtype=float)
    s, k = float(skew), float(excess_kurtosis)
    return (
        z
        + (z**2 - 1.0) * s / 6.0
        + (z**3 - 3.0 * z) * k / 24.0
        - (2.0 * z**3 - 5.0 * z) * s**2 / 36.0
    )


def cornish_fisher_domain_ok(
    skew: float, excess_kurtosis: float, z_range: float = 4.0, n_grid: int = 801
) -> bool:
    """True if the CF expansion is monotone increasing on ``[-z_range, z_range]``.

    Monotonicity of ``z -> z_cf`` is the validity condition for the
    expansion to define a quantile function (Maillard 2012); it is checked
    numerically on a dense grid.
    """
    z = np.linspace(-z_range, z_range, n_grid)
    zcf = cornish_fisher_z(z, skew, excess_kurtosis)
    return bool(np.all(np.diff(zcf) > 0))


def cornish_fisher_var(
    sigma: float,
    skew: float,
    excess_kurtosis: float,
    alpha: float = 0.99,
    mean: float = 0.0,
    horizon_days: float = 1.0,
    check_domain: bool = True,
) -> float:
    """Cornish-Fisher VaR (positive loss) with an explicit domain check.

    Parameters
    ----------
    sigma, mean : float
        1-day P&L std and mean.
    skew, excess_kurtosis : float
        Sample skewness and *excess* kurtosis of daily P&L.
    check_domain : bool
        If True (default), raise ``ValueError`` when (S, K) are outside the
        monotonicity domain - a silently non-monotone CF "quantile" can
        report 99% VaR below 95% VaR.

    Raises
    ------
    ValueError
        If the domain check fails and ``check_domain`` is True.
    """
    validate_alpha(alpha)
    validate_horizon(horizon_days)
    validate_finite(sigma=sigma, skew=skew, excess_kurtosis=excess_kurtosis,
                    mean=mean)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if check_domain and not cornish_fisher_domain_ok(skew, excess_kurtosis):
        raise ValueError(
            f"Cornish-Fisher expansion is non-monotone for skew={skew:.3f}, "
            f"excess_kurtosis={excess_kurtosis:.3f}: outside validity domain; "
            "fall back to historical or t VaR (set check_domain=False to force)"
        )
    # loss quantile: use the lower tail of the P&L distribution
    z = norm.ppf(1.0 - alpha)
    zcf = float(cornish_fisher_z(z, skew, excess_kurtosis))
    scale = np.sqrt(horizon_days)
    return float(-(mean * horizon_days) - sigma * scale * zcf)
