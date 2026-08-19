"""Monte Carlo VaR: multivariate normal / Student-t / jump-mixture factors.

Factor returns are simulated from a daily covariance matrix (Cholesky with
jitter escalation for the singular case - two perfectly correlated pegged
currencies are a legitimate FX input) and the book is **fully revalued**
scenario by scenario (options via Garman-Kohlhagen, forwards via their
deposit legs).  Because the revaluation is vectorised, 100k scenarios on a
ten-factor book take well under a second.

Distributions
-------------
* ``"normal"`` - MVN(0, Sigma h).
* ``"t"``      - multivariate Student-t scaled to *match Sigma exactly*
  (``X = Z sqrt((df-2)/df) / sqrt(W/df)``), so the comparison with normal
  MC is at equal covariance: any 99% VaR difference is pure tail shape.
  EM currency returns are the textbook case (df 4-6).
* ``"jump"``   - normal diffusion plus a Bernoulli(p) common jump event
  with per-factor jump sizes ~ N(mu_J, sigma_J): the devaluation /
  peg-break overlay.  The jump *adds* variance on top of Sigma by design -
  it models exactly the risk the covariance matrix cannot see.

Standard error
--------------
``var_standard_error`` uses the asymptotic order-statistic formula
``SE = sqrt(a(1-a)/n) / f(q)`` with a Gaussian-KDE density estimate at the
quantile.  Convergence tests accept MC vs closed form within 3 SE.
"""

from __future__ import annotations

import warnings as _w
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .book import Book, Market
from .common import (NumericalWarning, validate_alpha, validate_finite,
                     validate_horizon)
from .expected_shortfall import empirical_var_es

__all__ = [
    "JumpSpec",
    "MonteCarloVaRResult",
    "robust_cholesky",
    "simulate_factor_returns",
    "var_standard_error",
    "monte_carlo_var",
]


@dataclass(frozen=True)
class JumpSpec:
    """Common-jump overlay for the ``"jump"`` distribution.

    Parameters
    ----------
    prob : float
        Per-scenario (per-day) probability of the jump event.
    mean : mapping factor -> float
        Jump mean per factor (log-return for FX factors); factors not
        listed do not jump.  E.g. ``{"FX:TRY": -0.15}`` - a 15% (log)
        devaluation of the lira against USD.
    std : mapping factor -> float
        Jump dispersion per factor (0 allowed = deterministic jump size).
    """

    prob: float
    mean: Mapping[str, float]
    std: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.prob <= 1.0):
            raise ValueError(f"jump prob must be in [0, 1], got {self.prob}")
        for k, v in self.mean.items():
            validate_finite(**{f"jump mean for {k}": v})
        for k, v in self.std.items():
            validate_finite(**{f"jump std for {k}": v})
            if v < 0:
                raise ValueError(f"jump std for {k} must be >= 0, got {v}")


@dataclass(frozen=True)
class MonteCarloVaRResult:
    """Monte Carlo VaR result (positive base-ccy losses)."""

    var: float
    es: float
    alpha: float
    horizon_days: float
    dist: str
    n_scenarios: int
    se_var: float
    pnl: np.ndarray


def robust_cholesky(cov: np.ndarray, max_tries: int = 8) -> np.ndarray:
    """Cholesky factor of ``cov`` with escalating diagonal jitter.

    A covariance containing pegged currencies is routinely singular or
    numerically indefinite (near-zero-vol factors, perfectly correlated
    pegs to the same anchor).  On failure, jitter ``1e-12 * mean(diag)``
    is added and escalated by x10 up to ``max_tries`` times; a
    :class:`NumericalWarning` reports the jitter actually used.

    Raises
    ------
    ValueError
        If factorisation still fails at maximum jitter.
    """
    a = np.asarray(cov, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("cov must be a square matrix")
    if not np.isfinite(a).all():
        raise ValueError(
            "cov contains NaN or infinite entries (NaN policy: refuse, "
            "never impute); a non-finite covariance would silently yield a "
            "NaN Cholesky factor and a NaN VaR"
        )
    if not np.allclose(a, a.T, atol=1e-12):
        raise ValueError("cov must be symmetric")
    try:
        return np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        pass
    base = float(np.mean(np.diag(a)))
    base = base if base > 0 else 1.0
    jitter = 1e-12 * base
    for _ in range(max_tries):
        try:
            chol = np.linalg.cholesky(a + jitter * np.eye(a.shape[0]))
            _w.warn(
                f"covariance not positive definite; Cholesky computed with "
                f"diagonal jitter {jitter:.2e} (singular/pegged factor block)",
                NumericalWarning,
                stacklevel=2,
            )
            return chol
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise ValueError("covariance matrix is not factorisable even with jitter")


def _as_rng(rng_or_seed) -> np.random.Generator:
    if isinstance(rng_or_seed, np.random.Generator):
        return rng_or_seed
    return np.random.default_rng(rng_or_seed)


def simulate_factor_returns(
    cov: pd.DataFrame,
    n_scenarios: int,
    dist: str = "normal",
    df: float = 6.0,
    jumps: JumpSpec | None = None,
    seed: int | np.random.Generator = 0,
    horizon_days: float = 1.0,
) -> pd.DataFrame:
    """Simulate factor-return scenarios from a daily covariance.

    Parameters
    ----------
    cov : pandas.DataFrame
        Daily factor covariance (columns = factor names).
    n_scenarios : int
        Number of scenarios (rows of the result).
    dist : {"normal", "t", "jump"}
        See module docstring.  ``"jump"`` requires ``jumps``.
    df : float
        Degrees of freedom for ``"t"`` (must be > 2; covariance-matched).
    jumps : JumpSpec, optional
        Common-jump overlay for ``"jump"``.
    seed : int or numpy.random.Generator
        Reproducibility (explicit by convention).
    horizon_days : float
        Scales the covariance by h (i.i.d. aggregation).

    Returns
    -------
    pandas.DataFrame
        ``n_scenarios x n_factors`` scenario matrix.
    """
    validate_horizon(horizon_days)
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be >= 1")
    rng = _as_rng(seed)
    factors = list(cov.columns)
    a = cov.to_numpy(dtype=float) * horizon_days
    chol = robust_cholesky(a)
    z = rng.standard_normal((n_scenarios, len(factors)))
    x = z @ chol.T
    if dist == "normal":
        pass
    elif dist == "t":
        validate_finite(df=df)
        if df <= 2:
            raise ValueError("Student-t df must be > 2 for finite variance")
        w = rng.chisquare(df, size=n_scenarios) / df
        x = x * np.sqrt((df - 2.0) / df) / np.sqrt(w)[:, None]
    elif dist == "jump":
        if jumps is None:
            raise ValueError("dist='jump' requires a JumpSpec")
        hit = rng.random(n_scenarios) < jumps.prob
        for j, f in enumerate(factors):
            if f in jumps.mean:
                size = jumps.mean[f] + jumps.std.get(f, 0.0) * rng.standard_normal(n_scenarios)
                x[:, j] += hit * size
    else:
        raise ValueError(f"dist must be 'normal', 't' or 'jump', got {dist!r}")
    return pd.DataFrame(x, columns=factors)


def var_standard_error(pnl: np.ndarray, alpha: float) -> float:
    """Asymptotic standard error of the empirical VaR estimate.

    ``SE = sqrt(alpha (1-alpha) / n) / f_hat(q)`` with ``f_hat`` a Gaussian
    KDE of the P&L density evaluated at the loss quantile.
    """
    validate_alpha(alpha)
    pnl = np.asarray(pnl, dtype=float).ravel()
    n = pnl.size
    if n < 10:
        raise ValueError("need at least 10 scenarios for a VaR standard error")
    q = np.quantile(pnl, 1.0 - alpha)
    dens = float(gaussian_kde(pnl)(q)[0])
    dens = max(dens, 1e-300)
    return float(np.sqrt(alpha * (1.0 - alpha) / n) / dens)


def monte_carlo_var(
    book: Book,
    market: Market,
    cov: pd.DataFrame,
    alpha: float = 0.99,
    horizon_days: float = 1.0,
    n_scenarios: int = 50_000,
    dist: str = "normal",
    df: float = 6.0,
    jumps: JumpSpec | None = None,
    seed: int | np.random.Generator = 0,
    option_method: str = "full",
) -> MonteCarloVaRResult:
    """Monte Carlo VaR/ES with full revaluation of the book.

    ``cov`` must cover every factor in ``book.factors(market)`` (extra
    columns are ignored).  See :func:`simulate_factor_returns` for the
    distributional choices and :func:`var_standard_error` for the SE.
    """
    validate_alpha(alpha)
    validate_horizon(horizon_days)
    factors = book.factors(market)
    if not factors:
        return MonteCarloVaRResult(0.0, 0.0, alpha, horizon_days, dist, n_scenarios,
                                   0.0, np.zeros(n_scenarios))
    missing = [f for f in factors if f not in cov.columns]
    if missing:
        raise ValueError(f"cov is missing required factor columns: {missing}")
    sub = cov.loc[factors, factors]
    scen = simulate_factor_returns(sub, n_scenarios, dist, df, jumps, seed, horizon_days)
    pnl = np.asarray(book.pnl(market, scen, option_method=option_method), dtype=float)
    var, es = empirical_var_es(pnl, alpha)
    se = var_standard_error(pnl, alpha)
    return MonteCarloVaRResult(var, es, alpha, horizon_days, dist, n_scenarios, se, pnl)
