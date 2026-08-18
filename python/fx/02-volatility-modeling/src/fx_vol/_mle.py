"""Shared maximum-likelihood machinery for the GARCH-family modules.

Everything here is implemented from scratch (numpy/scipy only): innovation
log-densities, arch-style exponential backcast initialization, numerical
Hessian standard errors, and the common fitted-model result container.

Scaling convention
------------------
All fitters internally rescale returns to unit sample variance
(``y = r / s`` with ``s = std(r)``) so the optimizer works on O(1)
parameters regardless of whether the caller passes decimal or percent
returns, then map estimates back:

* ``omega``, exogenous ``gamma_x`` and ``sigma2`` scale by ``s^2``;
* ``alpha``, ``beta``, leverage and tail parameters are scale-invariant;
* ``loglik`` maps back via ``L_r = L_y - n * log(s)`` (change of variables).

This is what makes the fitters robust on pegged-currency series with
near-zero volatility (HKD-style) without any special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import lgamma, log, pi, sqrt
from typing import Callable

import numpy as np

__all__ = [
    "FitResult",
    "validate_returns",
    "backcast",
    "gaussian_loglik",
    "student_t_loglik",
    "student_t_abs_moment",
    "numerical_hessian",
    "hessian_std_errors",
    "MIN_OBS",
    "MIN_NU",
]

MIN_OBS = 100          # minimum observations for any MLE fit
MIN_NU = 2.05          # Student-t dof lower bound (variance must exist)
LOG2PI = log(2.0 * pi)


def validate_returns(returns, min_obs: int = MIN_OBS) -> np.ndarray:
    """Validate a return series for MLE fitting.

    Raises ``ValueError`` on NaN/inf (explicit NaN policy: reject), on series
    shorter than ``min_obs`` and on constant series (zero variance -- the
    likelihood is degenerate; pegged pairs with tiny-but-nonzero vol are fine
    thanks to internal rescaling).
    """
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"returns must be 1-dimensional, got ndim={arr.ndim}")
    if not np.isfinite(arr).all():
        raise ValueError(
            "returns contain NaN or infinite values; fx_vol rejects rather than "
            "imputes -- clean the series explicitly first"
        )
    if arr.size < min_obs:
        raise ValueError(
            f"need at least {min_obs} returns for a stable GARCH-family fit, got {arr.size}"
        )
    if float(np.std(arr)) == 0.0:
        raise ValueError(
            "returns are constant (zero variance): conditional-variance MLE is "
            "degenerate. For hard-pegged currencies use a tiny positive vol floor "
            "or model the peg explicitly (see docs/VALIDATION.md)."
        )
    return arr


def backcast(y: np.ndarray, decay: float = 0.94, max_obs: int = 75) -> float:
    """arch-style exponentially weighted backcast of the initial variance.

    ``b = sum_i w_i * y_i^2`` over the first ``max_obs`` observations with
    ``w_i propto decay^i``. Used to seed ``sigma2_0`` (and ``eps2_0``) in all
    recursions, matching the `arch` package convention so the cross-check
    tests compare like for like.
    """
    tau = min(max_obs, y.size)
    w = decay ** np.arange(tau)
    w = w / w.sum()
    return float(w @ (y[:tau] ** 2))


def gaussian_loglik(y: np.ndarray, sigma2: np.ndarray) -> float:
    """Gaussian log-likelihood sum for zero-mean returns with variance path sigma2."""
    return float(-0.5 * np.sum(LOG2PI + np.log(sigma2) + y * y / sigma2))


def student_t_loglik(y: np.ndarray, sigma2: np.ndarray, nu: float) -> float:
    """Standardized Student-t log-likelihood (unit-variance t, ``nu > 2``).

    Density of ``z = y / sigma``:
    ``f(z) = Gamma((nu+1)/2) / (Gamma(nu/2) sqrt(pi (nu-2))) (1 + z^2/(nu-2))^-((nu+1)/2)``
    """
    if nu <= 2.0:
        raise ValueError(f"Student-t dof must exceed 2 for finite variance, got {nu}")
    const = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * log(pi * (nu - 2.0))
    z2 = y * y / sigma2
    return float(
        np.sum(const - 0.5 * np.log(sigma2) - (nu + 1.0) / 2.0 * np.log1p(z2 / (nu - 2.0)))
    )


def student_t_abs_moment(nu: float) -> float:
    """``E|z|`` for the standardized (unit-variance) Student-t, used by EGARCH.

    ``E|z| = 2 sqrt(nu-2) Gamma((nu+1)/2) / ((nu-1) Gamma(nu/2) sqrt(pi))``;
    tends to ``sqrt(2/pi)`` as nu -> inf (the Gaussian value).
    """
    if nu <= 2.0:
        raise ValueError(f"nu must exceed 2, got {nu}")
    return (
        2.0
        * sqrt(nu - 2.0)
        * np.exp(lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0))
        / ((nu - 1.0) * sqrt(pi))
    )


def numerical_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, rel_step: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of scalar function ``f`` at ``x``."""
    x = np.asarray(x, dtype=float)
    k = x.size
    h = np.maximum(np.abs(x) * rel_step, 1e-6)
    H = np.empty((k, k))
    f0 = f(x)
    for i in range(k):
        for j in range(i, k):
            if i == j:
                xp = x.copy(); xp[i] += h[i]
                xm = x.copy(); xm[i] -= h[i]
                H[i, i] = (f(xp) - 2.0 * f0 + f(xm)) / h[i] ** 2
            else:
                xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
                xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
                xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
                xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
                H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h[i] * h[j])
    return H


def hessian_std_errors(nll_natural: Callable[[np.ndarray], float], theta: np.ndarray) -> np.ndarray:
    """Standard errors from the inverse Hessian of the negative log-likelihood.

    Evaluated in the *natural* (untransformed) parameter space at the optimum.
    Returns NaNs where the Hessian is not positive definite (typical at
    boundaries, e.g. IGARCH-like fits on pegged series -- documented rather
    than hidden).
    """
    try:
        H = numerical_hessian(nll_natural, theta)
        cov = np.linalg.pinv(H)
        diag = np.diag(cov).copy()
        diag[diag < 0] = np.nan
        return np.sqrt(diag)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return np.full(theta.size, np.nan)


@dataclass
class FitResult:
    """Container for a fitted conditional-variance model.

    Attributes
    ----------
    model : str
        'garch', 'garch-x', 'egarch' or 'gjr'.
    dist : str
        'gaussian' or 't'.
    params : dict[str, float]
        Natural-scale estimates (omega in return-variance units).
    std_errors : dict[str, float]
        Hessian standard errors (NaN where the Hessian is singular).
    loglik : float
        Maximized log-likelihood in the *caller's* return units.
    sigma2 : numpy.ndarray
        In-sample conditional variance path, same length/units as returns.
    returns : numpy.ndarray
        The (raw) return series the model was fitted to.
    x : numpy.ndarray or None
        Exogenous variance regressors (GARCH-X), shape (n, k).
    converged : bool
        Optimizer convergence flag.
    n_obs : int
        Number of observations.
    persistence : float
        Model-specific persistence (alpha+beta; alpha+gamma/2+beta; beta).
    unconditional_variance : float
        Implied long-run variance (exogenous terms at zero); ``inf`` if
        persistence >= 1.
    """

    model: str
    dist: str
    params: dict[str, float]
    std_errors: dict[str, float]
    loglik: float
    sigma2: np.ndarray
    returns: np.ndarray
    x: np.ndarray | None
    converged: bool
    n_obs: int
    persistence: float
    unconditional_variance: float
    extra: dict = field(default_factory=dict)

    @property
    def std_resid(self) -> np.ndarray:
        """Standardized residuals ``z_t = r_t / sigma_t``."""
        return self.returns / np.sqrt(self.sigma2)

    def annualized_uncond_vol(self, periods_per_year: int = 252) -> float:
        """Annualized unconditional volatility implied by the fit."""
        return float(np.sqrt(self.unconditional_variance * periods_per_year))

    def summary(self) -> str:
        """Human-readable parameter table."""
        lines = [
            f"{self.model.upper()} ({self.dist}), n={self.n_obs}, "
            f"loglik={self.loglik:.3f}, persistence={self.persistence:.4f}, "
            f"converged={self.converged}"
        ]
        for k, v in self.params.items():
            se = self.std_errors.get(k, np.nan)
            lines.append(f"  {k:>10s} = {v: .6g}  (se {se:.3g})")
        return "\n".join(lines)
