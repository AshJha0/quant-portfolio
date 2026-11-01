"""Shared utilities: input validation, numerical Hessian, exceptions.

Conventions (used throughout the package)
-----------------------------------------
* Returns are **daily log-returns in decimal units** (0.01 = 1%).
* Variances are daily; volatilities are annualised with ``sqrt(252)`` unless
  stated otherwise (ACT/365F day count is irrelevant for return-count-based
  annualisation; we use the trading-day convention of 252 periods/year).
* Every stochastic routine takes an explicit ``seed`` or
  ``numpy.random.Generator``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

TRADING_DAYS: int = 252
"""Trading-day annualisation factor for daily data."""


class ConvergenceError(RuntimeError):
    """Raised when a maximum-likelihood optimisation fails to converge.

    Failures are surfaced loudly (never silently returned as if valid); pass
    ``raise_on_failure=False`` to the fit functions to receive the result with
    ``converged=False`` and a diagnostic ``message`` instead.
    """


def as_generator(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` from a seed or pass through."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def validate_returns(
    returns: Sequence[float] | np.ndarray,
    min_obs: int = 2,
    name: str = "returns",
) -> np.ndarray:
    """Validate and coerce a return series to a 1-D float array.

    Policy (documented + tested):

    * NaN / inf values **raise** ``ValueError``: silent dropping reorders the
      time series and corrupts recursions, so cleaning is the caller's job.
    * Series shorter than ``min_obs`` raise an informative ``ValueError``.

    Parameters
    ----------
    returns : array-like
        Daily log-returns in decimal units.
    min_obs : int
        Minimum number of observations required.
    name : str
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        1-D ``float64`` copy of the input.
    """
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < min_obs:
        raise ValueError(
            f"{name} has {r.size} observations; at least {min_obs} are "
            f"required for this estimator."
        )
    if not np.all(np.isfinite(r)):
        n_bad = int(np.sum(~np.isfinite(r)))
        raise ValueError(
            f"{name} contains {n_bad} NaN/inf value(s). This package does not "
            f"silently drop or impute; clean the series first (policy: "
            f"time-series recursions are order-sensitive)."
        )
    return r


def checked_sample_variance(returns: np.ndarray, model_name: str) -> float:
    """Sample variance with a degeneracy guard for MLE fitters.

    An exactly-constant series has ``np.var`` equal to floating-point
    round-off (e.g. 1.9e-37 for a constant 0.001), not exactly zero, so the
    check must be on the *range* of the series, plus a numerical floor far
    below any real daily return variance.
    """
    var = float(np.var(returns))
    if np.ptp(returns) == 0.0 or var < 1e-20:
        raise ValueError(
            f"return series has zero variance (constant series); a "
            f"{model_name} model cannot be estimated. Check the data."
        )
    return var


def _fd_steps(x: np.ndarray, rel_step: float, floor: np.ndarray | float | None) -> np.ndarray:
    """Per-parameter finite-difference steps ``rel_step * max(|x_i|, floor_i)``.

    ``floor`` is the parameter's natural scale. Without it a parameter that
    sits near zero (GJR gamma, a small alpha) gets a step of ~1e-12 and its
    Hessian diagonal is round-off. Fitters pass the scale of each parameter
    (omega's own magnitude, 1e-2 for the dimensionless ones).
    """
    if floor is None:
        floor = 1e-8
    return rel_step * np.maximum(np.abs(x), np.asarray(floor, dtype=float))


def numerical_hessian(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    rel_step: float = 1e-4,
    floor: np.ndarray | float | None = None,
) -> np.ndarray:
    """Numerical Hessian of ``func`` at ``x`` via central finite differences.

    Uses per-parameter steps ``h_i = rel_step * max(|x_i|, floor_i)`` so
    that parameters on very different scales (``omega ~ 1e-6`` vs
    ``beta ~ 0.9`` for decimal returns) are each perturbed proportionally,
    and parameters near zero are still perturbed by a usable amount.

    Returns
    -------
    numpy.ndarray
        Symmetrised ``(k, k)`` Hessian matrix.
    """
    x = np.asarray(x, dtype=float)
    k = x.size
    h = _fd_steps(x, rel_step, floor)
    hess = np.empty((k, k))
    f0 = func(x)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        # diagonal: (f(x+h) - 2 f(x) + f(x-h)) / h^2
        hess[i, i] = (func(x + ei) - 2.0 * f0 + func(x - ei)) / h[i] ** 2
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = h[j]
            fpp = func(x + ei + ej)
            fpm = func(x + ei - ej)
            fmp = func(x - ei + ej)
            fmm = func(x - ei - ej)
            hess[i, j] = hess[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
    return 0.5 * (hess + hess.T)


def numerical_scores(
    loglik_obs: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    rel_step: float = 1e-4,
    floor: np.ndarray | float | None = None,
) -> np.ndarray:
    """Per-observation score matrix ``S[t, i] = d l_t / d x_i`` (central differences)."""
    x = np.asarray(x, dtype=float)
    h = _fd_steps(x, rel_step, floor)
    cols = []
    for i in range(x.size):
        e = np.zeros(x.size)
        e[i] = h[i]
        cols.append((loglik_obs(x + e) - loglik_obs(x - e)) / (2.0 * h[i]))
    return np.column_stack(cols)


def _inverse(hess: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(hess)


def _se_from_cov(cov: np.ndarray) -> np.ndarray:
    diag = np.diag(cov).copy()
    with np.errstate(invalid="ignore"):
        return np.where(diag > 0, np.sqrt(np.abs(diag)), np.nan)


def std_errors_from_hessian(hess: np.ndarray) -> np.ndarray:
    """Standard errors as sqrt of the diagonal of the inverse Hessian of the
    negative log-likelihood. Falls back to the pseudo-inverse if the Hessian
    is numerically singular; non-positive diagonal entries yield ``nan``."""
    return _se_from_cov(_inverse(hess))


def mle_std_errors(
    nll: Callable[[np.ndarray], float],
    loglik_obs: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    floor: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classic and Bollerslev-Wooldridge (1992) robust standard errors.

    With ``A`` the Hessian of the negative log-likelihood and ``B = S'S``
    the outer product of the per-observation scores, the classic covariance
    is ``A^-1`` and the robust (sandwich) covariance is ``A^-1 B A^-1``.
    The classic form is valid only if the innovation density is correctly
    specified. The sandwich form stays valid for Gaussian QMLE when the true
    innovations are fat-tailed, which is the usual case for daily returns.

    Returns
    -------
    (classic_se, robust_se) : tuple of numpy.ndarray
        NaN where the corresponding variance is not positive.
    """
    x = np.asarray(x, dtype=float)
    a_inv = _inverse(numerical_hessian(nll, x, floor=floor))
    scores = numerical_scores(loglik_obs, x, floor=floor)
    if not np.all(np.isfinite(scores)):
        return _se_from_cov(a_inv), np.full(x.size, np.nan)
    b = scores.T @ scores
    return _se_from_cov(a_inv), _se_from_cov(a_inv @ b @ a_inv)


def arch_style_backcast(returns: np.ndarray, decay: float = 0.94, tau_max: int = 75) -> float:
    """Pre-sample variance via exponentially weighted backcast.

    Matches the initialisation used by the ``arch`` package (Sheppard):
    ``tau = min(75, n)`` observations, weights proportional to ``0.94**i``.
    Using the same backcast lets our GARCH log-likelihood agree with ``arch``'s
    to numerical precision on identical parameters (see tests).
    """
    tau = min(tau_max, returns.size)
    w = decay ** np.arange(tau)
    w /= w.sum()
    return float(w @ (returns[:tau] ** 2))


def initial_variance(returns: np.ndarray, method: str = "backcast") -> float:
    """Pre-sample variance ``b`` used to start variance recursions.

    Parameters
    ----------
    method : {"backcast", "sample"}
        ``"backcast"``: arch-compatible exponentially weighted backcast
        (default, robust to early-sample outliers);
        ``"sample"``: full-sample mean of squared returns.
    """
    if method == "backcast":
        return arch_style_backcast(returns)
    if method == "sample":
        return float(np.mean(returns**2))
    raise ValueError(f"unknown init method {method!r}; use 'backcast' or 'sample'")
