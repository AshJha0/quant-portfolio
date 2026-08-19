"""GARCH(1,1) and GARCH-X with exogenous variance dummies, from scratch.

Model (zero-mean returns r_t, information set F_{t-1}):

    r_t = sigma_t * z_t,   z_t ~ N(0,1) or standardized Student-t(nu)
    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1} + gamma_x' x_t

The exogenous regressors ``x_t`` (GARCH-X) are **variance dummies known at
t-1** -- e.g. scheduled central-bank event days (FOMC, ECB, BoJ): the
announcement calendar is public, so the extra variance ``gamma_x`` on event
day t is in the t-1 information set. This is the natural FX specification:
event risk is calendar-driven and priced ex ante (overnight implied vols
routinely double before FOMC).

Estimation is maximum likelihood implemented from scratch:

* constraint transforms -- ``omega = exp(u)``; stationarity enforced via
  persistence ``p = expit(u_p) in (0,1)`` split as ``alpha = p*s``,
  ``beta = p*(1-s)`` with ``s = expit(u_s)``; ``gamma_x = exp(u)`` (>= 0 keeps
  variance positive); ``nu = 2.05 + exp(u)``;
* multi-start L-BFGS-B on the unconstrained parameters;
* standard errors from the numerical Hessian in natural parameter space;
* optional variance targeting (``omega`` fixed by the sample variance).

The `arch` package is used in the test suite ONLY to cross-validate the
estimates -- see tests/test_arch_crosscheck.py.
"""

from __future__ import annotations

from math import log
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit, logit

from ._mle import (
    MIN_NU,
    MIN_OBS,
    FitResult,
    backcast,
    gaussian_loglik,
    hessian_std_errors,
    student_t_loglik,
    validate_filter_params,
    validate_returns,
)

__all__ = ["garch_filter", "fit_garch"]

_CLIP = 30.0  # bound on unconstrained parameters (expit/exp saturation guard)


def _validate_x(x, n: int) -> np.ndarray:
    """Validate exogenous variance regressors: shape (n,) or (n, k), finite, >= 0."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError(f"x must have shape ({n},) or ({n}, k), got {np.shape(x)}")
    if not np.isfinite(arr).all():
        raise ValueError("x contains NaN or infinite values")
    if (arr < 0).any():
        raise ValueError(
            "x must be non-negative (variance regressors with gamma_x >= 0 keep "
            "sigma2 positive); recode signed regressors as separate dummies"
        )
    return arr


def garch_filter(
    returns: Sequence[float] | np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    gamma_x: Sequence[float] | np.ndarray | None = None,
    x: np.ndarray | None = None,
    initial_variance: float | None = None,
) -> np.ndarray:
    """Run the GARCH(1,1)(-X) variance recursion.

    ``sigma2[0] = omega + (alpha + beta) * b + gamma_x' x_0`` with backcast
    ``b`` (arch convention: the pre-sample ``eps^2`` and ``sigma2`` are both
    set to the backcast), then
    ``sigma2[t] = omega + alpha r_{t-1}^2 + beta sigma2[t-1] + gamma_x' x_t``.

    Implemented with :func:`scipy.signal.lfilter` (the recursion is linear in
    ``sigma2``), which keeps 20k-observation likelihood evaluations fast
    enough for from-scratch MLE.

    Parameters
    ----------
    returns : array-like
        Zero-mean log returns.
    omega, alpha, beta : float
        Variance-equation parameters (omega > 0, alpha, beta >= 0).
    gamma_x, x : optional
        Exogenous coefficient vector (k,) and regressor matrix (n, k).
    initial_variance : float, optional
        Override the backcast seed.

    Returns
    -------
    numpy.ndarray
        Conditional variance path, same length as ``returns``.
    """
    y = np.asarray(returns, dtype=float)
    validate_filter_params(omega=omega, alpha=alpha, beta=beta,
                           initial_variance=initial_variance)
    if omega <= 0 or alpha < 0 or beta < 0 or beta >= 1:
        raise ValueError(
            f"require omega > 0, alpha >= 0, 0 <= beta < 1; got "
            f"omega={omega}, alpha={alpha}, beta={beta}"
        )
    b = backcast(y) if initial_variance is None else float(initial_variance)
    if b <= 0.0:
        raise ValueError(f"initial_variance must be positive, got {b!r}")
    n = y.size
    u = np.empty(n)
    u[0] = omega + alpha * b
    u[1:] = omega + alpha * y[:-1] ** 2
    if x is not None:
        if gamma_x is None:
            raise ValueError(
                "x was supplied without gamma_x; the exogenous coefficients are "
                "required (passing only x would give an all-NaN variance path)"
            )
        gx = np.atleast_1d(np.asarray(gamma_x, dtype=float))
        if not np.isfinite(gx).all():
            raise ValueError(f"gamma_x contains NaN or infinite values: {gamma_x!r}")
        xm = _validate_x(x, n)
        if gx.size != xm.shape[1]:
            raise ValueError(f"gamma_x has {gx.size} entries but x has {xm.shape[1]} columns")
        u += xm @ gx
    elif gamma_x is not None:
        raise ValueError(
            "gamma_x was supplied without x; exogenous coefficients have no "
            "regressors to multiply -- pass x as well or drop gamma_x"
        )
    sigma2, _ = lfilter([1.0], [1.0, -beta], u, zi=np.array([beta * b]))
    return sigma2


def _nll_scaled(theta_nat: np.ndarray, y: np.ndarray, x: np.ndarray | None, dist: str, b: float) -> float:
    """Negative log-likelihood in natural parameter space on unit-variance data."""
    k = 0 if x is None else x.shape[1]
    omega, alpha, beta = theta_nat[0], theta_nat[1], theta_nat[2]
    gamma_x = theta_nat[3 : 3 + k] if k else None
    if omega <= 0 or alpha < 0 or beta < 0 or beta >= 1 or alpha + beta >= 1:
        return 1e10
    if k and (gamma_x < 0).any():
        return 1e10
    sigma2 = garch_filter(y, omega, alpha, beta, gamma_x, x, initial_variance=b)
    if not np.all(sigma2 > 0) or not np.all(np.isfinite(sigma2)):
        return 1e10
    if dist == "gaussian":
        ll = gaussian_loglik(y, sigma2)
    else:
        nu = theta_nat[3 + k]
        if nu <= 2.0:
            return 1e10
        ll = student_t_loglik(y, sigma2, nu)
    return -ll if np.isfinite(ll) else 1e10


def fit_garch(
    returns: Sequence[float] | np.ndarray,
    dist: str = "gaussian",
    x: Sequence[float] | np.ndarray | None = None,
    variance_targeting: bool = False,
    min_obs: int = MIN_OBS,
) -> FitResult:
    """Fit GARCH(1,1) (or GARCH-X) by maximum likelihood, from scratch.

    Parameters
    ----------
    returns : array-like
        Zero-mean log returns (decimal or percent -- internally rescaled to
        unit variance; estimates are mapped back to the caller's units).
        NaNs raise; series shorter than ``min_obs`` raise; constant series
        raise (degenerate likelihood).
    dist : {'gaussian', 't'}
        Innovation distribution. 't' is the standardized Student-t and adds a
        dof parameter ``nu`` -- the workhorse for G10 FX, whose returns are
        fat-tailed but (for most pairs) close to symmetric.
    x : array-like, optional
        (n,) or (n, k) non-negative exogenous variance regressors known at
        t-1, e.g. central-bank event dummies (GARCH-X).
    variance_targeting : bool
        Fix ``omega = Var(r) * (1 - alpha - beta)`` instead of estimating it
        (one fewer parameter; the implied unconditional variance matches the
        sample variance exactly). Not available together with ``x``.
    min_obs : int
        Minimum sample size (default 100).

    Returns
    -------
    FitResult
        Parameters ``omega, alpha, beta [, gamma_x_i][, nu]`` with Hessian
        standard errors, log-likelihood, conditional-variance path.
    """
    if dist not in ("gaussian", "t"):
        raise ValueError(f"dist must be 'gaussian' or 't', got {dist!r}")
    r = validate_returns(returns, min_obs=min_obs)
    n = r.size
    xm = None if x is None else _validate_x(x, n)
    if variance_targeting and xm is not None:
        raise ValueError("variance targeting is not supported together with exogenous x")
    k = 0 if xm is None else xm.shape[1]

    s = float(np.std(r))
    y = r / s
    b = backcast(y)
    var_y = float(np.var(y))

    def unpack(u: np.ndarray) -> tuple[float, float, float, np.ndarray | None, float | None]:
        u = np.clip(u, -_CLIP, _CLIP)
        if variance_targeting:
            p = expit(u[0]); sh = expit(u[1])
            alpha, beta = p * sh, p * (1.0 - sh)
            omega = var_y * (1.0 - p)
            rest = 2
        else:
            omega = np.exp(u[0])
            p = expit(u[1]); sh = expit(u[2])
            alpha, beta = p * sh, p * (1.0 - sh)
            rest = 3
        gamma_x = np.exp(u[rest : rest + k]) if k else None
        nu = MIN_NU + np.exp(u[rest + k]) if dist == "t" else None
        return omega, alpha, beta, gamma_x, nu

    def nll_u(u: np.ndarray) -> float:
        omega, alpha, beta, gamma_x, nu = unpack(u)
        if omega <= 0:
            return 1e10
        sigma2 = garch_filter(y, omega, alpha, beta, gamma_x, xm, initial_variance=b)
        if not np.all(sigma2 > 1e-12):
            return 1e10
        if dist == "gaussian":
            ll = gaussian_loglik(y, sigma2)
        else:
            ll = student_t_loglik(y, sigma2, nu)
        return -ll if np.isfinite(ll) else 1e10

    starts = []
    for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.02, 0.955)):
        p0, sh0 = a0 + b0, a0 / (a0 + b0)
        core = [logit(p0), logit(sh0)] if variance_targeting else [
            log(var_y * (1.0 - p0)), logit(p0), logit(sh0)
        ]
        core += [log(0.5)] * k
        if dist == "t":
            core += [log(8.0 - MIN_NU)]
        starts.append(np.array(core))

    best = None
    for u0 in starts:
        res = minimize(nll_u, u0, method="L-BFGS-B", options={"maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, beta, gamma_x, nu = unpack(best.x)
    sigma2_y = garch_filter(y, omega, alpha, beta, gamma_x, xm, initial_variance=b)
    ll_y = -best.fun

    # ----- natural-space Hessian standard errors (on scaled data) -----
    theta_nat = [omega, alpha, beta]
    names = ["omega", "alpha", "beta"]
    if k:
        theta_nat += list(gamma_x)
        names += [f"gamma_x{i}" for i in range(k)] if k > 1 else ["gamma_x"]
    if dist == "t":
        theta_nat.append(nu)
        names.append("nu")
    theta_nat = np.array(theta_nat)
    se_nat = hessian_std_errors(lambda t: _nll_scaled(t, y, xm, dist, b), theta_nat)

    # ----- map back to the caller's return units -----
    scale2 = s * s
    params = {"omega": omega * scale2, "alpha": alpha, "beta": beta}
    se = {"omega": se_nat[0] * scale2, "alpha": se_nat[1], "beta": se_nat[2]}
    if k:
        for i in range(k):
            nm = names[3 + i]
            params[nm] = float(gamma_x[i] * scale2)
            se[nm] = float(se_nat[3 + i] * scale2)
    if dist == "t":
        params["nu"] = float(nu)
        se["nu"] = float(se_nat[-1])

    persistence = alpha + beta
    uncond = params["omega"] / (1.0 - persistence) if persistence < 1.0 else np.inf
    return FitResult(
        model="garch-x" if k else "garch",
        dist=dist,
        params=params,
        std_errors=se,
        loglik=ll_y - n * log(s),
        sigma2=sigma2_y * scale2,
        returns=r,
        x=xm,
        converged=bool(best.success),
        n_obs=n,
        persistence=float(persistence),
        unconditional_variance=float(uncond),
        extra={"variance_targeting": variance_targeting, "backcast": b * scale2},
    )
