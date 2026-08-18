"""EGARCH(1,1) (Nelson 1991) from scratch.

    ln sigma2_t = omega + beta * ln sigma2_{t-1}
                  + alpha * (|z_{t-1}| - E|z|) + gamma * z_{t-1},
    z_t = r_t / sigma_t

Why EGARCH matters for FX: the leverage parameter ``gamma`` is
**unconstrained in sign** (the log form guarantees positive variance without
any positivity constraints), which is exactly what FX asymmetry requires --
G10 asymmetry is weak and can point either way (USDJPY: safe-haven yen
buying makes *negative* pair returns the high-vol direction, gamma < 0 in
this quote direction; invert the pair and gamma flips sign), while EM pairs
quoted USD/EM show strong asymmetry on EM depreciation. GJR constrains its
asymmetry term to be non-negative for positivity; EGARCH does not need to.

Stationarity of ln sigma2 requires ``|beta| < 1`` (enforced via tanh
transform). ``E|z| = sqrt(2/pi)`` for Gaussian innovations and the analytic
Student-t absolute moment otherwise (:func:`fx_vol._mle.student_t_abs_moment`).

The likelihood recursion is inherently sequential (z_t depends on the
filtered sigma_t), so the filter is a tight scalar loop; a 20k-observation
likelihood costs ~5 ms, keeping from-scratch MLE practical.
"""

from __future__ import annotations

from math import exp, log, sqrt
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from ._mle import (
    MIN_NU,
    MIN_OBS,
    FitResult,
    backcast,
    gaussian_loglik,
    hessian_std_errors,
    student_t_abs_moment,
    student_t_loglik,
    validate_returns,
)

__all__ = ["egarch_filter", "fit_egarch", "GAUSSIAN_ABS_MOMENT"]

GAUSSIAN_ABS_MOMENT = sqrt(2.0 / np.pi)
_CLIP = 30.0
_LOGS2_BOUND = 60.0  # |ln sigma2| clip inside the recursion (overflow guard)


def egarch_filter(
    returns: Sequence[float] | np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    abs_moment: float = GAUSSIAN_ABS_MOMENT,
    initial_variance: float | None = None,
) -> np.ndarray:
    """Run the EGARCH(1,1) log-variance recursion.

    ``sigma2[0]`` is the backcast (arch convention:
    ``ln sigma2_0 = ln(backcast)``); subsequent values follow the recursion in
    the module docstring.

    Parameters
    ----------
    returns : array-like
        Zero-mean log returns.
    omega, alpha, gamma, beta : float
        Log-variance parameters; requires ``|beta| < 1``.
    abs_moment : float
        ``E|z|`` of the innovation distribution (``sqrt(2/pi)`` Gaussian).
    initial_variance : float, optional
        Override the backcast seed (variance units, not log).
    """
    y = np.asarray(returns, dtype=float)
    if not abs(beta) < 1.0:
        raise ValueError(f"require |beta| < 1 for log-variance stationarity, got beta={beta}")
    b = backcast(y) if initial_variance is None else float(initial_variance)
    if b <= 0:
        raise ValueError("initial variance must be positive")
    n = y.size
    sigma2 = np.empty(n)
    ls2 = log(b)
    sigma2[0] = b
    yl = y.tolist()  # scalar-math loop is ~4x faster than numpy scalars
    for t in range(1, n):
        z = yl[t - 1] / sqrt(sigma2[t - 1])
        ls2 = omega + beta * ls2 + alpha * (abs(z) - abs_moment) + gamma * z
        if ls2 > _LOGS2_BOUND:
            ls2 = _LOGS2_BOUND
        elif ls2 < -_LOGS2_BOUND:
            ls2 = -_LOGS2_BOUND
        sigma2[t] = exp(ls2)
    return sigma2


def _nll_natural(theta: np.ndarray, y: np.ndarray, dist: str, b: float) -> float:
    omega, alpha, gamma, beta = theta[0], theta[1], theta[2], theta[3]
    if not abs(beta) < 1.0:
        return 1e10
    if dist == "t":
        nu = theta[4]
        if nu <= 2.0:
            return 1e10
        am = student_t_abs_moment(nu)
    else:
        nu, am = None, GAUSSIAN_ABS_MOMENT
    sigma2 = egarch_filter(y, omega, alpha, gamma, beta, abs_moment=am, initial_variance=b)
    ll = gaussian_loglik(y, sigma2) if dist == "gaussian" else student_t_loglik(y, sigma2, nu)
    return -ll if np.isfinite(ll) else 1e10


def fit_egarch(
    returns: Sequence[float] | np.ndarray,
    dist: str = "gaussian",
    min_obs: int = MIN_OBS,
) -> FitResult:
    """Fit EGARCH(1,1) by from-scratch MLE.

    Same conventions as :func:`fx_vol.garch.fit_garch`: internal unit-variance
    rescaling (``omega`` maps back via ``omega_r = omega_y + (1-beta) ln s^2``
    -- additive in log-variance space), transforms (``beta = tanh(u)``;
    ``alpha``, ``gamma`` unconstrained -- the log form needs no positivity
    constraints), multi-start L-BFGS-B, natural-space Hessian standard errors.

    Returns
    -------
    FitResult
        Parameters ``omega, alpha, gamma, beta [, nu]``; ``persistence`` is
        ``beta``; ``unconditional_variance`` is the log-normal-corrected
        Gaussian approximation ``exp(omega/(1-beta))`` reported for
        orientation only (exact only in the median sense -- use the
        simulation-based forecaster for long horizons).
    """
    if dist not in ("gaussian", "t"):
        raise ValueError(f"dist must be 'gaussian' or 't', got {dist!r}")
    r = validate_returns(returns, min_obs=min_obs)
    n = r.size
    s = float(np.std(r))
    y = r / s
    b = backcast(y)

    def unpack(u: np.ndarray) -> tuple[float, float, float, float, float | None, float]:
        u = np.clip(u, -_CLIP, _CLIP)
        omega = float(u[0])
        alpha = float(u[1])
        gamma = float(u[2])
        beta = float(np.tanh(u[3]))
        if dist == "t":
            nu = float(MIN_NU + np.exp(u[4]))
            am = student_t_abs_moment(nu)
        else:
            nu, am = None, GAUSSIAN_ABS_MOMENT
        return omega, alpha, gamma, beta, nu, am

    def nll_u(u: np.ndarray) -> float:
        omega, alpha, gamma, beta, nu, am = unpack(u)
        sigma2 = egarch_filter(y, omega, alpha, gamma, beta, abs_moment=am, initial_variance=b)
        ll = gaussian_loglik(y, sigma2) if dist == "gaussian" else student_t_loglik(y, sigma2, nu)
        return -ll if np.isfinite(ll) else 1e10

    var_y = float(np.var(y))
    starts = []
    for a0, g0, b0 in ((0.10, 0.0, 0.95), (0.20, -0.05, 0.90), (0.15, 0.05, 0.98)):
        u0 = [log(var_y) * (1.0 - b0), a0, g0, float(np.arctanh(b0))]
        if dist == "t":
            u0.append(log(8.0 - MIN_NU))
        starts.append(np.array(u0))

    best = None
    for u0 in starts:
        res = minimize(nll_u, u0, method="L-BFGS-B", options={"maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, gamma, beta, nu, am = unpack(best.x)
    sigma2_y = egarch_filter(y, omega, alpha, gamma, beta, abs_moment=am, initial_variance=b)

    theta_nat = [omega, alpha, gamma, beta] + ([nu] if dist == "t" else [])
    se_nat = hessian_std_errors(lambda t: _nll_natural(t, y, dist, b), np.array(theta_nat))

    # omega maps additively: ln sigma2_r = ln sigma2_y + ln s^2, so
    # omega_r = omega_y + (1 - beta) * ln s^2; alpha/gamma/beta/nu invariant.
    log_s2 = log(s * s)
    omega_r = omega + (1.0 - beta) * log_s2
    params = {"omega": omega_r, "alpha": alpha, "gamma": gamma, "beta": beta}
    se = {"omega": se_nat[0], "alpha": se_nat[1], "gamma": se_nat[2], "beta": se_nat[3]}
    if dist == "t":
        params["nu"] = float(nu)
        se["nu"] = float(se_nat[4])

    uncond = exp(omega_r / (1.0 - beta))
    return FitResult(
        model="egarch",
        dist=dist,
        params=params,
        std_errors=se,
        loglik=-best.fun - n * log(s),
        sigma2=sigma2_y * s * s,
        returns=r,
        x=None,
        converged=bool(best.success),
        n_obs=n,
        persistence=float(beta),
        unconditional_variance=float(uncond),
        extra={"abs_moment": am, "backcast": b * s * s},
    )
