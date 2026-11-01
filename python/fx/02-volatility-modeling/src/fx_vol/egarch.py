"""EGARCH(1,1) (Nelson 1991) from scratch.

    ln sigma2_t = omega + beta * ln sigma2_{t-1}
                  + alpha * (|z_{t-1}| - E|z|) + gamma * z_{t-1},
    z_t = r_t / sigma_t

Why EGARCH matters for FX: the leverage parameter ``gamma`` is
**unconstrained in sign** (the log form guarantees positive variance without
any positivity constraints), which is exactly what FX asymmetry requires:
G10 asymmetry is weak and can point either way (USDJPY: safe-haven yen
buying makes *negative* pair returns the high-vol direction, gamma < 0 in
this quote direction; invert the pair and gamma flips sign), while EM pairs
quoted USD/EM show strong asymmetry on EM depreciation. GJR needs
``alpha >= 0`` and ``alpha + gamma >= 0`` for positivity; EGARCH needs no
sign constraint at all.

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
    ConvergenceError,
    loglik_obs,
    map_std_errors,
    mle_covariances,
    student_t_abs_moment,
    student_t_loglik,
    validate_filter_params,
    validate_returns,
)

__all__ = ["egarch_filter", "fit_egarch", "egarch_unconditional_variance", "GAUSSIAN_ABS_MOMENT"]

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

    ``ln sigma2[0] = omega + beta ln(b)`` with backcast ``b`` (the pre-sample
    shock at its expectation, the ``arch`` package convention); subsequent
    values follow the recursion in the module docstring.

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
    validate_filter_params(omega=omega, alpha=alpha, gamma=gamma, beta=beta,
                           abs_moment=abs_moment,
                           initial_variance=initial_variance)
    if not abs(beta) < 1.0:
        raise ValueError(f"require |beta| < 1 for log-variance stationarity, got beta={beta}")
    b = backcast(y) if initial_variance is None else float(initial_variance)
    if not b > 0:
        raise ValueError(f"initial variance must be positive, got {b!r}")
    n = y.size
    sigma2 = np.empty(n)
    # pre-sample shock at its expectation: ln sigma2_0 = omega + beta ln b
    # (arch's convention; GARCH uses the same one-step-from-backcast rule)
    ls2 = min(max(omega + beta * log(b), -_LOGS2_BOUND), _LOGS2_BOUND)
    sigma2[0] = exp(ls2)
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


def _ll_obs_natural(theta: np.ndarray, y: np.ndarray, dist: str, b: float) -> np.ndarray:
    omega, alpha, gamma, beta = theta[0], theta[1], theta[2], theta[3]
    if not abs(beta) < 1.0:
        return np.full(y.size, np.nan)
    if dist == "t":
        nu = theta[4]
        if nu <= 2.0:
            return np.full(y.size, np.nan)
        am = student_t_abs_moment(nu)
    else:
        nu, am = None, GAUSSIAN_ABS_MOMENT
    sigma2 = egarch_filter(y, omega, alpha, gamma, beta, abs_moment=am, initial_variance=b)
    return loglik_obs(y, sigma2, dist, nu)


def _nll_natural(theta: np.ndarray, y: np.ndarray, dist: str, b: float) -> float:
    ll = float(np.sum(_ll_obs_natural(theta, y, dist, b)))
    return -ll if np.isfinite(ll) else 1e10


def egarch_unconditional_variance(
    omega: float, alpha: float, gamma: float, beta: float, dist: str = "gaussian"
) -> float:
    """Exact unconditional mean ``E[sigma2]`` of a stationary EGARCH(1,1).

    With ``g(z) = alpha (|z| - E|z|) + gamma z`` (Nelson 1991):
    ``E[sigma2] = exp(omega/(1-beta)) prod_i E[exp(beta^i g(z))]``. For
    Gaussian z each factor is closed form; the product is truncated once
    ``|beta|^i < 1e-16``. For Student-t innovations ``E exp(c|z|)`` is
    infinite, so ``E[sigma2]`` does not exist (``inf``) unless
    ``alpha = gamma = 0``. ``exp(omega/(1-beta))`` is the median-type level
    and is below ``E[sigma2]`` by Jensen's inequality.
    """
    from scipy.special import log_ndtr

    if not abs(beta) < 1.0:
        raise ValueError(f"|beta| must be < 1, got {beta}")
    level = omega / (1.0 - beta)
    if alpha == 0.0 and gamma == 0.0:
        return float(exp(level))
    if dist == "t":
        return float("inf")
    if dist != "gaussian":
        raise ValueError(f"dist must be 'gaussian' or 't', got {dist!r}")
    log_e, c = level, 1.0
    while abs(c) >= 1e-16:
        a, d = c * alpha, c * gamma
        log_e += -a * GAUSSIAN_ABS_MOMENT + float(np.logaddexp(
            0.5 * (a + d) ** 2 + log_ndtr(a + d), 0.5 * (a - d) ** 2 + log_ndtr(a - d)
        ))
        c *= beta
    return float(exp(log_e))


def fit_egarch(
    returns: Sequence[float] | np.ndarray,
    dist: str = "gaussian",
    min_obs: int = MIN_OBS,
    raise_on_failure: bool = True,
) -> FitResult:
    """Fit EGARCH(1,1) by from-scratch MLE.

    Same conventions as :func:`fx_vol.garch.fit_garch`: internal unit-variance
    rescaling (``omega`` maps back via ``omega_r = omega_y + (1-beta) ln s^2``
   : additive in log-variance space), transforms (``beta = tanh(u)``;
    ``alpha``, ``gamma`` unconstrained: the log form needs no positivity
    constraints), multi-start L-BFGS-B, classic and robust standard errors.
    Because ``omega_r`` depends on ``beta``, its standard error comes from
    the delta method: ``Var(omega_r) = Var(omega_y) + L^2 Var(beta)
    - 2 L Cov(omega_y, beta)`` with ``L = ln s^2``.

    Returns
    -------
    FitResult
        Parameters ``omega, alpha, gamma, beta [, nu]``; ``persistence`` is
        ``beta``; ``unconditional_variance`` is the exact ``E[sigma2]``
        (:func:`egarch_unconditional_variance`: closed form for Gaussian
        innovations, ``inf`` for Student-t where it does not exist);
        ``extra["median_variance"]`` is ``exp(omega/(1-beta))``.
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

    # L-BFGS-B on the average NLL from each start. On some samples its line
    # search stalls at the start point (the first trial step overshoots into
    # the explosive region and it reports "convergence" after one or two
    # iterations); those starts are re-run with BFGS and central-difference
    # gradients. Convergence is judged by the optimiser flag or a gradient
    # below 1e-5 on the average scale (BFGS often reports precision loss at
    # an optimum where the Student-t nu direction is flat).
    def avg(u: np.ndarray) -> float:
        return nll_u(u) / n

    best = None
    for u0 in starts:
        res = minimize(avg, u0, method="L-BFGS-B", options={"maxiter": 500})
        if res.nit <= 2 or not res.success:
            res = minimize(avg, res.x, method="BFGS", jac="3-point",
                           options={"maxiter": 400, "gtol": 1e-6})
            res.success = bool(res.success or np.max(np.abs(res.jac)) < 1e-5)
        res.fun = res.fun * n
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, gamma, beta, nu, am = unpack(best.x)
    sigma2_y = egarch_filter(y, omega, alpha, gamma, beta, abs_moment=am, initial_variance=b)

    if not best.success and raise_on_failure:
        raise ConvergenceError(f"EGARCH MLE failed to converge: {best.message} (nit={best.nit})")
    names = ["omega", "alpha", "gamma", "beta"] + (["nu"] if dist == "t" else [])
    theta_nat = np.array([omega, alpha, gamma, beta] + ([nu] if dist == "t" else []))
    cov_c, cov_r = mle_covariances(
        lambda t: _nll_natural(t, y, dist, b),
        lambda t: _ll_obs_natural(t, y, dist, b),
        theta_nat,
    )

    # omega maps additively: ln sigma2_r = ln sigma2_y + ln s^2, so
    # omega_r = omega_y + (1 - beta) * ln s^2; alpha/gamma/beta/nu invariant.
    log_s2 = log(s * s)
    omega_r = omega + (1.0 - beta) * log_s2
    jac = np.eye(theta_nat.size)
    jac[0, 3] = -log_s2
    se = dict(zip(names, [float(v) for v in map_std_errors(cov_c, jac)]))
    robust = dict(zip(names, [float(v) for v in map_std_errors(cov_r, jac)]))
    params = {"omega": omega_r, "alpha": alpha, "gamma": gamma, "beta": beta}
    if dist == "t":
        params["nu"] = float(nu)

    uncond = egarch_unconditional_variance(omega_r, alpha, gamma, beta, dist)
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
        extra={
            "abs_moment": am,
            "backcast": b * s * s,
            "median_variance": exp(omega_r / (1.0 - beta)),
            "at_boundary": bool(abs(beta) > 0.999),
        },
        robust_std_errors=robust,
    )
