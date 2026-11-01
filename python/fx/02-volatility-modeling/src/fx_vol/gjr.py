"""GJR-GARCH(1,1,1) (Glosten-Jagannathan-Runkle) from scratch.

    sigma2_t = omega + (alpha + gamma * 1[r_{t-1} < 0]) * r_{t-1}^2 + beta * sigma2_{t-1}

``gamma > 0`` means *negative* returns of the quoted pair raise next-day
variance more than positive ones.

FX-specific reading of the asymmetry (see docs/METHODOLOGY.md): for G10
pairs the equity-style leverage story does not apply and gamma is usually
small and can take either sign depending on which currency is the risk /
safe-haven leg: e.g. USDJPY *falls* in risk-off (safe-haven yen bid), so
the asymmetry appears on negative USDJPY returns, but on JPYUSD the same
economics flips the sign. EM pairs quoted EM-per-USD (USDMXN, USDTRY)
typically show strong positive gamma: depreciation of the EM currency (pair
rallies... note the pair *rises* when MXN weakens, so risk-off shows up on
*positive* returns of USDMXN: quote direction matters; fit the pair the
desk trades and check the sign). Positivity of the variance needs
``alpha >= 0`` and ``alpha + gamma >= 0`` only, so ``gamma`` may take either
sign. The estimator imposes exactly those constraints, so asymmetry of
either sign is estimated on the pair as quoted.

Stationarity (symmetric innovations, so ``E 1[z<0] = 1/2``):
``alpha + gamma/2 + beta < 1``, enforced by transform.
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
    ConvergenceError,
    loglik_obs,
    map_std_errors,
    mle_covariances,
    student_t_loglik,
    validate_filter_params,
    validate_returns,
)

__all__ = ["gjr_filter", "fit_gjr"]

_CLIP = 30.0


def gjr_filter(
    returns: Sequence[float] | np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    initial_variance: float | None = None,
) -> np.ndarray:
    """Run the GJR-GARCH variance recursion (lfilter-based, arch-style seed).

    ``sigma2[0] = omega + (alpha + gamma/2 + beta) * b`` with backcast ``b``
    (the pre-sample indicator takes its expectation 1/2), then the recursion
    above.
    """
    y = np.asarray(returns, dtype=float)
    validate_filter_params(omega=omega, alpha=alpha, gamma=gamma, beta=beta,
                           initial_variance=initial_variance)
    if omega <= 0 or alpha < 0 or alpha + gamma < 0 or beta < 0 or beta >= 1:
        raise ValueError(
            f"require omega > 0, alpha >= 0, alpha + gamma >= 0, 0 <= beta < 1; got "
            f"omega={omega}, alpha={alpha}, gamma={gamma}, beta={beta}"
        )
    b = backcast(y) if initial_variance is None else float(initial_variance)
    if b <= 0.0:
        raise ValueError(f"initial_variance must be positive, got {b!r}")
    n = y.size
    u = np.empty(n)
    u[0] = omega + (alpha + 0.5 * gamma) * b
    neg = (y[:-1] < 0.0).astype(float)
    u[1:] = omega + (alpha + gamma * neg) * y[:-1] ** 2
    sigma2, _ = lfilter([1.0], [1.0, -beta], u, zi=np.array([beta * b]))
    return sigma2


def _ll_obs_natural(theta: np.ndarray, y: np.ndarray, dist: str, b: float) -> np.ndarray:
    omega, alpha, gamma, beta = theta[0], theta[1], theta[2], theta[3]
    bad = np.full(y.size, np.nan)
    if omega <= 0 or alpha < 0 or alpha + gamma < 0 or beta < 0 or beta >= 1:
        return bad
    if alpha + 0.5 * gamma + beta >= 1:
        return bad
    sigma2 = gjr_filter(y, omega, alpha, gamma, beta, initial_variance=b)
    return loglik_obs(y, sigma2, dist, theta[4] if dist == "t" else None)


def _nll_natural(theta: np.ndarray, y: np.ndarray, dist: str, b: float) -> float:
    ll = float(np.sum(_ll_obs_natural(theta, y, dist, b)))
    return -ll if np.isfinite(ll) else 1e10


def fit_gjr(
    returns: Sequence[float] | np.ndarray,
    dist: str = "gaussian",
    min_obs: int = MIN_OBS,
    raise_on_failure: bool = True,
) -> FitResult:
    """Fit GJR-GARCH(1,1,1) by from-scratch MLE.

    Same conventions as :func:`fx_vol.garch.fit_garch`: internal unit-variance
    rescaling, multi-start L-BFGS-B, classic and robust standard errors,
    ``raise_on_failure``. Transforms: persistence ``p = expit(u1)`` is split
    into the average ARCH loading ``a = alpha + gamma/2`` and ``beta``; ``a``
    is then split into the positive-shock loading ``alpha = 2 a w`` and the
    negative-shock loading ``alpha + gamma = 2 a (1 - w)``. This imposes
    ``alpha >= 0``, ``alpha + gamma >= 0`` and ``alpha + gamma/2 + beta < 1``
    and nothing else, so gamma can be negative.

    Returns
    -------
    FitResult
        Parameters ``omega, alpha, gamma, beta [, nu]``; ``persistence`` is
        ``alpha + gamma/2 + beta``.
    """
    if dist not in ("gaussian", "t"):
        raise ValueError(f"dist must be 'gaussian' or 't', got {dist!r}")
    r = validate_returns(returns, min_obs=min_obs)
    n = r.size
    s = float(np.std(r))
    y = r / s
    b = backcast(y)
    var_y = float(np.var(y))

    def unpack(u: np.ndarray) -> tuple[float, float, float, float, float | None]:
        u = np.clip(u, -_CLIP, _CLIP)
        omega = np.exp(u[0])
        p = expit(u[1])
        avg = p * expit(u[2])               # alpha + gamma/2
        beta = p - avg
        w = expit(u[3])
        alpha = 2.0 * avg * w               # loading on positive shocks
        gamma = 2.0 * avg * (1.0 - 2.0 * w)  # (alpha + gamma) - alpha
        nu = MIN_NU + np.exp(u[4]) if dist == "t" else None
        return float(omega), float(alpha), float(gamma), float(beta), nu

    def nll_u(u: np.ndarray) -> float:
        omega, alpha, gamma, beta, nu = unpack(u)
        sigma2 = gjr_filter(y, omega, alpha, gamma, beta, initial_variance=b)
        if not np.all(sigma2 > 1e-12):
            return 1e10
        ll = gaussian_loglik(y, sigma2) if dist == "gaussian" else student_t_loglik(y, sigma2, nu)
        return -ll if np.isfinite(ll) else 1e10

    starts = []
    for a0, g0, b0 in ((0.03, 0.08, 0.88), (0.05, 0.0, 0.90), (0.08, -0.06, 0.85)):
        avg0 = a0 + 0.5 * g0
        p0 = avg0 + b0
        w0 = a0 / (2.0 * avg0)
        u0 = [log(var_y * (1.0 - p0)), logit(p0), logit(avg0 / p0), logit(w0)]
        if dist == "t":
            u0.append(log(8.0 - MIN_NU))
        starts.append(np.array(u0))

    best = None
    for u0 in starts:
        res = minimize(nll_u, u0, method="L-BFGS-B", options={"maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, gamma, beta, nu = unpack(best.x)
    sigma2_y = gjr_filter(y, omega, alpha, gamma, beta, initial_variance=b)

    if not best.success and raise_on_failure:
        raise ConvergenceError(f"GJR MLE failed to converge: {best.message} (nit={best.nit})")
    names = ["omega", "alpha", "gamma", "beta"] + (["nu"] if dist == "t" else [])
    theta_nat = np.array([omega, alpha, gamma, beta] + ([nu] if dist == "t" else []))
    cov_c, cov_r = mle_covariances(
        lambda t: _nll_natural(t, y, dist, b),
        lambda t: _ll_obs_natural(t, y, dist, b),
        theta_nat,
    )
    scale2 = s * s
    jac = np.diag([scale2, 1.0, 1.0, 1.0] + ([1.0] if dist == "t" else []))
    se = dict(zip(names, [float(v) for v in map_std_errors(cov_c, jac)]))
    robust = dict(zip(names, [float(v) for v in map_std_errors(cov_r, jac)]))
    params = {"omega": omega * scale2, "alpha": alpha, "gamma": gamma, "beta": beta}
    if dist == "t":
        params["nu"] = float(nu)

    persistence = alpha + 0.5 * gamma + beta
    uncond = params["omega"] / (1.0 - persistence) if persistence < 1.0 else np.inf
    return FitResult(
        model="gjr",
        dist=dist,
        params=params,
        std_errors=se,
        loglik=-best.fun - n * log(s),
        sigma2=sigma2_y * scale2,
        returns=r,
        x=None,
        converged=bool(best.success),
        n_obs=n,
        persistence=float(persistence),
        unconditional_variance=float(uncond),
        extra={
            "backcast": b * scale2,
            "at_boundary": bool(persistence > 0.999 or alpha < 1e-6 or alpha + gamma < 1e-6),
        },
        robust_std_errors=robust,
    )
