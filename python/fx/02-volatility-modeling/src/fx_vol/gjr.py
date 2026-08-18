"""GJR-GARCH(1,1,1) (Glosten-Jagannathan-Runkle) from scratch.

    sigma2_t = omega + (alpha + gamma * 1[r_{t-1} < 0]) * r_{t-1}^2 + beta * sigma2_{t-1}

``gamma > 0`` means *negative* returns of the quoted pair raise next-day
variance more than positive ones.

FX-specific reading of the asymmetry (see docs/METHODOLOGY.md): for G10
pairs the equity-style leverage story does not apply and gamma is usually
small and can take either sign depending on which currency is the risk /
safe-haven leg -- e.g. USDJPY *falls* in risk-off (safe-haven yen bid), so
the asymmetry appears on negative USDJPY returns, but on JPYUSD the same
economics flips the sign. EM pairs quoted EM-per-USD (USDMXN, USDTRY)
typically show strong positive gamma: depreciation of the EM currency (pair
rallies... note the pair *rises* when MXN weakens, so risk-off shows up on
*positive* returns of USDMXN -- quote direction matters; fit the pair the
desk trades and check the sign). This estimator constrains ``gamma >= 0``
(sufficient for positivity); to capture asymmetry of the opposite sign,
either invert the pair (log-return sign flip, :func:`fx_vol.returns.invert_returns`)
or use EGARCH, whose leverage parameter is unconstrained in sign.

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
    hessian_std_errors,
    student_t_loglik,
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
    if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0 or beta >= 1:
        raise ValueError(
            f"require omega > 0, alpha, gamma >= 0, 0 <= beta < 1; got "
            f"omega={omega}, alpha={alpha}, gamma={gamma}, beta={beta}"
        )
    b = backcast(y) if initial_variance is None else float(initial_variance)
    n = y.size
    u = np.empty(n)
    u[0] = omega + (alpha + 0.5 * gamma) * b
    neg = (y[:-1] < 0.0).astype(float)
    u[1:] = omega + (alpha + gamma * neg) * y[:-1] ** 2
    sigma2, _ = lfilter([1.0], [1.0, -beta], u, zi=np.array([beta * b]))
    return sigma2


def _nll_natural(theta: np.ndarray, y: np.ndarray, dist: str, b: float) -> float:
    omega, alpha, gamma, beta = theta[0], theta[1], theta[2], theta[3]
    if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0 or beta >= 1:
        return 1e10
    if alpha + 0.5 * gamma + beta >= 1:
        return 1e10
    sigma2 = gjr_filter(y, omega, alpha, gamma, beta, initial_variance=b)
    if not np.all(sigma2 > 0):
        return 1e10
    if dist == "gaussian":
        ll = gaussian_loglik(y, sigma2)
    else:
        nu = theta[4]
        if nu <= 2.0:
            return 1e10
        ll = student_t_loglik(y, sigma2, nu)
    return -ll if np.isfinite(ll) else 1e10


def fit_gjr(
    returns: Sequence[float] | np.ndarray,
    dist: str = "gaussian",
    min_obs: int = MIN_OBS,
) -> FitResult:
    """Fit GJR-GARCH(1,1,1) by from-scratch MLE.

    Same conventions as :func:`fx_vol.garch.fit_garch`: internal unit-variance
    rescaling, constraint transforms (persistence ``p = expit`` split across
    ``alpha``, ``gamma/2``, ``beta`` by a softmax, guaranteeing
    ``alpha + gamma/2 + beta < 1`` and non-negativity), multi-start L-BFGS-B,
    natural-space Hessian standard errors.

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
        e = np.exp(np.array([u[2], u[3], 0.0]))
        w = e / e.sum()                      # softmax shares (alpha, gamma/2, beta)
        alpha, gamma, beta = p * w[0], 2.0 * p * w[1], p * w[2]
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
    for a0, g0, b0 in ((0.03, 0.08, 0.88), (0.05, 0.0, 0.90), (0.08, 0.15, 0.75)):
        p0 = a0 + 0.5 * max(g0, 1e-3) + b0
        wa = a0 / p0
        wg = 0.5 * max(g0, 1e-3) / p0
        wb = 1.0 - wa - wg
        u0 = [log(var_y * (1.0 - p0)), logit(p0), log(wa / wb), log(wg / wb)]
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

    theta_nat = [omega, alpha, gamma, beta] + ([nu] if dist == "t" else [])
    se_nat = hessian_std_errors(
        lambda t: _nll_natural(t, y, dist, b), np.array(theta_nat)
    )

    scale2 = s * s
    params = {"omega": omega * scale2, "alpha": alpha, "gamma": gamma, "beta": beta}
    se = {"omega": se_nat[0] * scale2, "alpha": se_nat[1], "gamma": se_nat[2], "beta": se_nat[3]}
    if dist == "t":
        params["nu"] = float(nu)
        se["nu"] = float(se_nat[4])

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
        extra={"backcast": b * scale2},
    )
