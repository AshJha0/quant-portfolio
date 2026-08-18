"""GJR-GARCH(1,1) implemented from scratch (Glosten-Jagannathan-Runkle 1993).

Model
-----
r_t = sigma_t z_t,  z_t ~ iid(0, 1)
sigma2_t = omega + (alpha + gamma 1[r_{t-1} < 0]) r_{t-1}^2 + beta sigma2_{t-1}

Sign convention (opposite of our EGARCH convention, both tested):
**gamma > 0 produces the leverage effect** — negative shocks load with
coefficient alpha + gamma > alpha.

Constraints: omega > 0, alpha >= 0, alpha + gamma >= 0 (positivity of the
variance; gamma itself may be negative), beta >= 0, and — for symmetric
innovations where E 1[z<0] = 1/2 — covariance stationarity requires

    alpha + gamma/2 + beta < 1,

with unconditional variance omega / (1 - alpha - gamma/2 - beta).

Constraints are enforced by a smooth reparameterisation: with
P = alpha + gamma/2 + beta in (0,1), split P into the average ARCH load
avg = (alpha + (alpha+gamma))/2 and beta via a sigmoid, then split avg between
the positive-shock load alpha >= 0 and negative-shock load alpha + gamma >= 0
via a second sigmoid. gamma is free to take either sign, so leverage-sign
recovery is a genuine test, while positivity and stationarity always hold.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from ._results import VolatilityFitResult
from ._utils import (
    ConvergenceError,
    checked_sample_variance,
    initial_variance,
    numerical_hessian,
    std_errors_from_hessian,
    validate_returns,
)
from .garch import _logit, _sigmoid, gaussian_loglik, student_t_loglik

__all__ = [
    "gjr_recursion",
    "fit_gjr",
    "gjr_persistence",
    "gjr_unconditional_variance",
    "news_impact_curve",
]

_PMAX = 0.9999
_MIN_OBS_FIT = 100


def gjr_recursion(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    init_var: float,
) -> np.ndarray:
    """Conditional variance path of GJR-GARCH(1,1), fully vectorised.

    Same linear-filter trick as GARCH: given returns, the effective ARCH
    coefficient alpha + gamma 1[r<0] is a known exogenous sequence, so the
    recursion is a first-order linear filter in sigma2.

    Pre-sample convention: r^2_{-1} = sigma2_{-1} = init_var with the
    indicator at its expectation 1/2, hence
    sigma2_0 = omega + (alpha + gamma/2 + beta) init_var (matches arch).
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    sigma2 = np.empty(n)
    sigma2[0] = omega + (alpha + 0.5 * gamma + beta) * init_var
    if n > 1:
        a_eff = alpha + gamma * (r[:-1] < 0)
        x = omega + a_eff * r[:-1] ** 2
        sigma2[1:] = lfilter([1.0], [1.0, -beta], x, zi=np.array([beta * sigma2[0]]))[0]
    return sigma2


def gjr_persistence(alpha: float, gamma: float, beta: float) -> float:
    """Persistence alpha + gamma/2 + beta (symmetric innovations)."""
    return float(alpha + 0.5 * gamma + beta)


def gjr_unconditional_variance(omega: float, alpha: float, gamma: float, beta: float) -> float:
    """Long-run daily variance omega / (1 - alpha - gamma/2 - beta).

    Raises ``ValueError`` if the stationarity condition
    alpha + gamma/2 + beta < 1 fails (no finite unconditional variance).
    """
    p = gjr_persistence(alpha, gamma, beta)
    if p >= 1.0:
        raise ValueError(
            f"alpha + gamma/2 + beta = {p:.6f} >= 1: no finite unconditional variance."
        )
    if omega <= 0:
        raise ValueError(f"omega must be > 0, got {omega}")
    return float(omega / (1.0 - p))


def fit_gjr(
    returns: np.ndarray,
    dist: str = "normal",
    init_method: str = "backcast",
    starting: dict[str, float] | None = None,
    raise_on_failure: bool = True,
) -> VolatilityFitResult:
    """Fit GJR-GARCH(1,1) by maximum likelihood.

    Parameters as :func:`eq_vol.garch.fit_garch`. ``extra`` fields:
    persistence (alpha + gamma/2 + beta), unconditional_variance,
    unconditional_vol_annual, halflife_days.
    """
    r = validate_returns(returns, min_obs=_MIN_OBS_FIT)
    sample_var = checked_sample_variance(r, "GJR-GARCH")
    if dist not in ("normal", "t"):
        raise ValueError(f"unknown dist {dist!r}; use 'normal' or 't'")
    b = initial_variance(r, init_method)

    s = {"omega": sample_var * 0.05, "alpha": 0.04, "gamma": 0.06, "beta": 0.88, "nu": 8.0}
    if starting:
        s.update(starting)

    def unpack(u: np.ndarray) -> tuple[float, ...]:
        omega = float(np.exp(u[0]))
        p = _PMAX * _sigmoid(u[1])
        avg = p * _sigmoid(u[2])          # (alpha + (alpha+gamma)) / 2
        beta = p - avg
        w = _sigmoid(u[3])
        alpha = 2.0 * avg * w             # load on positive shocks, >= 0
        a_neg = 2.0 * avg * (1.0 - w)     # load on negative shocks, >= 0
        gamma = a_neg - alpha
        if dist == "t":
            return omega, alpha, gamma, beta, 2.05 + float(np.exp(u[4]))
        return omega, alpha, gamma, beta

    p0 = gjr_persistence(s["alpha"], s["gamma"], s["beta"])
    avg0 = s["alpha"] + 0.5 * s["gamma"]
    u0 = [
        np.log(s["omega"]),
        _logit(p0 / _PMAX),
        _logit(avg0 / p0),
        _logit(s["alpha"] / (2.0 * avg0)),
    ]
    if dist == "t":
        u0.append(np.log(s["nu"] - 2.05))
    u0 = np.array(u0)

    def nll_theta(th) -> float:
        sigma2 = gjr_recursion(r, th[0], th[1], th[2], th[3], b)
        ll = (
            gaussian_loglik(r, sigma2)
            if dist == "normal"
            else student_t_loglik(r, sigma2, th[4])
        )
        return -ll if np.isfinite(ll) else 1e10

    def nll_u(u: np.ndarray) -> float:
        return nll_theta(unpack(u))

    opt = minimize(nll_u, u0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    theta = unpack(opt.x)
    if not opt.success and raise_on_failure:
        raise ConvergenceError(
            f"GJR-GARCH MLE failed to converge: {opt.message} "
            f"(nit={opt.nit}, final params={theta})"
        )

    names = ["omega", "alpha", "gamma", "beta"] + (["nu"] if dist == "t" else [])
    params = dict(zip(names, [float(x) for x in theta]))
    sigma2 = gjr_recursion(r, params["omega"], params["alpha"], params["gamma"], params["beta"], b)
    ll = (
        gaussian_loglik(r, sigma2)
        if dist == "normal"
        else student_t_loglik(r, sigma2, params["nu"])
    )

    se = std_errors_from_hessian(numerical_hessian(nll_theta, np.array(list(theta))))
    std_errors = dict(zip(names, [float(x) for x in se]))

    pers = gjr_persistence(params["alpha"], params["gamma"], params["beta"])
    extra = {"persistence": pers}
    if pers < 1:
        uv = gjr_unconditional_variance(params["omega"], params["alpha"], params["gamma"], params["beta"])
        extra["unconditional_variance"] = uv
        extra["unconditional_vol_annual"] = float(np.sqrt(uv * 252.0))
        extra["halflife_days"] = float(np.log(0.5) / np.log(pers))

    return VolatilityFitResult(
        model="GJR-GARCH",
        dist=dist,
        params=params,
        std_errors=std_errors,
        loglik=float(ll),
        n_obs=r.size,
        sigma2=sigma2,
        returns=r,
        converged=bool(opt.success),
        message=str(opt.message),
        init_var=b,
        extra=extra,
    )


def news_impact_curve(
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    z_grid: np.ndarray | None = None,
    base_var: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """News impact curve: next-period variance vs standardised shock z.

    sigma2(z) = omega + beta base_var + (alpha + gamma 1[z<0]) base_var z^2,
    anchoring the lagged variance and shock scale at ``base_var`` (default:
    the unconditional variance, Engle-Ng convention). With gamma > 0 the
    left branch (z < 0) lies above the right branch — the leverage effect.
    """
    if z_grid is None:
        z_grid = np.linspace(-5.0, 5.0, 201)
    z = np.asarray(z_grid, dtype=float)
    if base_var is None:
        base_var = gjr_unconditional_variance(omega, alpha, gamma, beta)
    a_eff = alpha + gamma * (z < 0)
    return z, omega + beta * base_var + a_eff * base_var * z**2
