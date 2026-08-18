"""EGARCH(1,1) implemented from scratch (Nelson 1991).

Model
-----
r_t = sigma_t z_t,  z_t ~ iid(0, 1)
ln sigma2_t = omega + beta ln sigma2_{t-1}
              + alpha (|z_{t-1}| - E|z|) + gamma z_{t-1}

Sign convention (used consistently in this package and tested):
**gamma < 0 produces the leverage effect** — a negative standardised shock
z < 0 raises next-period log-variance by alpha(|z| - E|z|) - gamma|z|, which
exceeds the response to +z whenever gamma < 0. (This mirrors the empirical
finding for equities; note GJR uses the opposite sign convention, gamma > 0.)

Why no positivity constraints are needed: the recursion is written in
**log-variance**, so sigma2_t = exp(ln sigma2_t) > 0 for *any* real
(omega, alpha, gamma, beta). This is EGARCH's key structural advantage over
GARCH/GJR, where omega > 0, alpha >= 0, beta >= 0 (and alpha + gamma >= 0)
must be imposed to keep the variance positive. The only constraint we impose
is |beta| < 1 for stationarity of ln sigma2 (an L-BFGS-B box bound).

The recursion cannot be vectorised with a linear filter because the input
z_{t-1} = r_{t-1}/sigma_{t-1} depends on the lagged output; it is computed in
an optimised scalar loop.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from ._results import VolatilityFitResult
from ._utils import (
    ConvergenceError,
    checked_sample_variance,
    initial_variance,
    numerical_hessian,
    std_errors_from_hessian,
    validate_returns,
)
from .garch import gaussian_loglik, student_t_loglik

__all__ = [
    "egarch_recursion",
    "fit_egarch",
    "egarch_unconditional_logvar",
    "news_impact_curve",
    "expected_abs_z",
]

_BMAX = 0.9999
_MIN_OBS_FIT = 100
_LOG_CLIP = 60.0  # |ln sigma2| cap to keep exp() finite during optimisation


def expected_abs_z(dist: str = "normal", nu: float = np.nan) -> float:
    """E|z| for a unit-variance innovation.

    Gaussian: sqrt(2/pi). Standardised Student-t(nu):
    2 sqrt(nu-2) Gamma((nu+1)/2) / (sqrt(pi) (nu-1) Gamma(nu/2)).
    """
    if dist == "normal":
        return math.sqrt(2.0 / math.pi)
    if dist == "t":
        if nu <= 2:
            raise ValueError("Student-t requires nu > 2")
        return float(
            2.0
            * math.sqrt(nu - 2.0)
            * math.exp(gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0))
            / (math.sqrt(math.pi) * (nu - 1.0))
        )
    raise ValueError(f"unknown dist {dist!r}")


def egarch_recursion(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    init_var: float,
    e_abs_z: float | None = None,
) -> np.ndarray:
    """Conditional variance path of EGARCH(1,1).

    Initialisation: ln sigma2_0 = ln(init_var) (pre-sample variance level;
    with |beta| < 1 the effect of the initial condition dies off
    geometrically and is negligible for the sample sizes used here).
    """
    r = np.asarray(returns, dtype=float)
    if e_abs_z is None:
        e_abs_z = math.sqrt(2.0 / math.pi)
    n = r.size
    out = np.empty(n)
    r_list = r.tolist()
    ls = math.log(init_var)
    exp_, log_, sqrt_ = math.exp, math.log, math.sqrt
    clip = _LOG_CLIP
    for t in range(n):
        if t > 0:
            ls = omega + beta * ls + alpha * (abs(z) - e_abs_z) + gamma * z  # noqa: F821
            if ls > clip:
                ls = clip
            elif ls < -clip:
                ls = -clip
        s2 = exp_(ls)
        out[t] = s2
        z = r_list[t] / sqrt_(s2)
    return out


def egarch_unconditional_logvar(omega: float, beta: float) -> float:
    """Unconditional mean of ln sigma2: omega / (1 - beta).

    Raises ``ValueError`` when |beta| >= 1 (non-stationary log-variance).
    Note exp(E[ln sigma2]) understates E[sigma2] by Jensen's inequality; it is
    the *median-like* level, which is what the news impact curve is anchored
    to by convention.
    """
    if abs(beta) >= 1:
        raise ValueError(f"|beta| = {abs(beta):.6f} >= 1: ln sigma2 is non-stationary")
    return float(omega / (1.0 - beta))


def fit_egarch(
    returns: np.ndarray,
    dist: str = "normal",
    init_method: str = "backcast",
    starting: dict[str, float] | None = None,
    raise_on_failure: bool = True,
) -> VolatilityFitResult:
    """Fit EGARCH(1,1) by maximum likelihood.

    Only |beta| < 1 is enforced (box bound); omega, alpha, gamma are
    genuinely unconstrained — see module docstring for why positivity needs
    no constraints under the log specification.

    Returns
    -------
    VolatilityFitResult
        ``extra``: persistence (= beta), unconditional_logvar,
        unconditional_vol_annual (from exp of the mean log-variance),
        halflife_days (of a shock to ln sigma2).
    """
    r = validate_returns(returns, min_obs=_MIN_OBS_FIT)
    sample_var = checked_sample_variance(r, "EGARCH")
    if dist not in ("normal", "t"):
        raise ValueError(f"unknown dist {dist!r}; use 'normal' or 't'")
    b = initial_variance(r, init_method)

    beta0 = 0.95
    s = {
        "omega": (1.0 - beta0) * math.log(sample_var),
        "alpha": 0.10,
        "gamma": -0.05,
        "beta": beta0,
        "nu": 8.0,
    }
    if starting:
        s.update(starting)

    def nll_theta(th) -> float:
        nu = th[4] if dist == "t" else np.nan
        if dist == "t" and nu <= 2.05:
            return 1e10
        eaz = expected_abs_z(dist, nu)
        sigma2 = egarch_recursion(r, th[0], th[1], th[2], th[3], b, eaz)
        ll = gaussian_loglik(r, sigma2) if dist == "normal" else student_t_loglik(r, sigma2, nu)
        return -ll if np.isfinite(ll) else 1e10

    # Optimise directly in natural parameters: the log specification needs no
    # positivity constraints, so the only box constraint is |beta| < 1.
    # SLSQP handles the explosive far-field of the EGARCH likelihood far more
    # robustly than L-BFGS-B's line search (verified empirically); a final
    # L-BFGS-B polish from the SLSQP optimum tightens the solution. The
    # objective is scaled to the *average* NLL so tolerances are sample-size
    # independent.
    u0 = [s["omega"], s["alpha"], s["gamma"], min(max(s["beta"], -_BMAX), _BMAX)]
    bounds = [(None, None), (None, None), (None, None), (-_BMAX, _BMAX)]
    if dist == "t":
        u0.append(s["nu"])
        bounds.append((2.1, 200.0))
    u0 = np.array(u0)

    n_obs = r.size

    def avg_nll(th) -> float:
        return nll_theta(th) / n_obs

    opt = minimize(
        avg_nll, u0, method="SLSQP", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    polish = minimize(
        avg_nll, opt.x, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 200},
    )
    if polish.fun <= opt.fun:
        best, best_success = polish, polish.success or opt.success
    else:
        best, best_success = opt, opt.success
    opt = best
    opt.success = best_success
    theta = tuple(float(x) for x in opt.x)
    if not opt.success and raise_on_failure:
        raise ConvergenceError(
            f"EGARCH MLE failed to converge: {opt.message} "
            f"(nit={opt.nit}, final params={theta})"
        )

    names = ["omega", "alpha", "gamma", "beta"] + (["nu"] if dist == "t" else [])
    params = dict(zip(names, [float(x) for x in theta]))
    nu = params.get("nu", np.nan)
    eaz = expected_abs_z(dist, nu)
    sigma2 = egarch_recursion(r, params["omega"], params["alpha"], params["gamma"], params["beta"], b, eaz)
    ll = gaussian_loglik(r, sigma2) if dist == "normal" else student_t_loglik(r, sigma2, nu)

    se = std_errors_from_hessian(numerical_hessian(nll_theta, np.array(list(theta))))
    std_errors = dict(zip(names, [float(x) for x in se]))

    extra: dict[str, float] = {"persistence": params["beta"]}
    if abs(params["beta"]) < 1:
        ulv = egarch_unconditional_logvar(params["omega"], params["beta"])
        extra["unconditional_logvar"] = ulv
        extra["unconditional_vol_annual"] = float(math.sqrt(math.exp(ulv) * 252.0))
        if 0 < params["beta"] < 1:
            extra["halflife_days"] = float(math.log(0.5) / math.log(params["beta"]))

    return VolatilityFitResult(
        model="EGARCH",
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
    dist: str = "normal",
    nu: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """News impact curve: next-period variance vs standardised shock z.

    sigma2(z) = exp(omega + beta ln(base_var) + alpha(|z| - E|z|) + gamma z),
    with the lagged variance anchored at ``base_var`` (default: the
    unconditional level exp(omega/(1-beta)), Engle-Ng convention).

    With gamma < 0 the curve is steeper for z < 0 than for z > 0
    (asymmetric response = leverage effect).

    Returns
    -------
    (z_grid, sigma2) : tuple of numpy.ndarray
    """
    if z_grid is None:
        z_grid = np.linspace(-5.0, 5.0, 201)
    z = np.asarray(z_grid, dtype=float)
    if base_var is None:
        base_var = math.exp(egarch_unconditional_logvar(omega, beta))
    eaz = expected_abs_z(dist, nu)
    log_s2 = omega + beta * math.log(base_var) + alpha * (np.abs(z) - eaz) + gamma * z
    return z, np.exp(log_s2)
