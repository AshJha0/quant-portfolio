"""GARCH(1,1) implemented from scratch (Bollerslev 1986).

Model
-----
r_t = sigma_t z_t,  z_t ~ iid(0, 1) (Gaussian or standardised Student-t)
sigma2_t = omega + alpha r_{t-1}^2 + beta sigma2_{t-1}

with omega > 0, alpha, beta >= 0 and alpha + beta < 1 (covariance
stationarity). The constraints are enforced with smooth parameter transforms
so the optimiser (L-BFGS-B) works on an unconstrained space:

* omega = exp(u0)                        -> omega > 0
* P = sigmoid(u1) * 0.9999               -> persistence alpha + beta in (0, 1)
* alpha = P sigmoid(u2), beta = P - alpha -> alpha, beta >= 0
* nu = 2.05 + exp(u3) (Student-t only)   -> nu > 2 (finite variance)

Recursion initialisation: pre-sample values r^2_{-1} and sigma2_{-1} are both
set to a backcast ``b`` (arch-package-compatible exponentially weighted
backcast by default, or the sample variance), i.e. sigma2_0 = omega +
(alpha + beta) b. Matching arch's initialisation exactly is what makes the
cross-validation against the arch package tight (see tests).

Scaling convention: this package works in decimal returns. The arch package
recommends percent returns (x100); parameters map as omega_pct = 1e4 * omega,
alpha/beta invariant, and log-likelihoods differ by n ln(100).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
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

__all__ = [
    "garch_recursion",
    "gaussian_loglik",
    "student_t_loglik",
    "garch_loglik",
    "fit_garch",
    "persistence",
    "unconditional_variance",
    "vol_halflife",
]

_PMAX = 0.9999
_MIN_OBS_FIT = 100  # MLE on fewer daily observations is not meaningful


def garch_recursion(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    init_var: float,
) -> np.ndarray:
    """Conditional variance path of GARCH(1,1), fully vectorised.

    The recursion sigma2_t = (omega + alpha r_{t-1}^2) + beta sigma2_{t-1} is
    an AR(1) in sigma2 with exogenous input, i.e. a first-order linear filter,
    so the whole path is one :func:`scipy.signal.lfilter` call (no Python
    loop; this is what makes 20k-observation MLE fast).

    Pre-sample convention: r^2_{-1} = sigma2_{-1} = ``init_var``, hence
    sigma2_0 = omega + (alpha + beta) init_var.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    sigma2 = np.empty(n)
    sigma2[0] = omega + (alpha + beta) * init_var
    if n > 1:
        x = omega + alpha * r[:-1] ** 2
        sigma2[1:] = lfilter([1.0], [1.0, -beta], x, zi=np.array([beta * sigma2[0]]))[0]
    return sigma2


def gaussian_loglik(returns: np.ndarray, sigma2: np.ndarray) -> float:
    """Gaussian log-likelihood sum(-0.5 (ln 2 pi + ln sigma2_t + r_t^2/sigma2_t))."""
    if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)):
        return -np.inf
    return float(
        -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
    )


def student_t_loglik(returns: np.ndarray, sigma2: np.ndarray, nu: float) -> float:
    """Standardised Student-t log-likelihood (unit-variance t with nu > 2).

    Density of r = sigma * z where z is t_nu scaled to unit variance:
    ln f = lnGamma((nu+1)/2) - lnGamma(nu/2) - 0.5 ln(pi (nu-2))
           - 0.5 ln sigma2 - (nu+1)/2 ln(1 + z^2/(nu-2)).
    """
    if nu <= 2:
        return -np.inf
    if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)):
        return -np.inf
    z2 = returns**2 / sigma2
    c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0))
    return float(
        np.sum(c - 0.5 * np.log(sigma2) - 0.5 * (nu + 1.0) * np.log1p(z2 / (nu - 2.0)))
    )


def garch_loglik(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    init_var: float | None = None,
    dist: str = "normal",
    nu: float = np.nan,
    init_method: str = "backcast",
) -> float:
    """Log-likelihood of GARCH(1,1) at given natural parameters.

    Exposed for tests and for evaluating our likelihood at parameters fitted
    by the ``arch`` package (cross-validation of the recursion itself).
    """
    r = validate_returns(returns, min_obs=2)
    if init_var is None:
        init_var = initial_variance(r, init_method)
    sigma2 = garch_recursion(r, omega, alpha, beta, init_var)
    if dist == "normal":
        return gaussian_loglik(r, sigma2)
    if dist == "t":
        return student_t_loglik(r, sigma2, nu)
    raise ValueError(f"unknown dist {dist!r}; use 'normal' or 't'")


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(np.log(p / (1.0 - p)))


def _to_natural(u: np.ndarray, dist: str) -> tuple[float, ...]:
    omega = float(np.exp(u[0]))
    p = _PMAX * _sigmoid(u[1])
    alpha = p * _sigmoid(u[2])
    beta = p - alpha
    if dist == "t":
        return omega, alpha, beta, 2.05 + float(np.exp(u[3]))
    return omega, alpha, beta


def _from_natural(omega: float, alpha: float, beta: float, dist: str, nu: float = 8.0) -> np.ndarray:
    p = alpha + beta
    u = [np.log(omega), _logit(p / _PMAX), _logit(alpha / max(p, 1e-12))]
    if dist == "t":
        u.append(np.log(max(nu - 2.05, 1e-6)))
    return np.array(u)


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def fit_garch(
    returns: np.ndarray,
    dist: str = "normal",
    variance_targeting: bool = False,
    init_method: str = "backcast",
    starting: dict[str, float] | None = None,
    raise_on_failure: bool = True,
) -> VolatilityFitResult:
    """Fit GARCH(1,1) by maximum likelihood.

    Parameters
    ----------
    returns : array-like
        Daily log-returns, decimal units. At least 100 observations.
    dist : {"normal", "t"}
        Innovation distribution. Student-t adds a shape parameter ``nu > 2``
        (fat tails; standard for daily equity returns).
    variance_targeting : bool
        If True, omega is not estimated freely but pinned to
        ``sample_var * (1 - alpha - beta)`` so the model's unconditional
        variance matches the sample variance exactly. Reduces the parameter
        space (more stable in short samples) at a small efficiency cost;
        omega's standard error is reported as NaN in this mode.
    init_method : {"backcast", "sample"}
        Pre-sample variance convention (see module docstring).
    starting : dict, optional
        Starting values (keys omega/alpha/beta[/nu]) overriding the defaults.
    raise_on_failure : bool
        Raise :class:`ConvergenceError` if the optimiser fails (default);
        otherwise return the result flagged ``converged=False``.

    Returns
    -------
    VolatilityFitResult
        With ``extra`` fields: persistence, unconditional_variance (daily),
        unconditional_vol_annual, halflife_days.
    """
    r = validate_returns(returns, min_obs=_MIN_OBS_FIT)
    sample_var = checked_sample_variance(r, "GARCH")
    if dist not in ("normal", "t"):
        raise ValueError(f"unknown dist {dist!r}; use 'normal' or 't'")
    b = initial_variance(r, init_method)

    s = {"omega": sample_var * 0.05, "alpha": 0.05, "beta": 0.90, "nu": 8.0}
    if starting:
        s.update(starting)

    if variance_targeting:
        # optimise (persistence split) only; omega = sample_var * (1 - a - b)
        def unpack(u: np.ndarray) -> tuple[float, ...]:
            p = _PMAX * _sigmoid(u[0])
            alpha = p * _sigmoid(u[1])
            beta = p - alpha
            omega = sample_var * (1.0 - p)
            if dist == "t":
                return omega, alpha, beta, 2.05 + float(np.exp(u[2]))
            return omega, alpha, beta

        u0 = [_logit((s["alpha"] + s["beta"]) / _PMAX), _logit(s["alpha"] / (s["alpha"] + s["beta"]))]
        if dist == "t":
            u0.append(np.log(s["nu"] - 2.05))
        u0 = np.array(u0)
    else:
        unpack = lambda u: _to_natural(u, dist)  # noqa: E731
        u0 = _from_natural(s["omega"], s["alpha"], s["beta"], dist, s["nu"])

    def nll_u(u: np.ndarray) -> float:
        th = unpack(u)
        sigma2 = garch_recursion(r, th[0], th[1], th[2], b)
        ll = (
            gaussian_loglik(r, sigma2)
            if dist == "normal"
            else student_t_loglik(r, sigma2, th[3])
        )
        return -ll if np.isfinite(ll) else 1e10

    opt = minimize(nll_u, u0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    theta = unpack(opt.x)
    if not opt.success and raise_on_failure:
        raise ConvergenceError(
            f"GARCH MLE failed to converge: {opt.message} "
            f"(nit={opt.nit}, final params={theta})"
        )

    names = ["omega", "alpha", "beta"] + (["nu"] if dist == "t" else [])
    params = dict(zip(names, [float(x) for x in theta]))
    omega, alpha, beta = theta[0], theta[1], theta[2]
    sigma2 = garch_recursion(r, omega, alpha, beta, b)
    ll = gaussian_loglik(r, sigma2) if dist == "normal" else student_t_loglik(r, sigma2, theta[3])

    # standard errors: numerical Hessian of NLL in *natural* parameter space
    def nll_nat(th: np.ndarray) -> float:
        s2 = garch_recursion(r, th[0], th[1], th[2], b)
        llv = (
            gaussian_loglik(r, s2) if dist == "normal" else student_t_loglik(r, s2, th[3])
        )
        return -llv if np.isfinite(llv) else 1e10

    if variance_targeting:
        # free parameters are (alpha, beta[, nu]); omega is derived
        def nll_vt(th: np.ndarray) -> float:
            om = sample_var * (1.0 - th[0] - th[1])
            full = np.concatenate(([om], th))
            return nll_nat(full)

        free = np.array([params[k] for k in names[1:]])
        se = std_errors_from_hessian(numerical_hessian(nll_vt, free))
        std_errors = dict(zip(names[1:], [float(x) for x in se]))
        std_errors["omega"] = np.nan
    else:
        se = std_errors_from_hessian(numerical_hessian(nll_nat, np.array(list(theta))))
        std_errors = dict(zip(names, [float(x) for x in se]))

    pers = persistence(alpha, beta)
    extra = {"persistence": pers}
    if pers < 1:
        uv = unconditional_variance(omega, alpha, beta)
        extra["unconditional_variance"] = uv
        extra["unconditional_vol_annual"] = float(np.sqrt(uv * 252.0))
        extra["halflife_days"] = vol_halflife(alpha, beta)

    return VolatilityFitResult(
        model="GARCH",
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


# ---------------------------------------------------------------------------
# derived quantities
# ---------------------------------------------------------------------------

def persistence(alpha: float, beta: float) -> float:
    """GARCH(1,1) persistence alpha + beta (AR(1) coefficient of sigma2)."""
    return float(alpha + beta)


def unconditional_variance(omega: float, alpha: float, beta: float) -> float:
    """Long-run (unconditional) daily variance omega / (1 - alpha - beta).

    Raises
    ------
    ValueError
        If alpha + beta >= 1 (integrated/explosive: the unconditional
        variance does not exist — do not silently return a negative number).
    """
    p = alpha + beta
    if p >= 1.0:
        raise ValueError(
            f"alpha + beta = {p:.6f} >= 1: the process is integrated (IGARCH) "
            f"or explosive and has no finite unconditional variance."
        )
    if omega <= 0:
        raise ValueError(f"omega must be > 0, got {omega}")
    return float(omega / (1.0 - p))


def vol_halflife(alpha: float, beta: float) -> float:
    """Half-life (days) of a variance shock: ln(1/2) / ln(alpha + beta).

    The variance forecast deviation from the long-run level decays as
    (alpha + beta)^k; the half-life is where that factor hits 1/2.
    """
    p = alpha + beta
    if not 0.0 < p < 1.0:
        raise ValueError(f"half-life requires 0 < alpha + beta < 1, got {p}")
    return float(np.log(0.5) / np.log(p))
