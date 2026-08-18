"""Spread construction, Ornstein-Uhlenbeck fitting, and adaptive hedge ratios.

Model
-----
The (dollar) spread s_t = y_t - beta x_t - alpha is modelled as an
Ornstein-Uhlenbeck process

    ds = kappa (mu - s) dt + sigma dW,

whose exact discretisation at spacing dt is the AR(1)

    s_{t+1} = c + b s_t + eps_t,   b = e^{-kappa dt},
    c = mu (1 - b),  Var(eps) = sigma^2 (1 - b^2) / (2 kappa).

Conventions: dt is in trading days (default 1.0), so kappa is per day and
half-life = ln(2)/kappa is in trading days. Stationary standard deviation is
sigma / sqrt(2 kappa) in dollars.

Two estimators are provided and cross-validated in tests:

* :func:`fit_ou_ols` — OLS on the discretised AR(1) (closed form).
* :func:`fit_ou_mle` — exact Gaussian maximum likelihood (numerical),
  conditioning on s_0. For the AR(1) with freely-varying (c, b, variance)
  the conditional MLE point estimates of c and b coincide with OLS; the MLE
  differs only in the variance normalisation (1/n vs 1/(n-2)), so the two
  must agree closely — a useful implementation check.

Adaptive hedge ratio ("Kalman-lite")
------------------------------------
:func:`rls_hedge_ratio` implements exponentially-weighted recursive least
squares: at each step the coefficient estimate minimises
sum_j lambda^{t-j} (y_j - a - b x_j)^2. This is algebraically the Kalman
filter for a random-walk-coefficient state-space model in the special case
where the implied state noise is tied to the forgetting factor, rather than
estimated. A full Kalman filter adds: an explicit observation/state noise
ratio (two extra hyperparameters to calibrate), proper state covariance
dynamics, and smoothing. RLS keeps one interpretable knob (lambda, an
effective memory of ~1/(1-lambda) observations) and is O(1) per step, which
is why desks often prefer it for production hedge-ratio tracking; see
docs/METHODOLOGY.md for the trade-off discussion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "compute_spread",
    "OUFit",
    "fit_ou_ols",
    "fit_ou_mle",
    "half_life_from_kappa",
    "rls_hedge_ratio",
    "rolling_ols_hedge_ratio",
]

ArrayLike = Union[np.ndarray, pd.Series, list]

#: AR(1) coefficient at/above which the spread is treated as non-mean-reverting.
_UNIT_ROOT_B = 1.0 - 1e-10


def compute_spread(
    y: ArrayLike, x: ArrayLike, beta: float, alpha: float = 0.0
) -> np.ndarray:
    """Dollar spread s_t = y_t - beta x_t - alpha.

    Accepts pandas Series (returns Series preserving the index) or arrays.
    """
    if isinstance(y, pd.Series) and isinstance(x, pd.Series):
        if not y.index.equals(x.index):
            raise ValueError("y and x indices differ; align prices first")
        return y - beta * x - alpha
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) != len(x):
        raise ValueError(f"length mismatch: {len(y)} vs {len(x)}")
    return y - beta * x - alpha


@dataclass(frozen=True)
class OUFit:
    """Fitted Ornstein-Uhlenbeck parameters.

    Attributes
    ----------
    kappa : float
        Mean-reversion speed per day (inf-half-life when ~0).
    mu : float
        Long-run mean (dollars).
    sigma : float
        Diffusion volatility, dollars per sqrt(day).
    half_life : float
        ln(2)/kappa in trading days; ``inf`` when the AR(1) coefficient is
        >= 1 (no mean reversion detected).
    b : float
        Fitted AR(1) coefficient e^{-kappa dt}.
    dt : float
        Sampling interval used (days).
    method : str
        "ols" or "mle".
    mean_reverting : bool
        False when the fit found b >= 1 (random walk or explosive); kappa is
        then 0 and downstream signal logic must not trade the pair.
    """

    kappa: float
    mu: float
    sigma: float
    half_life: float
    b: float
    dt: float
    method: str
    mean_reverting: bool

    @property
    def stationary_std(self) -> float:
        """Stationary standard deviation sigma/sqrt(2 kappa), dollars."""
        if not self.mean_reverting:
            return np.inf
        return self.sigma / np.sqrt(2.0 * self.kappa)


def half_life_from_kappa(kappa: float, dt_units: float = 1.0) -> float:
    """Half-life of mean reversion, ln(2)/kappa, in the units of 1/kappa.

    kappa <= 0 returns ``inf`` (no mean reversion).
    """
    if kappa <= 0.0:
        return np.inf
    return float(np.log(2.0) / kappa) * dt_units


def _ou_from_ar1(c: float, b: float, resid_var: float, dt: float, method: str) -> OUFit:
    """Map AR(1) estimates (c, b, Var(eps)) to OU parameters."""
    if b <= 0.0:
        # negative AR coefficient: not an OU discretisation; treat as very
        # fast mean reversion with kappa from |b| floor to keep maths finite
        raise ValueError(
            f"AR(1) coefficient b={b:.4f} <= 0: series is not OU-like "
            "(oscillatory); inspect the spread before fitting"
        )
    if b >= _UNIT_ROOT_B:
        # random walk / explosive: flag, do not fabricate a half-life
        mu = np.nan
        sigma = float(np.sqrt(max(resid_var, 0.0) / dt))
        return OUFit(
            kappa=0.0,
            mu=mu,
            sigma=sigma,
            half_life=np.inf,
            b=float(b),
            dt=dt,
            method=method,
            mean_reverting=False,
        )
    kappa = -np.log(b) / dt
    mu = c / (1.0 - b)
    sigma2 = resid_var * 2.0 * kappa / (1.0 - b**2)
    return OUFit(
        kappa=float(kappa),
        mu=float(mu),
        sigma=float(np.sqrt(max(sigma2, 0.0))),
        half_life=half_life_from_kappa(kappa),
        b=float(b),
        dt=dt,
        method=method,
        mean_reverting=True,
    )


def fit_ou_ols(spread: ArrayLike, dt: float = 1.0) -> OUFit:
    """Fit OU by OLS on the discretised AR(1) s_{t+1} = c + b s_t + eps.

    Parameters
    ----------
    spread : array-like (n,)
        Spread observations (dollars), n >= 10.
    dt : float
        Sampling interval in days (default 1.0).

    Returns
    -------
    OUFit
    """
    s = np.asarray(spread, dtype=float)
    if s.ndim != 1 or len(s) < 10:
        raise ValueError(f"need a 1-D spread with n >= 10, got n={s.size}")
    if np.any(~np.isfinite(s)):
        raise ValueError("spread contains NaN/inf")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if np.std(s) == 0.0:
        raise ValueError("spread has zero variance; OU fit undefined")
    y, x = s[1:], s[:-1]
    X = np.column_stack([np.ones_like(x), x])
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    c, b = float(params[0]), float(params[1])
    resid = y - X @ params
    resid_var = float(resid @ resid) / (len(y) - 2)
    return _ou_from_ar1(c, b, resid_var, dt, method="ols")


def fit_ou_mle(spread: ArrayLike, dt: float = 1.0) -> OUFit:
    """Fit OU by exact Gaussian MLE (conditional on s_0), numerically.

    Maximises the exact transition likelihood
    s_{t+1} | s_t ~ N(mu + (s_t - mu) e^{-kappa dt}, sigma^2 (1-e^{-2 kappa
    dt})/(2 kappa)) over (kappa, mu, sigma) with Nelder-Mead, warm-started
    at the OLS estimate. Falls back to the OLS result (with the
    non-mean-reverting flag) when OLS already finds b >= 1.

    Returns
    -------
    OUFit
    """
    s = np.asarray(spread, dtype=float)
    start = fit_ou_ols(s, dt=dt)
    if not start.mean_reverting:
        return OUFit(**{**start.__dict__, "method": "mle"})

    y, x = s[1:], s[:-1]
    n = len(y)

    def nll(theta: np.ndarray) -> float:
        log_kappa, mu, log_sigma = theta
        kappa = np.exp(log_kappa)
        sigma = np.exp(log_sigma)
        b = np.exp(-kappa * dt)
        v = sigma**2 * (1.0 - b**2) / (2.0 * kappa)
        if v <= 0 or not np.isfinite(v):
            return 1e12
        mean = mu + (x - mu) * b
        z = y - mean
        return 0.5 * n * np.log(2.0 * np.pi * v) + 0.5 * float(z @ z) / v

    theta0 = np.array([np.log(start.kappa), start.mu, np.log(start.sigma)])
    res = minimize(nll, theta0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    log_kappa, mu, log_sigma = res.x
    kappa = float(np.exp(log_kappa))
    sigma = float(np.exp(log_sigma))
    b = float(np.exp(-kappa * dt))
    return OUFit(
        kappa=kappa,
        mu=float(mu),
        sigma=sigma,
        half_life=half_life_from_kappa(kappa),
        b=b,
        dt=dt,
        method="mle",
        mean_reverting=True,
    )


def rls_hedge_ratio(
    y: ArrayLike,
    x: ArrayLike,
    lam: float = 0.995,
    delta: float = 1e4,
    intercept: bool = True,
) -> pd.DataFrame:
    """Recursive least squares hedge ratio with forgetting factor (Kalman-lite).

    At each t the estimate minimises sum_{j<=t} lam^{t-j} (y_j - a - b x_j)^2,
    updated in O(1) per observation:

        k_t   = P_{t-1} z_t / (lam + z_t' P_{t-1} z_t)
        theta = theta + k_t (y_t - z_t' theta)
        P_t   = (P_{t-1} - k_t z_t' P_{t-1}) / lam

    Parameters
    ----------
    y, x : array-like (n,)
        Dependent and hedge legs (price levels, dollars).
    lam : float
        Forgetting factor in (0, 1]; effective memory ~ 1/(1-lam)
        observations (lam=1 reproduces expanding-window OLS).
    delta : float
        Initial covariance scale P_0 = delta I (large = diffuse prior).
    intercept : bool
        Estimate an intercept alongside the slope. CAVEAT: with a short
        effective memory, prices vary little relative to their level within
        the memory window, so the constant and the slope become nearly
        collinear and the slope estimate degrades badly. For hedge-ratio
        tracking prefer ``intercept=False`` (let the OU mean mu absorb the
        level) or a long memory (lam close to 1). This identification issue
        is demonstrated in tests/test_spread_ou.py.

    Returns
    -------
    DataFrame with columns ``beta`` (and ``alpha`` when intercept) indexed
    like ``y`` if it is a Series, else RangeIndex. The first few estimates
    are prior-dominated; discard a burn-in before trading on them.
    """
    if not 0.0 < lam <= 1.0:
        raise ValueError(f"lam must be in (0, 1], got {lam}")
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    idx = y.index if isinstance(y, pd.Series) else None
    yv = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)
    if len(yv) != len(xv):
        raise ValueError(f"length mismatch: {len(yv)} vs {len(xv)}")
    n = len(yv)
    k = 2 if intercept else 1
    theta = np.zeros(k)
    P = np.eye(k) * delta
    betas = np.empty(n)
    alphas = np.empty(n)
    for t in range(n):
        z = np.array([1.0, xv[t]]) if intercept else np.array([xv[t]])
        Pz = P @ z
        denom = lam + z @ Pz
        gain = Pz / denom
        err = yv[t] - z @ theta
        theta = theta + gain * err
        P = (P - np.outer(gain, Pz)) / lam
        if intercept:
            alphas[t], betas[t] = theta[0], theta[1]
        else:
            alphas[t], betas[t] = 0.0, theta[0]
    out = {"beta": betas}
    if intercept:
        out["alpha"] = alphas
    return pd.DataFrame(out, index=idx)


def rolling_ols_hedge_ratio(
    y: pd.Series, x: pd.Series, window: int, intercept: bool = True
) -> pd.DataFrame:
    """Rolling-window OLS hedge ratio (the static-per-window alternative).

    Parameters
    ----------
    y, x : pandas.Series
        Price levels, identical index.
    window : int
        Window length in observations (>= 3).

    Returns
    -------
    DataFrame with ``beta`` (and ``alpha``); NaN during the warm-up
    (first window-1 rows).
    """
    if window < 3:
        raise ValueError(f"window must be >= 3, got {window}")
    if not y.index.equals(x.index):
        raise ValueError("y and x indices differ; align prices first")
    n = len(y)
    betas = np.full(n, np.nan)
    alphas = np.full(n, np.nan)
    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    for t in range(window - 1, n):
        ys = yv[t - window + 1 : t + 1]
        xs = xv[t - window + 1 : t + 1]
        if np.std(xs) == 0.0:
            continue
        if intercept:
            X = np.column_stack([np.ones(window), xs])
            p, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
            alphas[t], betas[t] = p[0], p[1]
        else:
            betas[t] = float(xs @ ys / (xs @ xs))
            alphas[t] = 0.0
    out = {"beta": betas}
    if intercept:
        out["alpha"] = alphas
    return pd.DataFrame(out, index=y.index)
