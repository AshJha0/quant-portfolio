"""Kalman-filter hedge ratio with the noise variances estimated by maximum likelihood.

Plain idea
----------
The hedge ratio between two prices drifts. Model it as a hidden state that
takes a small random step each day, and the observed price as the state times
the hedge leg plus noise. The Kalman filter gives the best estimate of the
state each day from the past only. How fast the estimate moves depends on the
ratio of the state noise to the observation noise. Here both are estimated
from the data by maximum likelihood, instead of being tied to one forgetting
factor as in recursive least squares (``spread.rls_hedge_ratio``).

Formal statement
----------------
State ``theta_t = (alpha_t, beta_t)'`` (or ``beta_t`` alone without an
intercept), ``h_t = (1, x_t)``:

    y_t     = h_t theta_t + e_t,           e_t ~ N(0, R)
    theta_t = theta_{t-1} + w_t,           w_t ~ N(0, diag(q_alpha, q_beta))
    theta_0 ~ N(0, P_0)

Filter (Harvey 1989, Forecasting, Structural Time Series Models and the
Kalman Filter, section 3.2; Durbin and Koopman 2012, Time Series Analysis by
State Space Methods, section 4.3):

    P_{t|t-1} = P_{t-1} + Q
    v_t = y_t - h_t theta_{t-1},   F_t = h_t P_{t|t-1} h_t' + R
    K_t = P_{t|t-1} h_t' / F_t
    theta_t = theta_{t-1} + K_t v_t,   P_t = P_{t|t-1} - K_t h_t P_{t|t-1}

Log-likelihood by the prediction-error decomposition (Durbin and Koopman,
section 7.2): ``-0.5 sum_{t > burn} (log 2 pi + log F_t + v_t^2 / F_t)``.
With a diffuse prior the first ``k`` (number of states) terms are dropped.
``(R, q)`` maximise it over log-variances (L-BFGS-B).

Worked example
--------------
With ``Q = 0`` and a diffuse prior the filter is recursive least squares
with no forgetting, so after the burn-in ``beta_t`` equals the expanding
OLS slope (``test_kalman.py``). The likelihood is also checked against the
exact multivariate normal density of ``y``, whose covariance is
``h_s (P_0 + min(s, t) Q) h_t' + R 1{s = t}`` (1-based s, t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize

__all__ = ["KalmanHedgeResult", "kalman_filter", "kalman_hedge_ratio", "kalman_loglik"]

_LOG2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class KalmanHedgeResult:
    """Filtered path and the variances used.

    ``beta``/``alpha`` are filtered after seeing t (use t-1 to trade t).
    ``error`` is the one-step prediction error ``v_t`` and ``error_var`` its
    variance ``F_t``, so ``error / sqrt(error_var)`` is a standardised spread.
    """

    beta: pd.Series
    alpha: pd.Series | None
    error: pd.Series
    error_var: pd.Series
    obs_var: float
    state_var: tuple[float, ...]
    loglik: float
    estimated: bool


def _design(x: np.ndarray, intercept: bool) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x]) if intercept else x[:, None]


def kalman_filter(y: np.ndarray, H: np.ndarray, obs_var: float, state_var: np.ndarray,
                  prior_var: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the filter. Returns (filtered states n x k, errors v, variances F)."""
    n, k = H.shape
    q = np.diag(state_var)
    theta = np.zeros(k)
    p = prior_var * np.eye(k)
    states = np.empty((n, k))
    v = np.empty(n)
    f = np.empty(n)
    for t in range(n):
        h = H[t]
        p = p + q
        ph = p @ h
        f[t] = h @ ph + obs_var
        v[t] = y[t] - h @ theta
        gain = ph / f[t]
        theta = theta + gain * v[t]
        p = p - np.outer(gain, ph)
        p = 0.5 * (p + p.T)  # keep it symmetric in floating point
        states[t] = theta
    return states, v, f


def kalman_loglik(y: ArrayLike, x: ArrayLike, obs_var: float, state_var: ArrayLike,
                  intercept: bool = True, prior_var: float = 1e7, burn: int | None = None) -> float:
    """Gaussian log-likelihood by the prediction-error decomposition."""
    yv, xv = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    H = _design(xv, intercept)
    _, v, f = kalman_filter(yv, H, float(obs_var), np.asarray(state_var, dtype=float), prior_var)
    b = H.shape[1] if burn is None else int(burn)
    return float(-0.5 * np.sum(_LOG2PI + np.log(f[b:]) + v[b:] ** 2 / f[b:]))


def kalman_hedge_ratio(
    y: ArrayLike,
    x: ArrayLike,
    obs_var: float | None = None,
    state_var: ArrayLike | None = None,
    intercept: bool = True,
    prior_var: float = 1e7,
    burn: int | None = None,
) -> KalmanHedgeResult:
    """Kalman hedge ratio of ``y`` on ``x``; noise variances by MLE unless given.

    Parameters
    ----------
    y, x : array-like (n,)
        Dependent and hedge legs (prices or log prices).
    obs_var, state_var : optional
        Fix ``R`` and ``diag(Q)`` instead of estimating them. Both or neither.
    intercept : bool
        Include a random-walk intercept state.
    prior_var : float
        ``P_0 = prior_var I``; the default is effectively diffuse.
    burn : int, optional
        Number of first prediction errors left out of the likelihood
        (default: the number of states, the diffuse-prior convention).
    """
    idx = y.index if isinstance(y, pd.Series) else None
    yv, xv = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if yv.ndim != 1 or yv.shape != xv.shape:
        raise ValueError("y and x must be equal-length 1-D arrays")
    if not (np.all(np.isfinite(yv)) and np.all(np.isfinite(xv))):
        raise ValueError("y and x must be finite")
    H = _design(xv, intercept)
    k = H.shape[1]
    if len(yv) < 3 * k + 10:
        raise ValueError(f"need at least {3 * k + 10} observations, got {len(yv)}")
    if (obs_var is None) != (state_var is None):
        raise ValueError("give both obs_var and state_var, or neither")
    if obs_var is None:
        # start: R from the OLS residual variance, Q a small share of it
        coef, *_ = np.linalg.lstsq(H, yv, rcond=None)
        r0 = max(float(np.var(yv - H @ coef)), 1e-12)
        x0 = np.log(np.r_[r0, np.full(k, r0 * 1e-4)])

        def nll(z: np.ndarray) -> float:
            e = np.exp(np.clip(z, -60.0, 30.0))
            return -kalman_loglik(yv, xv, e[0], e[1:], intercept, prior_var, burn)

        res = minimize(nll, x0, method="L-BFGS-B", bounds=[(-60.0, 30.0)] * (k + 1))
        est = np.exp(res.x)
        r_hat, q_hat, estimated = float(est[0]), est[1:], True
    else:
        r_hat = float(obs_var)
        q_hat = np.asarray(state_var, dtype=float).ravel()
        if r_hat <= 0 or q_hat.shape != (k,) or np.any(q_hat < 0):
            raise ValueError(f"need obs_var > 0 and {k} non-negative state variances")
        estimated = False
    states, v, f = kalman_filter(yv, H, r_hat, q_hat, prior_var)
    ll = kalman_loglik(yv, xv, r_hat, q_hat, intercept, prior_var, burn)
    beta = pd.Series(states[:, -1], index=idx, name="beta")
    alpha = pd.Series(states[:, 0], index=idx, name="alpha") if intercept else None
    return KalmanHedgeResult(beta=beta, alpha=alpha, error=pd.Series(v, index=idx, name="error"),
                             error_var=pd.Series(f, index=idx, name="error_var"),
                             obs_var=r_hat, state_var=tuple(float(q) for q in q_hat),
                             loglik=ll, estimated=estimated)
