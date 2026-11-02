"""Models the Phase 4 studies need that the packages do not have yet.

- ``johansen_trace``: Johansen (1991) trace test, constant unrestricted.
- ``kalman_hedge``: random-walk hedge ratio by the Kalman filter.
- ``training_only``: the hold-out guard every study calls on its data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOLDOUT_START = pd.Timestamp("2022-01-01")

# Trace critical values (90 %, 95 %, 99 %), constant in the VAR (statsmodels
# det_order=0), from MacKinnon, Haug and Michelis (1999), J. Appl. Econ. 14,
# table II, as tabulated in statsmodels ``c_sjt``. Row r = hypothesised rank
# for a 2-variable system.
TRACE_CV_2VAR = np.array([[13.4294, 15.4943, 19.9349], [2.7055, 3.8415, 6.6349]])


class HoldoutViolation(RuntimeError):
    """A study was handed data on or after 2022-01-01 without a signature."""


def training_only(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Raise if ``obj`` has any row on or after the hold-out start."""
    idx = pd.DatetimeIndex(obj.index)
    if len(idx) and idx.max() >= HOLDOUT_START:
        raise HoldoutViolation(f"data reaches {idx.max().date()}; hold-out starts {HOLDOUT_START.date()}")
    return obj


def _demean(a: np.ndarray) -> np.ndarray:
    return a - a.mean(axis=0) if a.size else a


def _resid(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    if z.shape[1] == 0:
        return y
    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
    return y - z @ coef


@dataclass(frozen=True)
class JohansenResult:
    trace: np.ndarray  # trace statistic for r = 0, 1, ...
    eigenvalues: np.ndarray
    cv95: np.ndarray

    @property
    def rank(self) -> int:
        """Largest r not rejected, testing r = 0, 1, ... in order at 95 %."""
        r = 0
        while r < len(self.trace) and self.trace[r] > self.cv95[r]:
            r += 1
        return r


def johansen_trace(x: np.ndarray, k_ar_diff: int = 1) -> JohansenResult:
    """Johansen trace test with an unrestricted constant.

    Reduced-rank regression of Johansen (1991), Econometrica 59(6), 1551-1580,
    section 2: concentrate out the lagged differences, form the moment
    matrices S00, S0k, Skk of the residuals R0 (from dx_t) and Rk (from
    x_{t-1}), solve |lambda Skk - Sk0 S00^-1 S0k| = 0, and
    trace(r) = -T sum_{i>r} log(1 - lambda_i) (eq. 2.8). The sample
    alignment follows statsmodels ``coint_johansen(x, 0, k_ar_diff)``, the
    oracle in the tests.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("johansen_trace supports two-variable systems")
    if not np.isfinite(x).all() or len(x) < 30 + k_ar_diff:
        raise ValueError("need at least 30 finite observations")
    k = k_ar_diff
    x = _demean(x)
    dx = np.diff(x, axis=0)
    z = np.column_stack([dx[k - j - 1 : len(dx) - j - 1] for j in range(k)]) if k else np.empty((len(dx), 0))
    dx = _demean(dx[k:])
    z = _demean(z)
    lx = _demean(x[: len(x) - k][1:])
    r0, rk = _resid(z, dx), _resid(z, lx)
    t = r0.shape[0]
    s00, s0k, skk = r0.T @ r0 / t, r0.T @ rk / t, rk.T @ rk / t
    m = np.linalg.solve(skk, s0k.T @ np.linalg.solve(s00, s0k))
    lam = np.sort(np.real(np.linalg.eigvals(m)))[::-1]
    lam = np.clip(lam, 0.0, 1.0 - 1e-15)
    trace = np.array([-t * np.log1p(-lam[r:]).sum() for r in range(2)])
    return JohansenResult(trace=trace, eigenvalues=lam, cv95=TRACE_CV_2VAR[:, 1])


@dataclass(frozen=True)
class KalmanHedge:
    alpha: np.ndarray  # filtered intercept after observing t
    beta: np.ndarray  # filtered hedge ratio after observing t
    error: np.ndarray  # one-step prediction error y_t - [1, x_t] theta_{t|t-1}
    error_var: np.ndarray  # its forecast variance


def kalman_hedge(
    y: np.ndarray,
    x: np.ndarray,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
    prior_var: float = 1e6,
) -> KalmanHedge:
    """Kalman filter for y_t = alpha_t + beta_t x_t + e_t with
    (alpha, beta) a random walk of covariance delta / (1 - delta) I.

    State-space form and recursions: Harvey (1989), Forecasting, Structural
    Time Series Models and the Kalman Filter, section 3.2; the delta
    parameterisation is Chan (2013), Algorithmic Trading, chapter 3. With
    delta = 0 and a diffuse prior the filter equals recursive least squares,
    which is the oracle test.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("y and x must be equal-length 1-D arrays")
    if not (0.0 <= delta < 1.0) or obs_var <= 0:
        raise ValueError("need 0 <= delta < 1 and obs_var > 0")
    n = len(y)
    q = delta / (1.0 - delta) * np.eye(2)
    theta = np.zeros(2)
    p = prior_var * np.eye(2)
    out = np.empty((n, 4))
    for t in range(n):
        h = np.array([1.0, x[t]])
        p = p + q
        e = y[t] - h @ theta
        s = h @ p @ h + obs_var
        gain = p @ h / s
        theta = theta + gain * e
        p = p - np.outer(gain, h @ p)
        out[t] = theta[0], theta[1], e, s
    return KalmanHedge(out[:, 0], out[:, 1], out[:, 2], out[:, 3])
