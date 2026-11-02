"""HAR-RV: heterogeneous autoregressive model of realised variance.

Corsi, F. (2009), "A simple approximate long-memory model of realized
volatility", J. Financial Econometrics 7(2), 174-196, eq. (8):

    RV_{t+1} = b0 + b_d RV_t + b_w RV_t^(w) + b_m RV_t^(m) + e_{t+1},

with ``RV^(w)`` and ``RV^(m)`` the means of the last 5 and 22 daily RVs.
Estimated by OLS; standard errors are Newey-West (1987) HAC with Bartlett
weights, as Corsi recommends for the overlapping regressors. Multi-step
forecasts iterate the fitted recursion, feeding forecasts back in.

Realised variance of one day is the sum of squared intraday log returns
(Andersen, Bollerslev, Diebold and Labys 2003, Econometrica 71(2), eq. 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WEEK, MONTH = 5, 22


def realized_variance(intraday_log_returns: np.ndarray) -> np.ndarray:
    """Daily RV from a ``(days, intraday_obs)`` array of log returns."""
    r = np.asarray(intraday_log_returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("intraday_log_returns must be 2-D (days, obs)")
    return np.sum(r * r, axis=1)


def har_design(rv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Regressor matrix ``[1, RV_d, RV_w, RV_m]`` and target ``RV_{t+1}``."""
    rv = np.asarray(rv, dtype=float)
    if rv.ndim != 1 or rv.size < MONTH + 10:
        raise ValueError(f"need a 1-D series of at least {MONTH + 10} days")
    c = np.concatenate(([0.0], np.cumsum(rv)))
    t = np.arange(MONTH - 1, rv.size - 1)  # last index usable as "today"
    d = rv[t]
    w = (c[t + 1] - c[t + 1 - WEEK]) / WEEK
    m = (c[t + 1] - c[t + 1 - MONTH]) / MONTH
    X = np.column_stack([np.ones_like(d), d, w, m])
    return X, rv[t + 1]


@dataclass(frozen=True)
class HARResult:
    params: np.ndarray  # b0, b_d, b_w, b_m
    se_hac: np.ndarray
    r2: float
    n_obs: int
    last_rv: np.ndarray  # last MONTH observations, oldest first

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Iterated 1..horizon-day-ahead RV forecasts."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        hist = list(self.last_rv)
        out = []
        b0, bd, bw, bm = self.params
        for _ in range(horizon):
            f = b0 + bd * hist[-1] + bw * np.mean(hist[-WEEK:]) + bm * np.mean(hist[-MONTH:])
            out.append(f)
            hist.append(f)
        return np.array(out)


def _newey_west(X: np.ndarray, e: np.ndarray, lags: int) -> np.ndarray:
    n = X.shape[0]
    xe = X * e[:, None]
    S = xe.T @ xe
    for L in range(1, lags + 1):
        g = xe[L:].T @ xe[:-L]
        S += (1.0 - L / (lags + 1.0)) * (g + g.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(XtX_inv @ S @ XtX_inv) * n / (n - X.shape[1]))


def fit_har(rv: np.ndarray, hac_lags: int = MONTH) -> HARResult:
    """OLS fit of HAR-RV with Newey-West standard errors."""
    X, y = har_design(rv)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    r2 = 1.0 - float(e @ e) / float(np.sum((y - y.mean()) ** 2))
    return HARResult(beta, _newey_west(X, e, hac_lags), r2, y.size,
                     np.asarray(rv, dtype=float)[-MONTH:].copy())


def simulate_har(params: np.ndarray, n: int, noise_sd: float, seed: int = 0) -> np.ndarray:
    """Simulate a HAR-RV path (Gaussian innovations, floored at a tiny positive)."""
    b0, bd, bw, bm = params
    rng = np.random.default_rng(seed)
    mean = b0 / (1.0 - bd - bw - bm)
    rv = list(np.full(MONTH, mean))
    for _ in range(n):
        f = b0 + bd * rv[-1] + bw * np.mean(rv[-WEEK:]) + bm * np.mean(rv[-MONTH:])
        rv.append(max(f + noise_sd * rng.standard_normal(), 1e-12))
    return np.array(rv[MONTH:])
