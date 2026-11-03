"""Exponential-kernel Hawkes processes for order flow (W25c).

Plain idea
----------
Order arrivals cluster: a trade makes another trade more likely for a short
while. A Hawkes process writes that as an intensity that jumps at each event
and decays back to a base rate.

Formal statement
----------------
For ``M`` event types with a shared decay ``beta``::

    lambda_m(t) = mu_m + sum_n sum_{t_j^n < t} alpha[m, n] * beta * exp(-beta (t - t_j^n))

``alpha[m, n]`` is the branching ratio: the expected number of type-``m``
events triggered by one type-``n`` event. The process is stationary when the
spectral radius of ``alpha`` is below 1 (Hawkes 1971; Bacry, Mastromatteo and
Muzy 2015 survey the finance uses). The log-likelihood on ``[0, T]`` is::

    sum_m [ sum_i log lambda_m(t_i^m) - integral_0^T lambda_m ]

and with the exponential kernel both terms have the O(N) recursion of Ozaki
(1979), used here. Goodness of fit is the time-rescaling theorem (Brown et al.
2002): the compensator increments between consecutive events of each type are
i.i.d. Exp(1) under the true model; a Kolmogorov-Smirnov test checks that.

Worked example
--------------
>>> import numpy as np
>>> t = simulate_univariate(mu=0.5, alpha=0.6, beta=2.0, horizon=4000.0, seed=3)
>>> fit = fit_univariate(t, horizon=4000.0)
>>> bool(abs(fit.alpha - 0.6) < 0.1 and abs(fit.mu - 0.5) < 0.1)
True
>>> bool(ks_time_rescaling(compensator_increments_univariate(t, fit.mu, fit.alpha, fit.beta)).pvalue > 0.01)
True
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

FloatArray = NDArray[np.float64]

__all__ = [
    "MultiFit", "UniFit", "compensator_increments_multivariate",
    "compensator_increments_univariate", "fit_multivariate", "fit_univariate",
    "ks_time_rescaling", "loglik_multivariate", "loglik_univariate",
    "simulate_multivariate", "simulate_univariate",
]


@dataclass(frozen=True)
class UniFit:
    mu: float
    alpha: float
    beta: float
    loglik: float
    stderr: tuple[float, float, float]  # from the inverse observed information


@dataclass(frozen=True)
class MultiFit:
    mu: FloatArray
    alpha: FloatArray
    beta: float
    loglik: float


def simulate_multivariate(mu: FloatArray, alpha: FloatArray, beta: float, horizon: float,
                          seed: int) -> list[FloatArray]:
    """Ogata (1981) thinning. Returns one sorted array of times per type."""
    mu = np.asarray(mu, float)
    alpha = np.asarray(alpha, float)
    m = mu.size
    if np.max(np.abs(np.linalg.eigvals(alpha))) >= 1:
        raise ValueError("alpha must have spectral radius < 1")
    rng = np.random.default_rng(seed)
    excite = np.zeros(m)  # sum over past events of alpha*beta*exp(-beta*(t - tj)) per target
    t = 0.0
    out: list[list[float]] = [[] for _ in range(m)]
    while True:
        lam_bar = float(np.sum(mu + excite))  # intensity only decays until the next event
        w = rng.exponential(1.0 / lam_bar)
        t += w
        if t > horizon:
            break
        excite *= np.exp(-beta * w)
        lam = mu + excite
        u = rng.random() * lam_bar
        c = np.cumsum(lam)
        if u < c[-1]:
            k = int(np.searchsorted(c, u, side="right"))
            out[k].append(t)
            excite += alpha[:, k] * beta
    return [np.asarray(x) for x in out]


def simulate_univariate(mu: float, alpha: float, beta: float, horizon: float, seed: int) -> FloatArray:
    return simulate_multivariate(np.array([mu]), np.array([[alpha]]), beta, horizon, seed)[0]


def _merge(times: list[FloatArray]) -> tuple[FloatArray, NDArray[np.int64]]:
    t = np.concatenate([np.asarray(x, float) for x in times])
    k = np.concatenate([np.full(len(x), i, dtype=np.int64) for i, x in enumerate(times)])
    o = np.argsort(t, kind="stable")
    return t[o], k[o]


def loglik_multivariate(times: list[FloatArray], horizon: float, mu: FloatArray,
                        alpha: FloatArray, beta: float) -> float:
    t, k = _merge(times)
    m = len(times)
    state = np.zeros(m)  # per source n: sum exp(-beta (t - t_j^n))
    last = 0.0
    ll = 0.0
    for ti, ki in zip(t, k, strict=True):
        state *= np.exp(-beta * (ti - last))
        lam = mu[ki] + beta * float(alpha[ki] @ state)
        if lam <= 0:
            return -np.inf
        ll += np.log(lam)
        state[ki] += 1.0
        last = ti
    # integral: mu_m T + sum_n alpha[m,n] sum_j (1 - exp(-beta (T - t_j^n)))
    tail = np.array([np.sum(1.0 - np.exp(-beta * (horizon - np.asarray(x, float)))) for x in times])
    ll -= float(np.sum(mu) * horizon + np.sum(alpha @ tail))
    return float(ll)


def loglik_univariate(t: FloatArray, horizon: float, mu: float, alpha: float, beta: float) -> float:
    """O(N) exponential-kernel log-likelihood (Ozaki 1979)."""
    t = np.asarray(t, float)
    if t.size == 0:
        return -mu * horizon
    a = np.zeros(t.size)
    d = np.exp(-beta * np.diff(t))
    for i in range(1, t.size):
        a[i] = d[i - 1] * (1.0 + a[i - 1])
    lam = mu + alpha * beta * a
    if np.any(lam <= 0):
        return -np.inf
    comp = mu * horizon + alpha * np.sum(1.0 - np.exp(-beta * (horizon - t)))
    return float(np.sum(np.log(lam)) - comp)


def fit_univariate(t: FloatArray, horizon: float, x0: tuple[float, float, float] | None = None) -> UniFit:
    t = np.asarray(t, float)
    if x0 is None:
        x0 = (0.5 * t.size / horizon, 0.5, 1.0)

    def nll(x: FloatArray) -> float:
        v = -loglik_univariate(t, horizon, float(x[0]), float(x[1]), float(x[2]))
        return v if np.isfinite(v) else 1e300

    res = optimize.minimize(nll, np.array(x0), method="L-BFGS-B",
                            bounds=[(1e-9, None), (0.0, 0.999), (1e-6, None)])
    x = res.x
    se = _stderr(nll, x)
    return UniFit(float(x[0]), float(x[1]), float(x[2]), float(-res.fun), se)


def _stderr(fn: Callable[[FloatArray], float], x: FloatArray) -> tuple[float, float, float]:
    """Standard errors from a central finite-difference Hessian."""
    n = x.size
    h = 1e-4 * np.maximum(np.abs(x), 1e-3)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i = np.eye(n)[i] * h[i]
            e_j = np.eye(n)[j] * h[j]
            hess[i, j] = (fn(x + e_i + e_j) - fn(x + e_i - e_j) - fn(x - e_i + e_j)
                          + fn(x - e_i - e_j)) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(hess)
        d = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        d = np.full(n, np.nan)
    return (float(d[0]), float(d[1]), float(d[2]))


def fit_multivariate(times: list[FloatArray], horizon: float, beta0: float = 1.0) -> MultiFit:
    """MLE of ``mu``, ``alpha`` and the shared ``beta`` (L-BFGS-B, box bounds)."""
    m = len(times)
    rates = np.array([len(x) / horizon for x in times])
    x0 = np.concatenate([0.5 * rates, np.full(m * m, 0.2), [beta0]])

    def unpack(x: FloatArray) -> tuple[FloatArray, FloatArray, float]:
        return x[:m], x[m:m + m * m].reshape(m, m), float(x[-1])

    def nll(x: FloatArray) -> float:
        mu, al, be = unpack(x)
        v = -loglik_multivariate(times, horizon, mu, al, be)
        return v if np.isfinite(v) else 1e300

    bounds = [(1e-9, None)] * m + [(0.0, 0.999)] * (m * m) + [(1e-6, None)]
    res = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    mu, al, be = unpack(res.x)
    return MultiFit(mu.copy(), al.copy(), be, float(-res.fun))


def compensator_increments_multivariate(times: list[FloatArray], mu: FloatArray, alpha: FloatArray,
                                        beta: float) -> list[FloatArray]:
    """Lambda_m(t_i^m) - Lambda_m(t_{i-1}^m) for each type ``m`` (time rescaling)."""
    t, k = _merge(times)
    m = len(times)
    state = np.zeros(m)
    big = np.zeros(m)  # compensator Lambda_m(t) so far
    last = 0.0
    prev_at = np.zeros(m)
    out: list[list[float]] = [[] for _ in range(m)]
    for ti, ki in zip(t, k, strict=True):
        dt = ti - last
        decay = np.exp(-beta * dt)
        # integral of mu + beta*alpha@state*exp(-beta s) over [0, dt]
        big += mu * dt + (alpha @ state) * (1.0 - decay)
        state *= decay
        out[ki].append(float(big[ki] - prev_at[ki]))
        prev_at[ki] = big[ki]
        state[ki] += 1.0
        last = ti
    return [np.asarray(x) for x in out]


def compensator_increments_univariate(t: FloatArray, mu: float, alpha: float, beta: float) -> FloatArray:
    return compensator_increments_multivariate([np.asarray(t, float)], np.array([mu]),
                                               np.array([[alpha]]), beta)[0]


def ks_time_rescaling(increments: FloatArray) -> Any:
    """KS test of the rescaled inter-event times against Exp(1)."""
    return stats.kstest(np.asarray(increments, float), "expon")
