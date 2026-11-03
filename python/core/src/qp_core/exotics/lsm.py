"""Longstaff-Schwartz least-squares Monte Carlo for American options.

Plain idea
----------
At each exercise date, the holder compares the exercise value with the
value of waiting. Monte Carlo gives the future cash flows path by path;
regressing them on functions of today's spot (in-the-money paths only)
estimates the value of waiting. Work backwards from expiry, exercising
where the immediate payoff beats the regression.

Formal statement
----------------
Longstaff and Schwartz (2001), "Valuing American options by simulation:
a simple least-squares approach", RFS 14(1). Basis: the first ``degree + 1``
weighted Laguerre polynomials of ``S / K`` (their eq. 2), constant
included. The estimator uses the realised cash flows (not the fitted
continuation values), so it is low-biased by the sub-optimality of the
estimated rule and high-biased by in-sample fitting; LS report both
effects small at 50 dates and 100k paths. A Cox-Ross-Rubinstein tree
(Cox, Ross and Rubinstein 1979) is provided as the independent reference.

Worked example
--------------
Longstaff-Schwartz Table 1, ``S = 36, K = 40, sigma = 0.2, T = 1,
r = 0.06``: the finite-difference value is 4.478 and LSM gives 4.472
(s.e. 0.010).

>>> round(crr_american("put", 36, 40, 1.0, 0.06, 0.0, 0.2, n_steps=2000), 2)
4.49
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["crr_american", "lsm_american"]


def _check(cp: str, S: float, K: float, T: float, sigma: float) -> float:
    if cp not in ("call", "put"):
        raise ValueError(f"cp must be 'call' or 'put', got {cp!r}")
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        raise ValueError("S, K, T and sigma must be positive")
    return 1.0 if cp == "call" else -1.0


def crr_american(
    cp: str, S: float, K: float, T: float, r: float, q: float, sigma: float, n_steps: int = 1000
) -> float:
    """American option on a Cox-Ross-Rubinstein binomial tree (reference)."""
    sign = _check(cp, S, K, T, sigma)
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("tree probability outside (0, 1); use more steps")
    disc = math.exp(-r * dt)
    j = np.arange(n_steps + 1)
    s = S * u ** (n_steps - 2 * j).astype(float)
    v = np.maximum(sign * (s - K), 0.0)
    for i in range(n_steps - 1, -1, -1):
        s = S * u ** (i - 2 * np.arange(i + 1)).astype(float)
        v = np.maximum(disc * (p * v[:-1] + (1.0 - p) * v[1:]), sign * (s - K))
    return float(v[0])


def _laguerre_basis(x: np.ndarray, degree: int) -> np.ndarray:
    # weighted Laguerre: exp(-x/2) L_n(x), L_0 = 1, L_1 = 1 - x, L_2 = 1 - 2x + x^2/2, ...
    cols = [np.ones_like(x)]
    w = np.exp(-0.5 * x)
    lm1, l0 = np.zeros_like(x), np.ones_like(x)
    for n in range(degree):
        l1 = ((2 * n + 1 - x) * l0 - n * lm1) / (n + 1)
        cols.append(w * l1)
        lm1, l0 = l0, l1
    cols.append(w)  # weighted L_0
    return np.column_stack(cols)


def lsm_american(
    cp: str, S: float, K: float, T: float, r: float, q: float, sigma: float,
    n_paths: int = 100_000, n_steps: int = 50, degree: int = 3, seed: int = 0,
) -> tuple[float, float]:
    """LSM price of an American call or put; returns ``(estimate, standard_error)``.

    Paths are antithetic pairs (``n_paths`` is rounded up to even); the
    standard error is over pair averages, so it is honest under the
    pairing.
    """
    sign = _check(cp, S, K, T, sigma)
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2
    dt = T / n_steps
    z = rng.standard_normal((half, n_steps))
    z = np.concatenate([z, -z])
    logs = math.log(S) + np.cumsum((r - q - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z, axis=1)
    s = np.exp(logs)
    disc = math.exp(-r * dt)
    cash = np.maximum(sign * (s[:, -1] - K), 0.0)
    for i in range(n_steps - 2, -1, -1):
        cash *= disc
        ex = np.maximum(sign * (s[:, i] - K), 0.0)
        itm = ex > 0.0
        if itm.sum() > degree + 2:
            X = _laguerre_basis(s[itm, i] / K, degree)
            coef, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)
            cont = X @ coef
            exercise = ex[itm] > cont
            idx = np.flatnonzero(itm)[exercise]
            cash[idx] = ex[idx]
    cash *= disc
    pv = np.maximum(cash, 0.0)
    pair = 0.5 * (pv[:half] + pv[half:])
    # immediate exercise at t = 0 is allowed
    est = max(float(pair.mean()), max(sign * (S - K), 0.0))
    return est, float(pair.std(ddof=1) / math.sqrt(half))
