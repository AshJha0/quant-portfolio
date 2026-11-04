"""Monte Carlo checks for the closed-form barrier and touch prices.

Plain idea
----------
Simulate the spot on a time grid with exact log-normal steps. A path
sampled at discrete times can cross a barrier and come back between two
samples; the Brownian bridge gives the probability of that, so each step
contributes a survival factor instead of a 0/1 test. With the bridge, the
estimator is unbiased for *continuous* monitoring, so it can be compared
with the closed forms to within Monte Carlo error.

Formal statement
----------------
For ``X = ln S`` with constant volatility, given ``X_i`` and ``X_{i+1}``
both above ``h = ln H``, the probability the bridge touched ``h`` in
between is ``exp(-2 (X_i - h)(X_{i+1} - h) / (sigma^2 dt))`` (Glasserman
2004, *Monte Carlo Methods in Financial Engineering*, s.6.4, eq. 6.50;
Beaglehole, Dybvig and Zhou 1997). For two barriers the step survival is
taken as the product of the two single-barrier survivals, which ignores
paths that hit both barriers inside one step; at 100 steps per year and
corridors wider than a few step standard deviations that term is below
1e-12 per step.

The estimator returns the mean and its standard error over paths.

Worked example
--------------
>>> est, se = barrier_mc("down_out:call", 100, 100, 90, 1.0, 0.05, 0.02, 0.25,
...                      n_paths=20_000, n_steps=50, seed=1)
>>> 0.0 < se < 0.2
True
"""

from __future__ import annotations

import math

import numpy as np

from qp_core.model import model

__all__ = ["barrier_mc", "double_barrier_mc"]


def _paths_step(
    rng: np.random.Generator, x: np.ndarray, mu_dt: float, vol_dt: float
) -> np.ndarray:
    return x + mu_dt + vol_dt * rng.standard_normal(x.shape[0])


def _bridge_survive(x0: np.ndarray, x1: np.ndarray, h: float, var_dt: float, down: bool) -> np.ndarray:
    d0 = (x0 - h) if down else (h - x0)
    d1 = (x1 - h) if down else (h - x1)
    out = np.where((d0 > 0.0) & (d1 > 0.0), 1.0 - np.exp(-2.0 * d0 * d1 / var_dt), 0.0)
    return out


def _payoff(cp: str, s: np.ndarray, X: float) -> np.ndarray:
    if cp == "call":
        return np.maximum(s - X, 0.0)
    if cp == "put":
        return np.maximum(X - s, 0.0)
    raise ValueError(f"cp must be 'call' or 'put', got {cp!r}")


@model(id="CORE-EXO-BARRIER-MC")
def barrier_mc(
    spec: str, S: float, X: float, H: float, T: float, r: float, b: float, sigma: float,
    n_paths: int = 100_000, n_steps: int = 100, seed: int = 0,
) -> tuple[float, float]:
    """Price a single barrier (``"down_out:call"`` etc.) or touch (``"touch:one_touch_up"``).

    Touches pay 1 at expiry. Returns ``(estimate, standard_error)``.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    mu_dt = (b - 0.5 * sigma * sigma) * dt
    vol_dt = sigma * math.sqrt(dt)
    var_dt = sigma * sigma * dt
    h = math.log(H)
    head, tail = spec.split(":")
    if head == "touch":
        down = tail.endswith("down")
    elif head in ("down_in", "down_out", "up_in", "up_out"):
        down = head.startswith("down")
    else:
        raise ValueError(f"unknown spec {spec!r}")
    x = np.full(n_paths, math.log(S))
    surv = np.ones(n_paths)
    for _ in range(n_steps):
        x1 = _paths_step(rng, x, mu_dt, vol_dt)
        surv *= _bridge_survive(x, x1, h, var_dt, down)
        x = x1
    df = math.exp(-r * T)
    if head == "touch":
        pay = surv if tail.startswith("no") else 1.0 - surv
    else:
        vanilla = _payoff(tail, np.exp(x), X)
        pay = vanilla * surv if head.endswith("out") else vanilla * (1.0 - surv)
    pay = df * pay
    return float(pay.mean()), float(pay.std(ddof=1) / math.sqrt(n_paths))


@model(id="CORE-EXO-BARRIER-MC")
def double_barrier_mc(
    spec: str, S: float, X: float, L: float, U: float, T: float, r: float, b: float,
    sigma: float, n_paths: int = 100_000, n_steps: int = 100, seed: int = 0,
) -> tuple[float, float]:
    """Double knock-out (``"dko:call"``, ``"dko:put"``) or double no-touch (``"dnt"``, pays 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    mu_dt = (b - 0.5 * sigma * sigma) * dt
    vol_dt = sigma * math.sqrt(dt)
    var_dt = sigma * sigma * dt
    lo, hi = math.log(L), math.log(U)
    x = np.full(n_paths, math.log(S))
    surv = np.ones(n_paths)
    for _ in range(n_steps):
        x1 = _paths_step(rng, x, mu_dt, vol_dt)
        surv *= _bridge_survive(x, x1, lo, var_dt, True) * _bridge_survive(x, x1, hi, var_dt, False)
        x = x1
    df = math.exp(-r * T)
    if spec == "dnt":
        pay = surv
    elif spec.startswith("dko:"):
        pay = _payoff(spec[4:], np.exp(x), X) * surv
    else:
        raise ValueError(f"unknown spec {spec!r}")
    pay = df * pay
    return float(pay.mean()), float(pay.std(ddof=1) / math.sqrt(n_paths))
