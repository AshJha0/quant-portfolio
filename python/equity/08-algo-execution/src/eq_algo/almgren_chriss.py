"""Almgren-Chriss optimal execution, discrete time, from scratch.

Setup (Almgren & Chriss 2000, "Optimal execution of portfolio transactions"):
sell/buy ``X`` shares over ``N`` intervals of length ``tau = T/N``.  Holdings
``x_0 = X, ..., x_N = 0``; trades ``n_k = x_{k-1} - x_k``.  Arithmetic price
dynamics with volatility ``sigma`` (currency / sqrt(time)), **linear
permanent impact** ``gamma`` (currency per share) and **linear temporary
impact** ``eta`` (currency per share per unit trade rate), fixed cost
``epsilon`` per share:

    E[cost] = 0.5*gamma*X^2 + epsilon*sum|n_k| + (eta_tilde/tau) * sum n_k^2
    V[cost] = sigma^2 * tau * sum_{k=1..N} x_k^2,   eta_tilde = eta - gamma*tau/2

Minimising ``E + lambda * V`` (risk aversion ``lambda >= 0``) gives the
discrete Euler-Lagrange recursion

    x_{k-1} - 2*cosh(kappa*tau)*x_k + x_{k+1} = 0

whose solution with the boundary conditions is the closed form

    x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T),   t_j = j*tau,

where ``kappa`` solves ``cosh(kappa*tau) = 1 + (kappa_tilde^2 * tau^2)/2``
and ``kappa_tilde^2 = lambda * sigma^2 / eta_tilde``.

Limits: ``lambda -> 0`` gives equal slices (TWAP); ``lambda -> inf`` dumps
everything in the first slice (urgency).  Both are tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .intraday import IntradayMarket

__all__ = ["ACParams", "ac_kappa", "ac_trajectory", "ac_trades", "ac_cost_moments",
           "efficient_frontier", "evaluate_schedules"]


@dataclass(frozen=True)
class ACParams:
    """Almgren-Chriss model parameters (arithmetic, absolute units).

    ``sigma``: price vol in currency per sqrt(day); ``eta``: temporary impact
    (currency per (share/day)); ``gamma``: permanent impact (currency per
    share); ``epsilon``: fixed cost per share (half-spread + fees);
    ``total_time``: horizon in days; ``n_slices``: number of intervals.
    """

    total_shares: float
    n_slices: int
    total_time: float = 1.0
    sigma: float = 1.0
    eta: float = 1e-6
    gamma: float = 0.0
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.total_shares <= 0:
            raise ValueError("total_shares must be > 0")
        if self.n_slices < 1:
            raise ValueError("n_slices must be >= 1")
        if self.total_time <= 0:
            raise ValueError("total_time must be > 0")
        if self.sigma < 0 or self.eta <= 0 or self.gamma < 0 or self.epsilon < 0:
            raise ValueError("require sigma, gamma, epsilon >= 0 and eta > 0")
        if self.eta_tilde <= 0:
            raise ValueError(
                "eta_tilde = eta - gamma*tau/2 must be > 0 "
                "(temporary impact too small vs permanent at this slicing)"
            )

    @property
    def tau(self) -> float:
        return self.total_time / self.n_slices

    @property
    def eta_tilde(self) -> float:
        return self.eta - 0.5 * self.gamma * self.tau


def ac_kappa(params: ACParams, lam: float) -> float:
    """Urgency parameter ``kappa`` (1/time) for risk aversion ``lam >= 0``.

    Solves ``cosh(kappa*tau) = 1 + kappa_tilde^2 * tau^2 / 2`` exactly via
    ``arccosh``; returns 0 for ``lam = 0`` (TWAP limit).
    """
    if lam < 0:
        raise ValueError("lambda (risk aversion) must be >= 0")
    if lam == 0:
        return 0.0
    kt2 = lam * params.sigma**2 / params.eta_tilde
    tau = params.tau
    return float(np.arccosh(1.0 + 0.5 * kt2 * tau**2) / tau)


def _sinh_ratio(a: np.ndarray, b: float) -> np.ndarray:
    """``sinh(a) / sinh(b)`` for ``0 <= a <= b``, overflow-free.

    The naive form overflows for ``b`` beyond ~710 (``sinh`` exceeds the
    double range) and returns ``inf/inf = NaN``.  Rewriting

        sinh(a)/sinh(b) = e^{a-b} * (1 - e^{-2a}) / (1 - e^{-2b})

    keeps every exponential argument non-positive, so the result is finite
    for arbitrarily large urgency; ``expm1`` preserves accuracy as
    ``a, b -> 0``.
    """
    a = np.asarray(a, dtype=float)
    if b <= 0.0:
        raise ValueError("b must be > 0 in _sinh_ratio")
    # -expm1(-2x) == 1 - e^{-2x}; the two minus signs cancel in the ratio.
    return np.exp(a - b) * np.expm1(-2.0 * a) / np.expm1(-2.0 * b)


def ac_trajectory(params: ACParams, lam: float) -> np.ndarray:
    """Optimal holdings ``x_0..x_N`` (length ``n_slices + 1``).

    ``x_j = X * sinh(kappa*(T - t_j)) / sinh(kappa*T)``; exactly linear
    (TWAP) when ``lam = 0``.  Monotone decreasing from ``X`` to 0.

    The sinh ratio is evaluated in an overflow-free exponential form, so
    extreme risk aversion (``kappa*T`` in the hundreds or thousands) returns
    the correct front-loaded schedule instead of NaN.
    """
    X, N, T = params.total_shares, params.n_slices, params.total_time
    t = np.arange(N + 1) * params.tau
    kappa = ac_kappa(params, lam)
    if kappa == 0.0:
        return X * (1.0 - t / T)
    x = X * _sinh_ratio(kappa * np.maximum(T - t, 0.0), kappa * T)
    x[0] = X
    x[-1] = 0.0
    return x


def ac_trades(params: ACParams, lam: float) -> np.ndarray:
    """Trade list ``n_k = x_{k-1} - x_k`` (length ``n_slices``); sums to X."""
    x = ac_trajectory(params, lam)
    return -np.diff(x)


def ac_cost_moments(params: ACParams, trajectory: np.ndarray) -> tuple[float, float]:
    """Expected cost and variance of cost for an arbitrary trajectory.

    ``E = 0.5*gamma*X^2 + epsilon*sum|n_k| + (eta_tilde/tau)*sum n_k^2``;
    ``V = sigma^2 * tau * sum_{k=1..N} x_k^2``.  Works for any admissible
    trajectory (x_0 = X, x_N = 0), not just the optimum.
    """
    x = np.asarray(trajectory, dtype=float)
    if x.shape != (params.n_slices + 1,):
        raise ValueError("trajectory must have length n_slices + 1")
    if not np.isclose(x[0], params.total_shares) or not np.isclose(x[-1], 0.0):
        raise ValueError("trajectory must run from X to 0")
    n = -np.diff(x)
    exp_cost = (0.5 * params.gamma * params.total_shares**2
                + params.epsilon * np.abs(n).sum()
                + params.eta_tilde / params.tau * (n**2).sum())
    var_cost = params.sigma**2 * params.tau * (x[1:] ** 2).sum()
    return float(exp_cost), float(var_cost)


def efficient_frontier(params: ACParams, lams: Sequence[float]) -> pd.DataFrame:
    """Expected-cost / variance frontier across risk aversions.

    Columns: ``kappa, expected_cost, variance, std``; indexed by lambda.
    Expected cost is increasing and variance decreasing in lambda (tested).
    """
    rows = []
    for lam in lams:
        x = ac_trajectory(params, lam)
        e, v = ac_cost_moments(params, x)
        rows.append({"lam": float(lam), "kappa": ac_kappa(params, lam),
                     "expected_cost": e, "variance": v, "std": np.sqrt(v)})
    return pd.DataFrame(rows).set_index("lam")


def evaluate_schedules(market: IntradayMarket,
                       schedules: Mapping[str, np.ndarray], side: int = 1,
                       n_reps: int = 200, seed: int = 0) -> pd.DataFrame:
    """Monte Carlo horse race of execution schedules on the simulator.

    Each replication draws one market day (price path + volumes) per seed and
    executes every schedule against the *same* seed (common random numbers,
    variance-reduced comparison).  Reports mean and std of implementation
    shortfall (bps vs the arrival mid) and slippage vs VWAP/TWAP.

    Returns a DataFrame indexed by strategy name with columns
    ``mean_is_bps, std_is_bps, mean_vs_vwap_bps, mean_vs_twap_bps``.
    """
    from .benchmarks import benchmark_slippage  # local import avoids cycles

    if n_reps < 2:
        raise ValueError("n_reps must be >= 2")
    names = list(schedules.keys())
    is_bps = {k: np.empty(n_reps) for k in names}
    vw = {k: np.empty(n_reps) for k in names}
    tw = {k: np.empty(n_reps) for k in names}
    for r in range(n_reps):
        for k in names:
            res = market.execute(np.asarray(schedules[k], dtype=float), side=side,
                                 seed=seed + r)
            s = benchmark_slippage(res)
            is_bps[k][r] = s["vs_arrival_bps"]
            vw[k][r] = s["vs_vwap_bps"]
            tw[k][r] = s["vs_twap_bps"]
    rows = {k: {
        "mean_is_bps": is_bps[k].mean(),
        "std_is_bps": is_bps[k].std(ddof=1),
        "mean_vs_vwap_bps": vw[k].mean(),
        "mean_vs_twap_bps": tw[k].mean(),
    } for k in names}
    return pd.DataFrame.from_dict(rows, orient="index")
