"""Optimal schedules under transient impact (Obizhaeva-Wang, Gatheral).

Plain idea
----------
Impact fades. If each trade pushes the price and the push decays over time
with a kernel ``G``, then trading in a burst is expensive (you pay your own
recent impact) and spreading out is cheaper. With exponential decay the best
schedule is a block at the start, a block at the end and a constant rate in
between (Obizhaeva and Wang 2013).

Formal statement
----------------
Discrete trades ``x_i`` at times ``t_i`` with linear propagator impact
(Gatheral 2010; Alfonsi, Schied and Slynko 2012). Expected cost::

    C(x) = 1/2 * x' G x,     G[i, j] = G(|t_i - t_j|)

The optimal schedule for a parent of ``X`` is ``x* = X G^{-1} 1 / (1' G^{-1} 1)``.
No price manipulation needs ``G`` positive definite (cost of any round trip
``>= 0``), and to rule out sign-alternating optima (transaction-triggered
manipulation) the kernel should be convex and decreasing (Alfonsi, Schied and
Slynko 2012). For the nonlinear form ``f(v) = v^delta`` with power-law decay
``t^-gamma`` Gatheral (2010) shows no dynamic arbitrage needs
``gamma + delta >= 1``; the square-root law with ``gamma = 1/2`` sits on
that boundary.

Worked example
--------------
>>> import numpy as np
>>> t = np.arange(6.0)
>>> x = optimal_schedule(1.0, t, exponential_kernel(rho=0.5))
>>> bool(x[0] > x[2] and x[-1] > x[2] and np.allclose(x[1:-1], x[2]))
True
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from qp_core.model import model

FloatArray = NDArray[np.float64]
Kernel = Callable[[FloatArray], FloatArray]


def exponential_kernel(rho: float, kappa: float = 1.0) -> Kernel:
    """Obizhaeva-Wang: ``G(t) = kappa * exp(-rho t)``."""
    return lambda t: kappa * np.exp(-rho * np.asarray(t, float))


def power_law_kernel(gamma: float, tau: float = 1.0, kappa: float = 1.0) -> Kernel:
    """Gatheral power law, regularised at 0: ``G(t) = kappa / (1 + t/tau)^gamma``."""
    return lambda t: kappa * (1.0 + np.asarray(t, float) / tau) ** (-gamma)


def gaussian_kernel(width: float) -> Kernel:
    """Not convex at 0; included to show a kernel that admits manipulation."""
    return lambda t: np.exp(-(np.asarray(t, float) / width) ** 2)


def impact_matrix(times: FloatArray, kernel: Kernel) -> FloatArray:
    t = np.asarray(times, float)
    m: FloatArray = kernel(np.abs(t[:, None] - t[None, :]))
    return m


@model(id="EXEC-IMPACT-TRANSIENT")
def expected_cost(x: FloatArray, times: FloatArray, kernel: Kernel) -> float:
    g = impact_matrix(times, kernel)
    return float(0.5 * x @ g @ x)


@model(id="EXEC-IMPACT-TRANSIENT")
def optimal_schedule(parent: float, times: FloatArray, kernel: Kernel) -> FloatArray:
    g = impact_matrix(times, kernel)
    w = np.linalg.solve(g, np.ones(len(times)))
    out: FloatArray = parent * w / w.sum()
    return out


def min_round_trip_cost(times: FloatArray, kernel: Kernel) -> float:
    """Smallest eigenvalue of ``G``: negative means a round trip with negative cost."""
    return float(np.linalg.eigvalsh(impact_matrix(times, kernel)).min())


def gatheral_admissible(delta: float, gamma: float) -> bool:
    """No dynamic arbitrage for ``f(v) = v^delta`` and ``G(t) ~ t^-gamma`` (Gatheral 2010)."""
    return gamma + delta >= 1.0
