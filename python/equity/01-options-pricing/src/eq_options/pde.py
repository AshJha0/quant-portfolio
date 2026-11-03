"""Crank-Nicolson finite-difference pricer for European and American options.

Plain idea: write the Black-Scholes equation in log-spot, put it on a grid,
and step backwards from the payoff. Crank-Nicolson averages the explicit
and implicit steps, which makes it second order in both time and space. Its
weakness is the kink in the payoff: the first few steps ring. Rannacher's
fix replaces the first Crank-Nicolson step with two fully implicit half
steps, which damp the ringing. For American exercise, each step solves the
tridiagonal system and then takes the larger of that value and the payoff,
in the order of Brennan and Schwartz, which gives the exact solution of the
discrete linear complementarity problem for a single exercise boundary.

Formal statement. With ``x = ln S`` and ``tau = T - t``:
``V_tau = 1/2 sigma^2 V_xx + (r - q - sigma^2/2) V_x - r V``. Central
differences in ``x`` on ``n_space + 1`` uniform nodes over
``ln S +/- width * sigma sqrt(T)``; theta-scheme in ``tau`` with
``theta = 1/2`` (Crank and Nicolson 1947, Proc. Camb. Phil. Soc. 43, 50-67)
after ``2`` implicit half steps (Rannacher 1984, Numer. Math. 43, 309-327;
Giles and Carter 2006, J. Comput. Finance 9(4)). American projection after
Brennan and Schwartz (1977), J. Finance 32(2), 449-462, which is exact for
a put (or call) whose exercise region is one interval at the low (high)
end of the grid (Jaillet, Lamberton and Lapeyre 1990, Acta Appl. Math. 21).

Worked example: ``S = K = 100, T = 1, r = 0.05, q = 0, sigma = 0.2``,
American put, ``n_space = n_time = 400`` gives 6.08929, against a 4001-step
Leisen-Reimer tree at 6.09030; the error falls about 4x per doubling of
both grids (1.0e-3 at 400, 2.3e-4 at 800).

Conventions are those of :mod:`eq_options.black_scholes`.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from qp_core.model import model

from .black_scholes import OptionType, _check_rates, validate_inputs

ExerciseStyle = Literal["european", "american"]

__all__ = ["cn_price"]


def _solve_tridiag(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
    floor: np.ndarray | None, exercise_low: bool,
) -> np.ndarray:
    """Solve ``a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i`` (Thomas algorithm).

    With ``floor`` given, apply ``max(x, floor)`` during back-substitution
    (Brennan-Schwartz). The back-substitution must run away from the
    exercise region: for an exercise region at the low end, eliminate from
    the top down and substitute from the bottom up, and vice versa.
    """
    n = b.size
    if exercise_low:
        # Reverse the system so the generic sweep runs top-down.
        x = _solve_tridiag(c[::-1], b[::-1], a[::-1], d[::-1],
                           None if floor is None else floor[::-1], False)
        return x[::-1]
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1] if floor is None else max(dp[-1], floor[-1])
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
        if floor is not None and x[i] < floor[i]:
            x[i] = floor[i]
    return x


@model(id="EQ-OPT-CN")
def cn_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "put",
    exercise: ExerciseStyle = "american",
    n_space: int = 400,
    n_time: int = 400,
    width: float = 6.0,
) -> float:
    """Crank-Nicolson (Rannacher-started) price of a European or American option.

    Parameters
    ----------
    S, K, T, sigma : float
        Spot, strike, expiry in years, volatility; all strictly positive.
    r, q : float
        Continuously compounded annualised rate and dividend yield.
    option_type : {"call", "put"}
    exercise : {"european", "american"}
    n_space : int
        Number of space intervals, ``>= 10``; made even so the spot is a node.
    n_time : int
        Number of time steps, ``>= 2``.
    width : float
        Half-width of the log-spot grid in standard deviations ``sigma sqrt(T)``.

    Returns
    -------
    float
        Present value at the spot node, in currency units.

    Raises
    ------
    ValueError
        On invalid inputs or grid sizes.
    """
    validate_inputs(S, K, T, sigma, option_type)
    _check_rates(r=r, q=q)
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")
    if min(S, K, T, sigma) <= 0.0:
        raise ValueError("cn_price needs S, K, T and sigma strictly positive")
    if n_space < 10 or n_time < 2 or not width > 0.0:
        raise ValueError("need n_space >= 10, n_time >= 2 and width > 0")
    n_space += n_space % 2

    half = width * sigma * math.sqrt(T) + abs(math.log(K / S))
    x = math.log(S) + np.linspace(-half, half, n_space + 1)
    dx = x[1] - x[0]
    spots = np.exp(x)
    sign = 1.0 if option_type == "call" else -1.0
    payoff = np.maximum(sign * (spots - K), 0.0)
    american = exercise == "american"

    # Generator L V = alpha V_{i-1} + beta V_i + gamma V_{i+1} on interior nodes.
    s2 = sigma * sigma
    mu = r - q - 0.5 * s2
    alpha = 0.5 * s2 / dx**2 - 0.5 * mu / dx
    gamma = 0.5 * s2 / dx**2 + 0.5 * mu / dx
    beta = -s2 / dx**2 - r

    def boundary(tau: float) -> tuple[float, float]:
        lo, hi = spots[0], spots[-1]
        if option_type == "put":
            low = K - lo if american else K * math.exp(-r * tau) - lo * math.exp(-q * tau)
            return max(low, 0.0), 0.0
        high = hi - K if american else hi * math.exp(-q * tau) - K * math.exp(-r * tau)
        return 0.0, max(high, 0.0)

    m = n_space - 1  # interior unknowns
    floor = payoff[1:-1] if american else None
    v = payoff.copy()
    dt_full = T / n_time
    steps = [(0.5 * dt_full, 1.0), (0.5 * dt_full, 1.0)] + [(dt_full, 0.5)] * (n_time - 1)
    tau = 0.0
    for dt, theta in steps:
        tau_new = tau + dt
        lo_new, hi_new = boundary(tau_new)
        explicit = v[1:-1] + (1.0 - theta) * dt * (
            alpha * v[:-2] + beta * v[1:-1] + gamma * v[2:])
        explicit[0] += theta * dt * alpha * lo_new
        explicit[-1] += theta * dt * gamma * hi_new
        a = np.full(m, -theta * dt * alpha)
        b = np.full(m, 1.0 - theta * dt * beta)
        c = np.full(m, -theta * dt * gamma)
        a[0] = 0.0
        c[-1] = 0.0
        v[1:-1] = _solve_tridiag(a, b, c, explicit, floor, option_type == "put")
        v[0], v[-1] = lo_new, hi_new
        tau = tau_new
    return float(v[n_space // 2])
