"""Bermudan swaptions on a Hull-White trinomial tree.

Plain idea. A Bermudan swaption lets the holder enter a swap on any of
several dates, usually every fixed-coupon date after a lockout. Whether to
exercise today depends on what the option is worth if kept, so it has no
closed form; a lattice that knows today's curve answers it by working
backwards: at each exercise date compare "exercise now" with "keep".

Formal statement.

* Model: one-factor Hull-White, ``r(t) = x(t) + alpha(t)``,
  ``dx = -a x dt + sigma dW`` (Brigo and Mercurio 2006, "Interest Rate
  Models - Theory and Practice", 2nd ed., section 3.3.3; Hull and White
  1994, "Numerical procedures for implementing term structure models I",
  J. Derivatives 2(1), 7-16).
* Tree for ``x`` on a time grid ``0 = t_0 < ... < t_N`` (steps may differ):
  over step ``i`` the OU transition has mean ``x e^{-a dt_i}`` and variance
  ``V_i = sigma^2 (1 - e^{-2 a dt_i}) / (2a)``; level ``i+1`` has spacing
  ``dx_{i+1} = sqrt(3 V_i)``. Node ``j`` branches to ``k-1, k, k+1`` with
  ``k = floor(m / dx_{i+1} + 1/2)``, ``e = m - k dx_{i+1}`` and
  ``p_{-,0,+} = (1 + e^2/V - e sqrt3 / sqrt V) / 6, (2 - e^2/V) / 3,
  (1 + e^2/V + e sqrt3 / sqrt V) / 6``, which match the mean and variance
  exactly and are positive since ``|e| <= dx/2``. This is the
  construction of QuantLib's ``TrinomialTree`` (Ballabio 2020,
  "Implementing QuantLib", ch. 7); the tree's width is bounded by mean
  reversion rather than by an explicit ``j_max``.
* Fit: ``alpha_i`` makes the tree reprice ``P(0, t_{i+1})`` exactly. With
  Arrow-Debreu prices ``Q_{i,j}``,
  ``alpha_i = ln(sum_j Q_{i,j} e^{-x_j dt_i} / P(0, t_{i+1})) / dt_i``
  (Brigo and Mercurio 2006, eq. 3.53 in discrete form), and
  ``Q_{i+1,k} = sum_j Q_{i,j} p_{j->k} e^{-(x_j + alpha_i) dt_i}``.
* Underlying (single curve): entering at ``T_m`` a payer swap with fixed
  dates ``T_{m+1..n}`` is worth ``1 - B_m``, ``B_m = sum_{k>m} K tau_k
  P(T_m, T_k) + P(T_m, T_n)``. ``B`` is rolled back on the tree with its
  cash flows, so the swap and the option share one lattice. The float leg
  is ``1 - P(T_m, T_n)`` per unit notional because it is projected and
  discounted on the same curve; a two-curve basis is not modelled here
  (stated, see ``hull_white`` for the deterministic-spread version of
  the European).
* Bounds (model free given the European prices): the Bermudan is worth at
  least the most valuable co-terminal European (one exercise date is a
  restriction) and at most their sum (the exercised payoff is one of the
  European payoffs on every path). :func:`bermudan_bounds` gives both from
  the analytic Jamshidian formula.

Worked example (``tests/test_bermudan*.py``): flat 3% (continuous),
``a = 0.03``, ``sigma = 0.01``, annual payer struck at 3% on a swap ending
at 5y, exercisable at 1y, 2y, 3y and 4y. The tree at 48 steps a year gives
0.019528 per unit notional; the co-terminal Europeans give the bounds
0.014982 (largest) and 0.048304 (sum); QuantLib's ``TreeSwaptionEngine``
gives 0.019508 at 1000 steps (0.2 bp of notional away) and matches our
tree to 1e-12 at equal step counts.

``bermudan_tree_core`` takes plain arrays (grid, discount factors, cash
flows, exercise flags) so the Rust (``qp_core::rates_vol``) and C++
(``qp/rates_vol.hpp``) ports can mirror it operation for operation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .curve import DiscountCurve
from .hull_white import HullWhite

__all__ = [
    "BermudanResult",
    "bermudan_bounds",
    "bermudan_swaption",
    "bermudan_tree_core",
    "build_time_grid",
]

_SQRT3 = 1.7320508075688772


def bermudan_tree_core(a: float, sigma: float, times: Sequence[float], dfs: Sequence[float],
                       cash: Sequence[float], exercise: Sequence[bool], payer: bool) -> float:
    """Bermudan on a Hull-White trinomial tree; all inputs on one time grid.

    ``times[0] = 0``, ``dfs[i] = P(0, times[i])``, ``cash[i]`` is the
    coupon-bond cash flow paid at ``times[i]`` and ``exercise[i]`` marks an
    exercise date (the swap entered there is worth ``1 - B`` to a payer,
    ``B`` the coupon bond of later cash flows).
    """
    n = len(times) - 1
    if n < 1 or not (len(dfs) == len(cash) == len(exercise) == n + 1):
        raise ValueError("times, dfs, cash and exercise must have equal length >= 2")
    if times[0] != 0.0 or any(times[i + 1] <= times[i] for i in range(n)):
        raise ValueError("times must start at 0 and increase strictly")
    if not (a > 0.0 and sigma > 0.0):
        raise ValueError("need a > 0 and sigma > 0")
    # ---- tree geometry: per level spacing, lowest index, branching.
    # Vectorised over the nodes of a level; every value is the same
    # elementwise expression as the scalar ports, so results match them.
    dx = [0.0] * (n + 1)
    jmin = [0] * (n + 1)
    width = [1] * (n + 1)
    kk: list[np.ndarray] = []
    pp: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(n):
        dt = times[i + 1] - times[i]
        v2 = sigma * sigma * (1.0 - math.exp(-2.0 * a * dt)) / (2.0 * a)
        v = math.sqrt(v2)
        dx[i + 1] = v * _SQRT3
        decay = math.exp(-a * dt)
        x = (jmin[i] + np.arange(width[i])) * dx[i]
        m = x * decay
        k = np.floor(m / dx[i + 1] + 0.5).astype(np.int64)
        e = m - k * dx[i + 1]
        e2 = e * e / v2
        e3 = e * _SQRT3 / v
        kk.append(k)
        pp.append(((1.0 + e2 - e3) / 6.0, (2.0 - e2) / 3.0, (1.0 + e2 + e3) / 6.0))
        jmin[i + 1] = int(k.min()) - 1
        width[i + 1] = int(k.max()) + 1 - jmin[i + 1] + 1
    # ---- forward induction: fit alpha_i to the discount factors.
    # The node sum is a left-to-right loop (not a pairwise np.sum) so the
    # ports reproduce it exactly.
    alpha = [0.0] * n
    q = np.ones(1)
    for i in range(n):
        dt = times[i + 1] - times[i]
        x = (jmin[i] + np.arange(width[i])) * dx[i]
        s = 0.0
        for term in (q * np.exp(-x * dt)).tolist():
            s += term
        alpha[i] = math.log(s / dfs[i + 1]) / dt
        d = q * np.exp(-(x + alpha[i]) * dt)
        c = kk[i] - jmin[i + 1]
        p0, p1, p2 = pp[i]
        qn = np.zeros(width[i + 1])
        np.add.at(qn, c - 1, d * p0)  # unbuffered, in node order
        np.add.at(qn, c, d * p1)
        np.add.at(qn, c + 1, d * p2)
        q = qn
    # ---- backward induction: coupon bond B and option O together
    bond = np.full(width[n], float(cash[n]))
    opt = np.zeros(width[n])
    for i in range(n - 1, -1, -1):
        dt = times[i + 1] - times[i]
        x = (jmin[i] + np.arange(width[i])) * dx[i]
        d = np.exp(-(x + alpha[i]) * dt)
        c = kk[i] - jmin[i + 1]
        p0, p1, p2 = pp[i]
        nb = d * (p0 * bond[c - 1] + p1 * bond[c] + p2 * bond[c + 1])
        no = d * (p0 * opt[c - 1] + p1 * opt[c] + p2 * opt[c + 1])
        if exercise[i]:
            pay = 1.0 - nb if payer else nb - 1.0
            no = np.maximum(no, pay)
        bond, opt = nb + cash[i], no
    return float(opt[0])


def build_time_grid(mandatory: Sequence[float], steps_per_year: int) -> np.ndarray:
    """``0`` plus the mandatory times, each gap cut into ``ceil(gap * steps_per_year)`` equal steps."""
    pts = sorted({0.0, *(round(float(t), 12) for t in mandatory)})
    if pts[0] < 0.0:
        raise ValueError("mandatory times must be >= 0")
    out = [0.0]
    for lo, hi in zip(pts, pts[1:], strict=False):
        m = max(1, math.ceil((hi - lo) * steps_per_year - 1e-9))
        out.extend(lo + (hi - lo) * np.arange(1, m + 1) / m)
        out[-1] = hi
    return np.asarray(out)


@dataclass(frozen=True)
class BermudanResult:
    price: float
    lower: float
    upper: float
    european: tuple[float, ...]


def _check_swap(fixed_times: Sequence[float], accruals: Sequence[float],
                exercise_times: Sequence[float]) -> None:
    if len(accruals) != len(fixed_times) - 1 or len(fixed_times) < 2:
        raise ValueError("need fixed_times T_0..T_n and n accruals")
    if any(b <= a for a, b in zip(fixed_times, fixed_times[1:], strict=False)):
        raise ValueError("fixed_times must increase strictly")
    starts = {round(t, 12) for t in fixed_times[:-1]}
    if not exercise_times or any(round(t, 12) not in starts for t in exercise_times):
        raise ValueError("every exercise time must be one of T_0..T_{n-1}")


def bermudan_swaption(a: float, sigma: float, curve: DiscountCurve,
                      fixed_times: Sequence[float], accruals: Sequence[float], strike: float,
                      exercise_times: Sequence[float], payer: bool = True,
                      steps_per_year: int = 48, notional: float = 1.0) -> float:
    """Bermudan swaption of ``notional`` (single curve) on the Hull-White tree.

    ``fixed_times`` are ``T_0 < ... < T_n`` (years), ``accruals[k-1]`` the
    fixed accrual of ``[T_{k-1}, T_k]``; exercise at ``T_m`` enters the swap
    over ``[T_m, T_n]``.
    """
    _check_swap(fixed_times, accruals, exercise_times)
    grid = build_time_grid([*fixed_times, *exercise_times], steps_per_year)
    key = {round(float(t), 12): i for i, t in enumerate(grid)}
    cash = np.zeros(grid.size)
    for k in range(1, len(fixed_times)):
        cash[key[round(fixed_times[k], 12)]] += strike * accruals[k - 1]
    cash[key[round(fixed_times[-1], 12)]] += 1.0
    ex = np.zeros(grid.size, dtype=bool)
    for t in exercise_times:
        ex[key[round(t, 12)]] = True
    dfs = np.ones(grid.size)
    dfs[1:] = np.asarray(curve.df(grid[1:]), dtype=float)
    return notional * bermudan_tree_core(a, sigma, list(grid), list(dfs), list(cash), list(ex), payer)


def _european(hw: HullWhite, expiry: float, pay_times: Sequence[float],
              coupons: Sequence[float], payer: bool) -> float:
    """Jamshidian on a coupon bond with arbitrary dates, strike 1 at ``expiry``."""
    ti = np.asarray(pay_times, dtype=float)
    c = np.asarray(coupons, dtype=float)
    if expiry <= 0.0:
        bond = float(np.sum(c * np.asarray(hw.curve.df(ti))))
        return max(1.0 - bond, 0.0) if payer else max(bond - 1.0, 0.0)

    def g(r: float) -> float:
        return float(np.sum(c * hw.zcb(expiry, ti, r))) - 1.0

    r_star = brentq(g, -2.0, 2.0, xtol=1e-15)
    ks = hw.zcb(expiry, ti, r_star)
    return float(sum(ci * hw.bond_option(expiry, s, k, call=not payer)
                     for ci, s, k in zip(c, ti, ks, strict=True)))


def bermudan_bounds(a: float, sigma: float, curve: DiscountCurve,
                    fixed_times: Sequence[float], accruals: Sequence[float], strike: float,
                    exercise_times: Sequence[float], payer: bool = True,
                    steps_per_year: int = 48) -> BermudanResult:
    """Tree price with its European lower (max) and upper (sum) bounds."""
    _check_swap(fixed_times, accruals, exercise_times)
    hw = HullWhite(a, sigma, curve)
    euro = []
    for te in exercise_times:
        m = next(i for i, t in enumerate(fixed_times) if abs(t - te) < 1e-12)
        coupons = [strike * accruals[k - 1] for k in range(m + 1, len(fixed_times))]
        coupons[-1] += 1.0
        euro.append(_european(hw, te, fixed_times[m + 1:], coupons, payer))
    price = bermudan_swaption(a, sigma, curve, fixed_times, accruals, strike, exercise_times,
                              payer, steps_per_year)
    return BermudanResult(price, max(euro), sum(euro), tuple(euro))
