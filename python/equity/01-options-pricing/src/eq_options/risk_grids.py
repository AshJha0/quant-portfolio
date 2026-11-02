"""Bucketed risk ladders and scenario grids for a book of European options.

Plain idea: a single delta or vega number hides where the risk sits. A
spot ladder reprices the book at a row of spot shifts and shows price,
delta and gamma at each. A vega ladder splits each option's vega between
the two neighbouring expiry buckets, so the book's vega reads per bucket.
A scenario grid revalues the whole book on a spot-by-vol matrix, full
revaluation with no Taylor terms, and reports P&L against today.

Formal statement. Positions are ``(quantity, K, T, option_type)`` on one
underlying with spot ``S``, flat ``r``, ``q`` and ``sigma``.

* Spot ladder at relative shifts ``h``: ``V(S(1+h))``, ``Delta(S(1+h))``,
  ``Gamma(S(1+h))`` by the analytic Black-Scholes formulas.
* Vega bucketing: with bucket tenors ``b_1 < ... < b_m``, an option with
  vega ``v`` and expiry ``T`` in ``[b_j, b_{j+1}]`` puts
  ``v (b_{j+1} - T) / (b_{j+1} - b_j)`` on ``b_j`` and the rest on
  ``b_{j+1}``; outside the range everything goes to the end bucket. This is
  the plain linear split; it is a convention, not a theorem. Bucket totals
  sum to the book vega exactly.
* Scenario grid: ``PnL[i, j] = V(S(1 + h_i), sigma + dv_j) - V(S, sigma)``
  with absolute vol shifts ``dv``.

The oracle is the pricer itself: every grid cell equals a direct
:func:`eq_options.bs_price` call at the shifted inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .black_scholes import bs_price
from .greeks import bs_greeks

__all__ = ["Position", "book_value", "scenario_grid", "spot_ladder", "vega_ladder"]

Position = tuple[float, float, float, str]  # (quantity, K, T, option_type)


def book_value(positions: Sequence[Position], S: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Sum of ``quantity * bs_price`` over the book."""
    return float(sum(n * bs_price(S, k, t, r, sigma, q, ot) for n, k, t, ot in positions))


def spot_ladder(positions: Sequence[Position], S: float, r: float, sigma: float, q: float = 0.0,
                shifts: Sequence[float] = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)) -> dict:
    """Return ``{"shift", "spot", "value", "delta", "gamma"}`` arrays, one entry per shift."""
    h = np.asarray(shifts, dtype=float)
    if np.any(h <= -1.0):
        raise ValueError("shifts must be > -1")
    spots = S * (1.0 + h)
    value, delta, gamma = [], [], []
    for s in spots:
        gs = [(n, bs_greeks(s, k, t, r, sigma, q, ot)) for n, k, t, ot in positions]
        value.append(sum(n * g.price for n, g in gs))
        delta.append(sum(n * g.delta for n, g in gs))
        gamma.append(sum(n * g.gamma for n, g in gs))
    return {"shift": h, "spot": spots, "value": np.array(value),
            "delta": np.array(delta), "gamma": np.array(gamma)}


def vega_ladder(positions: Sequence[Position], S: float, r: float, sigma: float, q: float = 0.0,
                buckets: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 5.0)) -> dict[float, float]:
    """Vega per expiry bucket (linear split between neighbouring buckets)."""
    b = np.asarray(buckets, dtype=float)
    if b.ndim != 1 or b.size == 0 or np.any(np.diff(b) <= 0.0):
        raise ValueError("buckets must be a non-empty increasing sequence")
    out = np.zeros(b.size)
    for n, k, t, ot in positions:
        v = n * bs_greeks(S, k, t, r, sigma, q, ot).vega
        if t <= b[0]:
            out[0] += v
        elif t >= b[-1]:
            out[-1] += v
        else:
            j = int(np.searchsorted(b, t)) - 1
            w = (b[j + 1] - t) / (b[j + 1] - b[j])
            out[j] += v * w
            out[j + 1] += v * (1.0 - w)
    return {float(x): float(y) for x, y in zip(b, out, strict=True)}


def scenario_grid(positions: Sequence[Position], S: float, r: float, sigma: float, q: float = 0.0,
                  spot_shifts: Sequence[float] = (-0.1, -0.05, 0.0, 0.05, 0.1),
                  vol_shifts: Sequence[float] = (-0.05, 0.0, 0.05)) -> np.ndarray:
    """Full-revaluation P&L matrix, rows = spot shifts, columns = vol shifts."""
    base = book_value(positions, S, r, sigma, q)
    grid = np.empty((len(spot_shifts), len(vol_shifts)))
    for i, h in enumerate(spot_shifts):
        for j, dv in enumerate(vol_shifts):
            if h <= -1.0 or sigma + dv < 0.0:
                raise ValueError("shifted spot and vol must stay >= 0")
            grid[i, j] = book_value(positions, S * (1.0 + h), r, sigma + dv, q) - base
    return grid
