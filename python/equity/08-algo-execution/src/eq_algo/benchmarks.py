"""Execution benchmarks (VWAP / TWAP / arrival) and child-order schedulers.

Benchmark conventions:

- **VWAP** = sum(p*v)/sum(v) over the *market* tape for the order's horizon
  (our own volume is part of the tape in the simulator — documented; on a
  real desk you would use the exchange tape which also includes you).
- **TWAP** = simple average of bucket mid prices over the horizon.
- **Arrival price** = mid at order release; slippage vs arrival is the
  implementation-shortfall style benchmark (Perold).

Slippage sign convention: positive = cost (buy above / sell below the
benchmark), in basis points of the benchmark.
"""

from __future__ import annotations

import numpy as np

from .intraday import ExecutionResult

__all__ = [
    "vwap",
    "twap",
    "arrival_price",
    "slippage_bps",
    "benchmark_slippage",
    "twap_schedule",
    "vwap_schedule",
    "pov_schedule",
]


def vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Volume-weighted average price of a tape.  Requires positive total volume."""
    p = np.asarray(prices, dtype=float)
    v = np.asarray(volumes, dtype=float)
    if p.shape != v.shape:
        raise ValueError("prices and volumes must have the same shape")
    if np.any(v < 0):
        raise ValueError("volumes must be >= 0")
    tot = v.sum()
    if tot <= 0:
        raise ValueError("total volume must be > 0 for VWAP")
    return float((p * v).sum() / tot)


def twap(prices: np.ndarray) -> float:
    """Time-weighted average price: simple mean of the bucket prices."""
    p = np.asarray(prices, dtype=float)
    if p.size == 0:
        raise ValueError("empty price tape")
    return float(p.mean())


def arrival_price(prices: np.ndarray) -> float:
    """Arrival (decision-to-trade) price: the first mid on the tape."""
    p = np.asarray(prices, dtype=float)
    if p.size == 0:
        raise ValueError("empty price tape")
    return float(p[0])


def slippage_bps(avg_exec: float, benchmark: float, side: int) -> float:
    """Signed slippage in bps of the benchmark; positive = cost.

    ``side`` is +1 for a buy (paying above the benchmark costs), -1 for a
    sell (receiving below the benchmark costs).
    """
    if benchmark <= 0:
        raise ValueError("benchmark price must be > 0")
    if side not in (1, -1):
        raise ValueError("side must be +1 or -1")
    return float(side * (avg_exec - benchmark) / benchmark * 1e4)


def benchmark_slippage(result: ExecutionResult) -> dict[str, float]:
    """Slippage of an executed parent order vs VWAP, TWAP and arrival (bps).

    Uses the simulated day's bucket mids as the price tape and the realised
    market volumes as VWAP weights.  Also reports IS vs the decision price.
    """
    mids = result.fills["mid"].to_numpy()
    vols = result.fills["market_volume"].to_numpy()
    avg = result.avg_price
    return {
        "avg_exec": avg,
        "vs_vwap_bps": slippage_bps(avg, vwap(mids, vols), result.side),
        "vs_twap_bps": slippage_bps(avg, twap(mids), result.side),
        "vs_arrival_bps": slippage_bps(avg, result.arrival_price, result.side),
        "vs_decision_bps": slippage_bps(avg, result.decision_price, result.side),
    }


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

def twap_schedule(parent_qty: float, n_buckets: int) -> np.ndarray:
    """Equal slices: ``X / n`` shares in every bucket."""
    if parent_qty <= 0:
        raise ValueError("parent_qty must be > 0")
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")
    return np.full(n_buckets, parent_qty / n_buckets)


def vwap_schedule(parent_qty: float, profile: np.ndarray) -> np.ndarray:
    """Slices proportional to the (expected) volume profile.

    ``q_j = X * p_j / sum(p)``; tracks market volume so the order's average
    price tracks VWAP by construction.
    """
    if parent_qty <= 0:
        raise ValueError("parent_qty must be > 0")
    p = np.asarray(profile, dtype=float)
    if p.size == 0 or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("profile must be non-negative with positive sum")
    return parent_qty * p / p.sum()


def pov_schedule(parent_qty: float, market_volumes: np.ndarray,
                 participation: float) -> np.ndarray:
    """Percentage-of-volume schedule: trade ``participation * V_j`` per bucket
    until done.

    Uses the (forecast or realised) bucket volumes; the cap is respected in
    every bucket.  If the parent cannot complete within the day at this cap,
    a ``ValueError`` explains the shortfall — split the order across days or
    raise the participation deliberately (see docs/DESK_GUIDE.md on caps).
    """
    if parent_qty <= 0:
        raise ValueError("parent_qty must be > 0")
    if not 0 < participation <= 1:
        raise ValueError("participation must be in (0, 1]")
    v = np.asarray(market_volumes, dtype=float)
    if np.any(v < 0):
        raise ValueError("market volumes must be >= 0")
    q = np.zeros(v.size)
    remaining = parent_qty
    for j in range(v.size):
        take = min(remaining, participation * v[j])
        q[j] = take
        remaining -= take
        if remaining <= 1e-12 * parent_qty:
            remaining = 0.0
            break
    if remaining > 0:
        max_qty = participation * v.sum()
        raise ValueError(
            f"parent order of {parent_qty:.0f} shares cannot complete at "
            f"{participation:.1%} participation: day capacity is {max_qty:.0f} "
            f"shares ({remaining:.0f} left). Split across days or raise the cap."
        )
    return q
