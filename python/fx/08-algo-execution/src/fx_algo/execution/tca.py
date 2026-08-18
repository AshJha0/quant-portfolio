"""Transaction-cost analysis for OTC FX executions.

Benchmarks are the FX-native set — **arrival price**, **interval TWAP**
and the **WM/R 4pm London fix** (5-minute window TWAP).  There is no
VWAP benchmark because there is no consolidated FX volume tape.

The implementation-shortfall decomposition is *exact by construction*
(the components are an algebraic partition of the average fill vs the
arrival mid, unit-tested to 1e-10):

``IS = spread+temporary  +  permanent impact  +  market drift``

* spread+temporary — fill price vs the pre-trade mid of its bucket
  (quoted half-spread, sqrt temporary impact, and any last-look
  requote/penalty cost);
* permanent impact — the part of the mid move caused by our own earlier
  child orders;
* market drift — everything else the mid did between arrival and each
  child order (session vol plus any alpha drift: the "cost of waiting").
"""

from __future__ import annotations

import numpy as np

from ..sessions import FIX_HOUR_LONDON, FIX_WINDOW_MINUTES, fix_window_mask
from .simulator import ExecutionResult

__all__ = [
    "decompose_implementation_shortfall",
    "twap_benchmark",
    "fix_benchmark",
    "slippage_vs_benchmark",
    "rejection_cost_pips",
    "venue_comparison",
]


def decompose_implementation_shortfall(result: ExecutionResult) -> dict[str, float]:
    """Exact IS decomposition in pips per unit of executed quantity.

    Parameters
    ----------
    result : ExecutionResult
        Output of ``MarketSimulator.execute``.

    Returns
    -------
    dict
        Keys ``total``, ``spread_temporary``, ``permanent_impact``,
        ``market_drift`` (all pips/unit, positive = cost to the client);
        components sum to ``total`` exactly.
    """
    q = np.abs(result.qty)
    X = q.sum()
    if X == 0:
        return {"total": 0.0, "spread_temporary": 0.0, "permanent_impact": 0.0, "market_drift": 0.0}
    s, pip = result.side, result.pip_size
    spread_temp = float(np.sum(q * s * (result.fills - result.mids_pre)) / (X * pip))
    permanent = float(np.sum(q * s * result.perm_cum_pips) / X)
    drift = float(
        np.sum(q * (s * (result.mids_pre - result.arrival_mid) / pip - s * result.perm_cum_pips)) / X
    )
    total = float(np.sum(q * s * (result.fills - result.arrival_mid)) / (X * pip))
    return {
        "total": total,
        "spread_temporary": spread_temp,
        "permanent_impact": permanent,
        "market_drift": drift,
    }


def twap_benchmark(result: ExecutionResult) -> float:
    """Interval TWAP of pre-trade mids over the execution horizon."""
    return float(result.mids_pre.mean())


def fix_benchmark(
    result: ExecutionResult,
    fix_hour: float = FIX_HOUR_LONDON,
    window_minutes: float = FIX_WINDOW_MINUTES,
) -> float:
    """WM/R-style fix print: TWAP of mids inside the fix window.

    Post-2015 WM/R methodology replaced the 60-second window with a
    5-minute window of (median-filtered) observations; here the print is
    the plain TWAP of simulated mids in the window.

    Raises
    ------
    ValueError
        If the execution grid does not cover the fix window.
    """
    mask = fix_window_mask(result.times_hours, result.dt_minutes, fix_hour, window_minutes)
    if not mask.any():
        raise ValueError("execution grid does not cover the fix window")
    return float(result.mids_pre[mask].mean())


def slippage_vs_benchmark(result: ExecutionResult, benchmark_price: float) -> float:
    """Signed slippage of the average fill vs a benchmark, in pips.

    Positive = the execution cost money relative to the benchmark
    (bought above / sold below it).
    """
    return float(result.side * (result.avg_fill - benchmark_price) / result.pip_size)


def rejection_cost_pips(result: ExecutionResult) -> float:
    """Cost attributable to last-look rejections, pips per unit executed.

    Sum over rejected children of (final fill - original quote), signed
    into the client's cost direction, per unit of total quantity.  Zero
    on a firm venue by construction.
    """
    q = np.abs(result.qty)
    X = q.sum()
    if X == 0:
        return 0.0
    rej = result.rejected
    return float(
        np.sum(q[rej] * result.side * (result.fills[rej] - result.quoted[rej]))
        / (X * result.pip_size)
    )


def venue_comparison(results: dict[str, ExecutionResult]) -> dict[str, dict[str, float]]:
    """Side-by-side venue scorecard (per venue label).

    Returns
    -------
    dict
        Per venue: ``quoted_half_spread_pips`` (qty-weighted average of
        *first-quote* half spreads), ``temp_impact_pips`` (qty-weighted
        temporary impact), ``effective_cost_pips`` (fills vs the
        contemporaneous pre-trade mids — the controllable "spread +
        temporary" cost, which is the venue-attributable part of IS),
        ``rejection_rate``, ``rejection_cost_pips``.  Exact identity
        (tested): ``effective = quoted_half_spread + temp_impact +
        rejection_cost``.
    """
    out: dict[str, dict[str, float]] = {}
    for label, r in results.items():
        q = np.abs(r.qty)
        X = q.sum()
        if X == 0:
            raise ValueError(f"venue {label!r}: empty execution")
        quoted_hs = float(
            np.sum(q * r.side * (r.quoted - r.mids_pre - r.side * r.pip_size * r.temp_pips))
            / (X * r.pip_size)
        )
        temp = float(np.sum(q * r.temp_pips) / X)
        effective = float(np.sum(q * r.side * (r.fills - r.mids_pre)) / (X * r.pip_size))
        out[label] = {
            "quoted_half_spread_pips": quoted_hs,
            "temp_impact_pips": temp,
            "effective_cost_pips": effective,
            "rejection_rate": r.rejection_rate,
            "rejection_cost_pips": rejection_cost_pips(r),
        }
    return out
