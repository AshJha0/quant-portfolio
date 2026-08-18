"""Transaction cost analysis: Perold implementation-shortfall decomposition.

Implementation shortfall (Perold 1988) compares the *paper* portfolio traded
instantly and costlessly at the decision price ``p_d`` with the *actual*
execution.  With side ``s`` (+1 buy / -1 sell), parent size ``X``, fills
``(q_j, p_j)`` totalling ``Q``, arrival price ``p_a`` (mid at order release)
and end-of-horizon price ``p_T``:

    delay        = s * (p_a - p_d) * X            (decision -> release drift)
    trading      = s * sum_j q_j * (p_j - p_a)    (spread + impact + intra-day drift)
    opportunity  = s * (X - Q) * (p_T - p_a)      (cost of the unfilled tail)
    -----------------------------------------------------------------------
    total IS     = s * [ sum_j q_j p_j + (X - Q) p_T - X p_d ]

The three components sum to the total **exactly** (algebraic identity,
tested to 1e-10).  All dollar figures are converted to bps of the decision
notional ``X * p_d``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .intraday import ExecutionResult

__all__ = ["ISReport", "is_decomposition", "tca_report", "slippage_attribution",
           "aggregate_tca"]


@dataclass(frozen=True)
class ISReport:
    """Implementation-shortfall decomposition, dollars and bps of decision notional."""

    side: int
    parent_qty: float
    filled_qty: float
    decision_price: float
    arrival_price: float
    final_price: float
    avg_fill_price: float
    delay_cost: float
    trading_cost: float
    opportunity_cost: float
    total_is: float

    @property
    def notional(self) -> float:
        return self.parent_qty * self.decision_price

    def bps(self) -> dict[str, float]:
        n = self.notional
        return {
            "delay_bps": self.delay_cost / n * 1e4,
            "trading_bps": self.trading_cost / n * 1e4,
            "opportunity_bps": self.opportunity_cost / n * 1e4,
            "total_is_bps": self.total_is / n * 1e4,
        }


def is_decomposition(side: int, parent_qty: float, decision_price: float,
                     arrival_price: float, final_price: float,
                     fill_qty: Sequence[float], fill_price: Sequence[float]) -> ISReport:
    """Perold IS decomposition of one parent order (see module docstring).

    Raises ``ValueError`` on inconsistent inputs (overfill, negative prices).
    """
    if side not in (1, -1):
        raise ValueError("side must be +1 or -1")
    if parent_qty <= 0:
        raise ValueError("parent_qty must be > 0")
    if min(decision_price, arrival_price, final_price) <= 0:
        raise ValueError("prices must be > 0")
    q = np.asarray(fill_qty, dtype=float)
    p = np.asarray(fill_price, dtype=float)
    if q.shape != p.shape:
        raise ValueError("fill_qty and fill_price must have the same shape")
    if np.any(q < 0):
        raise ValueError("fill quantities must be >= 0")
    filled = float(q.sum())
    if filled > parent_qty * (1 + 1e-9):
        raise ValueError(f"overfill: filled {filled:.2f} > parent {parent_qty:.2f}")
    delay = side * (arrival_price - decision_price) * parent_qty
    trading = side * float((q * (p - arrival_price)).sum())
    opportunity = side * (parent_qty - filled) * (final_price - arrival_price)
    total = side * (float((q * p).sum()) + (parent_qty - filled) * final_price
                    - parent_qty * decision_price)
    avg = float((q * p).sum() / filled) if filled > 0 else np.nan
    return ISReport(
        side=side, parent_qty=parent_qty, filled_qty=filled,
        decision_price=decision_price, arrival_price=arrival_price,
        final_price=final_price, avg_fill_price=avg,
        delay_cost=delay, trading_cost=trading, opportunity_cost=opportunity,
        total_is=total,
    )


def tca_report(result: ExecutionResult) -> ISReport:
    """IS decomposition of a simulated execution (uses only filled buckets)."""
    f = result.fills[result.fills["qty"] > 0]
    return is_decomposition(
        side=result.side, parent_qty=result.parent_qty,
        decision_price=result.decision_price, arrival_price=result.arrival_price,
        final_price=result.final_price,
        fill_qty=f["qty"].to_numpy(), fill_price=f["price"].to_numpy(),
    )


def slippage_attribution(result: ExecutionResult) -> pd.DataFrame:
    """Per-bucket slippage attribution vs arrival, in currency per share.

    For each filled bucket: ``fill - arrival = drift + spread + temporary``
    where ``drift = mid_j - arrival`` (market noise + accumulated permanent
    impact of earlier child orders), ``spread`` is the half-spread paid and
    ``temporary`` the square-root impact.  A ``TOTAL`` row aggregates with
    fill-quantity weights; components sum to the per-share trading slippage
    exactly.
    """
    f = result.fills[result.fills["qty"] > 0].copy()
    if f.empty:
        raise ValueError("no filled buckets to attribute")
    s = result.side
    arr = result.arrival_price
    out = pd.DataFrame(index=f.index)
    out["qty"] = f["qty"]
    out["drift"] = s * (f["mid"] - arr)
    out["spread"] = f["half_spread_cost"]
    out["temporary"] = f["temp_cost"]
    out["total"] = s * (f["price"] - arr)
    w = f["qty"] / f["qty"].sum()
    total_row = (out[["drift", "spread", "temporary", "total"]].mul(w, axis=0)).sum()
    total_row["qty"] = f["qty"].sum()
    out.loc["TOTAL"] = total_row
    return out


def aggregate_tca(reports: Sequence[ISReport]) -> pd.DataFrame:
    """Aggregate TCA stats over a set of orders (all figures in bps).

    Rows: delay / trading / opportunity / total IS; columns: mean, std, min,
    max across orders, plus the qty-weighted mean.
    """
    if len(reports) == 0:
        raise ValueError("no reports to aggregate")
    comp = pd.DataFrame([r.bps() for r in reports])
    notionals = np.array([r.notional for r in reports])
    w = notionals / notionals.sum()
    out = pd.DataFrame({
        "mean": comp.mean(),
        "std": comp.std(ddof=1) if len(reports) > 1 else np.nan,
        "min": comp.min(),
        "max": comp.max(),
        "notional_weighted": comp.mul(w, axis=0).sum(),
    })
    out.index = [c.replace("_bps", "") for c in comp.columns]
    return out
