"""P&L explain: risk-based P&L against full revaluation.

The plain idea
--------------
At the end of a day the desk knows two numbers for yesterday's book: what it
actually made by full revaluation, and what its risk report said it should
make given the market moves (delta times move, plus half gamma times move
squared, plus the time effect). The gap is the unexplained P&L. A small,
noisy gap means the risk report describes the book; a large or trending gap
means a missing factor, a wrong Greek or a wrong smile dynamic. FRTB turns
this into a pass/fail test (PLA).

The formal statement
--------------------
For start-of-day positions with value ``V(t, x)`` over factors ``x``::

    HPL (hypothetical P&L) = V(t1, x1) + C(t0, t1) - V(t0, x0)
    theta                  = V(t1, x0') + C(t0, t1) - V(t0, x0)
    RTPL (risk-theoretical) = theta + sum_i delta_i dx_i
                              + 0.5 sum_ij Gamma_ij dx_i dx_j
    unexplained            = HPL - RTPL

``C`` are cashflows paid in the period and ``x0'`` is "unchanged market"
rolled to ``t1`` (the caller states what that means: for a curve, unchanged
forwards gives carry and unchanged zero rates by tenor gives carry plus
roll-down). New trades booked in the period are reported separately and
are not part of HPL (BCBS MAR32.24-32.29 define HPL on static positions).
The second-order term is split between factors symmetrically, half of each
cross term to each factor, so the per-factor attribution sums to the total.

PLA statistics (Spearman correlation and Kolmogorov-Smirnov distance between
HPL and RTPL over 250 days, MAR32.42-32.44) live in
``eq_var.frtb_ima.pla_test`` and ``fx_var.frtb_ima.pla_test``; this module
produces their inputs.

Worked example
--------------
A book with delta 100 and gamma 40 to one factor that moves by 0.1, with
theta -1 and a full-revaluation P&L of 9.25, has
RTPL = -1 + 100 x 0.1 + 0.5 x 40 x 0.01 = 9.2 and unexplained 0.05.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from qp_core.model import model

__all__ = ["PnLExplain", "explain", "taylor_attribution"]


@model(id="CORE-RISK-PNL")
def taylor_attribution(
    delta: np.ndarray | Sequence[float],
    move: np.ndarray | Sequence[float],
    gamma: np.ndarray | None = None,
) -> np.ndarray:
    """Per-factor second-order P&L ``delta_i dx_i + 0.5 dx_i (Gamma dx)_i``.

    The vector sums to ``delta . dx + 0.5 dx^T Gamma dx``.
    """
    d = np.asarray(delta, dtype=float)
    dx = np.asarray(move, dtype=float)
    if d.shape != dx.shape or d.ndim != 1:
        raise ValueError("delta and move must be 1-D vectors of equal length")
    out: np.ndarray = d * dx
    if gamma is not None:
        g = np.asarray(gamma, dtype=float)
        if g.shape != (d.size, d.size):
            raise ValueError(f"gamma must be {d.size}x{d.size}, got {g.shape}")
        out = out + 0.5 * dx * np.asarray(0.5 * (g + g.T) @ dx, dtype=float)
    return out


@dataclass(frozen=True)
class PnLExplain:
    """One period's explain. All values in the book's reporting currency."""

    hpl: float
    time: Mapping[str, float]
    risk: Mapping[str, float]
    new_trades: float = 0.0
    notes: tuple[str, ...] = field(default=())

    @property
    def theta(self) -> float:
        """Total time effect (carry, roll-down, option decay)."""
        return float(sum(self.time.values()))

    @property
    def rtpl(self) -> float:
        return self.theta + float(sum(self.risk.values()))

    @property
    def unexplained(self) -> float:
        return self.hpl - self.rtpl

    @property
    def total(self) -> float:
        """HPL plus new-trade P&L: what the desk's book actually made."""
        return self.hpl + self.new_trades

    def as_dict(self) -> dict[str, float]:
        out = {"hpl": self.hpl, **{f"time:{k}": v for k, v in self.time.items()}}
        out.update({f"risk:{k}": v for k, v in self.risk.items()})
        out.update(
            {"theta": self.theta, "rtpl": self.rtpl, "unexplained": self.unexplained,
             "new_trades": self.new_trades, "total": self.total}
        )
        return out


@model(id="CORE-RISK-PNL")
def explain(
    *,
    hpl: float,
    time: Mapping[str, float],
    delta: np.ndarray | Sequence[float],
    move: np.ndarray | Sequence[float],
    labels: Sequence[str],
    gamma: np.ndarray | None = None,
    new_trades: float = 0.0,
) -> PnLExplain:
    """Assemble an explain from full revaluation, time effects and Greeks."""
    attrib = taylor_attribution(delta, move, gamma)
    if len(labels) != attrib.size:
        raise ValueError(f"{len(labels)} labels for {attrib.size} factors")
    values = [hpl, new_trades, *time.values()]
    if not all(np.isfinite(v) for v in values) or not np.all(np.isfinite(attrib)):
        raise ValueError("explain inputs must be finite (NaN policy: refuse)")
    return PnLExplain(
        hpl=float(hpl),
        time={k: float(v) for k, v in time.items()},
        risk={lab: float(v) for lab, v in zip(labels, attrib, strict=True)},
        new_trades=float(new_trades),
    )
