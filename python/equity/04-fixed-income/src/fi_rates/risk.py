"""Duration, convexity, DV01 and portfolio risk aggregation.

Definitions (street-convention yield ``y`` compounded ``m`` times/yr, dirty
price ``P``):

* Macaulay duration  ``D_mac = (1/P) sum t_i cf_i (1+y/m)^(-m t_i)`` with
  ``t_i`` in years (period exponents / m).
* Modified duration  ``D_mod = D_mac / (1 + y/m)`` = ``-(1/P) dP/dy``.
* Convexity          ``C = (1/P) d2P/dy2``
  = ``(1/P) sum cf_i n_i (n_i + 1) / m^2 * (1+y/m)^(-n_i-2)``,
  ``n_i`` = exponent in periods.
* DV01 (per bond)    ``DV01 = D_mod * P * 1e-4`` (price change for 1bp).

Curve-based ("effective") measures bump all pillar zero rates in parallel and
reprice: ``DV01_eff = (P(-h) - P(+h)) / 2`` with ``h = 1bp``.

Taylor P&L approximation for a yield move ``dy``:
``dP ~ -D_mod P dy + 0.5 C P dy^2`` — compared against full repricing in
:func:`pnl_approximation_table`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bond import (
    FixedRateBond,
    _period_exponents,
    accrued_interest,
    dirty_price_from_curve,
    price_from_ytm,
    ytm_from_price,
)
from .curve import DiscountCurve

__all__ = [
    "macaulay_duration",
    "modified_duration",
    "convexity",
    "dv01",
    "dv01_curve",
    "convexity_curve",
    "numerical_modified_duration",
    "numerical_convexity",
    "pnl_approximation_table",
    "Position",
    "portfolio_value",
    "portfolio_risk",
]


# ------------------------------------------------------------- YTM analytics
def macaulay_duration(bond: FixedRateBond, settlement: dt.date, ytm: float) -> float:
    """Macaulay duration in years (weighted average cashflow time)."""
    m = bond.frequency
    exps, cf = _period_exponents(bond, settlement)
    disc = (1.0 + ytm / m) ** (-exps)
    pv = cf * disc
    price = pv.sum()
    return float(np.sum((exps / m) * pv) / price)


def modified_duration(bond: FixedRateBond, settlement: dt.date, ytm: float) -> float:
    """Modified duration = Macaulay / (1 + y/m), in years per unit yield."""
    return macaulay_duration(bond, settlement, ytm) / (1.0 + ytm / bond.frequency)


def convexity(bond: FixedRateBond, settlement: dt.date, ytm: float) -> float:
    """Analytic convexity ``(1/P) d2P/dy2`` in 1/yield^2 units."""
    m = bond.frequency
    exps, cf = _period_exponents(bond, settlement)
    disc = (1.0 + ytm / m) ** (-exps - 2.0)
    price = float(np.sum(cf * (1.0 + ytm / m) ** (-exps)))
    c = np.sum(cf * exps * (exps + 1.0) / m**2 * disc)
    return float(c / price)


def dv01(bond: FixedRateBond, settlement: dt.date, ytm: float) -> float:
    """Analytic DV01 per bond: price change for a 1bp yield fall (positive)."""
    p = price_from_ytm(bond, settlement, ytm)
    return modified_duration(bond, settlement, ytm) * p * 1e-4


# ------------------------------------------------------ numerical / off-curve
def numerical_modified_duration(
    bond: FixedRateBond, settlement: dt.date, ytm: float, h: float = 1e-6
) -> float:
    """Central-difference modified duration from the YTM pricer."""
    p0 = price_from_ytm(bond, settlement, ytm)
    up = price_from_ytm(bond, settlement, ytm + h)
    dn = price_from_ytm(bond, settlement, ytm - h)
    return float(-(up - dn) / (2.0 * h) / p0)


def numerical_convexity(
    bond: FixedRateBond, settlement: dt.date, ytm: float, h: float = 1e-5
) -> float:
    """Central second difference convexity from the YTM pricer."""
    p0 = price_from_ytm(bond, settlement, ytm)
    up = price_from_ytm(bond, settlement, ytm + h)
    dn = price_from_ytm(bond, settlement, ytm - h)
    return float((up - 2.0 * p0 + dn) / (h * h) / p0)


def dv01_curve(
    bond: FixedRateBond,
    settlement: dt.date,
    curve: DiscountCurve,
    z_spread: float = 0.0,
    bump: float = 1e-4,
) -> float:
    """Effective DV01: central parallel 1bp zero-curve bump, full repricing."""
    up = dirty_price_from_curve(bond, settlement, curve.bumped_parallel(bump), z_spread)
    dn = dirty_price_from_curve(bond, settlement, curve.bumped_parallel(-bump), z_spread)
    return float((dn - up) / 2.0 * (1e-4 / bump))


def convexity_curve(
    bond: FixedRateBond,
    settlement: dt.date,
    curve: DiscountCurve,
    z_spread: float = 0.0,
    bump: float = 1e-4,
) -> float:
    """Effective convexity off the curve (parallel bump, second difference)."""
    p0 = dirty_price_from_curve(bond, settlement, curve, z_spread)
    up = dirty_price_from_curve(bond, settlement, curve.bumped_parallel(bump), z_spread)
    dn = dirty_price_from_curve(bond, settlement, curve.bumped_parallel(-bump), z_spread)
    return float((up - 2.0 * p0 + dn) / (bump * bump) / p0)


# --------------------------------------------------------- Taylor P&L table
def pnl_approximation_table(
    bond: FixedRateBond,
    settlement: dt.date,
    ytm: float,
    shocks_bp: tuple[float, ...] = (-200, -100, -50, -25, 25, 50, 100, 200),
) -> pd.DataFrame:
    """Duration-only and duration+convexity P&L vs full repricing.

    Returns a DataFrame indexed by shock (bp) with columns
    ``full_repricing, duration_only, duration_convexity, err_duration,
    err_dur_conv`` (all per bond, in price units).
    """
    p0 = price_from_ytm(bond, settlement, ytm)
    d_mod = modified_duration(bond, settlement, ytm)
    conv = convexity(bond, settlement, ytm)
    rows = []
    for bp in shocks_bp:
        dy = bp * 1e-4
        full = price_from_ytm(bond, settlement, ytm + dy) - p0
        dur_only = -d_mod * p0 * dy
        dur_conv = dur_only + 0.5 * conv * p0 * dy * dy
        rows.append(
            {
                "shock_bp": bp,
                "full_repricing": full,
                "duration_only": dur_only,
                "duration_convexity": dur_conv,
                "err_duration": dur_only - full,
                "err_dur_conv": dur_conv - full,
            }
        )
    return pd.DataFrame(rows).set_index("shock_bp")


# ------------------------------------------------------------------ portfolio
@dataclass(frozen=True)
class Position:
    """A holding of ``quantity`` bonds with an optional z-spread and label."""

    bond: FixedRateBond
    quantity: float
    z_spread: float = 0.0
    label: str = ""


def portfolio_value(
    positions: list[Position], settlement: dt.date, curve: DiscountCurve
) -> float:
    """Total dirty market value (currency units)."""
    return float(
        sum(
            pos.quantity
            * dirty_price_from_curve(pos.bond, settlement, curve, pos.z_spread)
            for pos in positions
        )
    )


def portfolio_risk(
    positions: list[Position], settlement: dt.date, curve: DiscountCurve
) -> pd.DataFrame:
    """Per-position and aggregate risk table.

    Columns: market value, weight, YTM, modified duration, convexity,
    DV01 (curve-based, per position).  The ``TOTAL`` row aggregates:
    market-value-weighted duration/convexity, summed MV and DV01.

    Returns an empty DataFrame (documented) for an empty portfolio.
    """
    cols = ["label", "mv", "weight", "ytm", "mod_duration", "convexity", "dv01"]
    if not positions:
        return pd.DataFrame(columns=cols).set_index("label")
    rows = []
    for i, pos in enumerate(positions):
        dirty = dirty_price_from_curve(pos.bond, settlement, curve, pos.z_spread)
        clean = dirty - accrued_interest(pos.bond, settlement)
        y = ytm_from_price(pos.bond, settlement, clean)
        mv = pos.quantity * dirty
        rows.append(
            {
                "label": pos.label or f"pos_{i}",
                "mv": mv,
                "ytm": y,
                "mod_duration": modified_duration(pos.bond, settlement, y),
                "convexity": convexity(pos.bond, settlement, y),
                "dv01": pos.quantity
                * dv01_curve(pos.bond, settlement, curve, pos.z_spread),
            }
        )
    df = pd.DataFrame(rows).set_index("label")
    total_mv = df["mv"].sum()
    df["weight"] = df["mv"] / total_mv
    total = pd.Series(
        {
            "mv": total_mv,
            "weight": 1.0,
            "ytm": float((df["ytm"] * df["weight"]).sum()),
            "mod_duration": float((df["mod_duration"] * df["weight"]).sum()),
            "convexity": float((df["convexity"] * df["weight"]).sum()),
            "dv01": df["dv01"].sum(),
        },
        name="TOTAL",
    )
    return pd.concat([df, total.to_frame().T])[
        ["mv", "weight", "ytm", "mod_duration", "convexity", "dv01"]
    ]
