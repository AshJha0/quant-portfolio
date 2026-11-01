"""Curve scenarios, historical episodes, scenario P&L and carry/roll-down.

A :class:`Scenario` is a vector of zero-rate shifts (in basis points) at
scenario tenors; the shift applied to each *curve pillar* is linearly
interpolated in tenor between scenario points and held flat beyond the ends.
Scenarios are applied to the pillar zero rates and the curve is rebuilt with
its own interpolation, full revaluation, no Taylor approximation.

Historical episodes (all **approximations** of published moves, encoded as
net pillar shifts over the episode compressed into a single instantaneous
shock; see docs/DESK_GUIDE.md for sources and narrative):

* ``taper_tantrum_2013``: bear steepener, May-Sep 2013: 10y UST roughly
  +130bp (≈1.6% → ≈2.9%), front end anchored by the Fed.
* ``hiking_2022``: bear flattener, calendar 2022: 2y roughly +370bp,
  10y roughly +235bp as the Fed hiked; curve inverted.
* ``gfc_2008``: calendar 2008 flight to quality. Year-end 2007 to year-end
  2008 changes in US Treasury constant-maturity yields (FRED series DGS3MO,
  DGS2, DGS5, DGS10, DGS30), rounded: 3m -325bp, 2y -229bp, 5y -190bp,
  10y -179bp, 30y -176bp. The long end rallied almost as much as the 10y.

Carry and roll-down
-------------------
:func:`carry_rolldown` decomposes the horizon P&L of a bond under an
unchanged (static) curve:

* coupon income = coupons received over the horizon + change in accrued
* funding       = dirty price x repo rate x ACT/360 horizon fraction
* carry         = coupon income - funding
* roll-down     = change in clean price on the static curve, split into
  pull-to-par (clean price change at the bond's own constant yield) and
  curve roll-down (the rest)

``carry + rolldown`` equals the static-curve horizon P&L of the position
net of financing. Coupons are not reinvested.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bond import (
    FixedRateBond,
    accrued_interest,
    bond_cashflows,
    dirty_price_from_curve,
)
from .curve import DiscountCurve
from .risk import Position, portfolio_risk, portfolio_value

__all__ = [
    "Scenario",
    "parallel_scenario",
    "steepener_scenario",
    "butterfly_scenario",
    "HISTORICAL_SCENARIOS",
    "apply_scenario",
    "scenario_pnl_table",
    "carry_rolldown",
]


@dataclass(frozen=True)
class Scenario:
    """Named zero-curve shift: ``shifts_bp[i]`` basis points at ``tenors[i]``,
    linearly interpolated in tenor across pillars, flat beyond the ends."""

    name: str
    tenors: tuple[float, ...]
    shifts_bp: tuple[float, ...]
    description: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if len(self.tenors) != len(self.shifts_bp):
            raise ValueError("tenors and shifts_bp must have equal length")
        if len(self.tenors) == 0:
            raise ValueError("scenario needs at least one tenor")
        if any(b - a <= 0 for a, b in zip(self.tenors, self.tenors[1:])):
            raise ValueError("scenario tenors must be strictly increasing")

    def pillar_shifts(self, times: np.ndarray) -> np.ndarray:
        """Absolute (decimal) zero-rate shifts at the given pillar times."""
        t = np.asarray(times, dtype=float)
        bp = np.interp(t, self.tenors, self.shifts_bp)  # flat beyond ends
        return bp * 1e-4


def parallel_scenario(bp: float, name: str | None = None) -> Scenario:
    """Parallel shift of ``bp`` basis points."""
    return Scenario(name or f"parallel_{bp:+g}bp", (1.0,), (bp,))


def steepener_scenario(
    bp_short: float, bp_long: float, pivot: float = 5.0, name: str | None = None
) -> Scenario:
    """Linear tilt around ``pivot`` (zero shift at the pivot): short end moves
    ``bp_short`` at 0y-2y, long end ``bp_long`` at 30y.  ``bp_short < 0 <
    bp_long`` is a steepener; the reverse is a flattener."""
    if not 2.0 < pivot < 30.0:
        raise ValueError(f"pivot must be in (2, 30) years, got {pivot}")
    return Scenario(
        name or f"tilt_{bp_short:+g}/{bp_long:+g}@{pivot:g}y",
        (2.0, pivot, 30.0),
        (bp_short, 0.0, bp_long),
    )


def butterfly_scenario(
    bp_wings: float, bp_belly: float, name: str | None = None
) -> Scenario:
    """Curvature: wings (2y, 30y) move ``bp_wings``, belly (10y) ``bp_belly``."""
    return Scenario(
        name or f"fly_wings{bp_wings:+g}_belly{bp_belly:+g}",
        (2.0, 10.0, 30.0),
        (bp_wings, bp_belly, bp_wings),
    )


HISTORICAL_SCENARIOS: dict[str, Scenario] = {
    "taper_tantrum_2013": Scenario(
        "taper_tantrum_2013",
        (0.25, 2.0, 5.0, 10.0, 30.0),
        (5.0, 30.0, 105.0, 130.0, 105.0),
        "Bear steepener, May-Sep 2013 (approx.): 10y UST +~130bp after the "
        "Fed signalled QE tapering; front end pinned by policy guidance.",
    ),
    "hiking_2022": Scenario(
        "hiking_2022",
        (0.25, 2.0, 5.0, 10.0, 30.0),
        (430.0, 370.0, 275.0, 235.0, 180.0),
        "Bear flattener, calendar 2022 (approx.): 2y +~370bp vs 10y +~235bp "
        "as the Fed hiked 425bp; 2s10s inverted.",
    ),
    "gfc_2008": Scenario(
        "gfc_2008",
        (0.25, 2.0, 5.0, 10.0, 30.0),
        (-325.0, -229.0, -190.0, -179.0, -176.0),
        "Flight to quality, calendar 2008: year-end 2007 to year-end 2008 "
        "changes in UST CMT yields (FRED DGS3MO/DGS2/DGS5/DGS10/DGS30), rounded.",
    ),
}


def apply_scenario(curve: DiscountCurve, scenario: Scenario) -> DiscountCurve:
    """Rebuild the curve with scenario shifts added to its pillar zeros."""
    return curve.bumped_pillars(scenario.pillar_shifts(curve.times))


def scenario_pnl_table(
    positions: list[Position],
    settlement: dt.date,
    curve: DiscountCurve,
    scenarios: list[Scenario],
) -> pd.DataFrame:
    """Full-revaluation scenario P&L vs the duration/convexity estimate.

    The estimate uses the portfolio's aggregate modified duration and
    convexity with one parallel-equivalent move per scenario: the average
    pillar shift weighted by ``P(t_i) * t_i``, the DV01 of a unit zero-coupon
    bond at each curve pillar. The weights belong to the curve, not to the
    portfolio, so the estimate is crude for non-parallel scenarios. That is
    the point of the comparison (docs/VALIDATION.md).

    Columns: ``pnl_full, pnl_dur_conv, error, avg_shift_bp``.
    """
    base_mv = portfolio_value(positions, settlement, curve)
    risk = portfolio_risk(positions, settlement, curve)
    d_mod = float(risk.loc["TOTAL", "mod_duration"]) if len(risk) else 0.0
    conv = float(risk.loc["TOTAL", "convexity"]) if len(risk) else 0.0
    rows = []
    for sc in scenarios:
        shocked = apply_scenario(curve, sc)
        pnl_full = portfolio_value(positions, settlement, shocked) - base_mv
        # unit-ZCB-DV01 (P(t) * t) weighted average pillar shift
        shifts = sc.pillar_shifts(curve.times)
        weights = np.asarray(curve.df(curve.times)) * curve.times
        avg_dy = float(np.sum(shifts * weights) / np.sum(weights))
        est = base_mv * (-d_mod * avg_dy + 0.5 * conv * avg_dy**2)
        rows.append(
            {
                "scenario": sc.name,
                "pnl_full": pnl_full,
                "pnl_dur_conv": est,
                "error": est - pnl_full,
                "avg_shift_bp": avg_dy * 1e4,
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def carry_rolldown(
    bond: FixedRateBond,
    settlement: dt.date,
    horizon: dt.date,
    curve: DiscountCurve,
    z_spread: float = 0.0,
    repo_rate: float = 0.0,
) -> dict[str, float]:
    """Carry and roll-down over ``[settlement, horizon]`` on a static curve.

    The horizon price re-discounts the bond's remaining cashflows off the
    *same* curve with times measured from the horizon date (static-curve /
    unchanged-world assumption; no coupon reinvestment).

    ``repo_rate`` (simple, ACT/360) finances the dirty price over the
    horizon. The default 0 reproduces unfunded carry.

    Returns
    -------
    dict with keys
        ``price_start, price_horizon`` (dirty), ``coupons`` received in
        ``(settlement, horizon]``, ``coupon_income`` (coupons + accrued
        change), ``funding`` (repo cost), ``carry`` (coupon income -
        funding), ``rolldown`` (clean price change), ``pull_to_par`` (clean
        price change at constant own yield), ``curve_rolldown``
        (rolldown - pull_to_par), ``total`` (= carry + rolldown).
    """
    if not np.isfinite(repo_rate):
        raise ValueError(f"repo_rate must be finite, got {repo_rate}")
    if not settlement < horizon <= bond.maturity:
        raise ValueError(
            f"require settlement < horizon <= maturity, got "
            f"{settlement} < {horizon} <= {bond.maturity}"
        )
    dirty0 = dirty_price_from_curve(bond, settlement, curve, z_spread)
    dirty_h = dirty_price_from_curve(bond, horizon, curve, z_spread)
    coupons = sum(
        amt
        for date, amt in bond_cashflows(bond, settlement)
        if date <= horizon and date != bond.maturity
    )
    if horizon == bond.maturity:
        # redemption + final coupon are cashflows, horizon dirty price is 0
        coupons = sum(
            amt for date, amt in bond_cashflows(bond, settlement) if date <= horizon
        )
    acc0 = accrued_interest(bond, settlement)
    acc_h = accrued_interest(bond, horizon)
    coupon_income = coupons + (acc_h - acc0)
    funding = dirty0 * repo_rate * (horizon - settlement).days / 360.0
    carry = coupon_income - funding
    rolldown = (dirty_h - acc_h) - (dirty0 - acc0)
    pull = _pull_to_par(bond, settlement, horizon, dirty0, acc0, acc_h)
    return {
        "price_start": dirty0,
        "price_horizon": dirty_h,
        "coupons": coupons,
        "coupon_income": coupon_income,
        "funding": funding,
        "carry": carry,
        "rolldown": rolldown,
        "pull_to_par": pull,
        "curve_rolldown": rolldown - pull,
        "total": carry + rolldown,
    }


def _pull_to_par(
    bond: FixedRateBond,
    settlement: dt.date,
    horizon: dt.date,
    dirty0: float,
    acc0: float,
    acc_h: float,
) -> float:
    """Clean-price change at the bond's own constant continuous yield."""
    from scipy.optimize import brentq

    flows = bond_cashflows(bond, settlement)
    t = np.array([(d - settlement).days / 365.0 for d, _ in flows])
    cf = np.array([a for _, a in flows])
    y = brentq(lambda r: float(np.sum(cf * np.exp(-r * t))) - dirty0, -0.5, 5.0,
               xtol=1e-15)
    keep = np.array([d > horizon for d, _ in flows])
    th = t[keep] - (horizon - settlement).days / 365.0
    dirty_h = float(np.sum(cf[keep] * np.exp(-y * th)))
    return (dirty_h - acc_h) - (dirty0 - acc0)
