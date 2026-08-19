"""Scenario engine: joint spot / curve / basis shocks, historical episodes,
and carry (forward-points roll) of FX forward positions.

Historical scenarios are **stylised** calibrations of well-documented
episodes (magnitudes rounded to round numbers of the observed moves):

- ``2008 USD funding squeeze``: Lehman aftermath.  EURUSD 3m cross-currency
  basis blew out beyond -150 bp (intraday prints wider), EUR fell ~12% vs
  USD over the quarter, and front-end USD rates collapsed as the Fed cut.
- ``2020 March dash-for-cash``: COVID USD scramble; EURUSD basis to ~-85 bp
  before Fed swap lines compressed it within weeks; both curves rallied.
- ``EUR year-end turn``: balance-sheet window dressing pushes the basis
  wider by tens of bp over the turn with no comparable move in rates/spot.

See docs/DESK_GUIDE.md for how a desk uses each scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from typing import Sequence

import pandas as pd

from ._validation import require_finite
from .fxforward import FXForward, MarketState, forward_points, market_forward
from .risk import Position, book_value

__all__ = [
    "Scenario",
    "apply_scenario",
    "scenario_table",
    "historical_scenarios",
    "forward_carry",
    "carry_table",
]


@dataclass(frozen=True)
class Scenario:
    """Joint shock: spot (percent), parallel curve shifts (bp) per currency,
    and a parallel basis spread shift (bp)."""

    name: str
    spot_pct: float = 0.0
    domestic_bp: float = 0.0
    foreign_bp: float = 0.0
    basis_bp: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        require_finite(spot_pct=self.spot_pct, domestic_bp=self.domestic_bp,
                       foreign_bp=self.foreign_bp, basis_bp=self.basis_bp)
        if self.spot_pct <= -100.0:
            raise ValueError(
                f"spot_pct must be > -100% (spot stays positive), got {self.spot_pct}"
            )


def apply_scenario(market: MarketState, scenario: Scenario) -> MarketState:
    """Return the shocked market state (curves rebuilt, basis reattached)."""
    spot = market.spot * (1.0 + scenario.spot_pct / 100.0)
    dom = market.domestic_curve
    if scenario.domestic_bp != 0.0:
        dom = dom.parallel_shift(scenario.domestic_bp)
    for_ = market.foreign_curve
    if scenario.foreign_bp != 0.0:
        for_ = for_.parallel_shift(scenario.foreign_bp)
    spreads = market.basis_spreads
    if scenario.basis_bp != 0.0:
        if not spreads:
            spreads = ((market.foreign_curve.times[-1], 0.0),)
        spreads = tuple((t, s + scenario.basis_bp * 1e-4) for t, s in spreads)
    return market.replace(
        spot=spot, domestic_curve=dom, foreign_curve=for_, basis_spreads=spreads
    )


def scenario_table(
    book: Sequence[Position],
    market: MarketState,
    scenarios: Sequence[Scenario],
) -> pd.DataFrame:
    """Full-revaluation scenario P&L table (quote currency).

    Columns: the shock components, scenario book PV and P&L vs base.
    """
    base_pv = book_value(book, market)
    rows = []
    for sc in scenarios:
        pv = book_value(book, apply_scenario(market, sc))
        rows.append(
            {
                "scenario": sc.name,
                "spot_pct": sc.spot_pct,
                "domestic_bp": sc.domestic_bp,
                "foreign_bp": sc.foreign_bp,
                "basis_bp": sc.basis_bp,
                "book_pv": pv,
                "pnl": pv - base_pv,
            }
        )
    df = pd.DataFrame(rows).set_index("scenario")
    df.attrs["base_pv"] = base_pv
    return df


def historical_scenarios() -> list[Scenario]:
    """Stylised historical episodes (magnitudes documented in the module
    docstring and DESK_GUIDE.md)."""
    return [
        Scenario(
            "2008 USD funding squeeze",
            spot_pct=-12.0, domestic_bp=-150.0, foreign_bp=-50.0, basis_bp=-150.0,
            description="Lehman aftermath: basis blowout beyond -150bp, Fed cuts, EUR -12%",
        ),
        Scenario(
            "2020 March dash-for-cash",
            spot_pct=-5.0, domestic_bp=-100.0, foreign_bp=-30.0, basis_bp=-85.0,
            description="COVID USD scramble; swap lines later compressed the basis",
        ),
        Scenario(
            "EUR year-end turn",
            spot_pct=0.0, domestic_bp=0.0, foreign_bp=0.0, basis_bp=-40.0,
            description="Balance-sheet window dressing widens the basis over the turn",
        ),
        Scenario(
            "Fed +100bp hiking surprise",
            spot_pct=2.0, domestic_bp=100.0, foreign_bp=25.0, basis_bp=-10.0,
            description="USD-led tightening; forward points collapse",
        ),
        Scenario(
            "ECB catch-up +75bp",
            spot_pct=3.0, domestic_bp=10.0, foreign_bp=75.0, basis_bp=5.0,
            description="EUR-led repricing; EUR rallies, points widen",
        ),
    ]


# ---------------------------------------------------------------------- #
# carry / roll
# ---------------------------------------------------------------------- #
def forward_carry(
    position: FXForward, market: MarketState, horizon: float
) -> dict[str, float]:
    """Carry of an FX forward over ``horizon`` years with the market frozen.

    The position ages from expiry T to T - h while spot, curves and basis
    are unchanged; the P&L is pure roll down the forward-points curve:

        carry = N * [ DF_d(T-h) * (F(T-h) - K) - DF_d(T) * (F(T) - K) ]

    A long position in the *low-yielding* base currency (positive points,
    e.g. long EURUSD forward pre-2022) rolls *down* — negative carry.

    Returns a dict with carry P&L (quote ccy), the points roll, and the
    start/end forwards.
    """
    require_finite(horizon=horizon)
    if not (0.0 < horizon < position.expiry):
        raise ValueError(
            f"horizon must be in (0, expiry={position.expiry}), got {horizon}"
        )
    aged = _dc_replace(position, expiry=position.expiry - horizon)
    v0 = position.value(market)
    v1 = aged.value(market)
    f0 = market_forward(market, position.expiry)
    f1 = market_forward(market, aged.expiry)
    return {
        "carry_pnl": v1 - v0,
        "points_roll": forward_points(market, aged.expiry)
        - forward_points(market, position.expiry),
        "forward_start": f0,
        "forward_end": f1,
    }


def carry_table(
    position: FXForward, market: MarketState, horizons: Sequence[float]
) -> pd.DataFrame:
    """Carry P&L over several horizons (rows indexed by horizon in years)."""
    rows = {h: forward_carry(position, market, h) for h in horizons}
    df = pd.DataFrame(rows).T
    df.index.name = "horizon_y"
    return df
