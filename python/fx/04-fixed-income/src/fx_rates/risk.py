"""Risk measures for FX-linked fixed income positions.

Definitions (all reported in the quote/domestic currency, e.g. USD):

- **FX delta**: dV/dS * S / 100 would be a percent delta; here we report the
  *cash* delta dV/dS in quote currency per unit of spot (so
  ``fx_delta * dS`` is the P&L of a spot move ``dS``).  For an FX forward
  the analytic value is ``N * DF_f_adj(T)`` — the base-currency equivalent
  position (tested).
- **DV01**: change in PV for a +1 bp *parallel* shift of one currency's
  zero curve, central-difference ``(V(+1bp) - V(-1bp)) / 2``.  Signed
  sensitivity (not absolute): a long-base forward has *positive* domestic
  DV01 and *negative* foreign DV01.
- **Key-rate DV01 (KRD)**: same, bumping one pillar zero rate at a time.
- **Basis DV01**: change in PV for a +1 bp shift of the entire cross-currency
  basis spread curve.

The foreign-curve bump shifts the *pure* foreign curve; the basis-adjusted
curve is rebuilt on top, so basis and rate risk are cleanly separated.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from .fxforward import MarketState

__all__ = [
    "Position",
    "fx_delta",
    "dv01",
    "key_rate_dv01",
    "basis_dv01",
    "position_risk",
    "book_value",
    "book_risk_report",
]


class Position(Protocol):
    """Anything with a label and a value in the quote currency."""

    label: str

    def value(self, market: MarketState) -> float: ...


# ---------------------------------------------------------------------- #
# bump helpers
# ---------------------------------------------------------------------- #
def _bumped_market(market: MarketState, currency: str, bp: float) -> MarketState:
    """Market with one currency's zero curve shifted in parallel by ``bp``."""
    if currency == market.quote_ccy or currency == "domestic":
        return market.replace(domestic_curve=market.domestic_curve.parallel_shift(bp))
    if currency == market.base_ccy or currency == "foreign":
        return market.replace(foreign_curve=market.foreign_curve.parallel_shift(bp))
    raise ValueError(
        f"unknown currency {currency!r}; expected one of "
        f"{market.pair} or 'domestic'/'foreign'"
    )


def _shift_basis(market: MarketState, bp: float) -> MarketState:
    spreads = market.basis_spreads
    if not spreads:
        # a flat zero-basis curve so the bump is well-defined
        spreads = ((market.foreign_curve.times[-1], 0.0),)
    return market.replace(
        basis_spreads=tuple((t, s + bp * 1e-4) for t, s in spreads)
    )


# ---------------------------------------------------------------------- #
# measures
# ---------------------------------------------------------------------- #
def fx_delta(position: Position, market: MarketState, rel_bump: float = 1e-5) -> float:
    """Cash FX delta dV/dS (quote ccy per unit of spot), central difference.

    Forwards and swaps are linear in spot, so the central difference is
    exact to machine precision; the FD form also covers any nonlinear
    position implementing the ``Position`` protocol.
    """
    ds = market.spot * rel_bump
    up = position.value(market.replace(spot=market.spot + ds))
    dn = position.value(market.replace(spot=market.spot - ds))
    return (up - dn) / (2.0 * ds)


def dv01(
    position: Position, market: MarketState, currency: str, bump_bp: float = 1.0
) -> float:
    """Signed PV change for a +1 bp parallel shift of ``currency``'s curve."""
    up = position.value(_bumped_market(market, currency, bump_bp))
    dn = position.value(_bumped_market(market, currency, -bump_bp))
    return (up - dn) / 2.0 / bump_bp


def key_rate_dv01(
    position: Position,
    market: MarketState,
    currency: str,
    bump_bp: float = 1.0,
) -> pd.Series:
    """Key-rate DV01 ladder: signed PV change per +1 bp bump of each pillar
    zero rate of ``currency``'s curve.  Index = pillar time (years)."""
    if currency == market.quote_ccy or currency == "domestic":
        curve, attr = market.domestic_curve, "domestic_curve"
    elif currency == market.base_ccy or currency == "foreign":
        curve, attr = market.foreign_curve, "foreign_curve"
    else:
        raise ValueError(f"unknown currency {currency!r}; expected one of {market.pair}")
    out = {}
    for i, t in enumerate(curve.times):
        up = position.value(market.replace(**{attr: curve.pillar_shift(i, bump_bp)}))
        dn = position.value(market.replace(**{attr: curve.pillar_shift(i, -bump_bp)}))
        out[float(t)] = (up - dn) / 2.0 / bump_bp
    return pd.Series(out, name=f"KRD_{currency}")


def basis_dv01(position: Position, market: MarketState, bump_bp: float = 1.0) -> float:
    """Signed PV change for a +1 bp shift of the whole basis spread curve."""
    up = position.value(_shift_basis(market, bump_bp))
    dn = position.value(_shift_basis(market, -bump_bp))
    return (up - dn) / 2.0 / bump_bp


# ---------------------------------------------------------------------- #
# book-level aggregation
# ---------------------------------------------------------------------- #
def position_risk(position: Position, market: MarketState) -> dict[str, float]:
    """PV + full risk vector for one position, in quote currency."""
    return {
        "pv": position.value(market),
        "fx_delta": fx_delta(position, market),
        f"dv01_{market.quote_ccy.lower()}": dv01(position, market, market.quote_ccy),
        f"dv01_{market.base_ccy.lower()}": dv01(position, market, market.base_ccy),
        "basis_dv01": basis_dv01(position, market),
    }


def book_value(book: Sequence[Position], market: MarketState) -> float:
    """Total book PV in the quote currency (0.0 for an empty book)."""
    return float(sum(p.value(market) for p in book))


def book_risk_report(book: Sequence[Position], market: MarketState) -> pd.DataFrame:
    """Per-position risk report with a TOTAL row, all in quote currency.

    Columns: pv, fx_delta, dv01_<quote>, dv01_<base>, basis_dv01.
    An empty book returns a report with only a zero TOTAL row.
    """
    cols = [
        "pv",
        "fx_delta",
        f"dv01_{market.quote_ccy.lower()}",
        f"dv01_{market.base_ccy.lower()}",
        "basis_dv01",
    ]
    rows, index = [], []
    for p in book:
        rows.append(position_risk(p, market))
        index.append(p.label)
    df = pd.DataFrame(rows, index=index, columns=cols)
    if len(df) == 0:
        df = pd.DataFrame(columns=cols, dtype=float)
    df.loc["TOTAL"] = df.sum(axis=0) if len(df) else pd.Series(0.0, index=cols)
    return df
