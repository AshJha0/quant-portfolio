"""``fx_rates`` (fx/04) market state and FX trades into the trade layer.

The snapshot takes the *adjusted* discount curves of the ``MarketState``
(the collateral-consistent curves the old pricers discount with), the spot
quote, and the spot-date year fraction as an explicit ``FXSPOTTIME``
override, so a ``MarketState`` built with ``spot_time`` and no date prices
the same. Discount-factor pillars are copied as stored, so the native curve
reproduces ``fx_rates.DiscountCurve.df`` to the last bit or one ulp of the
log (tested to 1e-12 relative).

Worked example (run in ``python/core/tests/test_trade_layer.py``; not a
doctest, because ``qp_core`` must import without ``fx_rates``)::

    ms = MarketState(1.10, DiscountCurve.from_zero_rates([5.0], [0.04]),
                     DiscountCurve.from_zero_rates([5.0], [0.02]),
                     valuation_date=dt.date(2025, 6, 2))
    old = fx_rates.FXForward(1e6, 1.12, 1.0)
    instrument_from_fx_rates(old).pv(snapshot_from_market_state(ms)).amount
    # == old.value(ms) to 1e-12 relative
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from ...errors import InputError
from ...market.curves import DiscountCurve
from ...market.snapshot import MarketSnapshot
from ..products import FXForward, FXSwap

__all__ = ["instrument_from_fx_rates", "snapshot_from_market_state"]


def _curve(c: Any) -> DiscountCurve:
    return DiscountCurve(c.times, c.dfs)


def snapshot_from_market_state(ms: Any, as_of: dt.date | None = None) -> MarketSnapshot:
    """``fx_rates.MarketState`` -> :class:`MarketSnapshot`.

    ``as_of`` defaults to ``ms.valuation_date``; one of them is required.
    """
    day = as_of or ms.valuation_date
    if day is None:
        raise InputError("MarketState has no valuation_date; pass as_of")
    pair = ms.base_ccy + ms.quote_ccy
    return MarketSnapshot(
        as_of=day,
        quotes={f"FX/{pair}": float(ms.spot), f"FXSPOTTIME/{pair}": float(ms.spot_time)},
        curves={ms.quote_ccy: _curve(ms.domestic_curve_adjusted), ms.base_ccy: _curve(ms.foreign_curve_adjusted)},
        label=f"fx_rates.MarketState {pair}",
    )


def instrument_from_fx_rates(obj: Any) -> FXForward | FXSwap:
    """``fx_rates.FXForward`` or ``fx_rates.FXSwap`` -> native instrument (times in years)."""
    name = type(obj).__name__
    pair = "".join(obj.pair)
    if name == "FXForward":
        return FXForward(pair, float(obj.notional_base), float(obj.strike), float(obj.expiry))
    if name == "FXSwap":
        return FXSwap(pair, float(obj.notional_base), float(obj.near_strike), float(obj.far_strike),
                      float(obj.near_expiry), float(obj.far_expiry))
    raise InputError(f"no trade-layer adapter for fx_rates.{name}")
