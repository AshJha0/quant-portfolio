"""Cross-currency swap initial exchange happens on the spot date.

A spot-starting cross-currency swap exchanges notionals on the spot date
t_s, at the quoted spot rate S, which is the price for delivery on t_s
(Clark 2011, ch. 1). Its value today is therefore

    sign * P_d(t_s) * (N_quote - S * N_base)

in quote currency. Before round 7 the exchange was valued undiscounted,
``sign * (N_quote - S * N_base)``, as if it happened today. The two agree
at inception (both zero when N_quote = S N_base) and differ for an
off-market exchange, by the factor P_d(t_s).
"""

from __future__ import annotations

import datetime as dt
import math

import pytest
from fx_rates.curve import DiscountCurve
from fx_rates.fxforward import MarketState
from fx_rates.xccy import BasisSwap, CrossCurrencySwap

S, R_D, R_F = 1.0850, 0.045, 0.025
VAL = dt.date(2025, 6, 5)          # Thu; EURUSD spot Mon 9 Jun 2025
TS = 4 / 365


def _flat(r, name):
    times = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    return DiscountCurve(times, [math.exp(-r * t) for t in times], name=name)


@pytest.fixture
def market():
    return MarketState(spot=S, domestic_curve=_flat(R_D, "USD"),
                       foreign_curve=_flat(R_F, "EUR"), valuation_date=VAL)


def _ccs(nq, **kw):
    return CrossCurrencySwap(notional_base=10e6, notional_quote=nq, rate_base=0.025,
                             rate_quote=0.045, maturity=2.0, **kw)


def test_off_market_initial_exchange_is_discounted_from_spot(market):
    nq = 11.0e6                              # S * N_base = 10.85m: 150k off market
    with_x = _ccs(nq, include_initial_exchange=True).value(market)
    without = _ccs(nq).value(market)
    exchange = with_x - without
    # before: 150000.00; after: 150000 * exp(-0.045 * 4/365) = 149926.04
    assert exchange == pytest.approx(150_000 * math.exp(-R_D * TS), rel=1e-12)
    assert exchange == pytest.approx(149_926.04, abs=0.01)


def test_at_market_exchange_is_worth_zero(market):
    x = _ccs(S * 10e6, include_initial_exchange=True).value(market)
    assert x == pytest.approx(_ccs(S * 10e6).value(market), abs=1e-6)


def test_explicit_exchange_time_and_cashflows(market):
    swap = _ccs(11.0e6, include_initial_exchange=True, exchange_time=0.0)
    assert swap.value(market) - _ccs(11.0e6).value(market) == pytest.approx(
        150_000 - 10e6 * (market.spot_today - S), rel=1e-12)
    cfs = _ccs(11.0e6, include_initial_exchange=True).cashflows(market)
    assert cfs[0][1] == pytest.approx(TS) and cfs[1][1] == pytest.approx(TS)


def test_pay_side_mirrors(market):
    rec = _ccs(11.0e6, include_initial_exchange=True)
    pay = _ccs(11.0e6, include_initial_exchange=True, receive_base=False)
    assert pay.value(market) == pytest.approx(-rec.value(market), rel=1e-12)


def test_basis_swap_initial_exchange_on_spot_date(market):
    def bs(nq, x):
        return BasisSwap(notional_base=10e6, notional_quote=nq, spread=0.0, maturity=2.0,
                         include_initial_exchange=x)
    # at market: exchange worth zero
    assert bs(S * 10e6, True).value(market) == pytest.approx(
        bs(S * 10e6, False).value(market), abs=1e-6)
    # off market: rec base pays N_base (worth S N_base P_d(t_s)) and receives N_quote P_d(t_s)
    diff = bs(11.0e6, True).value(market) - bs(11.0e6, False).value(market)
    assert diff == pytest.approx(150_000 * math.exp(-R_D * TS), rel=1e-12)
