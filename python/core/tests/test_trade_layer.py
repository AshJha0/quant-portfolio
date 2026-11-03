"""W10 trade layer: snapshot, instruments, trades, portfolios, JSON, adapters."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math

import pytest
from qp_core.black_scholes import carry_greeks, carry_price
from qp_core.errors import InputError
from qp_core.instruments import (
    Cash,
    EquityOption,
    EquityPosition,
    FixedCashflows,
    FXForward,
    FXSwap,
    Instrument,
    Portfolio,
    Trade,
    dumps,
    loads,
)
from qp_core.market import DiscountCurve, MarketSnapshot, Shock, VolSurface

AS_OF = dt.date(2025, 6, 2)  # Monday; EURUSD spot Wed 4 Jun


def snap() -> MarketSnapshot:
    return MarketSnapshot(
        AS_OF,
        quotes={"FX/EURUSD": 1.10, "FX/USDJPY": 150.0, "EQ/SPX": 5000.0, "DIV/SPX": 0.013},
        curves={
            "USD": DiscountCurve.from_zero_rates([0.5, 1.0, 2.0, 5.0], [0.043, 0.041, 0.039, 0.038]),
            "EUR": DiscountCurve.from_zero_rates([0.5, 1.0, 2.0, 5.0], [0.021, 0.020, 0.021, 0.023]),
            "JPY": DiscountCurve.from_zero_rates([1.0, 5.0], [0.004, 0.008]),
        },
        surfaces={"EQ/SPX": VolSurface([0.5, 1.0], [4000.0, 5000.0, 6000.0], [[0.25, 0.2, 0.17], [0.24, 0.2, 0.18]])},
        fixings={"SOFR": {dt.date(2025, 5, 30): 0.0431}},
    )


def book() -> Portfolio:
    return Portfolio([
        Trade("FWD1", FXForward("EURUSD", 1e6, 1.12, dt.date(2026, 6, 4)), book="FX", trade_date=dt.date(2025, 6, 2)),
        Trade("SWP1", FXSwap("EURUSD", 5e6, 1.1, 1.115, dt.date(2025, 6, 4), dt.date(2025, 12, 4)), book="FX"),
        Trade("BND1", FixedCashflows.bullet_bond("USD", 1e6, 0.04, [dt.date(2026, 6, 2), dt.date(2027, 6, 2)],
                                                 [1.0, 1.0]), book="RATES"),
        Trade("EQ1", EquityPosition("SPX", "USD"), quantity=10.0, book="EQ"),
        Trade("OPT1", EquityOption("SPX", "USD", 5100.0, dt.date(2026, 3, 2), call=False), quantity=-5.0, book="EQ"),
        Trade("CASH1", Cash("JPY", 1e7), book="FX"),
    ], name="mixed")


# -- curves -----------------------------------------------------------------
def test_curve_matches_log_linear_rule_and_extrapolates_last_forward() -> None:
    c = DiscountCurve([1.0, 2.0], [0.96, 0.92])
    assert c.df(0.0) == 1.0
    assert c.df(1.5) == pytest.approx(math.sqrt(0.96 * 0.92), rel=1e-15)
    f = math.log(0.96 / 0.92)
    assert c.df(3.0) == pytest.approx(0.92 * math.exp(-f), rel=1e-14)
    with pytest.raises(InputError):
        c.df(-0.1)
    with pytest.raises(InputError):
        DiscountCurve([1.0, 1.0], [0.9, 0.8])


def test_curve_is_bit_identical_to_fx_rates_curve() -> None:
    fx_rates = pytest.importorskip("fx_rates")
    old = fx_rates.DiscountCurve.from_zero_rates([0.25, 1.0, 3.0, 10.0], [0.05, 0.045, 0.04, 0.041])
    new = DiscountCurve(old.times, old.dfs)
    for t in (0.0, 0.1, 0.25, 0.7, 1.0, 2.2, 10.0, 14.5):
        assert new.df(t) == pytest.approx(old.df(t), rel=1e-15, abs=0.0)


def test_vol_surface_linear_total_variance_in_time() -> None:
    s = VolSurface([1.0, 2.0], [100.0], [[0.2], [0.3]])
    w = 0.5 * (0.04 * 1.0) + 0.5 * (0.09 * 2.0)
    assert s.vol(1.5, 100.0) == pytest.approx(math.sqrt(w / 1.5), rel=1e-15)
    assert s.vol(0.5, 100.0) == 0.2 and s.vol(3.0, 100.0) == 0.3


# -- snapshot ---------------------------------------------------------------
def test_snapshot_is_immutable_and_shocks_copy() -> None:
    m = snap()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.as_of = dt.date(2025, 1, 1)  # type: ignore[misc]
    with pytest.raises(TypeError):
        m.quotes["FX/EURUSD"] = 2.0  # type: ignore[index]
    up = m.shock(Shock("curve", "USD", 0.0001, name="usd+1bp"), Shock("vol", "*", 0.01))
    assert m.curve("USD").df(1.0) == pytest.approx(math.exp(-0.041), rel=1e-15)
    assert up.curve("USD").df(1.0) == pytest.approx(math.exp(-0.0411), rel=1e-14)
    assert up.surface("EQ/SPX").vol(1.0, 5000.0) == pytest.approx(0.21)
    assert up.label.endswith("|usd+1bp|vol:*")
    with pytest.raises(InputError):
        m.shock(Shock("quote", "FX/GBPUSD", 0.01))


def test_snapshot_json_round_trip_is_exact() -> None:
    m = snap()
    back = MarketSnapshot.from_json(m.to_json())
    assert back == m and back.to_json() == m.to_json()
    assert back.fixing("SOFR", dt.date(2025, 5, 30)) == 0.0431


def test_fx_today_uses_spot_date_and_crosses() -> None:
    m = snap()
    ts = 2 / 365
    assert m.spot_time("EURUSD") == ts
    eur = 1.10 * m.curve("USD").df(ts) / m.curve("EUR").df(ts)
    assert m.fx_today("EUR", "USD") == eur
    assert m.fx_today("USD", "EUR") == pytest.approx(1 / eur, rel=1e-15)
    jpy_usd = 1 / (150.0 * m.curve("JPY").df(ts) / m.curve("USD").df(ts))
    assert m.fx_today("EUR", "JPY") == pytest.approx(eur / jpy_usd, rel=1e-14)
    with pytest.raises(InputError):
        m.fx_today("GBP", "USD")


# -- instruments ------------------------------------------------------------
def test_every_product_satisfies_the_protocol() -> None:
    for t in book():
        assert isinstance(t.instrument, Instrument) and isinstance(t, Instrument)


def test_fx_forward_closed_form() -> None:
    m = snap()
    f = FXForward("EURUSD", 1e6, 1.12, dt.date(2026, 6, 4))
    t = 367 / 365
    want = 1e6 * m.fx_today("EUR", "USD") * m.curve("EUR").df(t) - 1e6 * 1.12 * m.curve("USD").df(t)
    assert f.pv(m).amount == want and f.pv(m).ccy == "USD"
    assert FXForward("EURUSD", 1e6, 1.12, dt.date(2025, 6, 1)).pv(m).amount == 0.0  # settled


def test_fx_forward_bump_risk_matches_analytic() -> None:
    m = snap()
    f = FXForward("EURUSD", 1e6, 1.12, 1.0)
    ts = m.spot_time("EURUSD")
    r = f.risk(m)
    d_ana = 1e6 * m.curve("USD").df(ts) / m.curve("EUR").df(ts) * m.curve("EUR").df(1.0)
    assert r["delta:FX/EURUSD"] == pytest.approx(d_ana, rel=1e-9)
    # pv01 of the USD leg: d/dz of -N K P_d(T) and of the spot conversion P_d(ts).
    s_t = m.fx_today("EUR", "USD")
    pv01_usd = (1e6 * 1.12 * m.curve("USD").df(1.0) * 1.0 - 1e6 * s_t * m.curve("EUR").df(1.0) * ts) * 1e-4
    assert r["pv01:USD"] == pytest.approx(pv01_usd, rel=1e-6)


def test_fx_swap_is_two_forwards() -> None:
    m = snap()
    s = FXSwap("EURUSD", 5e6, 1.1, 1.115, 0.5, 1.0, far_notional_base=4e6)
    want = FXForward("EURUSD", 5e6, 1.1, 0.5).pv(m).amount + FXForward("EURUSD", -4e6, 1.115, 1.0).pv(m).amount
    assert s.pv(m).amount == want
    assert len(s.cashflows()) == 4


def test_equity_option_equals_scalar_black_scholes() -> None:
    m = snap()
    o = EquityOption("SPX", "USD", 5100.0, dt.date(2026, 3, 2), call=False)
    t = (dt.date(2026, 3, 2) - AS_OF).days / 365
    r = m.curve("USD").zero_rate(t)
    sig = m.surface("EQ/SPX").vol(t, 5100.0)
    want = carry_price(5000.0, 5100.0, t, r, 0.013, sig, -1.0)
    assert abs(o.pv(m).amount - want) <= 1e-12 * abs(want)
    g = carry_greeks(5000.0, 5100.0, t, r, 0.013, sig, False)
    risk = o.risk(m)
    assert risk["delta:EQ/SPX"] == pytest.approx(g.delta, rel=1e-6)
    # vega is per vol point of a parallel grid shift. Between expiries total
    # variance is interpolated, so sigma(t, K) moves by d_sig != 1 per unit shift.
    surf = m.surface("EQ/SPX")
    d_sig = (surf.shifted(1e-6).vol(t, 5100.0) - surf.shifted(-1e-6).vol(t, 5100.0)) / 2e-6
    assert d_sig != pytest.approx(1.0, rel=1e-7)
    assert risk["vega:EQ/SPX"] == pytest.approx(g.vega * 0.01 * d_sig, rel=1e-6)


def test_bond_and_cash() -> None:
    m = snap()
    b = FixedCashflows.bullet_bond("USD", 100.0, 0.05, [1.0, 2.0], [1.0, 1.0])
    want = 5.0 * m.curve("USD").df(1.0) + 105.0 * m.curve("USD").df(2.0)
    assert b.pv(m).amount == want
    pv01 = -(5.0 * m.curve("USD").df(1.0) * 1.0 + 105.0 * m.curve("USD").df(2.0) * 2.0) * 1e-4
    assert b.risk(m)["pv01:USD"] == pytest.approx(pv01, rel=1e-7)
    assert Cash("EUR", 7.0).pv(m).amount == 7.0
    assert Cash("EUR", 7.0, -0.5).pv(m).amount == 0.0


# -- trades and portfolios --------------------------------------------------
def test_trade_scales_and_portfolio_converts() -> None:
    m = snap()
    p = book()
    total = sum(t.pv(m).amount * m.fx_today(t.pv(m).ccy, "EUR") for t in p)
    assert p.pv(m, "EUR").amount == pytest.approx(total, rel=1e-15)
    eq = p["EQ1"]
    assert eq.pv(m).amount == 50000.0 and eq.risk(m)["delta:EQ/SPX"] == pytest.approx(10.0, rel=1e-12)
    assert len(p.filter("FX")) == 3 and len(p.remove("EQ1")) == 5
    with pytest.raises(InputError):
        Portfolio([p["EQ1"], p["EQ1"]])
    assert sum(1 for c in p.cashflows() if c.ccy == "EUR") == 3


def test_mixed_book_round_trips_unchanged() -> None:
    m = snap()
    p = book()
    text = dumps(p)
    back = loads(text)
    assert back == p
    assert dumps(back) == text
    assert back.pv(m, "USD").amount == p.pv(m, "USD").amount


def test_unknown_type_rejected() -> None:
    with pytest.raises(InputError):
        loads('{"type":"Nope"}')


# -- adapters: PV through the trade layer equals the scalar functions -------
def test_fx_rates_adapter_matches_scalar_pricer() -> None:
    fx_rates = pytest.importorskip("fx_rates")
    from qp_core.instruments.adapters.fx_rates_adapter import (
        instrument_from_fx_rates,
        snapshot_from_market_state,
    )

    dom = fx_rates.DiscountCurve.from_zero_rates([0.5, 1.0, 2.0, 5.0], [0.043, 0.041, 0.039, 0.038])
    fgn = fx_rates.DiscountCurve.from_zero_rates([0.5, 1.0, 2.0, 5.0], [0.021, 0.020, 0.021, 0.023])
    states = [
        fx_rates.MarketState(1.10, dom, fgn, valuation_date=AS_OF),
        fx_rates.MarketState(1.10, dom, fgn, basis_spreads=((1.0, -0.002), (5.0, -0.0025)), spot_time=2 / 365),
    ]
    olds = [fx_rates.FXForward(1e6, 1.12, 0.75), fx_rates.FXForward(-3e6, 1.05, 4.0),
            fx_rates.FXSwap(2e6, 1.1, 0.5, 1.12, 2.5)]
    for ms in states:
        m = snapshot_from_market_state(ms, as_of=AS_OF)
        for old in olds:
            got = instrument_from_fx_rates(old).pv(m).amount
            want = old.value(ms)
            assert abs(got - want) <= 1e-12 * abs(old.notional_base), (old, got, want)
    with pytest.raises(InputError):
        snapshot_from_market_state(states[1])
