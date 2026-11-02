"""Phase 5 FX: smile, premium styles, cuts and dates, each against an oracle.

Oracles: the defining identities of the quotes (RR/BF to pillar vols,
pillar deltas), GK inverse-pair symmetry for foreign pips, and hand-checked
calendar dates (weekday arithmetic is checked with ``datetime`` itself).
"""

import datetime as dt
import math

import pytest
from fx_options.cuts import (
    add_business_days,
    cut_datetime_utc,
    expiry_and_delivery,
    spot_date,
    spot_lag,
    time_to_expiry,
)
from fx_options.deltas import delta
from fx_options.garman_kohlhagen import gk_price
from fx_options.premium import convert_premium, premium_amount
from fx_options.smile import smile_from_quotes

MKT = dict(S=1.10, T=0.5, r_d=0.04, r_f=0.02)


# ---------------------------------------------------------------- smile ---
@pytest.mark.parametrize("conv", ["spot", "forward", "spot_pa", "forward_pa"])
def test_smile_pillars_reproduce_quotes(conv):
    sm = smile_from_quotes(**MKT, atm=0.08, rr25=-0.01, bf25=0.003, convention=conv)
    kp, ka, kc = sm.strikes
    assert sm.vol(kc) == pytest.approx(0.078, abs=1e-15)
    assert sm.vol(kp) == pytest.approx(0.088, abs=1e-15)
    assert sm.vol(ka) == pytest.approx(0.080, abs=1e-15)
    rr = sm.vol(kc) - sm.vol(kp)
    bf = 0.5 * (sm.vol(kc) + sm.vol(kp)) - sm.vol(ka)
    assert rr == pytest.approx(-0.01, abs=1e-15) and bf == pytest.approx(0.003, abs=1e-15)
    # ATM is the delta-neutral straddle.
    assert sm.delta(ka, "call") + sm.delta(ka, "put") == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("conv", ["spot", "forward", "spot_pa", "forward_pa"])
def test_smile_consistent_strike_from_delta(conv):
    sm = smile_from_quotes(**MKT, atm=0.08, rr25=-0.01, bf25=0.003, convention=conv)
    assert sm.strike_from_delta(0.25, "call") == pytest.approx(sm.strikes[2], rel=1e-12)
    assert sm.strike_from_delta(-0.25, "put") == pytest.approx(sm.strikes[0], rel=1e-12)
    for target, ot in ((0.10, "call"), (0.35, "call"), (-0.10, "put"), (-0.40, "put")):
        k = sm.strike_from_delta(target, ot)
        assert delta(MKT["S"], k, MKT["T"], MKT["r_d"], MKT["r_f"], sm.vol(k), ot, conv) == \
            pytest.approx(target, abs=1e-12)


def test_smile_flat_wings_and_bad_quotes():
    sm = smile_from_quotes(**MKT, atm=0.08, rr25=-0.01, bf25=0.003)
    assert sm.vol(0.5) == sm.vols[0] == pytest.approx(0.088, abs=1e-15)
    assert sm.vol(3.0) == sm.vols[2] == pytest.approx(0.078, abs=1e-15)
    with pytest.raises(ValueError):
        smile_from_quotes(**MKT, atm=0.01, rr25=0.05, bf25=0.0)


# -------------------------------------------------------------- premium ---
def test_premium_worked_example_and_round_trip():
    S, K = 1.10, 1.12
    assert convert_premium(0.025, "dom_pips", "for_pct", S, K) == pytest.approx(0.025 / 1.10)
    assert convert_premium(0.025, "dom_pips", "dom_pct", S, K) == pytest.approx(0.025 / 1.12)
    v = convert_premium(0.025, "dom_pips", "for_pips", S, K)
    assert v == pytest.approx(0.020292207792207792, rel=1e-14)
    assert convert_premium(v, "for_pips", "dom_pips", S, K) == pytest.approx(0.025, rel=1e-15)
    assert premium_amount(0.025, 1e6, S, K, "foreign") == pytest.approx(25000 / 1.10)


@pytest.mark.parametrize(("K", "ot", "mirror"), [(1.12, "call", "put"), (1.05, "put", "call")])
def test_foreign_pips_equals_inverse_pair_price(K, ot, mirror):
    S, T, rd, rf, vol = 1.10, 0.75, 0.045, 0.025, 0.09
    v = gk_price(S, K, T, rd, rf, vol, ot)
    inv = gk_price(1.0 / S, 1.0 / K, T, rf, rd, vol, mirror)
    assert convert_premium(v, "dom_pips", "for_pips", S, K) == pytest.approx(inv, rel=1e-13)


def test_premium_rejects_unknown_style():
    with pytest.raises(ValueError):
        convert_premium(1.0, "pips", "for_pct", 1.0, 1.0)


# ----------------------------------------------------------------- dates ---
def test_spot_lags_and_weekends():
    mon = dt.date(2026, 11, 2)
    assert mon.weekday() == 0
    assert spot_date(mon, "EURUSD") == dt.date(2026, 11, 4)
    assert spot_date(mon, "USDCAD") == dt.date(2026, 11, 3) and spot_lag("USD/CAD") == 1
    assert spot_date(dt.date(2026, 11, 6), "EURUSD") == dt.date(2026, 11, 10)  # Fri -> Tue
    assert add_business_days(mon, 1, {dt.date(2026, 11, 3)}) == dt.date(2026, 11, 4)


def test_month_tenor_worked_example():
    exp, dlv = expiry_and_delivery(dt.date(2026, 11, 2), "1M", "EURUSD")
    assert (exp, dlv) == (dt.date(2026, 12, 2), dt.date(2026, 12, 4))
    assert spot_date(exp, "EURUSD") == dlv


def test_end_of_month_rule():
    # Trade Tue 2026-04-28, spot Thu 04-30 = last business day of April.
    exp, dlv = expiry_and_delivery(dt.date(2026, 4, 28), "1M", "EURUSD")
    assert dlv == dt.date(2026, 5, 29)  # 05-31 is a Sunday
    assert exp == dt.date(2026, 5, 27)


def test_week_tenor_and_holiday_roll():
    exp, dlv = expiry_and_delivery(dt.date(2026, 11, 2), "1W", "EURUSD", {dt.date(2026, 11, 9)})
    assert exp == dt.date(2026, 11, 10) and dlv == dt.date(2026, 11, 12)


def test_cut_follows_daylight_saving():
    assert cut_datetime_utc(dt.date(2026, 12, 2), "NY").hour == 15  # EST
    assert cut_datetime_utc(dt.date(2026, 7, 1), "NY").hour == 14  # EDT
    assert cut_datetime_utc(dt.date(2026, 7, 1), "TOK").hour == 6  # JST, no DST
    assert cut_datetime_utc(dt.date(2026, 7, 1), "ECB") == dt.datetime(
        2026, 7, 1, 12, 15, tzinfo=dt.UTC)


def test_time_to_expiry_act365f():
    val = dt.datetime(2026, 12, 1, 15, 0, tzinfo=dt.UTC)
    assert time_to_expiry(val, dt.date(2026, 12, 2), "NY") == pytest.approx(1.0 / 365.0, rel=1e-15)
    with pytest.raises(ValueError):
        time_to_expiry(dt.datetime(2026, 12, 1), dt.date(2026, 12, 2))
    assert math.isfinite(time_to_expiry(val, dt.date(2027, 6, 1), "TOK"))
