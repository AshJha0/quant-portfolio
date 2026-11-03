"""Broker-to-smile strangle conversion and shipped holiday calendars.

Oracles: the defining market-strangle condition itself (the smile must
reprice the broker strangle), the RR = 0 identity (broker strikes are the
smile pillars, so the two butterflies coincide), the inverse round trip,
and hand-checked 2026 calendars against the published lists (Federal
Reserve, gov.uk, Cabinet Office of Japan, ECB TARGET2).
"""

import datetime as dt

import pytest
from fx_options.cuts import expiry_and_delivery, spot_date
from fx_options.garman_kohlhagen import gk_price
from fx_options.holidays import CURRENCIES, easter_sunday, holidays, pair_holidays
from fx_options.smile import (
    broker_strangle_from_smile,
    broker_strangle_premium,
    smile_from_quotes,
    smile_strangle_from_broker,
)

MKT = dict(S=1.10, T=0.5, r_d=0.04, r_f=0.02)
CONVS = ["spot", "forward", "spot_pa", "forward_pa"]
D = dt.date


# ------------------------------------------------------------ strangle ---
@pytest.mark.parametrize("conv", CONVS)
def test_zero_rr_broker_equals_smile(conv):
    bf = smile_strangle_from_broker(**MKT, atm=0.08, rr25=0.0, bf_mkt=0.003, convention=conv)
    assert bf == pytest.approx(0.003, abs=1e-13)


@pytest.mark.parametrize("conv", CONVS)
@pytest.mark.parametrize("rr", [-0.03, -0.01, 0.015])
def test_smile_reprices_broker_strangle(conv, rr):
    from fx_options.deltas import strike_from_delta
    bf_s = smile_strangle_from_broker(**MKT, atm=0.08, rr25=rr, bf_mkt=0.003, convention=conv)
    sm = smile_from_quotes(**MKT, atm=0.08, rr25=rr, bf25=bf_s, convention=conv)
    v = 0.083
    kc = strike_from_delta(0.25, MKT["S"], MKT["T"], MKT["r_d"], MKT["r_f"], v, "call", conv)
    kp = strike_from_delta(-0.25, MKT["S"], MKT["T"], MKT["r_d"], MKT["r_f"], v, "put", conv)
    smile_px = (gk_price(MKT["S"], kc, MKT["T"], MKT["r_d"], MKT["r_f"], sm.vol(kc), "call")
                + gk_price(MKT["S"], kp, MKT["T"], MKT["r_d"], MKT["r_f"], sm.vol(kp), "put"))
    target = broker_strangle_premium(**MKT, atm=0.08, bf_mkt=0.003, convention=conv)
    assert smile_px == pytest.approx(target, rel=1e-12)
    # With skew the smile strangle exceeds the broker one (Clark 2011, s. 3.6).
    assert bf_s > 0.003


@pytest.mark.parametrize("conv", CONVS)
@pytest.mark.parametrize(("rr", "bf"), [(-0.02, 0.002), (0.01, 0.006), (-0.04, 0.0005)])
def test_round_trip(conv, rr, bf):
    bf_s = smile_strangle_from_broker(**MKT, atm=0.10, rr25=rr, bf_mkt=bf, convention=conv)
    back = broker_strangle_from_smile(**MKT, atm=0.10, rr25=rr, bf_smile=bf_s, convention=conv)
    assert back == pytest.approx(bf, abs=1e-13)


def test_strangle_bad_inputs():
    with pytest.raises(ValueError):
        broker_strangle_premium(**MKT, atm=0.05, bf_mkt=-0.06)
    with pytest.raises(ValueError):
        smile_strangle_from_broker(**MKT, atm=0.08, rr25=0.0, bf_mkt=0.003, convention="bad")


# ------------------------------------------------------------ holidays ---
def test_easter_known_dates():
    assert [easter_sunday(y) for y in (2019, 2024, 2025, 2026, 2038)] == [
        D(2019, 4, 21), D(2024, 3, 31), D(2025, 4, 20), D(2026, 4, 5), D(2038, 4, 25)]


def test_usd_2026_fed_calendar():
    # Jul 4 is a Saturday: the Fed does not observe it on Friday.
    assert sorted(holidays("USD", 2026)) == [
        D(2026, 1, 1), D(2026, 1, 19), D(2026, 2, 16), D(2026, 5, 25), D(2026, 6, 19),
        D(2026, 9, 7), D(2026, 10, 12), D(2026, 11, 11), D(2026, 11, 26), D(2026, 12, 25)]
    assert D(2021, 6, 18) not in holidays("USD", 2021)  # Juneteenth from 2022
    assert D(2023, 1, 2) in holidays("USD", 2023)  # Sunday New Year -> Monday


def test_gbp_eur_2026_and_specials():
    assert sorted(holidays("GBP", 2026)) == [
        D(2026, 1, 1), D(2026, 4, 3), D(2026, 4, 6), D(2026, 5, 4), D(2026, 5, 25),
        D(2026, 8, 31), D(2026, 12, 25), D(2026, 12, 28)]
    g22 = holidays("GBP", 2022)
    assert {D(2022, 6, 2), D(2022, 6, 3), D(2022, 9, 19), D(2022, 12, 26), D(2022, 12, 27)} <= g22
    assert D(2022, 5, 30) not in g22 and D(2023, 5, 8) in holidays("GBP", 2023)
    assert sorted(holidays("EUR", 2026)) == [
        D(2026, 1, 1), D(2026, 4, 3), D(2026, 4, 6), D(2026, 5, 1), D(2026, 12, 25)]


def test_jpy_2026_substitute_and_citizens_day():
    j = holidays("JPY", 2026)
    assert sorted(j) == [
        D(2026, 1, 1), D(2026, 1, 2), D(2026, 1, 12), D(2026, 2, 11), D(2026, 2, 23),
        D(2026, 3, 20), D(2026, 4, 29), D(2026, 5, 4), D(2026, 5, 5), D(2026, 5, 6),
        D(2026, 7, 20), D(2026, 8, 11), D(2026, 9, 21), D(2026, 9, 22), D(2026, 9, 23),
        D(2026, 10, 12), D(2026, 11, 3), D(2026, 11, 23), D(2026, 12, 31)]
    j21 = holidays("JPY", 2021)
    assert {D(2021, 7, 22), D(2021, 7, 23), D(2021, 8, 9)} <= j21 and D(2021, 10, 11) not in j21


def test_other_centres_spot_checks():
    assert {D(2026, 5, 14), D(2026, 5, 25), D(2026, 8, 1) - dt.timedelta(0)} - {D(2026, 8, 1)} \
        <= holidays("CHF", 2026)  # Ascension, Whit Monday
    assert {D(2026, 5, 18), D(2026, 9, 30), D(2026, 12, 28)} <= holidays("CAD", 2026)
    assert {D(2026, 1, 26), D(2026, 6, 8), D(2026, 8, 3), D(2026, 10, 5)} <= holidays("AUD", 2026)


@pytest.mark.parametrize("ccy", CURRENCIES)
def test_calendars_deterministic_weekdays_in_range(ccy):
    for y in (2010, 2026, 2099):
        h = holidays(ccy, y)
        assert h == holidays(ccy.lower(), y) and all(d.weekday() < 5 and d.year == y for d in h)
        assert 3 <= len(h) <= 20  # EUR 2010 has 3: May 1 and Christmas on weekends


def test_holiday_errors():
    with pytest.raises(ValueError):
        holidays("NZD", 2026)
    with pytest.raises(ValueError):
        holidays("USD", 1999)
    with pytest.raises(ValueError):
        holidays("USD", 2100)
    with pytest.raises(ValueError):
        pair_holidays("EURUSD", 2027, 2026)
    with pytest.raises(ValueError):
        pair_holidays("EUR", 2026, 2026)


# ------------------------------------------------- wired into cuts/dates ---
def test_auto_calendar_in_dates():
    # Trade Wed 2026-11-25: Thanksgiving Thu 11-26 is a USD holiday, spot Mon 11-30.
    assert spot_date(D(2026, 11, 25), "EURUSD", "auto") == D(2026, 11, 30)
    assert spot_date(D(2026, 11, 25), "EURUSD") == D(2026, 11, 27)
    assert pair_holidays("EURUSD", 2026, 2026) == holidays("EUR", 2026) | holidays("USD", 2026)
    # 1M from Mon 2026-11-23 (spot Wed 11-25 under USD calendar too): delivery 12-28 is a
    # GBP holiday, so GBPUSD rolls to Tue 12-29.
    exp, dlv = expiry_and_delivery(D(2026, 11, 23), "1M", "GBPUSD", "auto")
    assert dlv == D(2026, 12, 29) and spot_date(exp, "GBPUSD", "auto") <= dlv
    with pytest.raises(ValueError):
        spot_date(D(2026, 11, 25), "EURUSD", "none")
