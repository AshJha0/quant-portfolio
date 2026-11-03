"""W2: FX forwards from the spot date, broken dates and NDFs.

Before round 6 ``MarketState`` treated the quoted spot rate as a t = 0
price. A quoted spot rate is for delivery on the spot date, two business
days out (one for USDCAD), so forwards and forward points were off by
about two days of carry. ``spot_date`` defaulted to T+2 for every pair and
let a cross settle on a USD holiday.

References: Clark, *Foreign Exchange Option Pricing* (2011) ch. 1-2;
Wystup, *FX Options and Structured Products* (2nd ed., 2017) ch. 1; EMTA
NDF templates. The spot-date table is checked case by case against the
same rule run on QuantLib calendars (``test_spot_table_matches_quantlib``).
"""

from __future__ import annotations

import datetime as dt
import math
import warnings

import pytest
from fx_rates.curve import DiscountCurve
from fx_rates.daycount import spot_date
from fx_rates.fxforward import (
    FXForward,
    MarketState,
    curve_from_forward_points,
    forward_for_date,
    forward_points,
    forward_points_for_date,
    market_forward,
)
from fx_rates.ndf import NDF, ndf_fixing_date

D = dt.date

# (trade date, pair, spot date). Comments give the reason when a holiday moves it.
SPOT_TABLE = [
    (D(2025, 6, 5), "EURUSD", D(2025, 6, 9)),     # Thu -> Mon over the weekend
    (D(2025, 4, 30), "EURUSD", D(2025, 5, 5)),    # 1 May TARGET holiday
    (D(2025, 7, 2), "EURUSD", D(2025, 7, 7)),     # T+2 = 4 Jul, USD closed
    (D(2025, 7, 3), "EURUSD", D(2025, 7, 7)),     # 4 Jul counts as T+1 for EUR
    (D(2025, 7, 2), "EURJPY", D(2025, 7, 7)),     # cross: no USD-holiday spot
    (D(2025, 7, 2), "GBPJPY", D(2025, 7, 7)),
    (D(2025, 6, 6), "USDCAD", D(2025, 6, 9)),     # T+1
    (D(2025, 6, 30), "USDCAD", D(2025, 7, 2)),    # 1 Jul Canada Day
    (D(2025, 4, 16), "GBPUSD", D(2025, 4, 22)),   # Good Friday, Easter Monday
    (D(2025, 4, 17), "GBPUSD", D(2025, 4, 23)),
    (D(2025, 4, 17), "EURUSD", D(2025, 4, 23)),
    (D(2025, 12, 23), "EURUSD", D(2025, 12, 29)), # 25, 26 Dec TARGET
    (D(2025, 12, 24), "GBPUSD", D(2025, 12, 30)),
    (D(2025, 12, 30), "USDJPY", D(2026, 1, 6)),   # Tokyo 31 Dec - 3 Jan
    (D(2025, 12, 31), "USDJPY", D(2026, 1, 6)),
    (D(2025, 11, 26), "EURUSD", D(2025, 11, 28)), # T+2 Thanksgiving -> Fri
    (D(2025, 11, 25), "EURUSD", D(2025, 11, 28)),
    (D(2025, 5, 1), "USDJPY", D(2025, 5, 7)),     # Golden Week 3-6 May
    (D(2025, 5, 2), "USDJPY", D(2025, 5, 8)),
    (D(2026, 1, 29), "EURUSD", D(2026, 2, 2)),    # month end
    (D(2026, 2, 26), "USDCHF", D(2026, 3, 2)),
    (D(2026, 3, 27), "AUDUSD", D(2026, 3, 31)),   # quarter end
    (D(2025, 4, 24), "AUDUSD", D(2025, 4, 29)),   # Anzac Day
    (D(2025, 8, 29), "GBPUSD", D(2025, 9, 2)),    # Labor Day (USD)
    (D(2025, 5, 22), "EURCHF", D(2025, 5, 27)),   # cross, USD Memorial Day 26 May
    (D(2025, 5, 28), "EURCHF", D(2025, 6, 2)),    # Ascension 29 May (CHF)
    (D(2025, 9, 26), "USDCAD", D(2025, 9, 29)),   # 30 Sep in Canada is later
    (D(2025, 10, 10), "USDCAD", D(2025, 10, 14)), # Thanksgiving CA + Columbus US
    (D(2026, 6, 18), "EURUSD", D(2026, 6, 22)),   # Juneteenth on T+1 counts
    (D(2025, 1, 17), "USDJPY", D(2025, 1, 21)),   # MLK on T+1 counts
    (D(2025, 10, 31), "EURUSD", D(2025, 11, 4)),
    (D(2025, 6, 17), "USDTRY", D(2025, 6, 18)),   # T+1, TRY weekends only
]


class TestSpotDates:
    @pytest.mark.parametrize(("trade", "pair", "expected"), SPOT_TABLE)
    def test_table(self, trade, pair, expected):
        assert spot_date(trade, pair=(pair[:3], pair[3:])) == expected

    def test_usdcad_defaults_to_t_plus_1(self):
        # Before: T+2 unless the caller passed spot_lag_days=1 (Tue 10 Jun).
        assert spot_date(D(2025, 6, 6), pair=("USD", "CAD")) == D(2025, 6, 9)

    def test_cross_does_not_settle_on_a_usd_holiday(self):
        # Before: Fri 4 Jul 2025, when New York is closed.
        assert spot_date(D(2025, 7, 2), pair=("EUR", "JPY")) == D(2025, 7, 7)

    def test_spot_table_matches_quantlib(self):
        ql = pytest.importorskip("QuantLib")
        cal = {
            "USD": ql.UnitedStates(ql.UnitedStates.FederalReserve),
            "GBP": ql.UnitedKingdom(), "EUR": ql.TARGET(), "JPY": ql.Japan(),
            "CHF": ql.Switzerland(), "CAD": ql.Canada(), "AUD": ql.Australia(),
        }
        for trade, pair, expected in SPOT_TABLE:
            b, q = pair[:3], pair[3:]
            if b not in cal or q not in cal:
                continue
            lag = 1 if pair in ("USDCAD",) else 2
            non_usd = [cal[c] for c in (b, q) if c != "USD"]
            count = non_usd[0] if len(non_usd) == 1 else ql.JointCalendar(*non_usd)
            d = ql.Date(trade.day, trade.month, trade.year)
            for _ in range(lag):
                d = count.adjust(d + 1, ql.Following)
            joint = ql.JointCalendar(cal[b], cal[q])
            joint = ql.JointCalendar(joint, cal["USD"])
            d = joint.adjust(d, ql.Following)
            assert D(d.year(), d.month(), d.dayOfMonth()) == expected, (trade, pair)


R_D, R_F, S = 0.045, 0.025, 1.0850
VAL = D(2025, 6, 5)                # Thursday; EURUSD spot Mon 9 Jun
TS = 4 / 365.0


def _flat(r, name):
    times = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    return DiscountCurve(times, [math.exp(-r * t) for t in times], name=name)


def _market(**kw):
    return MarketState(spot=S, domestic_curve=_flat(R_D, "USD"),
                       foreign_curve=_flat(R_F, "EUR"), pair=("EUR", "USD"), **kw)


class TestForwardFromSpotDate:
    def test_spot_time_from_valuation_date(self):
        m = _market(valuation_date=VAL)
        assert m.spot_date == D(2025, 6, 9)
        assert m.spot_time == pytest.approx(TS, abs=1e-15)

    def test_forward_carries_from_spot_not_today(self):
        m = _market(valuation_date=VAL)
        for t in (0.25, 1.0, 5.0):
            assert market_forward(m, t) == pytest.approx(
                S * math.exp((R_D - R_F) * (t - TS)), rel=1e-13)
        # Before: S * exp((r_d - r_f) * 1.0) = 1.106918, which is 2.43 pips
        # too high at 1y: four calendar days of carry at a 2% differential.
        before = S * math.exp((R_D - R_F) * 1.0)
        assert (before - market_forward(m, 1.0)) * 1e4 == pytest.approx(
            S * math.exp(0.02) * (1 - math.exp(-0.02 * TS)) * 1e4, rel=1e-9)

    def test_spot_forward_is_spot(self):
        m = _market(valuation_date=VAL)
        assert market_forward(m, TS) == pytest.approx(S, rel=1e-14)
        assert forward_points(m, TS) == pytest.approx(0.0, abs=1e-9)

    def test_outright_value_identity(self):
        m = _market(valuation_date=VAL)
        fwd = FXForward(1e6, 1.09, 1.0)
        a, b = fwd.value(m, "cashflows"), fwd.value(m, "forward")
        assert a == pytest.approx(b, rel=1e-12)
        assert a == pytest.approx(1e6 * math.exp(-R_D) * (market_forward(m, 1.0) - 1.09),
                                  rel=1e-12)

    def test_explicit_spot_time(self):
        assert market_forward(_market(spot_time=TS), 1.0) == pytest.approx(
            market_forward(_market(valuation_date=VAL), 1.0), rel=1e-15)

    def test_implicit_t0_is_deprecated_and_unchanged(self):
        with pytest.warns(DeprecationWarning, match="spot"):
            m = _market()
        assert m.spot_time == 0.0
        assert market_forward(m, 1.0) == pytest.approx(S * math.exp(R_D - R_F), rel=1e-13)

    def test_spot_at_t0_flag_is_silent_but_deprecated(self):
        with pytest.warns(DeprecationWarning):
            m = _market(spot_at_t0=True)
        assert m.spot_time == 0.0

    def test_conflicting_inputs_rejected(self):
        with pytest.raises(ValueError):
            _market(valuation_date=VAL, spot_time=0.01)
        with pytest.raises(ValueError):
            _market(spot_time=-0.01)


class TestBrokenDates:
    def test_forward_for_any_value_date(self):
        m = _market(valuation_date=VAL)
        for d in (D(2025, 7, 17), D(2025, 11, 3), D(2026, 2, 27)):
            t = (d - VAL).days / 365.0
            assert forward_for_date(m, d) == pytest.approx(
                S * math.exp((R_D - R_F) * (t - TS)), rel=1e-13)

    def test_broken_date_points(self):
        m = _market(valuation_date=VAL)
        d = D(2025, 8, 20)
        assert forward_points_for_date(m, d) == pytest.approx(
            (forward_for_date(m, d) - S) * 1e4, rel=1e-13)

    def test_needs_valuation_date(self):
        with pytest.raises(ValueError, match="valuation_date"):
            forward_for_date(_market(spot_time=TS), D(2025, 8, 20))

    def test_value_date_before_spot_rejected_for_points_strip(self):
        m = _market(valuation_date=VAL)
        with pytest.raises(ValueError):
            forward_for_date(m, D(2025, 6, 4))


class TestPointsStrip:
    def test_reprices_input_strip(self):
        # A forward-points strip (pips) by value date, as a desk quotes it.
        strip = [(D(2025, 7, 9), 18.3), (D(2025, 9, 9), 55.1), (D(2025, 12, 9), 111.0),
                 (D(2026, 6, 9), 221.7), (D(2027, 6, 9), 430.2), (D(2030, 6, 10), 1012.5)]
        dom = _flat(R_D, "USD")
        cur = curve_from_forward_points(S, dom, strip, valuation_date=VAL,
                                        pair=("EUR", "USD"))
        m = MarketState(spot=S, domestic_curve=dom, foreign_curve=cur,
                        pair=("EUR", "USD"), valuation_date=VAL)
        for d, pts in strip:
            assert forward_points_for_date(m, d) == pytest.approx(pts, abs=1e-10)


class TestNDF:
    # USDBRL: BRL per USD, settled in USD. Rates are flat for a closed form.
    R_USD, R_BRL, X = 0.045, 0.11, 5.40

    def _m(self):
        return MarketState(spot=self.X, domestic_curve=_flat(self.R_BRL, "BRL"),
                           foreign_curve=_flat(self.R_USD, "USD"), pair=("USD", "BRL"),
                           valuation_date=VAL)

    def test_fixing_date_two_business_days_before_settlement(self):
        # Settlement Mon 8 Dec 2025: BRL counts weekends only here, USD open
        assert ndf_fixing_date(D(2025, 12, 8), ("USD", "BRL")) == D(2025, 12, 4)
        # Settlement Tue 2 Sep 2025, Mon 1 Sep is Labor Day (USD): fixing Thu 28 Aug
        assert ndf_fixing_date(D(2025, 9, 2), ("USD", "BRL")) == D(2025, 8, 28)

    def test_usd_settled_value_closed_form(self):
        m = self._m()
        settle = D(2025, 12, 8)
        ndf = NDF(10e6, 5.55, ndf_fixing_date(settle, ("USD", "BRL")), settle,
                  pair=("USD", "BRL"), settlement_ccy="USD", fixing_source="BRL09 (PTAX)")
        t = (settle - VAL).days / 365.0
        f = forward_for_date(m, settle)
        assert f == pytest.approx(self.X * math.exp((self.R_BRL - self.R_USD) * (t - TS)),
                                  rel=1e-13)
        expected = 10e6 * math.exp(-self.R_USD * t) * (1.0 - 5.55 / f)
        assert ndf.value(m) == pytest.approx(expected, rel=1e-12)
        assert ndf.settlement_amount(5.70) == pytest.approx(10e6 * (5.70 - 5.55) / 5.70)

    def test_quote_settled_equals_deliverable_forward(self):
        m = self._m()
        settle = D(2026, 3, 9)
        ndf = NDF(10e6, 5.6, ndf_fixing_date(settle, ("USD", "BRL")), settle,
                  pair=("USD", "BRL"), settlement_ccy="BRL")
        t = (settle - VAL).days / 365.0
        fwd = FXForward(10e6, 5.6, t, pair=("USD", "BRL"))
        assert ndf.value(m) == pytest.approx(fwd.value(m), rel=1e-12)

    def test_validation(self):
        with pytest.raises(ValueError):
            NDF(1e6, 5.5, D(2025, 12, 9), D(2025, 12, 8), pair=("USD", "BRL"))
        with pytest.raises(ValueError):
            NDF(1e6, 5.5, D(2025, 12, 4), D(2025, 12, 8), pair=("USD", "BRL"),
                settlement_ccy="EUR")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            m = MarketState(spot=5.4, domestic_curve=_flat(0.1, "BRL"),
                            foreign_curve=_flat(0.04, "USD"), pair=("USD", "BRL"))
        ndf = NDF(1e6, 5.5, D(2025, 12, 4), D(2025, 12, 8), pair=("USD", "BRL"))
        with pytest.raises(ValueError, match="valuation_date"):
            ndf.value(m)
