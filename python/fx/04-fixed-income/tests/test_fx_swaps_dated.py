"""W2b: dated FX swaps: near and far legs, swap points, tom-next, rolls,
mark-to-market of both legs, carry and uneven swaps.

Every expected number is a closed form on flat continuously compounded
curves, P(t) = exp(-r t), so the forward to a value date T (years from
today, ACT/365F) is F(T) = S exp((r_d - r_f)(T - t_s)), with t_s the spot
time (Clark 2011, ch. 1-2; Wystup 2017, ch. 1). Valuation date Thu
5-Jun-2025: EURUSD tom is Fri 6-Jun, spot Mon 9-Jun.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest
from fx_rates.curve import DiscountCurve
from fx_rates.fxforward import FXForward, MarketState, forward_for_date
from fx_rates.swaps import (
    DatedFXSwap,
    fair_swap_points,
    price_fx_swap,
    roll_position,
    tom_next_swap,
)

D = dt.date
S, R_D, R_F = 1.0850, 0.045, 0.025
VAL = D(2025, 6, 5)
TOM, SPOT = D(2025, 6, 6), D(2025, 6, 9)
TS = 4 / 365


def _flat(r, name):
    times = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    return DiscountCurve(times, [math.exp(-r * t) for t in times], name=name)


def _fwd(days):
    return S * math.exp((R_D - R_F) * (days / 365 - TS))


@pytest.fixture
def market():
    return MarketState(spot=S, domestic_curve=_flat(R_D, "USD"),
                       foreign_curve=_flat(R_F, "EUR"), valuation_date=VAL)


def _y(d):
    return (d - VAL).days


class TestSwapPoints:
    def test_spot_3m_points_closed_form(self, market):
        far = D(2025, 9, 9)
        pts = fair_swap_points(market, SPOT, far)
        assert pts == pytest.approx((_fwd(_y(far)) - S) * 1e4, rel=1e-10)
        # 92 days at a 2% differential: about 54.8 pips
        assert pts == pytest.approx(54.83, abs=0.01)

    def test_forward_forward_points(self, market):
        near, far = D(2025, 9, 9), D(2025, 12, 9)
        pts = fair_swap_points(market, near, far)
        assert pts == pytest.approx((_fwd(_y(far)) - _fwd(_y(near))) * 1e4, rel=1e-10)


class TestTomNext:
    def test_dates_and_points(self, market):
        sw = tom_next_swap(market, 1e6)
        assert (sw.near_date, sw.far_date) == (TOM, SPOT)
        # F(tom) = S exp(-(r_d - r_f) * 3/365); points = (S - F(tom)) * 1e4 = 1.78
        assert sw.contract_points() == pytest.approx((S - _fwd(1)) * 1e4, rel=1e-10)
        assert sw.contract_points() == pytest.approx(1.784, abs=1e-3)
        assert sw.far_rate == pytest.approx(S, rel=1e-14)
        assert sw.value(market) == pytest.approx(0.0, abs=1e-8)

    def test_tom_next_over_a_us_holiday(self):
        # Wed 2-Jul-2025: tom Thu 3-Jul, spot Mon 7-Jul (4-Jul is a USD holiday).
        m = MarketState(spot=S, domestic_curve=_flat(R_D, "USD"),
                        foreign_curve=_flat(R_F, "EUR"), valuation_date=D(2025, 7, 2))
        sw = tom_next_swap(m, 1e6)
        assert (sw.near_date, sw.far_date) == (D(2025, 7, 3), D(2025, 7, 7))
        # four nights of carry, not one
        assert sw.contract_points() == pytest.approx(
            S * (1 - math.exp(-(R_D - R_F) * 4 / 365)) * 1e4, rel=1e-10)


class TestLegsAndMtm:
    def test_at_market_swap_is_worth_zero_and_legs_offset(self, market):
        sw = price_fx_swap(market, SPOT, D(2026, 6, 9), 10e6)
        near, far = sw.leg_values(market)
        assert sw.value(market) == pytest.approx(0.0, abs=1e-6)
        assert near == pytest.approx(0.0, abs=1e-6) and far == pytest.approx(0.0, abs=1e-6)

    def test_legs_are_outright_forwards(self, market):
        sw = DatedFXSwap(SPOT, D(2026, 6, 9), near_rate=1.0800, far_rate=1.1000,
                         notional_base=10e6)
        near, far = sw.leg_values(market)
        tn, tf = _y(SPOT) / 365, _y(D(2026, 6, 9)) / 365
        assert near == pytest.approx(
            FXForward(10e6, 1.0800, tn).value(market), rel=1e-12)
        assert far == pytest.approx(
            FXForward(-10e6, 1.1000, tf).value(market), rel=1e-12)
        # near leg: buy 10m EUR at 1.0800 vs spot 1.0850 -> 50,000 USD on spot date
        assert near == pytest.approx(10e6 * (S - 1.0800) * math.exp(-R_D * TS), rel=1e-12)
        assert sw.value(market) == pytest.approx(near + far, rel=1e-15)

    def test_settled_near_leg_drops_out(self, market):
        sw = DatedFXSwap(D(2025, 6, 4), D(2025, 9, 9), 1.08, 1.09, 1e6)
        near, far = sw.leg_values(market)
        assert near == 0.0 and far != 0.0

    def test_sell_buy_mirrors(self, market):
        a = DatedFXSwap(SPOT, D(2026, 6, 9), 1.08, 1.10, 10e6)
        b = DatedFXSwap(SPOT, D(2026, 6, 9), 1.08, 1.10, -10e6)
        assert b.value(market) == pytest.approx(-a.value(market), rel=1e-12)


class TestUneven:
    def test_uneven_swap_carries_an_outright(self, market):
        sw = price_fx_swap(market, SPOT, D(2025, 12, 9), 10e6, far_notional_base=12e6)
        assert sw.outright_exposure == pytest.approx(-2e6)  # sells 2m more EUR far
        assert sw.value(market) == pytest.approx(0.0, abs=1e-6)
        # a spot move hits the near leg through P_d(t_s) and the far leg
        # through P_d(t_s) P_f(T)/P_f(t_s): 10m - 12m x 0.988, about a 1.85m outright
        up = market.replace(spot=S * 1.01)
        far_t = _y(D(2025, 12, 9)) / 365
        want = S * 0.01 * math.exp(-R_D * TS) * (10e6 - 12e6 * math.exp(-R_F * (far_t - TS)))
        assert sw.value(up) == pytest.approx(want, rel=1e-9)
        assert want == pytest.approx(-1.85e6 * S * 0.01 * math.exp(-R_D * TS), rel=0.01)

    def test_off_market_near_rate_solves_far(self, market):
        sw = price_fx_swap(market, SPOT, D(2025, 12, 9), 10e6, near_rate=1.0700)
        assert sw.value(market) == pytest.approx(0.0, abs=1e-6)
        # near leg is worth +150k on spot date; far rate absorbs it with interest
        far_t = _y(D(2025, 12, 9)) / 365
        k_far = _fwd(_y(D(2025, 12, 9))) - (S - 1.0700) * math.exp(R_D * (far_t - TS))
        assert sw.far_rate == pytest.approx(k_far, rel=1e-12)


class TestRolls:
    def test_market_rate_roll(self, market):
        # long 5m EUR forward at 1.0700 maturing on spot; roll to 3M.
        sw = roll_position(market, SPOT, D(2025, 9, 9), 5e6, original_rate=1.0700)
        assert sw.notional_base == -5e6          # sell EUR near (closes the long)
        assert sw.near_rate == pytest.approx(S, rel=1e-14)
        assert sw.far_rate == pytest.approx(_fwd(_y(D(2025, 9, 9))), rel=1e-12)
        assert sw.value(market) == pytest.approx(0.0, abs=1e-6)
        # realised on the near date: 5m * (1.0850 - 1.0700) = 75,000 USD
        assert sw.realised_on_near(1.0700) == pytest.approx(75_000.0, rel=1e-12)

    def test_historical_rate_roll_defers_pnl_into_far_rate(self, market):
        far = D(2025, 9, 9)
        sw = roll_position(market, SPOT, far, 5e6, original_rate=1.0700, historical=True)
        assert sw.near_rate == 1.0700
        assert sw.value(market) == pytest.approx(0.0, abs=1e-6)
        assert sw.realised_on_near(1.0700) == 0.0
        far_t = _y(far) / 365
        k_far = _fwd(_y(far)) - (S - 1.0700) * math.exp(R_D * (far_t - TS))
        assert sw.far_rate == pytest.approx(k_far, rel=1e-12)


def _pv_closed_form(sw, horizon, settled_at_spot=True):
    """PV at ``horizon`` with spot and flat zero rates unchanged; legs before
    ``horizon`` count as settled cash n (S - K), undiscounted."""
    spot_h = {D(2025, 6, 5): SPOT, D(2025, 6, 6): D(2025, 6, 10),
              D(2025, 6, 16): D(2025, 6, 18)}[horizon]
    ts_h = (spot_h - horizon).days / 365
    s_today = S * math.exp(-(R_D - R_F) * ts_h)
    v = 0.0
    for n, k, d in ((sw.notional_base, sw.near_rate, sw.near_date),
                    (-sw.far_notional, sw.far_rate, sw.far_date)):
        if d < horizon:
            v += n * (S - k)
        else:
            t = (d - horizon).days / 365
            v += n * (s_today * math.exp(-R_F * t) - k * math.exp(-R_D * t))
    return v


class TestCarry:
    @pytest.mark.parametrize("horizon", [D(2025, 6, 6), D(2025, 6, 16)])
    def test_carry_is_the_pv_change_with_spot_and_curves_unchanged(self, market, horizon):
        for sw in (price_fx_swap(market, SPOT, D(2026, 6, 9), 10e6),
                   DatedFXSwap(SPOT, D(2026, 6, 9), 1.08, 1.10, 10e6)):
            assert sw.value(market) == pytest.approx(_pv_closed_form(sw, VAL), rel=1e-12,
                                                     abs=1e-6)
            want = _pv_closed_form(sw, horizon) - _pv_closed_form(sw, VAL)
            assert sw.carry(market, horizon) == pytest.approx(want, rel=1e-9, abs=1e-6)

    def test_near_leg_carry_is_one_day_of_rate_differential(self, market):
        # buy-sell at market, one day: the near leg loses about
        # N S (r_d - r_f) / 365 = 594.5 USD and the far leg earns about the
        # same, discounted; the net is small (spot held fixed).
        sw = price_fx_swap(market, SPOT, D(2026, 6, 9), 10e6)
        h = D(2025, 6, 6)
        near_only = DatedFXSwap(sw.near_date, sw.far_date, sw.near_rate, sw.far_rate,
                                sw.notional_base, far_notional_base=0.0)
        assert near_only.carry(market, h) == pytest.approx(-10e6 * S * 0.02 / 365, rel=0.01)
        assert abs(sw.carry(market, h)) < 0.05 * abs(near_only.carry(market, h))

    def test_points_per_day(self, market):
        sw = price_fx_swap(market, SPOT, D(2025, 9, 9), 1e6)
        assert sw.points_per_day() == pytest.approx(sw.contract_points() / 92, rel=1e-14)


def test_validation(market):
    with pytest.raises(ValueError):
        DatedFXSwap(D(2025, 9, 9), SPOT, 1.08, 1.09, 1e6)
    with pytest.raises(ValueError):
        DatedFXSwap(SPOT, D(2025, 9, 9), -1.0, 1.09, 1e6)
    old = MarketState(spot=S, domestic_curve=_flat(R_D, "USD"),
                      foreign_curve=_flat(R_F, "EUR"), spot_time=TS)
    with pytest.raises(ValueError, match="valuation_date"):
        DatedFXSwap(SPOT, D(2025, 9, 9), 1.08, 1.09, 1e6).value(old)
    assert forward_for_date(market, SPOT) == pytest.approx(S, rel=1e-14)
