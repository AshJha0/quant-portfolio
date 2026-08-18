"""Bond pricing: ZCB/DF identity, YTM round-trips, clean/dirty, FRN, z-spread."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

import fi_rates as fr

D = dt.date


class TestZeroCoupon:
    def test_zcb_price_equals_df_exactly(self, settlement, curve):
        zcb = fr.ZeroCouponBond(maturity=D(2031, 8, 18), face=100.0)
        t = fr.curve_time(settlement, zcb.maturity)
        assert fr.zcb_price_from_curve(zcb, settlement, curve) == 100.0 * curve.df(t)

    def test_coupon_bond_is_sum_of_zcbs(self, settlement, curve, bond_5y):
        flows = fr.bond_cashflows(bond_5y, settlement)
        total = sum(
            amt / 100.0 * fr.zcb_price_from_curve(
                fr.ZeroCouponBond(maturity=date), settlement, curve
            )
            for date, amt in flows
        )
        got = fr.dirty_price_from_curve(bond_5y, settlement, curve)
        assert got == pytest.approx(total, abs=1e-10)


class TestYTM:
    def test_round_trip_price_ytm_price(self, settlement, curve, bond_5y):
        clean = fr.clean_price_from_curve(bond_5y, settlement, curve)
        y = fr.ytm_from_price(bond_5y, settlement, clean)
        back = fr.price_from_ytm(bond_5y, settlement, y) - fr.accrued_interest(
            bond_5y, settlement
        )
        assert back == pytest.approx(clean, abs=1e-10)

    def test_round_trip_mid_period(self, curve, bond_5y):
        mid = D(2027, 11, 3)  # between coupon dates
        clean = fr.clean_price_from_curve(bond_5y, mid, curve)
        y = fr.ytm_from_price(bond_5y, mid, clean)
        back = fr.price_from_ytm(bond_5y, mid, y) - fr.accrued_interest(bond_5y, mid)
        assert back == pytest.approx(clean, abs=1e-10)

    def test_par_bond_annual_ytm_equals_coupon(self):
        bond = fr.FixedRateBond(
            effective=D(2026, 8, 18),
            maturity=D(2031, 8, 18),
            coupon=0.05,
            frequency=1,
            daycount="30/360US",
        )
        assert fr.price_from_ytm(bond, D(2026, 8, 18), 0.05) == pytest.approx(
            100.0, abs=1e-12
        )
        assert fr.ytm_from_price(bond, D(2026, 8, 18), 100.0) == pytest.approx(
            0.05, abs=1e-12
        )

    def test_par_bond_semiannual_ytm_equals_coupon(self):
        bond = fr.FixedRateBond(
            effective=D(2026, 2, 15),
            maturity=D(2036, 2, 15),
            coupon=0.04,
            frequency=2,
            daycount="30/360US",
        )
        # street convention: at a coupon date, price(y=coupon) == par
        assert fr.price_from_ytm(bond, D(2026, 2, 15), 0.04) == pytest.approx(
            100.0, abs=1e-12
        )
        assert fr.ytm_from_price(bond, D(2028, 2, 15), 100.0) == pytest.approx(
            0.04, abs=1e-12
        )

    def test_price_decreasing_in_yield(self, settlement, bond_5y):
        ys = np.linspace(-0.02, 0.30, 25)
        ps = [fr.price_from_ytm(bond_5y, settlement, y) for y in ys]
        assert all(a > b for a, b in zip(ps, ps[1:]))

    def test_premium_discount_relation(self, settlement, bond_5y):
        # yield below coupon -> premium; above -> discount (at a coupon date)
        assert fr.price_from_ytm(bond_5y, settlement, 0.03) > 100.0
        assert fr.price_from_ytm(bond_5y, settlement, 0.05) < 100.0


class TestCleanDirtyAccrued:
    def test_accrued_zero_at_coupon_dates(self, bond_5y):
        assert fr.accrued_interest(bond_5y, D(2026, 8, 18)) == 0.0
        assert fr.accrued_interest(bond_5y, D(2028, 2, 18)) == 0.0

    def test_accrued_zero_at_maturity(self, bond_5y):
        assert fr.accrued_interest(bond_5y, bond_5y.maturity) == 0.0

    def test_accrued_quarter_period_30360(self, bond_5y):
        # 3 months into a semiannual 4% period: accrued = 100 * 4% * 0.25 = 1.0
        assert fr.accrued_interest(bond_5y, D(2026, 11, 18)) == pytest.approx(
            1.0, abs=1e-12
        )

    def test_clean_plus_accrued_is_dirty(self, curve, bond_5y):
        s = D(2027, 5, 3)
        dirty = fr.dirty_price_from_curve(bond_5y, s, curve)
        clean = fr.clean_price_from_curve(bond_5y, s, curve)
        assert dirty == pytest.approx(clean + fr.accrued_interest(bond_5y, s), abs=1e-12)

    def test_dirty_continuous_across_coupon_date(self, flat_curve, bond_5y):
        """Dirty price drops by the coupon across a payment date; clean is
        continuous (approximately, over one day)."""
        before = D(2027, 2, 17)
        on = D(2027, 2, 18)  # coupon date: coupon no longer in remaining flows
        dirty_before = fr.dirty_price_from_curve(bond_5y, before, flat_curve)
        dirty_on = fr.dirty_price_from_curve(bond_5y, on, flat_curve)
        assert dirty_before - dirty_on == pytest.approx(2.0, abs=0.02)
        clean_before = fr.clean_price_from_curve(bond_5y, before, flat_curve)
        clean_on = fr.clean_price_from_curve(bond_5y, on, flat_curve)
        assert clean_before == pytest.approx(clean_on, abs=0.02)


class TestFRN:
    @pytest.mark.parametrize("freq", [1, 2, 4])
    def test_frn_at_par_at_reset(self, settlement, curve, freq):
        price = fr.frn_price_from_curve(
            settlement, D(2031, 8, 18), curve, frequency=freq
        )
        assert price == pytest.approx(100.0, abs=1e-10)

    def test_frn_at_par_on_flat_curve(self, settlement, flat_curve):
        price = fr.frn_price_from_curve(settlement, D(2036, 8, 18), flat_curve)
        assert price == pytest.approx(100.0, abs=1e-10)

    def test_frn_with_margin_above_par(self, settlement, curve):
        price = fr.frn_price_from_curve(
            settlement, D(2031, 8, 18), curve, quoted_margin=0.005
        )
        assert price > 100.0

    def test_frn_invalid_frequency_raises(self, settlement, curve):
        with pytest.raises(ValueError, match="frequency"):
            fr.frn_price_from_curve(settlement, D(2031, 8, 18), curve, frequency=3)


class TestZSpread:
    def test_round_trip(self, settlement, curve, bond_5y):
        z_true = 0.0175
        clean = fr.clean_price_from_curve(bond_5y, settlement, curve, z_true)
        z = fr.z_spread_from_price(bond_5y, settlement, curve, clean)
        assert z == pytest.approx(z_true, abs=1e-10)

    def test_zero_spread_for_curve_price(self, settlement, curve, bond_5y):
        clean = fr.clean_price_from_curve(bond_5y, settlement, curve)
        z = fr.z_spread_from_price(bond_5y, settlement, curve, clean)
        assert z == pytest.approx(0.0, abs=1e-12)

    def test_negative_spread_recovered(self, settlement, curve, bond_5y):
        clean = fr.clean_price_from_curve(bond_5y, settlement, curve, -0.004)
        z = fr.z_spread_from_price(bond_5y, settlement, curve, clean)
        assert z == pytest.approx(-0.004, abs=1e-10)


class TestAnnuity:
    def test_annuity_pv_is_sum_of_dfs(self, settlement, curve):
        maturity = D(2031, 8, 18)
        pv = fr.annuity_pv(10.0, settlement, maturity, curve, frequency=1)
        dates = fr.generate_schedule(settlement, maturity, 1)
        manual = 10.0 * sum(
            float(np.asarray(curve.df(fr.curve_time(settlement, d)))) for d in dates
        )
        assert pv == pytest.approx(manual, abs=1e-12)

    def test_annuity_flat_curve_closed_form(self, settlement):
        # annual annuity on flat continuous curve: sum of exp(-z t_i)
        times = [1.0, 2.0, 3.0]
        c = fr.DiscountCurve.from_zero_rates([1.0, 2.0, 3.0], [0.05] * 3)
        pv = fr.annuity_pv(1.0, settlement, D(2029, 8, 18), c, frequency=1)
        # schedule dates are anniversaries; ACT/365F times differ slightly from
        # integers (leap days), so compare against the schedule-based sum
        dates = fr.generate_schedule(settlement, D(2029, 8, 18), 1)
        manual = sum(
            np.exp(-0.05 * fr.curve_time(settlement, d)) for d in dates
        )
        assert pv == pytest.approx(manual, abs=1e-12)
