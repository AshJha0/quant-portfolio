"""Edge cases: expiry boundaries, extreme yields, degenerate curves/portfolios."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

import fi_rates as fr

D = dt.date


class TestExpiryBoundaries:
    def test_settlement_equals_maturity_price_zero(self, curve, bond_5y):
        # all cashflows (incl. redemption) have been paid at maturity
        assert fr.dirty_price_from_curve(bond_5y, bond_5y.maturity, curve) == 0.0
        assert fr.bond_cashflows(bond_5y, bond_5y.maturity) == []
        assert fr.accrued_interest(bond_5y, bond_5y.maturity) == 0.0

    def test_settlement_equals_maturity_ytm_raises(self, bond_5y):
        with pytest.raises(ValueError, match="no remaining cashflows"):
            fr.price_from_ytm(bond_5y, bond_5y.maturity, 0.04)

    def test_bond_past_maturity_raises(self, curve, bond_5y):
        after = bond_5y.maturity + dt.timedelta(days=1)
        with pytest.raises(ValueError, match="expired"):
            fr.dirty_price_from_curve(bond_5y, after, curve)
        with pytest.raises(ValueError, match="expired"):
            fr.accrued_interest(bond_5y, after)

    def test_zcb_past_maturity_raises(self, curve):
        zcb = fr.ZeroCouponBond(maturity=D(2020, 1, 1))
        with pytest.raises(ValueError, match="expired"):
            fr.zcb_price_from_curve(zcb, D(2026, 8, 18), curve)

    def test_frn_at_maturity_price_zero(self, curve):
        assert fr.frn_price_from_curve(D(2031, 8, 18), D(2031, 8, 18), curve) == 0.0


class TestConstructionValidation:
    def test_zero_coupon_frequency_validation(self, settlement):
        with pytest.raises(ValueError, match="frequency"):
            fr.FixedRateBond(
                effective=settlement,
                maturity=D(2031, 8, 18),
                coupon=0.04,
                frequency=0,
            )
        with pytest.raises(ValueError, match="frequency"):
            fr.FixedRateBond(
                effective=settlement,
                maturity=D(2031, 8, 18),
                coupon=0.04,
                frequency=12,
            )

    def test_negative_coupon_raises(self, settlement):
        with pytest.raises(ValueError, match="coupon"):
            fr.FixedRateBond(
                effective=settlement,
                maturity=D(2031, 8, 18),
                coupon=-0.01,
                frequency=2,
            )

    def test_nonpositive_face_raises(self, settlement):
        with pytest.raises(ValueError, match="face"):
            fr.FixedRateBond(
                effective=settlement,
                maturity=D(2031, 8, 18),
                coupon=0.04,
                frequency=2,
                face=0.0,
            )

    def test_inverted_dates_raise(self, settlement):
        with pytest.raises(ValueError, match="after"):
            fr.FixedRateBond(
                effective=settlement,
                maturity=D(2020, 1, 1),
                coupon=0.04,
                frequency=2,
            )


class TestExtremeYields:
    def test_negative_yield_round_trip(self, settlement, bond_5y):
        p = fr.price_from_ytm(bond_5y, settlement, -0.01)
        assert p > 100.0  # negative yield -> price above par + coupons effect
        y = fr.ytm_from_price(bond_5y, settlement, p)  # settlement on coupon date
        assert y == pytest.approx(-0.01, abs=1e-12)

    def test_deeply_negative_yield_bounded(self, settlement, bond_5y):
        with pytest.raises(ValueError, match="1 \\+ ytm/frequency"):
            fr.price_from_ytm(bond_5y, settlement, -2.5)

    def test_huge_yield_round_trip(self, settlement, bond_5y):
        p = fr.price_from_ytm(bond_5y, settlement, 5.0)  # 500%
        assert 0 < p < 1.0
        y = fr.ytm_from_price(bond_5y, settlement, p)
        assert y == pytest.approx(5.0, rel=1e-10)

    def test_nonpositive_price_raises(self, settlement, bond_5y):
        with pytest.raises(ValueError, match="clean price"):
            fr.ytm_from_price(bond_5y, settlement, -5.0)


class TestNegativeRateCurves:
    def test_bond_pricing_on_negative_curve(self, settlement):
        from fi_rates.data import market_quotes

        curve = fr.bootstrap_curve(market_quotes("negative", noise_bp=0.0))
        bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2031, 8, 18), coupon=0.001, frequency=1
        )
        p = fr.dirty_price_from_curve(bond, settlement, curve)
        assert p > 100.0  # discounting at negative rates
        y = fr.ytm_from_price(bond, settlement, p - fr.accrued_interest(bond, settlement))
        assert y < 0.0

    def test_negative_curve_krd_sum(self, settlement):
        from fi_rates.data import market_quotes

        curve = fr.bootstrap_curve(market_quotes("negative", noise_bp=0.0))
        bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2036, 8, 18), coupon=0.005, frequency=1
        )
        krdv = fr.key_rate_dv01s(bond, settlement, curve)
        assert krdv.sum() == pytest.approx(
            fr.dv01_curve(bond, settlement, curve), rel=1e-6
        )


class TestDegenerate:
    def test_single_pillar_curve_bond_pricing(self, settlement):
        curve = fr.DiscountCurve([30.0], [np.exp(-0.04 * 30)])
        bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2031, 8, 18), coupon=0.04, frequency=2
        )
        p = fr.dirty_price_from_curve(bond, settlement, curve)
        assert np.isfinite(p) and p > 0

    def test_empty_portfolio_value_zero(self, settlement, curve):
        assert fr.portfolio_value([], settlement, curve) == 0.0

    def test_empty_portfolio_risk_empty_frame(self, settlement, curve):
        rpt = fr.portfolio_risk([], settlement, curve)
        assert rpt.empty

    def test_zero_pv_target_krd_raises(self, settlement, curve):
        with pytest.raises(ValueError, match="zero PV"):
            fr.key_rate_durations([], settlement, curve)

    def test_scenario_on_empty_portfolio(self, settlement, curve):
        pnl = fr.scenario_pnl_table([], settlement, curve, [fr.parallel_scenario(100)])
        assert pnl["pnl_full"].iloc[0] == 0.0
