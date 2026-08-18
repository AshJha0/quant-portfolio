"""Duration/convexity/DV01 identities, Taylor accuracy, portfolio aggregation."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

import fi_rates as fr

D = dt.date


def _zcb_as_bond(settlement: dt.date, years: int) -> fr.FixedRateBond:
    return fr.FixedRateBond(
        effective=settlement,
        maturity=D(settlement.year + years, settlement.month, settlement.day),
        coupon=0.0,
        frequency=1,
        daycount="30/360US",
    )


class TestDurationIdentities:
    def test_macaulay_of_zcb_equals_maturity(self, settlement):
        bond = _zcb_as_bond(settlement, 7)
        assert fr.macaulay_duration(bond, settlement, 0.04) == pytest.approx(
            7.0, abs=1e-12
        )

    def test_modified_equals_macaulay_over_one_plus_y(self, settlement, bond_5y):
        y = 0.045
        mac = fr.macaulay_duration(bond_5y, settlement, y)
        mod = fr.modified_duration(bond_5y, settlement, y)
        assert mod == pytest.approx(mac / (1 + y / 2), abs=1e-14)

    def test_par_bond_duration_decreasing_in_coupon(self, settlement):
        durs = []
        for cpn in [0.02, 0.04, 0.06, 0.08]:
            bond = fr.FixedRateBond(
                effective=settlement,
                maturity=D(2036, 8, 18),
                coupon=cpn,
                frequency=2,
                daycount="30/360US",
            )
            durs.append(fr.modified_duration(bond, settlement, cpn))  # at par
        assert all(a > b for a, b in zip(durs, durs[1:]))

    def test_duration_increasing_in_maturity(self, settlement):
        durs = []
        for years in [2, 5, 10, 30]:
            bond = fr.FixedRateBond(
                effective=settlement,
                maturity=D(settlement.year + years, 8, 18),
                coupon=0.04,
                frequency=2,
            )
            durs.append(fr.modified_duration(bond, settlement, 0.04))
        assert all(a < b for a, b in zip(durs, durs[1:]))


class TestAnalyticVsNumerical:
    def test_modified_duration_matches_finite_difference(self, settlement, bond_5y):
        y = 0.042
        analytic = fr.modified_duration(bond_5y, settlement, y)
        numerical = fr.numerical_modified_duration(bond_5y, settlement, y)
        assert analytic == pytest.approx(numerical, rel=1e-8)

    def test_convexity_matches_finite_difference(self, settlement, bond_5y):
        y = 0.042
        analytic = fr.convexity(bond_5y, settlement, y)
        numerical = fr.numerical_convexity(bond_5y, settlement, y)
        assert analytic == pytest.approx(numerical, rel=1e-6)

    def test_convexity_positive_vanilla(self, settlement, portfolio):
        for pos in portfolio:
            y = 0.05
            assert fr.convexity(pos.bond, settlement, y) > 0.0

    def test_analytic_vs_curve_dv01_close(self, settlement, curve, bond_5y):
        """YTM-based and curve-based DV01 agree closely for a flat-ish move."""
        clean = fr.clean_price_from_curve(bond_5y, settlement, curve)
        y = fr.ytm_from_price(bond_5y, settlement, clean)
        analytic = fr.dv01(bond_5y, settlement, y)
        eff = fr.dv01_curve(bond_5y, settlement, curve)
        # a 1bp continuous-zero bump moves the periodic ytm by ~(1+y/m) bp,
        # so the conventions differ by exactly that Jacobian to first order
        assert analytic * (1 + y / 2) == pytest.approx(eff, rel=5e-3)
        # raw numbers still within ~2.5%
        assert analytic == pytest.approx(eff, rel=0.025)

    def test_curve_convexity_close_to_ytm_convexity(self, settlement, curve, bond_5y):
        clean = fr.clean_price_from_curve(bond_5y, settlement, curve)
        y = fr.ytm_from_price(bond_5y, settlement, clean)
        # convexity in continuous-zero space vs periodic-yield space differ by
        # the squared Jacobian (1+y/m)^2 plus a first-order duration term
        assert fr.convexity_curve(bond_5y, settlement, curve) == pytest.approx(
            fr.convexity(bond_5y, settlement, y), rel=0.10
        )


class TestTaylorApproximation:
    @pytest.mark.parametrize("shock_bp", [100, -100, 200])
    def test_dur_conv_beats_dur_only(self, settlement, bond_5y, shock_bp):
        tbl = fr.pnl_approximation_table(
            bond_5y, settlement, 0.04, shocks_bp=(shock_bp,)
        )
        row = tbl.loc[shock_bp]
        assert abs(row["err_dur_conv"]) < abs(row["err_duration"])

    def test_duration_only_overstates_losses(self, settlement, bond_5y):
        """Positive convexity: duration-only P&L is always below full."""
        tbl = fr.pnl_approximation_table(bond_5y, settlement, 0.04)
        assert (tbl["duration_only"] <= tbl["full_repricing"] + 1e-12).all()

    def test_full_pnl_reproduced_by_repricing(self, settlement, bond_5y):
        tbl = fr.pnl_approximation_table(bond_5y, settlement, 0.04, shocks_bp=(50,))
        p0 = fr.price_from_ytm(bond_5y, settlement, 0.04)
        p1 = fr.price_from_ytm(bond_5y, settlement, 0.045)
        assert tbl.loc[50, "full_repricing"] == pytest.approx(p1 - p0, abs=1e-12)


class TestPortfolio:
    def test_portfolio_value_sums_positions(self, settlement, curve, portfolio):
        total = fr.portfolio_value(portfolio, settlement, curve)
        manual = sum(
            p.quantity
            * fr.dirty_price_from_curve(p.bond, settlement, curve, p.z_spread)
            for p in portfolio
        )
        assert total == pytest.approx(manual, abs=1e-8)

    def test_total_duration_is_mv_weighted_average(self, settlement, curve, portfolio):
        rpt = fr.portfolio_risk(portfolio, settlement, curve)
        body = rpt.drop(index="TOTAL")
        manual = float((body["mod_duration"] * body["weight"]).sum())
        assert rpt.loc["TOTAL", "mod_duration"] == pytest.approx(manual, abs=1e-12)

    def test_total_dv01_is_sum(self, settlement, curve, portfolio):
        rpt = fr.portfolio_risk(portfolio, settlement, curve)
        body = rpt.drop(index="TOTAL")
        assert rpt.loc["TOTAL", "dv01"] == pytest.approx(
            float(body["dv01"].sum()), abs=1e-10
        )

    def test_weights_sum_to_one(self, settlement, curve, portfolio):
        rpt = fr.portfolio_risk(portfolio, settlement, curve)
        body = rpt.drop(index="TOTAL")
        assert float(body["weight"].sum()) == pytest.approx(1.0, abs=1e-12)

    def test_two_bond_hand_check(self, settlement, curve):
        b1 = _zcb_as_bond(settlement, 2)
        b2 = _zcb_as_bond(settlement, 10)
        positions = [fr.Position(b1, 100.0, label="a"), fr.Position(b2, 100.0, label="b")]
        rpt = fr.portfolio_risk(positions, settlement, curve)
        w = rpt.loc["a", "weight"]
        expected = w * rpt.loc["a", "mod_duration"] + (1 - w) * rpt.loc[
            "b", "mod_duration"
        ]
        assert rpt.loc["TOTAL", "mod_duration"] == pytest.approx(expected, abs=1e-12)
