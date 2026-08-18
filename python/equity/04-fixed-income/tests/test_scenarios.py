"""Scenario application, P&L consistency, historical episodes, carry/roll."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

import fi_rates as fr

D = dt.date


def _single_bond_book(settlement: dt.date, years: int, coupon: float):
    bond = fr.FixedRateBond(
        effective=settlement,
        maturity=D(settlement.year + years, settlement.month, settlement.day),
        coupon=coupon,
        frequency=2,
    )
    return [fr.Position(bond, 1000.0, label=f"{years}y")]


class TestScenarioApplication:
    def test_parallel_scenario_matches_direct_bump(self, settlement, curve, portfolio):
        sc = fr.parallel_scenario(100)
        shocked = fr.apply_scenario(curve, sc)
        direct = curve.bumped_parallel(100e-4)
        np.testing.assert_allclose(shocked.dfs, direct.dfs, rtol=0, atol=1e-15)
        pnl = fr.scenario_pnl_table(portfolio, settlement, curve, [sc])
        manual = fr.portfolio_value(portfolio, settlement, direct) - fr.portfolio_value(
            portfolio, settlement, curve
        )
        assert pnl.loc[sc.name, "pnl_full"] == pytest.approx(manual, abs=1e-6)

    def test_pillar_shifts_interpolated_and_flat_beyond(self, curve):
        sc = fr.Scenario("test", (2.0, 10.0), (0.0, 80.0))
        shifts = sc.pillar_shifts(curve.times)
        # flat before 2y, linear to 10y, flat after
        assert shifts[curve.times <= 2.0].max() == 0.0
        i6 = int(np.argmin(np.abs(curve.times - 5.0)))
        assert shifts[i6] == pytest.approx((5.0 - 2.0) / 8.0 * 80e-4, abs=1e-12)
        assert shifts[curve.times >= 10.0].min() == pytest.approx(80e-4, abs=1e-15)

    def test_applied_zeros_shift_consistently(self, curve):
        sc = fr.HISTORICAL_SCENARIOS["taper_tantrum_2013"]
        shocked = fr.apply_scenario(curve, sc)
        np.testing.assert_allclose(
            shocked.zero_rates - curve.zero_rates,
            sc.pillar_shifts(curve.times),
            atol=1e-15,
        )

    def test_butterfly_shifts(self, curve):
        sc = fr.butterfly_scenario(-25, +50)
        s = sc.pillar_shifts(np.array([2.0, 10.0, 30.0]))
        np.testing.assert_allclose(s, [-25e-4, 50e-4, -25e-4], atol=1e-15)


class TestScenarioSigns:
    def test_rates_up_long_book_loses(self, settlement, curve, portfolio):
        pnl = fr.scenario_pnl_table(
            portfolio, settlement, curve, [fr.parallel_scenario(100)]
        )
        assert pnl["pnl_full"].iloc[0] < 0

    def test_steepener_hurts_long_duration_book(self, settlement, curve):
        long_book = _single_bond_book(settlement, 30, 0.045)
        short_book = _single_bond_book(settlement, 2, 0.03)
        sc = fr.steepener_scenario(-50, +50)
        pnl_long = fr.scenario_pnl_table(long_book, settlement, curve, [sc])
        pnl_short = fr.scenario_pnl_table(short_book, settlement, curve, [sc])
        assert pnl_long["pnl_full"].iloc[0] < 0  # long end sold off
        assert pnl_short["pnl_full"].iloc[0] > 0  # front end rallied

    def test_flattener_signs_reverse(self, settlement, curve):
        long_book = _single_bond_book(settlement, 30, 0.045)
        short_book = _single_bond_book(settlement, 2, 0.03)
        sc = fr.steepener_scenario(+50, -50, name="flattener")
        assert (
            fr.scenario_pnl_table(long_book, settlement, curve, [sc])["pnl_full"].iloc[0]
            > 0
        )
        assert (
            fr.scenario_pnl_table(short_book, settlement, curve, [sc])["pnl_full"].iloc[
                0
            ]
            < 0
        )


class TestHistoricalEpisodes:
    def test_all_episodes_present(self):
        assert set(fr.HISTORICAL_SCENARIOS) == {
            "taper_tantrum_2013",
            "hiking_2022",
            "gfc_2008",
        }

    def test_taper_tantrum_is_bear_steepener(self):
        sc = fr.HISTORICAL_SCENARIOS["taper_tantrum_2013"]
        s = dict(zip(sc.tenors, sc.shifts_bp))
        assert s[10.0] == pytest.approx(130.0)  # ~+130bp 10y
        assert s[2.0] < s[10.0]  # steepening

    def test_hiking_2022_is_bear_flattener(self):
        sc = fr.HISTORICAL_SCENARIOS["hiking_2022"]
        s = dict(zip(sc.tenors, sc.shifts_bp))
        assert s[2.0] > s[10.0] > s[30.0] > 0

    def test_gfc_2008_is_bull_steepener(self):
        sc = fr.HISTORICAL_SCENARIOS["gfc_2008"]
        s = dict(zip(sc.tenors, sc.shifts_bp))
        assert s[2.0] < s[10.0] < s[30.0] < 0

    def test_episode_pnl_signs(self, settlement, curve, portfolio):
        pnl = fr.scenario_pnl_table(
            portfolio, settlement, curve, list(fr.HISTORICAL_SCENARIOS.values())
        )
        assert pnl.loc["taper_tantrum_2013", "pnl_full"] < 0
        assert pnl.loc["hiking_2022", "pnl_full"] < 0
        assert pnl.loc["gfc_2008", "pnl_full"] > 0

    def test_duration_estimate_fails_for_non_parallel(self, settlement, curve):
        """Documented failure: for a pure steepener the parallel-equivalent
        duration estimate misses most of the P&L of a barbell-ish book."""
        book = _single_bond_book(settlement, 30, 0.045) + _single_bond_book(
            settlement, 2, 0.03
        )
        sc = fr.steepener_scenario(-50, +50)
        pnl = fr.scenario_pnl_table(book, settlement, curve, [sc])
        row = pnl.iloc[0]
        assert abs(row["error"]) > 0.25 * abs(row["pnl_full"])


class TestScenarioValidation:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="equal length"):
            fr.Scenario("bad", (1.0, 2.0), (10.0,))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            fr.Scenario("bad", (), ())

    def test_non_increasing_tenors_raise(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            fr.Scenario("bad", (5.0, 2.0), (1.0, 2.0))

    def test_bad_pivot_raises(self):
        with pytest.raises(ValueError, match="pivot"):
            fr.steepener_scenario(-50, 50, pivot=40.0)


class TestCarryRolldown:
    def test_identity_total_equals_static_pnl(self, settlement, curve, bond_5y):
        horizon = D(2027, 8, 18)
        cr = fr.carry_rolldown(bond_5y, settlement, horizon, curve)
        dirty0 = fr.dirty_price_from_curve(bond_5y, settlement, curve)
        dirty_h = fr.dirty_price_from_curve(bond_5y, horizon, curve)
        coupons = sum(
            amt
            for date, amt in fr.bond_cashflows(bond_5y, settlement)
            if date <= horizon
        )
        assert cr["total"] == pytest.approx(dirty_h + coupons - dirty0, abs=1e-12)
        assert cr["total"] == pytest.approx(cr["carry"] + cr["rolldown"], abs=1e-12)

    def test_upward_curve_positive_rolldown(self, settlement, curve, bond_5y):
        """On an upward curve a bond rolls down to lower yields: rolldown > 0."""
        cr = fr.carry_rolldown(bond_5y, settlement, D(2027, 8, 18), curve)
        assert cr["rolldown"] > 0

    def test_horizon_at_maturity_pull_to_par(self, settlement, curve):
        bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2028, 8, 18), coupon=0.04, frequency=2
        )
        cr = fr.carry_rolldown(bond, settlement, bond.maturity, curve)
        dirty0 = fr.dirty_price_from_curve(bond, settlement, curve)
        total_cf = sum(a for _, a in fr.bond_cashflows(bond, settlement))
        assert cr["price_horizon"] == 0.0
        assert cr["total"] == pytest.approx(total_cf - dirty0, abs=1e-12)

    def test_flat_curve_zcb_carry_is_pull_to_par(self, settlement, flat_curve):
        zcb = fr.FixedRateBond(
            effective=settlement, maturity=D(2031, 8, 18), coupon=0.0, frequency=1
        )
        cr = fr.carry_rolldown(zcb, settlement, D(2027, 8, 18), flat_curve)
        # no coupons: total = price appreciation from discount unwind
        assert cr["coupons"] == 0.0
        assert cr["carry"] == 0.0
        assert cr["total"] == pytest.approx(cr["rolldown"], abs=1e-14)
        assert cr["total"] > 0

    def test_invalid_horizon_raises(self, settlement, curve, bond_5y):
        with pytest.raises(ValueError, match="horizon"):
            fr.carry_rolldown(bond_5y, settlement, settlement, curve)
        with pytest.raises(ValueError, match="horizon"):
            fr.carry_rolldown(bond_5y, settlement, D(2040, 1, 1), curve)
