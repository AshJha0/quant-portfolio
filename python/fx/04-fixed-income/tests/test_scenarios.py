"""Scenario engine: consistency vs direct rebuild, historical episodes,
carry / forward-points roll."""

import pytest

from fx_rates.fxforward import FXForward, market_forward
from fx_rates.risk import book_value
from fx_rates.scenarios import (
    Scenario,
    apply_scenario,
    carry_table,
    forward_carry,
    historical_scenarios,
    scenario_table,
)


@pytest.fixture
def joint_scenario():
    return Scenario("joint", spot_pct=-5.0, domestic_bp=-75.0,
                    foreign_bp=25.0, basis_bp=-60.0)


class TestApplyScenario:
    def test_consistency_vs_direct_rebuild(self, market, book, joint_scenario):
        shocked = apply_scenario(market, joint_scenario)
        manual = market.replace(
            spot=market.spot * 0.95,
            domestic_curve=market.domestic_curve.parallel_shift(-75.0),
            foreign_curve=market.foreign_curve.parallel_shift(25.0),
            basis_spreads=tuple(
                (t, s - 0.0060) for t, s in market.basis_spreads
            ),
        )
        assert book_value(book, shocked) == pytest.approx(
            book_value(book, manual), abs=1e-8
        )

    def test_null_scenario_is_identity(self, market, book):
        shocked = apply_scenario(market, Scenario("null"))
        assert book_value(book, shocked) == pytest.approx(
            book_value(book, market), abs=1e-8
        )

    def test_spot_shock_only_moves_spot(self, market):
        shocked = apply_scenario(market, Scenario("spot", spot_pct=10.0))
        assert shocked.spot == pytest.approx(market.spot * 1.1, abs=1e-12)
        assert shocked.domestic_curve is market.domestic_curve
        assert shocked.basis_spreads == market.basis_spreads

    def test_basis_shock_widens_forward_points(self, market):
        shocked = apply_scenario(market, Scenario("basis", basis_bp=-50.0))
        assert market_forward(shocked, 5.0) > market_forward(market, 5.0)

    def test_scenario_on_empty_basis_market(self, market):
        m0 = market.replace(basis_spreads=())
        shocked = apply_scenario(m0, Scenario("b", basis_bp=-25.0))
        assert shocked.basis_spreads  # a flat spread curve was synthesised
        assert market_forward(shocked, 5.0) > market_forward(m0, 5.0)


class TestScenarioTable:
    def test_table_pnl_columns_consistent(self, market, book, joint_scenario):
        tbl = scenario_table(book, market, [joint_scenario, Scenario("null")])
        base = tbl.attrs["base_pv"]
        assert base == pytest.approx(book_value(book, market), abs=1e-8)
        assert ((tbl["book_pv"] - tbl["pnl"]) - base).abs().max() < 1e-6
        assert tbl.loc["null", "pnl"] == pytest.approx(0.0, abs=1e-8)

    def test_historical_scenarios_present_and_stress_the_book(self, market, book):
        scens = historical_scenarios()
        names = [s.name for s in scens]
        assert "2008 USD funding squeeze" in names
        assert "2020 March dash-for-cash" in names
        assert any("year-end" in n for n in names)
        s2008 = next(s for s in scens if "2008" in s.name)
        assert s2008.basis_bp <= -150.0  # blowout beyond -150bp
        tbl = scenario_table(book, market, scens)
        assert abs(tbl.loc["2008 USD funding squeeze", "pnl"]) > 0.0

    def test_2008_basis_blowout_hits_short_usd_funding_position(self, market):
        # a party long EUR forward (lending USD via the FX swap market)
        # gains when the basis blows out; the mirror position loses
        long_eur = FXForward(100e6, market_forward(market, 0.25), 0.25)
        s2008 = next(s for s in historical_scenarios() if "2008" in s.name)
        basis_only = Scenario("basis only", basis_bp=s2008.basis_bp)
        pnl = long_eur.value(apply_scenario(market, basis_only)) - long_eur.value(market)
        assert pnl > 0.0


class TestCarry:
    def test_long_low_yielder_has_negative_carry(self, market):
        # EUR is the low-yielding currency: positive points, long forward
        # rolls down the curve => negative carry
        fwd = FXForward(10e6, market_forward(market, 1.0), 1.0)
        res = forward_carry(fwd, market, 0.25)
        assert res["carry_pnl"] < 0.0
        assert res["points_roll"] < 0.0

    def test_short_position_carry_is_mirror(self, market):
        long = FXForward(10e6, market_forward(market, 1.0), 1.0)
        short = FXForward(-10e6, market_forward(market, 1.0), 1.0)
        assert forward_carry(short, market, 0.25)["carry_pnl"] == pytest.approx(
            -forward_carry(long, market, 0.25)["carry_pnl"], abs=1e-8
        )

    def test_carry_equals_aged_minus_fresh_value(self, market):
        from dataclasses import replace
        fwd = FXForward(10e6, 1.12, 2.0)
        res = forward_carry(fwd, market, 0.5)
        aged = replace(fwd, expiry=1.5)
        assert res["carry_pnl"] == pytest.approx(
            aged.value(market) - fwd.value(market), abs=1e-8
        )

    def test_carry_table_monotone_horizons(self, market):
        fwd = FXForward(10e6, market_forward(market, 1.0), 1.0)
        tbl = carry_table(fwd, market, [0.1, 0.25, 0.5])
        # rolling further down a premium curve loses more
        assert tbl["carry_pnl"].is_monotonic_decreasing

    def test_bad_horizon_raises(self, market):
        fwd = FXForward(10e6, 1.1, 1.0)
        with pytest.raises(ValueError, match="horizon"):
            forward_carry(fwd, market, 1.5)
        with pytest.raises(ValueError, match="horizon"):
            forward_carry(fwd, market, 0.0)
