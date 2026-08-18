"""Stress testing: replay, ladders, reverse stress closed form vs optimizer."""

import numpy as np
import pytest

from eq_var import (
    HISTORICAL_SCENARIOS,
    EquityPosition,
    FuturePosition,
    Portfolio,
    RiskFactor,
    StressScenario,
    apply_scenario,
    reverse_stress_delta,
    reverse_stress_delta_gamma,
    scenario_shock_vector,
    scenario_table,
    sensitivity_ladder,
)
from eq_var.data import demo_covariance, demo_portfolio


def linear_portfolio() -> Portfolio:
    factors = {
        "STK": RiskFactor("STK", "equity", 100.0),
        "IDX": RiskFactor("IDX", "index", 2000.0),
    }
    return Portfolio(
        [
            EquityPosition(name="s", factor="STK", shares=1000.0),
            FuturePosition(name="f", factor="IDX", contracts=3.0, multiplier=50.0),
        ],
        factors,
    )


class TestScenarioReplay:
    def test_replay_pnl_hand_computed_linear(self):
        pf = linear_portfolio()
        sc = StressScenario("crash", "test", shocks_by_kind={"equity": -0.20, "index": -0.10})
        # 1000*100*(-0.20) + 3*50*2000*(-0.10) = -20000 - 30000 = -50000
        assert apply_scenario(pf, sc) == pytest.approx(-50_000.0, abs=1e-9)

    def test_shock_vector_kind_mapping(self):
        pf = demo_portfolio()
        sc = HISTORICAL_SCENARIOS["1987_black_monday"]
        vec = scenario_shock_vector(pf, sc)
        assert vec[0] == pytest.approx(-0.225)  # AAPL: equity kind
        assert vec[2] == pytest.approx(-0.205)  # SPX: index kind
        assert vec[3] == pytest.approx(0.60)  # SPX_IV: vol kind

    def test_name_override_beats_kind(self):
        pf = linear_portfolio()
        sc = StressScenario(
            "custom", "", shocks_by_kind={"equity": -0.10}, shocks_by_name={"STK": -0.50}
        )
        vec = scenario_shock_vector(pf, sc)
        assert vec[0] == pytest.approx(-0.50)

    def test_all_historical_scenarios_present(self):
        for key in ("1987_black_monday", "2008_lehman", "2020_covid"):
            assert key in HISTORICAL_SCENARIOS

    def test_crash_scenarios_lose_money_for_demo_book(self):
        pf = demo_portfolio()
        for key in ("1987_black_monday", "2008_lehman", "2020_covid"):
            assert apply_scenario(pf, HISTORICAL_SCENARIOS[key]) < 0

    def test_scenario_table_columns(self):
        table = scenario_table(demo_portfolio())
        assert set(table.columns) >= {"scenario", "pnl_full", "pnl_delta_gamma", "approx_error"}
        assert len(table) == len(HISTORICAL_SCENARIOS)
        np.testing.assert_allclose(
            table["approx_error"], table["pnl_delta_gamma"] - table["pnl_full"], atol=1e-8
        )


class TestSensitivityLadder:
    def test_linear_ladder_is_linear_and_zero_at_origin(self):
        pf = linear_portfolio()
        ladder = sensitivity_ladder(pf, "STK", shocks=np.array([-0.1, -0.05, 0.0, 0.05, 0.1]))
        np.testing.assert_allclose(ladder["pnl"], 1000 * 100.0 * ladder["shock"], atol=1e-9)
        assert ladder.loc[ladder["shock"] == 0.0, "pnl"].iloc[0] == pytest.approx(0.0)

    def test_default_shocks_for_vol_factor(self):
        pf = demo_portfolio()
        ladder = sensitivity_ladder(pf, "SPX_IV")
        assert (ladder["shock"] >= -0.10).all()
        # long vega book: vol up = profit, monotone increasing
        assert np.all(np.diff(ladder["pnl"]) > 0)

    def test_unknown_factor_raises(self):
        with pytest.raises(ValueError, match="unknown factor"):
            sensitivity_ladder(linear_portfolio(), "GHOST")


class TestReverseStress:
    COV = np.array([[0.0004, 0.00015, -0.0001],
                    [0.00015, 0.0009, -0.0002],
                    [-0.0001, -0.0002, 0.0016]])
    W = np.array([1000.0, -400.0, 250.0])

    def test_closed_form_loss_is_radius_times_sigma(self):
        res = reverse_stress_delta(self.W, self.COV, radius=3.0)
        sigma_p = float(np.sqrt(self.W @ self.COV @ self.W))
        assert res["loss"] == pytest.approx(3.0 * sigma_p, rel=1e-12)

    def test_closed_form_shock_satisfies_mahalanobis_constraint(self):
        res = reverse_stress_delta(self.W, self.COV, radius=2.5)
        x = res["shock"]
        maha = float(x @ np.linalg.solve(self.COV, x))
        assert maha == pytest.approx(2.5**2, rel=1e-10)

    def test_closed_form_matches_numerical_optimizer(self):
        """Spec check: delta closed form == constrained optimiser (gamma=0)."""
        res_cf = reverse_stress_delta(self.W, self.COV, radius=3.0)
        res_num = reverse_stress_delta_gamma(
            self.W, np.zeros((3, 3)), self.COV, radius=3.0, seed=0
        )
        assert res_num["loss"] == pytest.approx(res_cf["loss"], rel=1e-6)
        np.testing.assert_allclose(res_num["shock"], res_cf["shock"], rtol=1e-4, atol=1e-8)

    def test_shock_realises_the_reported_loss(self):
        res = reverse_stress_delta(self.W, self.COV, radius=3.0)
        assert float(-self.W @ res["shock"]) == pytest.approx(res["loss"], rel=1e-12)

    def test_loss_scales_linearly_with_radius(self):
        l1 = reverse_stress_delta(self.W, self.COV, 1.0)["loss"]
        l3 = reverse_stress_delta(self.W, self.COV, 3.0)["loss"]
        assert l3 == pytest.approx(3.0 * l1, rel=1e-12)

    def test_long_gamma_trims_worst_case_short_gamma_worsens_it(self):
        g_long = np.diag([50_000.0, 0.0, 0.0])
        g_short = -g_long
        base = reverse_stress_delta(self.W, self.COV, 3.0)["loss"]
        long_loss = reverse_stress_delta_gamma(self.W, g_long, self.COV, 3.0, seed=1)["loss"]
        short_loss = reverse_stress_delta_gamma(self.W, g_short, self.COV, 3.0, seed=1)["loss"]
        assert long_loss <= base + 1e-6
        assert short_loss >= base - 1e-6

    def test_demo_portfolio_gamma_matters(self):
        pf = demo_portfolio()
        cov = demo_covariance()
        w = pf.delta_exposures()
        delta_loss = reverse_stress_delta(w, cov, 3.0)["loss"]
        dg_loss = reverse_stress_delta_gamma(w, pf.gamma_matrix(), cov, 3.0, seed=2)["loss"]
        assert dg_loss == pytest.approx(delta_loss, rel=0.10)  # small book gamma
        assert dg_loss <= delta_loss  # long gamma protection

    def test_zero_exposure_portfolio(self):
        res = reverse_stress_delta(np.zeros(3), self.COV, 3.0)
        assert res["loss"] == 0.0

    def test_invalid_radius_raises(self):
        with pytest.raises(ValueError, match="radius"):
            reverse_stress_delta(self.W, self.COV, radius=0.0)
        with pytest.raises(ValueError, match="radius"):
            reverse_stress_delta_gamma(self.W, np.zeros((3, 3)), self.COV, radius=-1.0)
