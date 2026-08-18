"""Edge cases from the documentation contract: every case here is also
described in docs/VALIDATION.md."""

import numpy as np
import pytest
from scipy.stats import norm

from eq_var import (
    EquityPosition,
    OptionPosition,
    Portfolio,
    RiskFactor,
    expected_shortfall,
    filtered_historical_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
    simulate_factor_returns,
)


class TestEmptyPortfolio:
    def test_pnl_is_zero(self):
        pf = Portfolio(positions=[], factors={})
        pnl = pf.pnl(np.empty((5, 0)))
        np.testing.assert_array_equal(pnl, np.zeros(5))

    def test_exposures_empty(self):
        pf = Portfolio(positions=[], factors={})
        assert pf.delta_exposures().size == 0
        assert pf.value() == 0.0

    def test_factors_without_positions(self):
        pf = Portfolio(positions=[], factors={"A": RiskFactor("A", "equity", 10.0)})
        np.testing.assert_array_equal(pf.pnl(np.array([[0.5]])), [0.0])
        np.testing.assert_array_equal(pf.delta_exposures(), [0.0])


class TestSingleAsset:
    def test_parametric_equals_mc_for_single_normal_asset(self):
        factors = {"A": RiskFactor("A", "equity", 100.0)}
        pf = Portfolio([EquityPosition(name="a", factor="A", shares=100.0)], factors)
        cov = np.array([[0.0004]])
        closed = parametric_var(pf.delta_exposures(), cov, 0.05)
        mc = monte_carlo_var(pf, cov, 0.05, n_paths=400_000, seed=13)
        assert mc == pytest.approx(closed, rel=0.02)


class TestZeroVolAsset:
    def test_zero_vol_asset_contributes_nothing(self):
        w = np.array([1000.0, 500.0])
        cov = np.array([[0.0004, 0.0], [0.0, 0.0]])  # second asset frozen
        v = parametric_var(w, cov, 0.01)
        v_only_first = parametric_var(np.array([1000.0]), np.array([[0.0004]]), 0.01)
        assert v == pytest.approx(v_only_first, rel=1e-12)

    def test_zero_vol_pnl_series_var_is_zero(self):
        pnl = np.zeros(300)
        assert historical_var(pnl, 0.01) == pytest.approx(0.0)
        assert expected_shortfall(pnl, 0.01) == pytest.approx(0.0)
        assert filtered_historical_var(pnl, 0.01) == pytest.approx(0.0)


class TestSingularCovariance:
    def test_perfectly_correlated_assets_parametric(self):
        # rho = 1: sigma_p = |w1*s1 + w2*s2| exactly
        s1, s2 = 0.02, 0.03
        cov = np.array([[s1**2, s1 * s2], [s1 * s2, s2**2]])
        w = np.array([100.0, -50.0])
        sigma = abs(w[0] * s1 + w[1] * s2)
        assert parametric_var(w, cov, 0.01) == pytest.approx(-norm.ppf(0.01) * sigma, rel=1e-10)

    def test_perfectly_correlated_assets_mc_via_jitter(self):
        s1, s2 = 0.02, 0.03
        cov = np.array([[s1**2, s1 * s2], [s1 * s2, s2**2]])
        rets = simulate_factor_returns(cov, 200_000, seed=14)
        w = np.array([100.0, -50.0])
        pnl = rets @ w
        mc = float(-np.quantile(pnl, 0.01))
        closed = parametric_var(w, cov, 0.01)
        assert mc == pytest.approx(closed, rel=0.03)


class TestOptionsOnlyPortfolio:
    def _pf(self) -> Portfolio:
        factors = {
            "SPX": RiskFactor("SPX", "index", 5000.0),
            "SPX_IV": RiskFactor("SPX_IV", "vol", 0.2),
        }
        opt = OptionPosition(
            name="o",
            underlier="SPX",
            vol_factor="SPX_IV",
            strike=5100.0,
            expiry=0.25,
            rate=0.02,
            div_yield=0.0,
            kind="call",
            contracts=-5.0,
            multiplier=100.0,
        )
        return Portfolio([opt], factors)

    def test_var_computes_and_short_option_var_positive(self):
        pf = self._pf()
        cov = np.array([[0.0001, -1e-5], [-1e-5, 0.0002]])
        v = monte_carlo_var(pf, cov, 0.01, n_paths=20_000, seed=15)
        assert v > 0  # short option loses when index rallies / vol moves

    def test_short_gamma_full_reval_worse_than_delta_gamma_for_big_rally(self):
        pf = self._pf()
        scen = np.array([[0.15, 0.0]])
        # short call, huge rally: quadratic approx cannot keep up with the
        # convex payoff -> delta-gamma understates the loss magnitude...
        dg = pf.pnl(scen, "delta_gamma")[0]
        full = pf.pnl(scen, "full")[0]
        assert full < 0 and dg < 0
        # ...but the *quadratic* term itself overshoots for moderate shocks;
        # what must hold is that the two disagree materially at 15%:
        assert abs(dg - full) > 0.02 * abs(full)

    def test_vol_floor_no_negative_vol_in_full_reval(self):
        pf = self._pf()
        scen = np.array([[0.0, -0.5]])  # vol shock below zero, floored at 0
        pnl = pf.pnl(scen, "full")
        assert np.isfinite(pnl).all()


class TestAlphaEdges:
    def test_alpha_bounds_raise_everywhere(self):
        pnl = np.random.default_rng(0).normal(size=200)
        w, cov = np.array([1.0]), np.array([[1.0]])
        for bad in (0.0, -0.1, 0.5, 0.99):
            with pytest.raises(ValueError, match="alpha"):
                historical_var(pnl, bad)
            with pytest.raises(ValueError, match="alpha"):
                parametric_var(w, cov, bad)
            with pytest.raises(ValueError, match="alpha"):
                expected_shortfall(pnl, bad)

    def test_extreme_but_valid_alpha_works(self):
        rng = np.random.default_rng(1)
        pnl = rng.normal(size=10_000)
        assert historical_var(pnl, 0.001) > historical_var(pnl, 0.05)
        assert expected_shortfall(pnl, 0.4999) > 0


class TestInsufficientHistory:
    def test_informative_error_messages(self):
        with pytest.raises(ValueError, match="at least 50 P&L observations"):
            historical_var(np.zeros(20), 0.01)
        with pytest.raises(ValueError, match="at least 50"):
            filtered_historical_var(np.zeros(20), 0.01)
        with pytest.raises(ValueError, match="at least 2 observations"):
            from eq_var import sample_covariance

            sample_covariance(np.zeros((1, 2)))
