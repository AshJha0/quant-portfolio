"""Multi-step forecasts and the rolling out-of-sample harness."""

import numpy as np
import pytest

from eq_vol._results import VolatilityFitResult
from eq_vol.data import synthetic as syn
from eq_vol.forecasting import (
    forecast,
    forecast_egarch,
    forecast_garch,
    forecast_gjr,
    forecast_historical,
    rolling_one_step_forecasts,
    term_structure,
)
from eq_vol.ewma import ewma_forecast
from eq_vol.garch import garch_recursion, unconditional_variance
from eq_vol.gjr import gjr_unconditional_variance


def _fake_garch_result(omega, alpha, beta, last_r, last_var):
    """Hand-built result object for exact forecast arithmetic tests."""
    return VolatilityFitResult(
        model="GARCH", dist="normal", params={"omega": omega, "alpha": alpha, "beta": beta},
        std_errors={}, loglik=0.0, n_obs=2, sigma2=np.array([last_var, last_var]),
        returns=np.array([0.0, last_r]), converged=True, message="", init_var=last_var,
    )


class TestGARCHForecast:
    def test_one_step_equals_recursion(self, garch_fit):
        r = garch_fit.returns
        p = garch_fit.params
        f1 = forecast_garch(garch_fit, horizon=1)[0]
        # appending any dummy return and re-running the recursion must give
        # the same next-period variance (it only depends on info up to T)
        extended = garch_recursion(
            np.append(r, 0.123), p["omega"], p["alpha"], p["beta"], garch_fit.init_var
        )
        assert f1 == pytest.approx(extended[-1], rel=1e-12)

    def test_converges_to_unconditional_from_above(self):
        omega, alpha, beta = 5e-6, 0.05, 0.90
        uncond = unconditional_variance(omega, alpha, beta)
        res = _fake_garch_result(omega, alpha, beta, last_r=0.05, last_var=4 * uncond)
        f = forecast_garch(res, horizon=500)
        assert f[0] > uncond
        assert np.all(np.diff(f) < 0)  # monotone decreasing
        assert f[-1] == pytest.approx(uncond, rel=1e-9)

    def test_converges_to_unconditional_from_below(self):
        omega, alpha, beta = 5e-6, 0.05, 0.90
        uncond = unconditional_variance(omega, alpha, beta)
        res = _fake_garch_result(omega, alpha, beta, last_r=0.0, last_var=0.2 * uncond)
        f = forecast_garch(res, horizon=500)
        assert f[0] < uncond
        assert np.all(np.diff(f) > 0)  # monotone increasing
        assert f[-1] == pytest.approx(uncond, rel=1e-9)

    def test_geometric_decay_rate(self):
        omega, alpha, beta = 5e-6, 0.05, 0.90
        uncond = unconditional_variance(omega, alpha, beta)
        res = _fake_garch_result(omega, alpha, beta, last_r=0.03, last_var=2 * uncond)
        f = forecast_garch(res, horizon=10)
        dev = f - uncond
        np.testing.assert_allclose(dev[1:] / dev[:-1], alpha + beta, rtol=1e-10)


class TestGJRForecast:
    def test_one_step_uses_indicator(self):
        p = {"omega": 5e-6, "alpha": 0.03, "gamma": 0.10, "beta": 0.88}
        base = dict(model="GJR-GARCH", dist="normal", params=p, std_errors={},
                    loglik=0.0, n_obs=2, converged=True, message="", init_var=1e-4)
        v = 1e-4
        neg = VolatilityFitResult(sigma2=np.array([v, v]), returns=np.array([0.0, -0.02]), **base)
        pos = VolatilityFitResult(sigma2=np.array([v, v]), returns=np.array([0.0, 0.02]), **base)
        f_neg = forecast_gjr(neg, 1)[0]
        f_pos = forecast_gjr(pos, 1)[0]
        assert f_neg == pytest.approx(p["omega"] + (p["alpha"] + p["gamma"]) * 0.02**2 + p["beta"] * v)
        assert f_pos == pytest.approx(p["omega"] + p["alpha"] * 0.02**2 + p["beta"] * v)
        assert f_neg > f_pos

    def test_converges_to_unconditional(self, gjr_fit):
        f = forecast_gjr(gjr_fit, horizon=2000)
        p = gjr_fit.params
        uncond = gjr_unconditional_variance(p["omega"], p["alpha"], p["gamma"], p["beta"])
        assert f[-1] == pytest.approx(uncond, rel=1e-6)
        dev = f - uncond
        assert np.all(dev[:-1] * dev[1:] >= 0)  # no oscillation across the level


class TestEGARCHForecast:
    def test_seeded_reproducibility(self, egarch_fit):
        f1 = forecast_egarch(egarch_fit, horizon=20, n_sims=2000, seed=7)
        f2 = forecast_egarch(egarch_fit, horizon=20, n_sims=2000, seed=7)
        np.testing.assert_array_equal(f1, f2)

    def test_seed_independence_within_mc_error(self, egarch_fit):
        f1 = forecast_egarch(egarch_fit, horizon=10, n_sims=40_000, seed=1)
        f2 = forecast_egarch(egarch_fit, horizon=10, n_sims=40_000, seed=2)
        np.testing.assert_allclose(f1, f2, rtol=0.02)

    def test_one_step_deterministic(self, egarch_fit):
        # horizon-1 forecast involves no simulation: identical across seeds
        a = forecast_egarch(egarch_fit, 1, seed=1)[0]
        b = forecast_egarch(egarch_fit, 1, seed=99)[0]
        assert a == b
        p = egarch_fit.params
        z = egarch_fit.returns[-1] / np.sqrt(egarch_fit.sigma2[-1])
        expected = np.exp(
            p["omega"] + p["beta"] * np.log(egarch_fit.sigma2[-1])
            + p["alpha"] * (abs(z) - np.sqrt(2 / np.pi)) + p["gamma"] * z
        )
        assert a == pytest.approx(expected, rel=1e-12)

    def test_long_horizon_stabilises(self, egarch_fit):
        f = forecast_egarch(egarch_fit, horizon=150, n_sims=20_000, seed=3)
        # far horizon: forecast levels off (stationary model)
        tail = f[-20:]
        assert tail.std() / tail.mean() < 0.05
        assert np.all(f > 0)


class TestFlatForecasts:
    def test_ewma_flat(self):
        r = syn.simulate_garch(500, seed=80).returns
        f = ewma_forecast(r, horizon=30)
        assert np.unique(f).size == 1

    def test_historical_flat_and_correct_level(self):
        r = syn.simulate_garch(500, seed=81).returns
        f = forecast_historical(r, horizon=15, window=21)
        np.testing.assert_allclose(f, np.mean(r[-21:] ** 2))


class TestDispatchAndTermStructure:
    def test_dispatch(self, garch_fit, gjr_fit, egarch_fit):
        assert forecast(garch_fit, 5).shape == (5,)
        assert forecast(gjr_fit, 5).shape == (5,)
        assert forecast(egarch_fit, 5, n_sims=1000, seed=0).shape == (5,)

    def test_invalid_horizon_raises(self, garch_fit):
        with pytest.raises(ValueError, match="horizon"):
            forecast_garch(garch_fit, horizon=0)

    def test_term_structure_frame(self, garch_fit):
        ts = term_structure(garch_fit, horizon=100)
        assert list(ts.columns) == ["forward_vol_annual", "avg_vol_annual"]
        assert ts.index[0] == 1 and ts.index[-1] == 100
        # average vol lies between the min and max forward vol
        assert ts["avg_vol_annual"].iloc[-1] <= ts["forward_vol_annual"].max() + 1e-12
        assert ts["avg_vol_annual"].iloc[-1] >= ts["forward_vol_annual"].min() - 1e-12
        # horizon-1 average vol equals horizon-1 forward vol
        assert ts["avg_vol_annual"].iloc[0] == pytest.approx(ts["forward_vol_annual"].iloc[0])


@pytest.fixture(scope="module")
def sim():
    return syn.simulate_garch(800, seed=90)


class TestRollingHarness:
    def test_garch_harness_tracks_true_variance(self, sim):
        res = rolling_one_step_forecasts(sim.returns, "garch", min_train=500, refit_every=100)
        assert res.forecasts.shape == (300,)
        assert np.all(res.forecasts > 0)
        assert res.n_refits == 3  # t = 0, 100, 200
        assert res.n_failed_refits == 0
        corr = np.corrcoef(res.forecasts, sim.sigma2[res.test_index])[0, 1]
        assert corr > 0.5

    def test_forecast_alignment_no_lookahead(self, sim):
        # the forecast for date t must not change if returns[t:] change
        res_a = rolling_one_step_forecasts(sim.returns, "ewma", min_train=700)
        tampered = sim.returns.copy()
        tampered[750:] = 0.05  # change the future
        res_b = rolling_one_step_forecasts(tampered, "ewma", min_train=700)
        np.testing.assert_allclose(res_a.forecasts[:50], res_b.forecasts[:50], rtol=1e-12)

    def test_rolling_vs_expanding_differ(self, sim):
        exp = rolling_one_step_forecasts(sim.returns, "historical", min_train=600, hist_window=21)
        rol = rolling_one_step_forecasts(
            sim.returns, "ewma", min_train=600, scheme="rolling", window=100
        )
        exp2 = rolling_one_step_forecasts(sim.returns, "ewma", min_train=600)
        assert exp.forecasts.shape == rol.forecasts.shape == exp2.forecasts.shape
        assert not np.allclose(rol.forecasts, exp2.forecasts)

    def test_test_returns_alignment(self, sim):
        res = rolling_one_step_forecasts(sim.returns, "historical", min_train=750)
        np.testing.assert_array_equal(res.test_returns, sim.returns[750:])
        np.testing.assert_array_equal(res.test_index, np.arange(750, 800))

    def test_invalid_args_raise(self, sim):
        with pytest.raises(ValueError, match="unknown model"):
            rolling_one_step_forecasts(sim.returns, "har", min_train=500)
        with pytest.raises(ValueError, match="unknown scheme"):
            rolling_one_step_forecasts(sim.returns, "ewma", min_train=500, scheme="walkforward")
        with pytest.raises(ValueError, match="refit_every"):
            rolling_one_step_forecasts(sim.returns, "garch", min_train=500, refit_every=0)
        with pytest.raises(ValueError, match="at least"):
            rolling_one_step_forecasts(sim.returns[:100], "ewma", min_train=500)
