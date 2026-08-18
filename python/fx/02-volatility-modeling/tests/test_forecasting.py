"""Forecasting: analytic multi-step, EGARCH simulation, rolling OOS harness."""

import numpy as np
import pytest

from fx_vol import (
    ewma_variance,
    forecast_egarch_simulated,
    forecast_variance,
    rolling_one_step,
)
from fx_vol.data import synthetic as syn


class TestGarchForecast:
    def test_one_step_identity(self, garch_fit):
        p = garch_fit.params
        f = forecast_variance(garch_fit, 1)
        expected = p["omega"] + p["alpha"] * garch_fit.returns[-1] ** 2 + p["beta"] * garch_fit.sigma2[-1]
        assert f[0] == pytest.approx(expected, rel=1e-12)

    def test_converges_to_unconditional_variance(self, garch_fit):
        f = forecast_variance(garch_fit, 800)
        assert f[-1] == pytest.approx(garch_fit.unconditional_variance, rel=1e-6)

    def test_geometric_decay_identity(self, garch_fit):
        f = forecast_variance(garch_fit, 50)
        ubar = garch_fit.unconditional_variance
        p = garch_fit.persistence
        expected = ubar + p ** np.arange(50) * (f[0] - ubar)
        assert np.allclose(f, expected, rtol=1e-10)

    def test_monotone_decay_from_elevated_state(self, garch_fit):
        f = forecast_variance(garch_fit, 100)
        diffs = np.diff(f)
        assert np.all(diffs > 0) or np.all(diffs < 0)  # monotone toward uncond

    def test_horizon_validation(self, garch_fit):
        with pytest.raises(ValueError, match="horizon"):
            forecast_variance(garch_fit, 0)


class TestGjrForecast:
    def test_one_step_uses_sign_indicator(self, gjr_fit):
        p = gjr_fit.params
        r_T, s2_T = gjr_fit.returns[-1], gjr_fit.sigma2[-1]
        ind = 1.0 if r_T < 0 else 0.0
        expected = p["omega"] + (p["alpha"] + p["gamma"] * ind) * r_T ** 2 + p["beta"] * s2_T
        assert forecast_variance(gjr_fit, 1)[0] == pytest.approx(expected, rel=1e-12)

    def test_converges_to_unconditional(self, gjr_fit):
        f = forecast_variance(gjr_fit, 800)
        assert f[-1] == pytest.approx(gjr_fit.unconditional_variance, rel=1e-5)


class TestGarchXForecast:
    def test_future_event_days_add_variance(self, garchx_fit):
        h = 10
        x_none = np.zeros((h, 1))
        x_event = np.zeros((h, 1)); x_event[4, 0] = 1.0
        f0 = forecast_variance(garchx_fit, h, x_future=x_none)
        f1 = forecast_variance(garchx_fit, h, x_future=x_event)
        gamma_x = garchx_fit.params["gamma_x"]
        assert f1[4] - f0[4] == pytest.approx(gamma_x, rel=1e-10)
        assert np.allclose(f1[:4], f0[:4], rtol=1e-12)
        assert f1[5] > f0[5]  # event variance persists via beta

    def test_default_assumes_no_events(self, garchx_fit):
        f_default = forecast_variance(garchx_fit, 5)
        f_zeros = forecast_variance(garchx_fit, 5, x_future=np.zeros((5, 1)))
        assert np.allclose(f_default, f_zeros, rtol=1e-12)

    def test_x_future_on_plain_garch_rejected(self, garch_fit):
        with pytest.raises(ValueError, match="exogenous"):
            forecast_variance(garch_fit, 5, x_future=np.ones((5, 1)))


class TestEgarchForecast:
    def test_one_step_deterministic(self, egarch_fit):
        p = egarch_fit.params
        am = egarch_fit.extra["abs_moment"]
        z_T = egarch_fit.returns[-1] / np.sqrt(egarch_fit.sigma2[-1])
        expected = np.exp(
            p["omega"] + p["beta"] * np.log(egarch_fit.sigma2[-1])
            + p["alpha"] * (abs(z_T) - am) + p["gamma"] * z_T
        )
        f = forecast_variance(egarch_fit, 1, rng=1)
        assert f[0] == pytest.approx(expected, rel=1e-12)

    def test_seed_reproducibility(self, egarch_fit):
        f1 = forecast_egarch_simulated(egarch_fit, 20, n_paths=500, rng=42)
        f2 = forecast_egarch_simulated(egarch_fit, 20, n_paths=500, rng=42)
        f3 = forecast_egarch_simulated(egarch_fit, 20, n_paths=500, rng=43)
        assert np.array_equal(f1, f2)
        assert not np.array_equal(f1, f3)

    def test_long_horizon_converges_to_sample_variance(self, egarch_fit, egarch_sim):
        """Simulated forecast should plateau near the series' long-run variance."""
        f = forecast_egarch_simulated(egarch_fit, 300, n_paths=4000, rng=7)
        long_run = np.var(egarch_sim)
        assert f[-1] == pytest.approx(long_run, rel=0.15)
        assert f[-1] == pytest.approx(f[-50], rel=0.05)  # plateau reached

    def test_wrong_model_rejected(self, garch_fit):
        with pytest.raises(ValueError, match="EGARCH"):
            forecast_egarch_simulated(garch_fit, 10)


class TestRollingHarness:
    def test_ewma_harness_matches_direct_recursion(self):
        r = syn.simulate_garch(1500, 1e-6, 0.05, 0.92, seed=81)
        out = rolling_one_step(r, model="ewma", window=1000)
        init = float(np.mean(r[:1000] ** 2))
        direct = ewma_variance(r, lam=0.94, init=init)
        assert np.allclose(out["forecast"], direct[1000:], rtol=1e-12)
        assert np.array_equal(out["realized"], r[1000:] ** 2)

    def test_garch_harness_shapes_and_refits(self):
        r = syn.simulate_garch(1400, 1e-6, 0.05, 0.92, seed=82)
        out = rolling_one_step(r, model="garch", window=1000, refit_every=150)
        assert out["forecast"].shape == (400,)
        assert out["refits"] == 3  # 400 OOS days / 150 per refit -> 3 blocks
        assert np.all(out["forecast"] > 0)
        assert out["start_index"] == 1000

    def test_forecasts_are_out_of_sample(self):
        """The day-t forecast must not depend on r_t: perturbing the last OOS
        return must leave its own forecast unchanged."""
        r = syn.simulate_garch(1200, 1e-6, 0.05, 0.92, seed=83)
        out1 = rolling_one_step(r, model="garch", window=1000, refit_every=500)
        r2 = r.copy(); r2[-1] *= 5.0
        out2 = rolling_one_step(r2, model="garch", window=1000, refit_every=500)
        assert out1["forecast"][-1] == pytest.approx(out2["forecast"][-1], rel=1e-12)

    def test_n_oos_cap(self):
        r = syn.simulate_garch(1600, 1e-6, 0.05, 0.92, seed=84)
        out = rolling_one_step(r, model="ewma", window=1000, n_oos=200)
        assert out["forecast"].shape == (200,)
        assert out["start_index"] == 1400

    def test_validation(self):
        r = syn.simulate_garch(1200, 1e-6, 0.05, 0.92, seed=85)
        with pytest.raises(ValueError, match="unknown model"):
            rolling_one_step(r, model="sv")
        with pytest.raises(ValueError, match="window"):
            rolling_one_step(r, model="ewma", window=10)
        with pytest.raises(ValueError, match="more than"):
            rolling_one_step(r[:900], model="ewma", window=1000)
