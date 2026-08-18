"""Edge cases and failure behaviour (documentation contract item 6):
degenerate series, outliers, short samples, NaNs, convergence failures,
crisis regime jumps. Every case here is also discussed in docs/VALIDATION.md.
"""

import numpy as np
import pytest

import eq_vol.garch as garch_mod
from eq_vol._utils import ConvergenceError
from eq_vol.data import synthetic as syn
from eq_vol.egarch import fit_egarch
from eq_vol.ewma import ewma_variance
from eq_vol.forecasting import rolling_one_step_forecasts
from eq_vol.garch import fit_garch
from eq_vol.gjr import fit_gjr
from eq_vol.historical import realized_vol


class TestDegenerateSeries:
    def test_constant_series_garch_raises(self):
        r = np.zeros(500)
        with pytest.raises(ValueError, match="zero variance"):
            fit_garch(r)

    def test_constant_nonzero_series_raises_all_models(self):
        r = np.full(500, 0.001)
        for fit in (fit_garch, fit_egarch, fit_gjr):
            with pytest.raises(ValueError, match="zero variance"):
                fit(r)

    def test_constant_series_historical_is_finite(self):
        # historical vol of a constant series is well-defined (not an error)
        r = np.full(100, 0.001)
        vol = realized_vol(r, window=21)
        assert np.all(np.isfinite(vol[20:]))
        vol0 = realized_vol(np.zeros(100), window=21)
        np.testing.assert_allclose(vol0[20:], 0.0)

    def test_constant_series_ewma_is_flat(self):
        sigma2 = ewma_variance(np.full(100, 0.01))
        np.testing.assert_allclose(sigma2, 1e-4, rtol=1e-10)

    def test_empty_series_raises(self):
        with pytest.raises(ValueError, match="observations"):
            realized_vol(np.array([]), window=21)


class TestOutliers:
    def test_single_huge_outlier_garch_survives(self):
        sim = syn.simulate_garch(3000, seed=110)
        r = sim.returns.copy()
        r[1500] = -0.25  # a -25% crash day in ~1% daily vol data
        res = fit_garch(r)
        assert res.converged
        assert all(np.isfinite(v) for v in res.params.values())
        assert np.all(res.sigma2 > 0) and np.all(np.isfinite(res.sigma2))
        assert res.extra["persistence"] < 1.0

    def test_single_huge_outlier_gjr_survives(self):
        sim = syn.simulate_gjr(3000, seed=111)
        r = sim.returns.copy()
        r[1500] = -0.25
        res = fit_gjr(r)
        assert res.converged and np.all(res.sigma2 > 0)

    def test_outlier_spikes_conditional_variance_then_decays(self):
        sim = syn.simulate_garch(3000, seed=112)
        r = sim.returns.copy()
        r[1500] = -0.25
        res = fit_garch(r)
        assert res.sigma2[1501] > 5.0 * res.sigma2[1500]  # spike after the crash
        assert res.sigma2[1600] < res.sigma2[1510]        # then mean reversion


class TestShortSeries:
    def test_garch_short_series_informative_error(self):
        with pytest.raises(ValueError, match="at least 100"):
            fit_garch(syn.simulate_garch(50, seed=1).returns)

    def test_all_fitters_reject_short_series(self):
        r = syn.simulate_garch(30, seed=2).returns
        for fit in (fit_garch, fit_egarch, fit_gjr):
            with pytest.raises(ValueError, match="at least"):
                fit(r)

    def test_realized_vol_window_longer_than_series(self):
        with pytest.raises(ValueError, match="at least"):
            realized_vol(np.full(10, 0.01), window=21)

    def test_ewma_single_observation_raises(self):
        with pytest.raises(ValueError, match="at least"):
            ewma_variance(np.array([0.01]))


class TestNaNPolicy:
    """Policy: NaN/inf inputs raise ValueError everywhere (documented in
    eq_vol._utils.validate_returns) — never silently dropped or imputed."""

    @pytest.fixture()
    def nan_returns(self):
        r = syn.simulate_garch(500, seed=120).returns.copy()
        r[250] = np.nan
        return r

    def test_fitters_reject_nan(self, nan_returns):
        for fit in (fit_garch, fit_egarch, fit_gjr):
            with pytest.raises(ValueError, match="NaN"):
                fit(nan_returns)

    def test_filters_reject_nan(self, nan_returns):
        with pytest.raises(ValueError, match="NaN"):
            ewma_variance(nan_returns)
        with pytest.raises(ValueError, match="NaN"):
            realized_vol(nan_returns, window=21)

    def test_harness_rejects_nan(self, nan_returns):
        with pytest.raises(ValueError, match="NaN"):
            rolling_one_step_forecasts(nan_returns, "ewma", min_train=300)

    def test_inf_rejected(self):
        r = syn.simulate_garch(500, seed=121).returns.copy()
        r[100] = np.inf
        with pytest.raises(ValueError, match="NaN"):
            fit_garch(r)


class TestConvergenceFailureSurfaced:
    """A failed optimisation must never be returned as if it succeeded."""

    @pytest.fixture()
    def force_failure(self, monkeypatch):
        real_minimize = garch_mod.minimize

        def failing_minimize(*args, **kwargs):
            res = real_minimize(*args, **kwargs)
            res.success = False
            res.message = "forced failure for testing"
            return res

        monkeypatch.setattr(garch_mod, "minimize", failing_minimize)

    def test_raises_convergence_error_by_default(self, force_failure):
        r = syn.simulate_garch(500, seed=130).returns
        with pytest.raises(ConvergenceError, match="failed to converge"):
            fit_garch(r)

    def test_flagged_when_not_raising(self, force_failure):
        r = syn.simulate_garch(500, seed=130).returns
        res = fit_garch(r, raise_on_failure=False)
        assert res.converged is False
        assert "forced failure" in res.message


@pytest.fixture(scope="module")
def crisis():
    return syn.simulate_crisis(n_pre=750, n_crisis=60, n_post=250, seed=140)


class TestCrisisRegimeJump:
    """COVID-Mar-2020-style structural break (see docs/VALIDATION.md)."""

    def test_garch_still_fits_through_break(self, crisis):
        res = fit_garch(crisis.returns)
        assert res.converged
        # a structural break masquerades as high persistence
        assert res.extra["persistence"] > 0.9

    def test_ewma_adapts_after_break(self, crisis):
        sigma2 = ewma_variance(crisis.returns)
        pre = np.sqrt(sigma2[740] * 252)
        during = np.sqrt(sigma2[805] * 252)  # ~55 days into the crisis
        assert during > 2.0 * pre  # vol estimate has repriced sharply upward

    def test_rolling_forecasts_reprice_upward(self, crisis):
        res = rolling_one_step_forecasts(crisis.returns, "ewma", min_train=700)
        pre_break = res.forecasts[:40].mean()      # forecasts for t in [700, 740)
        in_crisis = res.forecasts[80:100].mean()   # deep in the crisis regime
        assert in_crisis > 5.0 * pre_break

    def test_short_rolling_window_adapts_faster_than_expanding_hist(self, crisis):
        short = rolling_one_step_forecasts(
            crisis.returns, "historical", min_train=780, hist_window=10
        )
        long = rolling_one_step_forecasts(
            crisis.returns, "historical", min_train=780, hist_window=250
        )
        # 30 days into the crisis the short window has repriced far more
        assert short.forecasts[30] > 2.0 * long.forecasts[30]
