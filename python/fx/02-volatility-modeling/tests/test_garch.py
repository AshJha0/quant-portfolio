"""GARCH(1,1) and GARCH-X: recursion identities, parameter recovery on 20k-obs
simulations, Student-t tails, event-dummy recovery, variance targeting,
standard errors, input validation."""

import numpy as np
import pytest

from conftest import GARCH_T_TRUE, GARCH_TRUE, GARCHX_TRUE
from fx_vol import fit_garch, garch_filter
from fx_vol.data import synthetic as syn


class TestFilter:
    def test_recursion_identity(self):
        r = syn.simulate_constant_vol(500, 0.006, seed=41)
        omega, alpha, beta = 2e-6, 0.07, 0.9
        s2 = garch_filter(r, omega, alpha, beta)
        lhs = s2[1:]
        rhs = omega + alpha * r[:-1] ** 2 + beta * s2[:-1]
        assert np.allclose(lhs, rhs, rtol=1e-12)

    def test_initialization_uses_backcast(self):
        r = syn.simulate_constant_vol(500, 0.006, seed=42)
        s2 = garch_filter(r, 2e-6, 0.07, 0.9, initial_variance=4e-5)
        assert s2[0] == pytest.approx(2e-6 + (0.07 + 0.9) * 4e-5, rel=1e-12)

    def test_exogenous_term_adds_variance(self):
        r = syn.simulate_constant_vol(300, 0.006, seed=43)
        x = np.zeros(300); x[100] = 1.0
        s2_base = garch_filter(r, 2e-6, 0.05, 0.9)
        s2_x = garch_filter(r, 2e-6, 0.05, 0.9, gamma_x=[5e-5], x=x)
        assert s2_x[100] > s2_base[100] + 4e-5  # direct effect plus decay of past
        assert np.allclose(s2_x[:100], s2_base[:100], rtol=1e-12)

    def test_parameter_validation(self):
        r = syn.simulate_constant_vol(300, 0.006, seed=44)
        with pytest.raises(ValueError, match="omega"):
            garch_filter(r, -1e-6, 0.05, 0.9)
        with pytest.raises(ValueError, match="beta"):
            garch_filter(r, 1e-6, 0.05, 1.0)


class TestRecoveryGaussian:
    def test_parameter_recovery_20k(self, garch_fit):
        p = garch_fit.params
        assert p["alpha"] == pytest.approx(GARCH_TRUE["alpha"], abs=0.010)
        assert p["beta"] == pytest.approx(GARCH_TRUE["beta"], abs=0.015)
        true_uncond = GARCH_TRUE["omega"] / (1 - GARCH_TRUE["alpha"] - GARCH_TRUE["beta"])
        assert garch_fit.unconditional_variance == pytest.approx(true_uncond, rel=0.10)
        assert garch_fit.converged

    def test_estimates_within_confidence_bands(self, garch_fit):
        """|estimate - truth| < 4 * Hessian SE for each parameter."""
        for name in ("omega", "alpha", "beta"):
            se = garch_fit.std_errors[name]
            assert np.isfinite(se) and se > 0
            assert abs(garch_fit.params[name] - GARCH_TRUE[name]) < 4.0 * se

    def test_filtered_variance_tracks_truth(self, garch_sim, garch_fit):
        _, true_sigma2 = garch_sim
        corr = np.corrcoef(garch_fit.sigma2, true_sigma2)[0, 1]
        assert corr > 0.98

    def test_scale_invariance_percent_vs_decimal(self, garch_sim):
        """Fitting percent returns must give identical alpha/beta and mapped
        omega/loglik (the arch percent-scaling convention)."""
        r = garch_sim[0][:5000]
        f1 = fit_garch(r)
        f2 = fit_garch(100.0 * r)
        assert f2.params["alpha"] == pytest.approx(f1.params["alpha"], abs=1e-6)
        assert f2.params["beta"] == pytest.approx(f1.params["beta"], abs=1e-6)
        assert f2.params["omega"] == pytest.approx(1e4 * f1.params["omega"], rel=1e-5)
        assert f2.loglik == pytest.approx(f1.loglik - r.size * np.log(100.0), abs=1e-3)


class TestStudentT:
    def test_df_recovery_on_fat_tails(self, garch_t_fit):
        assert garch_t_fit.params["nu"] == pytest.approx(GARCH_T_TRUE["nu"], abs=1.0)
        assert garch_t_fit.params["alpha"] == pytest.approx(GARCH_T_TRUE["alpha"], abs=0.015)
        assert garch_t_fit.params["beta"] == pytest.approx(GARCH_T_TRUE["beta"], abs=0.02)

    def test_t_loglik_beats_gaussian_on_t_data(self, garch_t_sim, garch_t_fit):
        """LR statistic for the extra dof parameter should be enormous."""
        gauss = fit_garch(garch_t_sim, dist="gaussian")
        lr = 2.0 * (garch_t_fit.loglik - gauss.loglik)
        assert lr > 100.0

    def test_gaussian_fit_underestimates_tail_risk(self, garch_t_sim):
        """The practical 'vol bias': a Gaussian fit on fat-tailed FX returns
        gets the *variance* roughly right (QMLE) but its VaR multiplier
        understates the true tail -- the empirical 1% quantile of standardized
        residuals is far beyond the Gaussian 2.326."""
        gauss = fit_garch(garch_t_sim, dist="gaussian")
        z = gauss.std_resid
        q01 = np.quantile(z, 0.01)
        gaussian_q01 = -2.3263478740408408
        assert q01 < gaussian_q01 * 1.05  # at least 5% beyond the Gaussian quantile
        # and excess kurtosis is clearly positive (nu=6 -> excess 3)
        kurt = np.mean(z ** 4) / np.mean(z ** 2) ** 2 - 3.0
        assert kurt > 1.0

    def test_nu_large_on_gaussian_data(self, garch_sim):
        """On Gaussian data the fitted t dof should run away to 'infinity'."""
        fit = fit_garch(garch_sim[0][:8000], dist="t")
        assert fit.params["nu"] > 20.0


class TestGarchX:
    def test_event_dummy_coefficient_recovery(self, garchx_fit):
        p = garchx_fit.params
        true = GARCHX_TRUE
        assert p["gamma_x"] == pytest.approx(true["gamma_x"], rel=0.20)
        assert p["alpha"] == pytest.approx(true["alpha"], abs=0.015)
        assert p["beta"] == pytest.approx(true["beta"], abs=0.03)
        se = garchx_fit.std_errors["gamma_x"]
        assert np.isfinite(se) and abs(p["gamma_x"] - true["gamma_x"]) < 4 * se

    def test_event_days_have_higher_fitted_variance(self, garchx_sim, garchx_fit):
        _, x = garchx_sim
        s2 = garchx_fit.sigma2
        mask = x.astype(bool)
        assert s2[mask].mean() > 1.5 * s2[~mask].mean()

    def test_x_validation(self, garch_sim):
        r = garch_sim[0][:500]
        with pytest.raises(ValueError, match="shape"):
            fit_garch(r, x=np.ones(10))
        with pytest.raises(ValueError, match="non-negative"):
            fit_garch(r, x=-np.ones(500))
        with pytest.raises(ValueError, match="NaN"):
            fit_garch(r, x=np.full(500, np.nan))


class TestVarianceTargeting:
    def test_unconditional_matches_sample_variance(self, garch_sim):
        r = garch_sim[0][:5000]
        fit = fit_garch(r, variance_targeting=True)
        assert fit.unconditional_variance == pytest.approx(np.var(r), rel=1e-6)

    def test_close_to_free_fit(self, garch_sim):
        r = garch_sim[0][:5000]
        free = fit_garch(r)
        vt = fit_garch(r, variance_targeting=True)
        assert vt.params["alpha"] == pytest.approx(free.params["alpha"], abs=0.02)
        assert vt.params["beta"] == pytest.approx(free.params["beta"], abs=0.03)
        assert vt.loglik <= free.loglik + 1e-6  # targeting is a restriction

    def test_targeting_with_x_rejected(self, garch_sim):
        r = garch_sim[0][:500]
        with pytest.raises(ValueError, match="targeting"):
            fit_garch(r, x=np.ones(500), variance_targeting=True)


class TestValidation:
    def test_constant_series_rejected(self):
        with pytest.raises(ValueError, match="constant"):
            fit_garch(np.zeros(500))

    def test_nan_rejected(self):
        r = syn.simulate_constant_vol(500, 0.006, seed=45)
        r[100] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            fit_garch(r)

    def test_short_series_rejected(self):
        with pytest.raises(ValueError, match="at least 100"):
            fit_garch(syn.simulate_constant_vol(50, 0.006, seed=46))

    def test_bad_dist_rejected(self, garch_sim):
        with pytest.raises(ValueError, match="dist"):
            fit_garch(garch_sim[0][:500], dist="cauchy")
