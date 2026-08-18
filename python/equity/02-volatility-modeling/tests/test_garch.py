"""GARCH(1,1): recursion, likelihood, MLE parameter recovery, derived stats."""

import numpy as np
import pytest
from scipy import stats

from true_params import GARCH_TRUE
from eq_vol.data import synthetic as syn
from eq_vol.garch import (
    fit_garch,
    garch_loglik,
    garch_recursion,
    gaussian_loglik,
    persistence,
    student_t_loglik,
    unconditional_variance,
    vol_halflife,
)


class TestRecursionAndLikelihood:
    def test_recursion_matches_explicit_loop(self):
        r = syn.simulate_garch(500, seed=50).returns
        omega, alpha, beta, b = 4e-6, 0.07, 0.88, 1.1e-4
        fast = garch_recursion(r, omega, alpha, beta, b)
        slow = np.empty(r.size)
        slow[0] = omega + (alpha + beta) * b
        for t in range(1, r.size):
            slow[t] = omega + alpha * r[t - 1] ** 2 + beta * slow[t - 1]
        np.testing.assert_allclose(fast, slow, rtol=1e-12)

    def test_gaussian_loglik_matches_scipy(self):
        rng = np.random.default_rng(51)
        r = rng.normal(0, 0.01, 200)
        sigma2 = np.full(200, 1.2e-4)
        expected = stats.norm.logpdf(r, scale=np.sqrt(sigma2)).sum()
        assert gaussian_loglik(r, sigma2) == pytest.approx(expected, rel=1e-12)

    def test_student_t_loglik_matches_scipy(self):
        rng = np.random.default_rng(52)
        r = rng.normal(0, 0.01, 200)
        sigma2 = np.full(200, 1.2e-4)
        nu = 7.0
        scale = np.sqrt(sigma2 * (nu - 2.0) / nu)  # unit-variance standardisation
        expected = stats.t.logpdf(r, df=nu, scale=scale).sum()
        assert student_t_loglik(r, sigma2, nu) == pytest.approx(expected, rel=1e-12)

    def test_loglik_at_true_params_beats_perturbed(self, garch_sim):
        r = garch_sim.returns
        ll_true = garch_loglik(r, **GARCH_TRUE)
        ll_bad = garch_loglik(r, omega=GARCH_TRUE["omega"], alpha=0.15, beta=0.80)
        assert ll_true > ll_bad


class TestParameterRecovery:
    def test_recovers_true_parameters(self, garch_fit):
        p = garch_fit.params
        assert p["alpha"] == pytest.approx(GARCH_TRUE["alpha"], abs=0.015)
        assert p["beta"] == pytest.approx(GARCH_TRUE["beta"], abs=0.025)
        assert p["omega"] == pytest.approx(GARCH_TRUE["omega"], rel=0.5)
        assert garch_fit.converged

    def test_recovered_unconditional_variance(self, garch_fit):
        true_uv = GARCH_TRUE["omega"] / (1 - GARCH_TRUE["alpha"] - GARCH_TRUE["beta"])
        assert garch_fit.extra["unconditional_variance"] == pytest.approx(true_uv, rel=0.15)

    def test_recovered_persistence(self, garch_fit):
        assert garch_fit.extra["persistence"] == pytest.approx(0.95, abs=0.02)

    def test_standard_errors_positive_finite(self, garch_fit):
        for k, se in garch_fit.std_errors.items():
            assert np.isfinite(se) and se > 0, k
        # estimates within ~4 SE of truth (asymptotic normality sanity check)
        for k in ("alpha", "beta"):
            err = abs(garch_fit.params[k] - GARCH_TRUE[k])
            assert err < 4.0 * garch_fit.std_errors[k]

    def test_student_t_fit_recovers_nu(self):
        sim = syn.simulate_garch(20_000, dist="t", nu=8.0, seed=5)
        res = fit_garch(sim.returns, dist="t")
        assert res.converged
        assert res.params["nu"] == pytest.approx(8.0, abs=1.5)
        assert res.params["alpha"] == pytest.approx(0.05, abs=0.02)

    def test_variance_targeting_matches_sample_variance(self, garch_sim):
        res = fit_garch(garch_sim.returns, variance_targeting=True)
        assert res.extra["unconditional_variance"] == pytest.approx(
            float(np.var(garch_sim.returns)), rel=1e-10
        )
        assert np.isnan(res.std_errors["omega"])  # not freely estimated
        assert np.isfinite(res.std_errors["alpha"]) and res.std_errors["alpha"] > 0

    def test_variance_targeting_close_to_free_fit(self, garch_sim, garch_fit):
        vt = fit_garch(garch_sim.returns, variance_targeting=True)
        assert vt.params["alpha"] == pytest.approx(garch_fit.params["alpha"], abs=0.01)
        assert vt.params["beta"] == pytest.approx(garch_fit.params["beta"], abs=0.02)

    def test_init_methods_agree_on_long_sample(self, garch_sim):
        res_s = fit_garch(garch_sim.returns, init_method="sample")
        res_b = fit_garch(garch_sim.returns, init_method="backcast")
        assert res_s.params["alpha"] == pytest.approx(res_b.params["alpha"], abs=0.005)
        assert res_s.params["beta"] == pytest.approx(res_b.params["beta"], abs=0.005)

    def test_std_residuals_unit_variance(self, garch_fit):
        z = garch_fit.std_residuals
        assert np.std(z) == pytest.approx(1.0, abs=0.03)

    def test_information_criteria_definitions(self, garch_fit):
        k, n, ll = 3, garch_fit.n_obs, garch_fit.loglik
        assert garch_fit.aic == pytest.approx(2 * k - 2 * ll)
        assert garch_fit.bic == pytest.approx(k * np.log(n) - 2 * ll)


class TestDerivedQuantities:
    def test_unconditional_variance_formula(self):
        assert unconditional_variance(5e-6, 0.05, 0.90) == pytest.approx(1e-4, rel=1e-12)

    def test_unconditional_variance_igarch_raises(self):
        with pytest.raises(ValueError, match="IGARCH|integrated"):
            unconditional_variance(5e-6, 0.10, 0.90)  # alpha + beta = 1 exactly
        with pytest.raises(ValueError, match="IGARCH|integrated"):
            unconditional_variance(5e-6, 0.30, 0.75)  # explosive

    def test_unconditional_variance_bad_omega_raises(self):
        with pytest.raises(ValueError, match="omega"):
            unconditional_variance(-1e-6, 0.05, 0.90)

    def test_persistence(self):
        assert persistence(0.05, 0.90) == pytest.approx(0.95)

    def test_halflife_definition(self):
        hl = vol_halflife(0.05, 0.90)
        assert 0.95**hl == pytest.approx(0.5, rel=1e-12)
        assert hl == pytest.approx(13.51, abs=0.01)

    def test_halflife_invalid_raises(self):
        with pytest.raises(ValueError):
            vol_halflife(0.5, 0.5)  # persistence == 1
