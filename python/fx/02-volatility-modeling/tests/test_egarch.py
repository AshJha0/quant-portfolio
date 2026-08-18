"""EGARCH: recursion identity, recovery including sign-flexible leverage
(the FX safe-haven case), Student-t absolute moment."""

import numpy as np
import pytest

from conftest import EGARCH_TRUE
from fx_vol import egarch_filter, fit_egarch
from fx_vol._mle import student_t_abs_moment
from fx_vol.data import synthetic as syn


class TestFilter:
    def test_recursion_identity(self):
        r = syn.simulate_constant_vol(300, 0.006, seed=61)
        omega, alpha, gamma, beta = -0.4, 0.12, -0.05, 0.96
        am = np.sqrt(2 / np.pi)
        s2 = egarch_filter(r, omega, alpha, gamma, beta)
        for t in [1, 50, 299]:
            z = r[t - 1] / np.sqrt(s2[t - 1])
            expected = np.exp(omega + beta * np.log(s2[t - 1]) + alpha * (abs(z) - am) + gamma * z)
            assert s2[t] == pytest.approx(expected, rel=1e-10)

    def test_variance_always_positive_without_constraints(self):
        """The log form guarantees positivity even for wild parameters."""
        r = syn.simulate_constant_vol(500, 0.02, seed=62)
        s2 = egarch_filter(r, -2.0, 0.9, -0.8, 0.5)
        assert np.all(s2 > 0) and np.all(np.isfinite(s2))

    def test_beta_bound_enforced(self):
        r = syn.simulate_constant_vol(300, 0.006, seed=63)
        with pytest.raises(ValueError, match="beta"):
            egarch_filter(r, -0.4, 0.1, 0.0, 1.0)


class TestRecovery:
    def test_parameter_recovery(self, egarch_fit):
        p = egarch_fit.params
        assert p["alpha"] == pytest.approx(EGARCH_TRUE["alpha"], abs=0.04)
        assert p["gamma"] == pytest.approx(EGARCH_TRUE["gamma"], abs=0.03)
        assert p["beta"] == pytest.approx(EGARCH_TRUE["beta"], abs=0.02)
        assert egarch_fit.converged

    def test_negative_leverage_detected_significant(self, egarch_fit):
        """USDJPY-style safe-haven asymmetry: gamma < 0 and significant."""
        se = egarch_fit.std_errors["gamma"]
        assert np.isfinite(se)
        assert egarch_fit.params["gamma"] < -2.0 * se

    def test_positive_leverage_recovered_too(self):
        """Asymmetry of the opposite sign (EM depreciation) -- gamma > 0."""
        r = syn.simulate_egarch(10_000, omega=-0.6, alpha=0.15, gamma=0.10, beta=0.94, seed=64)
        fit = fit_egarch(r)
        assert fit.params["gamma"] == pytest.approx(0.10, abs=0.04)
        assert fit.params["gamma"] > 0

    def test_leverage_sign_flips_under_pair_inversion(self, egarch_sim, egarch_fit):
        """Inverting the pair negates returns, so the fitted gamma flips sign
        while alpha/beta are unchanged -- the safe-haven quote-direction fact."""
        fit_inv = fit_egarch(-egarch_sim)
        assert fit_inv.params["gamma"] == pytest.approx(-egarch_fit.params["gamma"], abs=0.01)
        assert fit_inv.params["alpha"] == pytest.approx(egarch_fit.params["alpha"], abs=0.01)
        assert fit_inv.params["beta"] == pytest.approx(egarch_fit.params["beta"], abs=0.005)

    def test_student_t_variant_fits(self, egarch_sim):
        fit = fit_egarch(egarch_sim[:6000], dist="t")
        assert fit.converged
        assert fit.params["nu"] > 8.0  # Gaussian data


class TestStudentTAbsMoment:
    def test_limits_to_gaussian(self):
        assert student_t_abs_moment(1e8) == pytest.approx(np.sqrt(2 / np.pi), rel=1e-6)

    def test_monte_carlo_agreement(self):
        nu = 5.0
        rng = np.random.default_rng(65)
        z = rng.standard_t(nu, 2_000_000) * np.sqrt((nu - 2) / nu)
        assert student_t_abs_moment(nu) == pytest.approx(np.mean(np.abs(z)), rel=0.005)

    def test_invalid_nu(self):
        with pytest.raises(ValueError):
            student_t_abs_moment(2.0)
