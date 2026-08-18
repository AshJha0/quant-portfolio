"""GJR-GARCH: recursion identity, asymmetry recovery, symmetric-data sanity."""

import numpy as np
import pytest

from conftest import GJR_TRUE
from fx_vol import fit_gjr, gjr_filter
from fx_vol.data import synthetic as syn


class TestFilter:
    def test_recursion_identity(self):
        r = syn.simulate_constant_vol(400, 0.006, seed=51)
        omega, alpha, gamma, beta = 2e-6, 0.03, 0.1, 0.85
        s2 = gjr_filter(r, omega, alpha, gamma, beta)
        neg = (r[:-1] < 0).astype(float)
        rhs = omega + (alpha + gamma * neg) * r[:-1] ** 2 + beta * s2[:-1]
        assert np.allclose(s2[1:], rhs, rtol=1e-12)

    def test_negative_returns_raise_variance_more(self):
        omega, alpha, gamma, beta = 2e-6, 0.03, 0.15, 0.8
        r_pos = np.array([0.0] * 200 + [0.02] + [0.0] * 10)
        r_neg = np.array([0.0] * 200 + [-0.02] + [0.0] * 10)
        s2_pos = gjr_filter(r_pos, omega, alpha, gamma, beta, initial_variance=4e-5)
        s2_neg = gjr_filter(r_neg, omega, alpha, gamma, beta, initial_variance=4e-5)
        assert s2_neg[201] > s2_pos[201]
        assert s2_neg[201] - s2_pos[201] == pytest.approx(gamma * 0.02 ** 2, rel=1e-10)

    def test_parameter_validation(self):
        r = syn.simulate_constant_vol(300, 0.006, seed=52)
        with pytest.raises(ValueError, match="gamma"):
            gjr_filter(r, 1e-6, 0.05, -0.1, 0.9)


class TestRecovery:
    def test_parameter_recovery_20k(self, gjr_fit):
        p = gjr_fit.params
        assert p["alpha"] == pytest.approx(GJR_TRUE["alpha"], abs=0.015)
        assert p["gamma"] == pytest.approx(GJR_TRUE["gamma"], abs=0.03)
        assert p["beta"] == pytest.approx(GJR_TRUE["beta"], abs=0.03)
        assert gjr_fit.persistence == pytest.approx(
            GJR_TRUE["alpha"] + 0.5 * GJR_TRUE["gamma"] + GJR_TRUE["beta"], abs=0.02
        )
        assert gjr_fit.converged

    def test_asymmetry_significant_on_asymmetric_data(self, gjr_fit):
        se = gjr_fit.std_errors["gamma"]
        assert np.isfinite(se)
        assert gjr_fit.params["gamma"] > 3.0 * se

    def test_gamma_near_zero_on_symmetric_data(self, garch_sim):
        """G10-style symmetric series: the asymmetry term should vanish."""
        fit = fit_gjr(garch_sim[0][:10000])
        assert fit.params["gamma"] < 0.02

    def test_gjr_beats_garch_on_asymmetric_data(self, gjr_fit, symmetric_garch_fit_on_gjr):
        lr = 2.0 * (gjr_fit.loglik - symmetric_garch_fit_on_gjr.loglik)
        assert lr > 25.0  # one restriction; chi2(1) 1% critical value is 6.63

    def test_student_t_variant_fits(self, gjr_sim):
        fit = fit_gjr(gjr_sim[:6000], dist="t")
        assert fit.converged
        assert fit.params["nu"] > 8.0  # data is Gaussian -> large fitted dof
        assert fit.params["gamma"] == pytest.approx(GJR_TRUE["gamma"], abs=0.05)


class TestValidation:
    def test_constant_and_short_rejected(self):
        with pytest.raises(ValueError, match="constant"):
            fit_gjr(np.zeros(500))
        with pytest.raises(ValueError, match="at least"):
            fit_gjr(syn.simulate_constant_vol(30, 0.006, seed=53))
