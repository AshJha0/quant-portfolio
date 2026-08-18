"""GJR-GARCH(1,1): recursion, recovery, leverage sign, news impact curve."""

import numpy as np
import pytest

from true_params import GJR_TRUE
from eq_vol.data import synthetic as syn
from eq_vol.gjr import (
    fit_gjr,
    gjr_persistence,
    gjr_recursion,
    gjr_unconditional_variance,
    news_impact_curve,
)


class TestRecursion:
    def test_matches_explicit_loop(self):
        r = syn.simulate_gjr(400, seed=70).returns
        omega, alpha, gamma, beta, b = 4e-6, 0.03, 0.09, 0.88, 1.1e-4
        fast = gjr_recursion(r, omega, alpha, gamma, beta, b)
        slow = np.empty(r.size)
        slow[0] = omega + (alpha + 0.5 * gamma + beta) * b
        for t in range(1, r.size):
            a_eff = alpha + (gamma if r[t - 1] < 0 else 0.0)
            slow[t] = omega + a_eff * r[t - 1] ** 2 + beta * slow[t - 1]
        np.testing.assert_allclose(fast, slow, rtol=1e-12)

    def test_indicator_asymmetry_in_recursion(self):
        # same |return|, opposite signs: negative must produce higher next var
        b = 1e-4
        r_neg = np.array([-0.03, 0.0])
        r_pos = np.array([0.03, 0.0])
        v_neg = gjr_recursion(r_neg, 5e-6, 0.03, 0.10, 0.88, b)[-1]
        v_pos = gjr_recursion(r_pos, 5e-6, 0.03, 0.10, 0.88, b)[-1]
        assert v_neg > v_pos


class TestParameterRecovery:
    def test_recovers_true_parameters(self, gjr_fit):
        p = gjr_fit.params
        assert p["alpha"] == pytest.approx(GJR_TRUE["alpha"], abs=0.02)
        assert p["gamma"] == pytest.approx(GJR_TRUE["gamma"], abs=0.03)
        assert p["beta"] == pytest.approx(GJR_TRUE["beta"], abs=0.03)
        assert p["omega"] == pytest.approx(GJR_TRUE["omega"], rel=0.5)
        assert gjr_fit.converged

    def test_leverage_sign_recovered(self, gjr_fit):
        # data generated with gamma > 0 (GJR leverage convention)
        g = gjr_fit.params["gamma"]
        se = gjr_fit.std_errors["gamma"]
        assert g > 0
        assert g / se > 3.0

    def test_symmetric_data_gives_small_gamma(self, garch_sim):
        # plain GARCH data (no leverage): gamma should be ~0 and alpha ~ true
        res = fit_gjr(garch_sim.returns[:10_000])
        assert abs(res.params["gamma"]) < 0.02
        assert res.params["alpha"] == pytest.approx(0.05, abs=0.02)

    def test_fitted_persistence_stationary(self, gjr_fit):
        p = gjr_fit.params
        pers = gjr_persistence(p["alpha"], p["gamma"], p["beta"])
        assert 0 < pers < 1
        assert pers == pytest.approx(0.98, abs=0.02)  # true: 0.03+0.05+0.88

    def test_standard_errors_positive_finite(self, gjr_fit):
        for k, se in gjr_fit.std_errors.items():
            assert np.isfinite(se) and se > 0, k


class TestDerivedQuantities:
    def test_persistence_formula(self):
        assert gjr_persistence(0.03, 0.10, 0.88) == pytest.approx(0.96)

    def test_unconditional_variance_formula(self):
        assert gjr_unconditional_variance(4e-6, 0.03, 0.10, 0.88) == pytest.approx(1e-4)

    def test_nonstationary_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            gjr_unconditional_variance(4e-6, 0.05, 0.10, 0.90)  # p = 1.0
        with pytest.raises(ValueError, match="omega"):
            gjr_unconditional_variance(0.0, 0.03, 0.10, 0.88)


class TestNewsImpactCurve:
    def test_asymmetry_with_positive_gamma(self):
        z, v = news_impact_curve(**GJR_TRUE)
        v_neg = v[np.isclose(z, -2.0)][0]
        v_pos = v[np.isclose(z, 2.0)][0]
        assert v_neg > v_pos
        # quantitatively: slopes ratio ~ (alpha + gamma) / alpha at the anchor
        base = gjr_unconditional_variance(**GJR_TRUE)
        expected_gap = GJR_TRUE["gamma"] * base * 4.0  # gamma * base * z^2
        assert v_neg - v_pos == pytest.approx(expected_gap, rel=1e-10)

    def test_symmetric_when_gamma_zero(self):
        z, v = news_impact_curve(omega=5e-6, alpha=0.05, gamma=0.0, beta=0.90)
        np.testing.assert_allclose(v, v[::-1], rtol=1e-12)

    def test_positive_everywhere(self):
        z, v = news_impact_curve(**GJR_TRUE)
        assert np.all(v > 0)
