"""EGARCH(1,1): recursion, recovery, leverage sign, news impact curve."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, t as t_dist

from true_params import EGARCH_TRUE
from eq_vol.data import synthetic as syn
from eq_vol.egarch import (
    egarch_recursion,
    egarch_unconditional_logvar,
    expected_abs_z,
    fit_egarch,
    news_impact_curve,
)


class TestRecursion:
    def test_matches_explicit_loop(self):
        r = syn.simulate_egarch(300, seed=60).returns
        omega, alpha, gamma, beta, b = -0.5, 0.12, -0.06, 0.94, 1e-4
        eaz = math.sqrt(2 / math.pi)
        fast = egarch_recursion(r, omega, alpha, gamma, beta, b, eaz)
        ls = math.log(b)
        slow = np.empty(r.size)
        slow[0] = math.exp(ls)
        for tt in range(1, r.size):
            z = r[tt - 1] / math.sqrt(slow[tt - 1])
            ls = omega + beta * ls + alpha * (abs(z) - eaz) + gamma * z
            slow[tt] = math.exp(ls)
        np.testing.assert_allclose(fast, slow, rtol=1e-12)

    def test_positivity_needs_no_constraints(self):
        # wild, "illegal-looking" parameters: variance still strictly positive
        # and finite because the recursion is in log-variance
        r = syn.simulate_garch(500, seed=61).returns
        sigma2 = egarch_recursion(r, omega=2.0, alpha=-3.0, gamma=5.0, beta=-0.7, init_var=1e-4)
        assert np.all(sigma2 > 0) and np.all(np.isfinite(sigma2))

    def test_expected_abs_z_normal(self):
        exact = quad(lambda z: abs(z) * norm.pdf(z), -np.inf, np.inf)[0]
        assert expected_abs_z("normal") == pytest.approx(exact, rel=1e-8)
        assert expected_abs_z("normal") == pytest.approx(math.sqrt(2 / math.pi), rel=1e-14)

    def test_expected_abs_z_student_t(self):
        nu = 7.0
        scale = math.sqrt((nu - 2) / nu)  # unit-variance standardisation
        exact = quad(lambda z: abs(z) * t_dist.pdf(z / scale, nu) / scale, -np.inf, np.inf)[0]
        assert expected_abs_z("t", nu) == pytest.approx(exact, rel=1e-6)


class TestParameterRecovery:
    def test_recovers_true_parameters(self, egarch_fit):
        p = egarch_fit.params
        assert p["alpha"] == pytest.approx(EGARCH_TRUE["alpha"], abs=0.03)
        assert p["gamma"] == pytest.approx(EGARCH_TRUE["gamma"], abs=0.02)
        assert p["beta"] == pytest.approx(EGARCH_TRUE["beta"], abs=0.02)
        assert p["omega"] == pytest.approx(EGARCH_TRUE["omega"], abs=0.15)
        assert egarch_fit.converged

    def test_leverage_sign_recovered(self, egarch_fit):
        # data generated with gamma < 0 (our leverage convention):
        # estimate must be negative and significantly so
        g = egarch_fit.params["gamma"]
        se = egarch_fit.std_errors["gamma"]
        assert g < 0
        assert g / se < -3.0

    def test_no_leverage_data_gives_small_gamma(self):
        sim = syn.simulate_egarch(8000, omega=-0.40, alpha=0.10, gamma=0.0, beta=0.96, seed=62)
        res = fit_egarch(sim.returns)
        assert abs(res.params["gamma"]) < 0.02

    def test_standard_errors_positive_finite(self, egarch_fit):
        for k, se in egarch_fit.std_errors.items():
            assert np.isfinite(se) and se > 0, k

    def test_unconditional_level_recovered(self, egarch_fit):
        true_ulv = EGARCH_TRUE["omega"] / (1 - EGARCH_TRUE["beta"])
        assert egarch_fit.extra["unconditional_logvar"] == pytest.approx(true_ulv, abs=0.15)


class TestUnconditional:
    def test_formula(self):
        assert egarch_unconditional_logvar(-0.4, 0.96) == pytest.approx(-10.0)

    def test_nonstationary_raises(self):
        with pytest.raises(ValueError, match="non-stationary"):
            egarch_unconditional_logvar(-0.4, 1.0)


class TestNewsImpactCurve:
    def test_asymmetry_with_negative_gamma(self):
        z, v = news_impact_curve(**EGARCH_TRUE)
        # negative shock raises variance more than positive shock of equal size
        v_neg = v[np.isclose(z, -2.0)][0]
        v_pos = v[np.isclose(z, 2.0)][0]
        assert v_neg > v_pos

    def test_symmetric_when_gamma_zero(self):
        z, v = news_impact_curve(omega=-0.4, alpha=0.1, gamma=0.0, beta=0.96)
        np.testing.assert_allclose(v, v[::-1], rtol=1e-10)

    def test_minimum_not_at_extremes_and_positive(self):
        z, v = news_impact_curve(**EGARCH_TRUE)
        assert np.all(v > 0)
        assert v[0] > v.min() and v[-1] > v.min()
