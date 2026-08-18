"""Expected Shortfall: exact empirical estimator, closed forms, ES >= VaR."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from eq_var import (
    es_standard_error_bootstrap,
    expected_shortfall,
    historical_var,
    normal_es,
    parametric_es,
    parametric_var,
    student_t_es,
)


class TestEmpiricalES:
    def test_exact_on_known_array_integer_tail(self):
        # pnl = -100..-1; alpha=0.05, n=100 -> mean of worst 5:
        # (100+99+98+97+96)/5 = 98
        pnl = np.arange(-100.0, 0.0)
        assert expected_shortfall(pnl, 0.05) == pytest.approx(98.0, abs=1e-12)

    def test_exact_on_known_array_fractional_tail(self):
        # n=10 scaled to meet min obs: use n=100, alpha=0.025 -> a*n=2.5:
        # ES = (100 + 99 + 0.5*98)/2.5 = 99.2
        pnl = np.arange(-100.0, 0.0)
        assert expected_shortfall(pnl, 0.025) == pytest.approx(99.2, abs=1e-12)

    def test_exact_alpha_001_smallest_tail(self):
        # a*n = 1: ES = worst observation
        pnl = np.arange(-100.0, 0.0)
        assert expected_shortfall(pnl, 0.01) == pytest.approx(100.0, abs=1e-12)

    def test_es_geq_var_everywhere(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            pnl = rng.standard_t(3, 500) * rng.uniform(1, 100)
            for alpha in (0.01, 0.025, 0.05, 0.1):
                assert expected_shortfall(pnl, alpha) >= historical_var(pnl, alpha) - 1e-12

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 10"):
            expected_shortfall(np.zeros(5), 0.05)

    def test_nan_raises(self):
        pnl = np.zeros(100)
        pnl[0] = np.inf
        with pytest.raises(ValueError, match="NaN or infinite"):
            expected_shortfall(pnl, 0.05)


class TestClosedForms:
    def test_normal_es_identity_to_1e10(self):
        """phi(z_alpha)/alpha identity vs direct numerical tail integration."""
        for alpha in (0.01, 0.025, 0.05):
            analytic = normal_es(1.0, alpha)
            q = norm.ppf(alpha)
            integral, _ = quad(lambda x: x * norm.pdf(x), -30.0, q, epsabs=1e-14)
            numeric = -integral / alpha
            assert analytic == pytest.approx(numeric, abs=1e-10)

    def test_normal_es_scales_with_sigma_and_mean(self):
        assert normal_es(3.0, 0.01, mean=0.5) == pytest.approx(3.0 * normal_es(1.0, 0.01) - 0.5)

    def test_normal_es_exceeds_normal_var(self):
        for alpha in (0.01, 0.05):
            var = parametric_var(np.array([1.0]), np.array([[1.0]]), alpha)
            assert normal_es(1.0, alpha) > var

    def test_t_es_identity_vs_numerical_integration(self):
        from scipy.stats import t as student

        df, alpha = 6.0, 0.025
        analytic = student_t_es(1.0, alpha, df)
        scale = np.sqrt((df - 2.0) / df)
        q = student.ppf(alpha, df)
        integral, _ = quad(lambda x: x * student.pdf(x, df), -300.0, q, epsabs=1e-13)
        numeric = -integral / alpha * scale
        assert analytic == pytest.approx(numeric, abs=1e-9)

    def test_t_es_fatter_than_normal(self):
        assert student_t_es(1.0, 0.01, df=4) > normal_es(1.0, 0.01)

    def test_t_es_converges_to_normal_as_df_grows(self):
        assert student_t_es(1.0, 0.01, df=1e6) == pytest.approx(normal_es(1.0, 0.01), rel=1e-4)

    def test_es975_close_to_var99_for_normal(self):
        """The FRTB calibration fact: normal ES97.5 ~ VaR99 (within ~1%)."""
        es975 = normal_es(1.0, 0.025)
        var99 = -norm.ppf(0.01)
        assert es975 == pytest.approx(var99, rel=0.01)

    def test_parametric_es_consistent_with_normal_es(self):
        w = np.array([100.0, -50.0])
        cov = np.array([[0.04, 0.01], [0.01, 0.02]])
        sigma = np.sqrt(w @ cov @ w)
        assert parametric_es(w, cov, 0.025) == pytest.approx(normal_es(sigma, 0.025), rel=1e-12)

    def test_parametric_es_geq_parametric_var(self):
        w = np.array([100.0])
        cov = np.array([[0.01]])
        for dist, df in (("normal", 6.0), ("t", 4.0)):
            es = parametric_es(w, cov, 0.01, dist=dist, df=df)
            var = parametric_var(w, cov, 0.01, dist=dist, df=df)
            assert es > var

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="sigma"):
            normal_es(-1.0, 0.01)
        with pytest.raises(ValueError, match="df"):
            student_t_es(1.0, 0.01, df=2.0)
        with pytest.raises(ValueError, match="alpha"):
            normal_es(1.0, 0.6)


class TestEstimationUncertainty:
    def test_bootstrap_se_positive_and_reproducible(self):
        rng = np.random.default_rng(2)
        pnl = rng.standard_t(5, 1000)
        se1 = es_standard_error_bootstrap(pnl, 0.025, n_boot=200, seed=3)
        se2 = es_standard_error_bootstrap(pnl, 0.025, n_boot=200, seed=3)
        assert se1 > 0
        assert se1 == se2

    def test_es_se_larger_for_smaller_alpha(self):
        """Deeper tail -> fewer effective observations -> larger SE."""
        rng = np.random.default_rng(4)
        pnl = rng.standard_t(5, 2000)
        se_deep = es_standard_error_bootstrap(pnl, 0.01, n_boot=300, seed=5)
        se_shallow = es_standard_error_bootstrap(pnl, 0.10, n_boot=300, seed=5)
        assert se_deep > se_shallow
