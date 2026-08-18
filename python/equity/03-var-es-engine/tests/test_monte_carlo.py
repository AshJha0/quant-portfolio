"""Monte Carlo VaR: convergence, reproducibility, singular covariances, SEs."""

import numpy as np
import pytest

from eq_var import (
    EquityPosition,
    Portfolio,
    RiskFactor,
    monte_carlo_pnl,
    monte_carlo_var,
    parametric_var,
    safe_cholesky,
    simulate_factor_returns,
    var_confidence_interval,
    var_standard_error_bootstrap,
)


def linear_portfolio() -> Portfolio:
    factors = {
        "A": RiskFactor("A", "equity", 100.0),
        "B": RiskFactor("B", "equity", 50.0),
    }
    return Portfolio(
        [
            EquityPosition(name="a", factor="A", shares=1000.0),
            EquityPosition(name="b", factor="B", shares=-500.0),
        ],
        factors,
    )


COV = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])


class TestConvergence:
    def test_mc_converges_to_parametric_closed_form(self):
        """Normal factors + linear portfolio: MC VaR -> closed form within 3 SE."""
        pf = linear_portfolio()
        closed = parametric_var(pf.delta_exposures(), COV, 0.01)
        pnl = monte_carlo_pnl(pf, COV, n_paths=400_000, dist="normal", seed=17)
        mc = float(-np.quantile(pnl, 0.01))
        se = var_standard_error_bootstrap(pnl, 0.01, n_boot=200, seed=1)
        assert abs(mc - closed) < 3.0 * se

    def test_mc_95_also_converges(self):
        pf = linear_portfolio()
        closed = parametric_var(pf.delta_exposures(), COV, 0.05)
        pnl = monte_carlo_pnl(pf, COV, n_paths=400_000, dist="normal", seed=18)
        se = var_standard_error_bootstrap(pnl, 0.05, n_boot=200, seed=2)
        assert abs(-np.quantile(pnl, 0.05) - closed) < 3.0 * se

    def test_t_mc_fatter_than_normal_at_99(self):
        pf = linear_portfolio()
        v_n = monte_carlo_var(pf, COV, 0.01, n_paths=200_000, dist="normal", seed=3)
        v_t = monte_carlo_var(pf, COV, 0.01, n_paths=200_000, dist="t", df=4, seed=3)
        assert v_t > 1.1 * v_n

    def test_simulated_covariance_matches_target(self):
        rets = simulate_factor_returns(COV, 300_000, dist="normal", seed=4)
        np.testing.assert_allclose(np.cov(rets.T), COV, rtol=0.05)

    def test_t_simulation_matches_target_covariance(self):
        rets = simulate_factor_returns(COV, 500_000, dist="t", df=6, seed=5)
        np.testing.assert_allclose(np.cov(rets.T), COV, rtol=0.05)


class TestReproducibility:
    def test_same_seed_same_result(self):
        pf = linear_portfolio()
        v1 = monte_carlo_var(pf, COV, 0.01, n_paths=10_000, seed=42)
        v2 = monte_carlo_var(pf, COV, 0.01, n_paths=10_000, seed=42)
        assert v1 == v2

    def test_different_seed_different_result(self):
        pf = linear_portfolio()
        v1 = monte_carlo_var(pf, COV, 0.01, n_paths=10_000, seed=1)
        v2 = monte_carlo_var(pf, COV, 0.01, n_paths=10_000, seed=2)
        assert v1 != v2

    def test_generator_object_accepted(self):
        rets = simulate_factor_returns(COV, 100, seed=np.random.default_rng(7))
        assert rets.shape == (100, 2)


class TestSafeCholesky:
    def test_plain_cholesky_on_spd_matrix(self):
        chol = safe_cholesky(COV)
        np.testing.assert_allclose(chol @ chol.T, COV, atol=1e-15)

    def test_jitter_path_on_singular_matrix(self):
        # rank-1: perfectly correlated factors
        v = np.array([1.0, 2.0])
        singular = np.outer(v, v)
        assert np.linalg.matrix_rank(singular) == 1
        chol = safe_cholesky(singular)
        np.testing.assert_allclose(chol @ chol.T, singular, atol=1e-6)

    def test_simulation_on_singular_covariance(self):
        v = np.array([0.02, 0.04])
        singular = np.outer(v, v)
        rets = simulate_factor_returns(singular, 50_000, seed=8)
        corr = np.corrcoef(rets.T)[0, 1]
        assert corr == pytest.approx(1.0, abs=1e-3)

    def test_zero_variance_factor(self):
        cov = np.array([[0.0004, 0.0], [0.0, 0.0]])
        rets = simulate_factor_returns(cov, 10_000, seed=9)
        assert float(np.std(rets[:, 1])) < 1e-4  # jitter noise only

    def test_asymmetric_matrix_raises(self):
        with pytest.raises(ValueError, match="symmetric"):
            safe_cholesky(np.array([[1.0, 0.5], [0.1, 1.0]]))

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            safe_cholesky(np.ones((2, 3)))


class TestStandardErrors:
    def test_bootstrap_se_shrinks_with_sample_size(self):
        rng = np.random.default_rng(10)
        small = rng.normal(0, 1, 2_000)
        large = rng.normal(0, 1, 50_000)
        se_small = var_standard_error_bootstrap(small, 0.01, n_boot=300, seed=1)
        se_large = var_standard_error_bootstrap(large, 0.01, n_boot=300, seed=1)
        assert se_large < se_small / 2.0  # ~ sqrt(25) = 5x in theory

    def test_order_statistic_ci_brackets_true_var(self):
        from scipy.stats import norm

        rng = np.random.default_rng(11)
        pnl = rng.normal(0, 1, 100_000)
        lo, hi = var_confidence_interval(pnl, 0.01, conf=0.95)
        true_var = -norm.ppf(0.01)
        assert lo < true_var < hi
        assert lo < hi

    def test_ci_ordering_and_width(self):
        rng = np.random.default_rng(12)
        pnl = rng.standard_t(5, 20_000)
        lo, hi = var_confidence_interval(pnl, 0.05, conf=0.99)
        lo2, hi2 = var_confidence_interval(pnl, 0.05, conf=0.90)
        assert hi - lo > hi2 - lo2  # higher confidence -> wider interval

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="at least 10"):
            var_standard_error_bootstrap(np.zeros(5), 0.01)
        with pytest.raises(ValueError, match="n_paths"):
            simulate_factor_returns(COV, 0)
        with pytest.raises(ValueError, match="df"):
            simulate_factor_returns(COV, 10, dist="t", df=1.5)
        with pytest.raises(ValueError, match="mean"):
            simulate_factor_returns(COV, 10, mean=np.zeros(3))
