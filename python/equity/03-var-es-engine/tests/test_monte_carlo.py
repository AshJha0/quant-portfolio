"""Monte Carlo VaR: convergence, reproducibility, singular covariances, SEs."""

import numpy as np
import pytest
from eq_var import (
    EquityPosition,
    Portfolio,
    RiskFactor,
    cholesky,
    higham_nearest_correlation,
    monte_carlo_pnl,
    monte_carlo_risk,
    monte_carlo_var,
    parametric_var,
    safe_cholesky,
    simulate_factor_returns,
    var_confidence_interval,
    var_standard_error_bootstrap,
    var_standard_error_kde,
)
from scipy.stats import norm


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
    @pytest.mark.parametrize("alpha, seed", [(0.01, 17), (0.05, 18)])
    def test_mc_converges_to_the_exact_lognormal_quantile(self, alpha, seed):
        """MC moves are log returns (spec S10). For a single long stock the
        P&L V * expm1(r) is increasing in r, so the exact VaR is
        -V * expm1(sigma * z_alpha). MC must land within 3 bootstrap SE."""
        pf = Portfolio([EquityPosition(name="a", factor="A", shares=1000.0)],
                       {"A": RiskFactor("A", "equity", 100.0)})
        cov = np.array([[4e-4]])
        exact = -1e5 * float(np.expm1(0.02 * norm.ppf(alpha)))
        pnl = monte_carlo_pnl(pf, cov, n_paths=400_000, dist="normal", seed=seed)
        mc = monte_carlo_var(pf, cov, alpha, n_paths=400_000, seed=seed)
        se = var_standard_error_bootstrap(pnl, alpha, n_boot=200, seed=1)
        assert abs(mc - exact) < 3.0 * se

    def test_linear_mc_is_close_to_parametric_up_to_the_log_return_convexity(self):
        """Parametric VaR is linear in returns; MC uses exp(r). The gap on a
        2 % daily vol book is of order sigma^2 terms, under 3 %."""
        pf = linear_portfolio()
        closed = parametric_var(pf.delta_exposures(), COV, 0.01)
        mc = monte_carlo_var(pf, COV, 0.01, n_paths=400_000, seed=17)
        assert mc == pytest.approx(closed, rel=0.03)

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
        with pytest.deprecated_call():
            chol = safe_cholesky(COV)
        np.testing.assert_allclose(chol @ chol.T, COV, atol=1e-15)

    def test_jitter_path_on_singular_matrix(self):
        # rank-1: perfectly correlated factors
        v = np.array([1.0, 2.0])
        singular = np.outer(v, v)
        assert np.linalg.matrix_rank(singular) == 1
        with pytest.warns(RuntimeWarning, match="jitter"):
            chol = cholesky(singular)
        np.testing.assert_allclose(chol @ chol.T, singular, atol=1e-10)

    def test_simulation_on_singular_covariance(self):
        v = np.array([0.02, 0.04])
        singular = np.outer(v, v)
        with pytest.warns(RuntimeWarning, match="jitter"):
            rets = simulate_factor_returns(singular, 50_000, seed=8)
        corr = np.corrcoef(rets.T)[0, 1]
        assert corr == pytest.approx(1.0, abs=1e-3)

    def test_zero_variance_factor(self):
        cov = np.array([[0.0004, 0.0], [0.0, 0.0]])
        with pytest.warns(RuntimeWarning, match="jitter"):
            rets = simulate_factor_returns(cov, 10_000, seed=9)
        # jitter <= 1e-6 * mean variance = 2e-10, so the noise sd is <= 1.5e-5
        assert float(np.std(rets[:, 1])) < 1.5e-5

    def test_asymmetric_matrix_raises(self):
        with pytest.raises(ValueError, match="symmetric"):
            cholesky(np.array([[1.0, 0.5], [0.1, 1.0]]))

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            cholesky(np.ones((2, 3)))

    # ---- spec S7 ------------------------------------------------------- #
    A = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])

    def test_indefinite_matrix_raises_instead_of_silent_10x_jitter(self):
        """C-1: eigenvalues -0.8, 1.9, 1.9. The old safe_cholesky returned L
        with L L' - A = I (jitter = 100 % of the mean variance) and no warning."""
        with pytest.raises(np.linalg.LinAlgError, match=r"-0.8"):
            cholesky(self.A)
        with pytest.deprecated_call(), pytest.raises(np.linalg.LinAlgError):
            safe_cholesky(self.A)

    def test_jitter_never_exceeds_the_cap(self):
        """Smallest eigenvalue -1e-7 x mean variance: the cap (1e-6) succeeds
        and the reconstruction error is at most the cap."""
        v = np.linalg.qr(np.random.default_rng(3).normal(size=(4, 4)))[0]
        lam = np.array([-1e-7, 1.0, 2.0, 1.0])
        a = (v * lam) @ v.T
        a = 0.5 * (a + a.T)
        with pytest.warns(RuntimeWarning, match="jitter"):
            chol = cholesky(a)
        assert np.abs(chol @ chol.T - a).max() <= 1e-6 * np.mean(np.diag(a)) * (1 + 1e-9)

    def test_symmetry_check_is_relative(self):
        """Entries of order 1e-5: an asymmetry of 1e-13 (above 1e-10 relative)
        must be refused; the old absolute 1e-12 * max(1, ...) accepted it."""
        a = np.array([[4e-4, 1e-4], [1e-4 + 1e-13, 9e-4]])
        with pytest.raises(ValueError, match="symmetric"):
            cholesky(a)

    def test_higham_repair(self):
        with pytest.warns(RuntimeWarning):
            chol = cholesky(self.A, repair="higham")
        rep = chol @ chol.T
        # The nearest correlation matrix to A is 0.5 off the diagonal (signs kept).
        np.testing.assert_allclose(rep, [[1, .5, .5], [.5, 1, -.5], [.5, -.5, 1]], atol=1e-6)
        corr = higham_nearest_correlation(self.A)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-15)
        assert np.linalg.eigvalsh(corr).min() > -1e-10

    def test_clip_repair_keeps_variances(self):
        cov = self.A * 4.0
        with pytest.warns(RuntimeWarning, match="clip"):
            chol = cholesky(cov, repair="clip")
        np.testing.assert_allclose(np.diag(chol @ chol.T), 4.0, rtol=1e-12)

    def test_unknown_repair_raises(self):
        with pytest.raises(ValueError, match="repair"):
            cholesky(COV, repair="nearest")


class TestSpecS10:
    def test_monte_carlo_risk_uses_one_scenario_set(self):
        pf = linear_portfolio()
        out = monte_carlo_risk(pf, COV, 0.01, n_paths=20_000, seed=4, n_boot=50)
        pnl = out["pnl"]
        losses = np.sort(-pnl)[::-1]
        assert out["var"] == losses[199]  # k = 200, m = 200
        assert out["es"] == pytest.approx(losses[:200].mean(), rel=1e-13)
        assert out["var_se"] > 0

    def test_heavy_t_paths_never_raise(self):
        """H-3: with arithmetic returns a df=2.2 path below -100 % gave a
        negative spot and bs_price raised. Log returns keep every spot > 0."""
        from eq_var.data import demo_covariance, demo_portfolio

        pf = demo_portfolio()
        cov = demo_covariance() * 25.0
        pnl = monte_carlo_pnl(pf, cov, n_paths=20_000, dist="t", df=2.2, seed=0)
        assert np.isfinite(pnl).all()

    def test_kde_se_is_close_to_bootstrap_for_normal_pnl(self):
        pnl = np.random.default_rng(5).standard_normal(50_000)
        kde = var_standard_error_kde(pnl, 0.01)
        exact = np.sqrt(0.01 * 0.99 / 50_000) / norm.pdf(norm.ppf(0.01))
        assert kde == pytest.approx(exact, rel=0.15)

    def test_confidence_interval_refuses_nan(self):
        """H-1: np.sort pushed NaN to the end and a finite interval came back."""
        p = np.random.default_rng(6).normal(size=500)
        p[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            var_confidence_interval(p, 0.01)

    def test_confidence_interval_refuses_impossible_coverage(self):
        """Old code fell back to rank 0 / n-1 and returned a narrower interval."""
        p = np.random.default_rng(7).normal(size=25)
        with pytest.raises(ValueError, match="too short"):
            var_confidence_interval(p, 0.01, conf=0.99)


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
