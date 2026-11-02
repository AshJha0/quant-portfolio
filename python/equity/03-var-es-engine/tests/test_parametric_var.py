"""Parametric VaR: closed forms, EWMA covariance, Cornish-Fisher, coherence."""

import numpy as np
import pytest
from eq_var import (
    cornish_fisher_domain_ok,
    cornish_fisher_valid,
    cornish_fisher_var,
    cornish_fisher_z,
    ewma_covariance,
    expected_shortfall,
    historical_var,
    parametric_var,
    portfolio_sigma,
    sample_covariance,
)
from scipy.stats import norm


class TestClosedForm:
    def test_var_matches_closed_form_known_covariance(self):
        w = np.array([100.0, 200.0])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        sigma = np.sqrt(w @ cov @ w)
        for alpha in (0.05, 0.01):
            expected = -norm.ppf(alpha) * sigma
            assert parametric_var(w, cov, alpha) == pytest.approx(expected, rel=1e-12)

    def test_single_asset(self):
        v = parametric_var(np.array([1000.0]), np.array([[0.02**2]]), 0.01)
        assert v == pytest.approx(-norm.ppf(0.01) * 1000.0 * 0.02, rel=1e-12)

    def test_mean_shifts_var(self):
        w, cov = np.array([1.0]), np.array([[1.0]])
        v0 = parametric_var(w, cov, 0.01, mean=0.0)
        v1 = parametric_var(w, cov, 0.01, mean=0.5)
        assert v1 == pytest.approx(v0 - 0.5, rel=1e-12)

    def test_horizon_scaling_sqrt_time(self):
        w, cov = np.array([1.0]), np.array([[4.0]])
        assert parametric_var(w, cov, 0.01, horizon_days=10) == pytest.approx(
            parametric_var(w, cov, 0.01) * np.sqrt(10), rel=1e-12
        )

    def test_t_var_fatter_than_normal_at_99(self):
        w, cov = np.array([1.0]), np.array([[1.0]])
        assert parametric_var(w, cov, 0.01, dist="t", df=4) > parametric_var(w, cov, 0.01)

    def test_t_var_thinner_than_normal_at_95(self):
        # variance-matched t has *less* mass at moderate quantiles
        w, cov = np.array([1.0]), np.array([[1.0]])
        assert parametric_var(w, cov, 0.05, dist="t", df=4) < parametric_var(w, cov, 0.05)

    def test_t_df_validation(self):
        with pytest.raises(ValueError, match="df"):
            parametric_var(np.array([1.0]), np.array([[1.0]]), 0.01, dist="t", df=2.0)

    def test_unknown_dist_raises(self):
        with pytest.raises(ValueError, match="dist"):
            parametric_var(np.array([1.0]), np.array([[1.0]]), 0.01, dist="cauchy")


class TestCovarianceEstimators:
    def test_sample_covariance_known_data(self):
        x = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        cov = sample_covariance(x)
        np.testing.assert_allclose(cov, np.cov(x.T, ddof=1))

    def test_ewma_covariance_constant_data_fixed_point(self):
        x = np.ones((100, 2))
        # zero-mean seed is J = r r' already, the fixed point of the recursion
        cov = ewma_covariance(x, 0.94)
        np.testing.assert_allclose(cov, np.ones((2, 2)), atol=1e-12)

    def test_ewma_covariance_seed_uses_first_window_only(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(80, 2)) * 0.01
        base = ewma_covariance(x, 0.94)
        y = x.copy()
        y[40:] *= 3.0  # rows after the seed window change only through the recursion
        lam = 0.94
        expect = lam**80 * (x[:30].T @ x[:30] / 30)
        for k, r in enumerate(y):
            expect = expect + (1 - lam) * lam ** (79 - k) * np.outer(r, r)
        np.testing.assert_allclose(ewma_covariance(y, lam), expect, rtol=1e-12)
        assert not np.allclose(base, ewma_covariance(y, lam))
        with pytest.raises(ValueError, match="seed_window"):
            ewma_covariance(x, 0.94, seed_window=0)

    def test_ewma_close_to_sample_on_iid_data(self):
        rng = np.random.default_rng(5)
        true = np.array([[4.0, 1.0], [1.0, 2.0]])
        chol = np.linalg.cholesky(true)
        x = rng.standard_normal((4000, 2)) @ chol.T
        ewma = ewma_covariance(x, 0.99)
        np.testing.assert_allclose(ewma, true, rtol=0.35)

    def test_ewma_reacts_to_recent_regime(self):
        rng = np.random.default_rng(6)
        calm = rng.normal(0, 1, (400, 1))
        wild = rng.normal(0, 5, (50, 1))
        data = np.vstack([calm, wild])
        cov = ewma_covariance(data, 0.94)
        assert cov[0, 0] > 2.0 * sample_covariance(data)[0, 0]  # tracks the wild regime

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            sample_covariance(np.ones((1, 3)))

    def test_portfolio_sigma_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            portfolio_sigma(np.ones(3), np.eye(2))


class TestCornishFisher:
    def test_reduces_to_normal_when_moments_are_gaussian(self):
        for alpha in (0.05, 0.025, 0.01):
            v_cf = cornish_fisher_var(sigma=2.0, alpha=alpha, skew=0.0, excess_kurt=0.0)
            v_n = -norm.ppf(alpha) * 2.0
            assert v_cf == pytest.approx(v_n, abs=1e-12)

    def test_negative_skew_increases_var(self):
        base = cornish_fisher_var(1.0, 0.01, 0.0, 0.0)
        skewed = cornish_fisher_var(1.0, 0.01, -0.3, 0.0)
        assert skewed > base

    def test_excess_kurtosis_increases_99_var(self):
        base = cornish_fisher_var(1.0, 0.01, 0.0, 0.0)
        fat = cornish_fisher_var(1.0, 0.01, 0.0, 2.0)
        assert fat > base

    def test_domain_ok_for_moderate_moments(self):
        assert cornish_fisher_valid(0.0, 0.0, 0.01)
        assert cornish_fisher_valid(-0.5, 1.0, 0.01)
        assert cornish_fisher_valid(0.3, 2.0, 0.01)

    def test_old_counterexample_is_valid_at_99(self):
        """Spec S6: (0.122, -0.427) has its minimum derivative -9.15e-4 at
        z = -4, an endpoint of the old fixed |z| <= 4 interval. On the
        interval implied by alpha = 0.01 (|z| <= 2.326) the map is increasing."""
        assert cornish_fisher_valid(0.122, -0.427, 0.01)
        assert cornish_fisher_valid(0.122, -0.427, 0.001)
        a = -0.427 / 8 - 0.122**2 / 6
        b, c = 0.122 / 3, 1 + 0.427 / 8 + 5 * 0.122**2 / 36
        assert a * 16 - 4 * b + c == pytest.approx(-9.15e-4, abs=1e-6)

    def test_validity_depends_on_alpha(self):
        """(S, K) = (0, -2): g'(z) = 1.25 - 0.25 z^2 is positive only for
        |z| < sqrt(5) = 2.236, so 95 % (z = 1.645) is valid and 99.9 % is not."""
        assert cornish_fisher_valid(0.0, -2.0, 0.05)
        assert not cornish_fisher_valid(0.0, -2.0, 0.001)

    def test_validity_rejects_non_finite(self):
        for bad in (np.nan, np.inf, -np.inf):
            with pytest.raises(ValueError):
                cornish_fisher_valid(bad, 0.0, 0.01)
            with pytest.raises(ValueError):
                cornish_fisher_valid(0.0, bad, 0.01)

    def test_old_predicate_is_deprecated(self):
        with pytest.deprecated_call():
            assert cornish_fisher_domain_ok(0.0, 0.0)
        assert cornish_fisher_domain_ok(-0.5, 1.0)
        assert cornish_fisher_domain_ok(0.3, 2.0)

    def test_domain_flags_pathological_inputs(self):
        # skew 3: dz_cf/dz = 1 + z - 1.5 z^2 + 1.25 goes negative near z = 2
        assert not cornish_fisher_valid(3.0, 0.0, 0.01)
        # kurtosis 10: derivative at z=0 is 1 - 30/24 < 0
        assert not cornish_fisher_valid(0.0, 10.0, 0.01)

    def test_var_raises_outside_domain(self):
        with pytest.raises(ValueError, match="non-monotone"):
            cornish_fisher_var(1.0, 0.01, skew=3.0, excess_kurt=0.0)

    def test_check_can_be_disabled(self):
        v = cornish_fisher_var(1.0, 0.01, skew=3.0, excess_kurt=0.0, check_domain=False)
        assert np.isfinite(v)

    def test_domain_check_is_exact_not_grid_resolution_dependent(self):
        # (S, K) placed so the derivative's parabola vertex falls between two
        # nodes of a 2001-point grid on [-3.5, 3.5]. A grid check misses the
        # dip; the closed form finds it.
        skew, excess_kurt = -0.010499946187942602, 8.000105998830488
        assert not cornish_fisher_valid(skew, excess_kurt, 0.01)
        with pytest.raises(ValueError, match="non-monotone"):
            cornish_fisher_var(1.0, 0.01, skew=skew, excess_kurt=excess_kurt)

    def test_cf_z_polynomial_hand_computed(self):
        # z=2, S=0.5, K=1: z + 3*0.5/6 + 2*1/24 - 11*0.25/36
        z = cornish_fisher_z(2.0, 0.5, 1.0)
        expected = 2.0 + (4 - 1) * 0.5 / 6 + (8 - 6) * 1.0 / 24 - (16 - 10) * 0.25 / 36
        assert z == pytest.approx(expected, abs=1e-12)


class TestCoherence:
    """The classic VaR non-subadditivity counterexample, and ES fixing it.

    Two independent 'defaultable bonds': each gains +5 with prob 0.96 and
    loses 100 with prob 0.04.  At 95 %: individual VaR = -5 (a gain!), but
    the diversified portfolio has VaR = 95 >> VaR_1 + VaR_2 = -10.
    ES is subadditive on the same book.
    """

    @staticmethod
    def _bond_scenarios() -> tuple[np.ndarray, np.ndarray]:
        n = 10_000
        pnl1 = np.full(n, 5.0)
        pnl2 = np.full(n, 5.0)
        pnl1[:400] = -100.0  # bond 1 defaults in scenarios 0..399
        idx2 = np.arange(0, n, 25)  # bond 2 defaults every 25th scenario (400)
        pnl2[idx2] = -100.0  # overlap = 16 scenarios = independence (400*400/10000)
        return pnl1, pnl2

    def test_var_non_subadditive(self):
        pnl1, pnl2 = self._bond_scenarios()
        v1 = historical_var(pnl1, 0.05)
        v2 = historical_var(pnl2, 0.05)
        v_p = historical_var(pnl1 + pnl2, 0.05)
        assert v1 == pytest.approx(-5.0)
        assert v2 == pytest.approx(-5.0)
        assert v_p > 90.0
        assert v_p > v1 + v2  # diversification 'penalty': VaR is not coherent

    def test_es_subadditive_on_same_book(self):
        pnl1, pnl2 = self._bond_scenarios()
        e1 = expected_shortfall(pnl1, 0.05)
        e2 = expected_shortfall(pnl2, 0.05)
        e_p = expected_shortfall(pnl1 + pnl2, 0.05)
        assert e_p <= e1 + e2 + 1e-9

    def test_es_subadditive_random_portfolios(self):
        rng = np.random.default_rng(9)
        for _ in range(5):
            a = rng.standard_t(4, 2000)
            b = 0.3 * a + rng.standard_t(4, 2000)
            ea, eb = expected_shortfall(a, 0.05), expected_shortfall(b, 0.05)
            eab = expected_shortfall(a + b, 0.05)
            assert eab <= ea + eb + 1e-9
