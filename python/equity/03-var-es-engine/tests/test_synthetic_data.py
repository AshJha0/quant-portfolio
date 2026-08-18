"""Synthetic data generators: shapes, seeds, moments, GARCH clustering."""

import numpy as np
import pytest
from scipy.stats import kurtosis

from eq_var.data import (
    default_covariance,
    demo_covariance,
    demo_portfolio,
    simulate_garch_returns,
    simulate_returns,
)


class TestGenerators:
    def test_shapes(self):
        cov = demo_covariance()
        for dist in ("normal", "t", "garch"):
            rets = simulate_returns(300, cov, dist=dist, seed=1)
            assert rets.shape == (300, 4)

    def test_seed_reproducibility(self):
        cov = demo_covariance()
        for dist in ("normal", "t", "garch"):
            a = simulate_returns(200, cov, dist=dist, seed=42)
            b = simulate_returns(200, cov, dist=dist, seed=42)
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        cov = demo_covariance()
        a = simulate_returns(200, cov, seed=1)
        b = simulate_returns(200, cov, seed=2)
        assert not np.array_equal(a, b)

    def test_normal_covariance_matches_target(self):
        cov = demo_covariance()
        rets = simulate_returns(200_000, cov, seed=3)
        np.testing.assert_allclose(np.cov(rets.T), cov, rtol=0.05)

    def test_t_has_fatter_tails_than_normal(self):
        cov = demo_covariance()
        rn = simulate_returns(100_000, cov, dist="normal", seed=4)
        rt = simulate_returns(100_000, cov, dist="t", df=5, seed=4)
        assert kurtosis(rt[:, 0]) > kurtosis(rn[:, 0]) + 1.0

    def test_invalid_dist_raises(self):
        with pytest.raises(ValueError, match="dist"):
            simulate_returns(100, demo_covariance(), dist="lognormal")

    def test_invalid_n_days_raises(self):
        with pytest.raises(ValueError, match="n_days"):
            simulate_returns(0, demo_covariance())


class TestGarch:
    def test_volatility_clustering_in_squared_returns(self):
        """GARCH squared returns are autocorrelated; iid normal's are not."""
        cov = demo_covariance()
        garch = simulate_garch_returns(4000, cov, seed=5)[:, 2]
        normal = simulate_returns(4000, cov, dist="normal", seed=5)[:, 2]

        def acf1(x):
            s = x**2
            return float(np.corrcoef(s[:-1], s[1:])[0, 1])

        assert acf1(garch) > 0.10
        assert abs(acf1(normal)) < 0.05

    def test_unconditional_variance_near_target(self):
        cov = demo_covariance()
        rets = simulate_garch_returns(60_000, cov, seed=6)
        np.testing.assert_allclose(np.var(rets, axis=0), np.diag(cov), rtol=0.25)

    def test_cross_correlation_preserved(self):
        cov = demo_covariance()
        rets = simulate_garch_returns(60_000, cov, seed=7)
        corr = np.corrcoef(rets.T)
        assert corr[0, 2] > 0.5  # AAPL-SPX strongly positive
        assert corr[2, 3] < -0.5  # SPX vs implied vol negative

    def test_nonstationary_params_raise(self):
        with pytest.raises(ValueError, match="stationarity"):
            simulate_garch_returns(100, demo_covariance(), alpha_g=0.3, beta_g=0.75)

    def test_invalid_df_raises(self):
        with pytest.raises(ValueError, match="df"):
            simulate_garch_returns(100, demo_covariance(), df=1.0)


class TestDemoObjects:
    def test_default_covariance_helper(self):
        cov = default_covariance([0.1, 0.2], [[1.0, 0.5], [0.5, 1.0]])
        np.testing.assert_allclose(cov, [[0.01, 0.01], [0.01, 0.04]])

    def test_default_covariance_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="correlation shape"):
            default_covariance([0.1, 0.2], [[1.0]])

    def test_demo_covariance_is_positive_definite(self):
        cov = demo_covariance()
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)
        np.testing.assert_allclose(cov, cov.T)

    def test_demo_portfolio_value_positive(self):
        assert demo_portfolio().value() > 0
