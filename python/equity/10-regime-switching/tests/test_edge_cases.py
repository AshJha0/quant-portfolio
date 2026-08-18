"""Edge cases from the documentation contract: every case here is also
described in docs/METHODOLOGY.md / docs/VALIDATION.md."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq_regime.detection import min_duration_filter
from eq_regime.features import expanding_zscore
from eq_regime.gmm import fit_gmm
from eq_regime.hmm import expected_durations, fit_hmm, stationary_distribution
from eq_regime.pca import fit_pca
from eq_regime.strategy import hysteresis_regime, naive_threshold_regime


class TestDegenerateData:
    def test_gmm_on_identical_observations(self):
        """All-identical data: regularization must keep EM finite."""
        x = np.ones((60, 2))
        fit = fit_gmm(x, 2, seed=0, n_init=1, reg_covar=1e-6, max_iter=50)
        assert np.isfinite(fit.log_likelihood)
        assert not np.isnan(fit.means).any()
        for c in fit.covariances:
            np.linalg.cholesky(c)

    def test_hmm_on_identical_observations(self):
        x = np.ones(80)
        fit = fit_hmm(x, 2, seed=0, n_init=1, reg_covar=1e-6, max_iter=30)
        assert np.isfinite(fit.log_likelihood)
        probs, _ = fit.filter(x)
        assert not np.isnan(probs).any()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_hmm_state_collapse_on_two_cluster_data(self):
        """Fitting 3 states to clearly 2-cluster data must not blow up."""
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(-3, 0.5, 300), rng.normal(3, 0.5, 300)])
        rng.shuffle(x)
        fit = fit_hmm(x, 3, seed=0, n_init=2, max_iter=60)
        assert np.isfinite(fit.log_likelihood)
        np.testing.assert_allclose(fit.transmat.sum(axis=1), 1.0, atol=1e-10)
        pi = stationary_distribution(fit.transmat)
        np.testing.assert_allclose(pi @ fit.transmat, pi, atol=1e-10)

    def test_gmm_more_components_than_clusters(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 200)
        fit = fit_gmm(x, 3, seed=0, n_init=2, max_iter=100)
        assert np.isfinite(fit.log_likelihood)
        assert fit.weights.sum() == pytest.approx(1.0, abs=1e-10)


class TestSmallAndShort:
    def test_k1_models(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, 100)
        g = fit_gmm(x, 1, seed=0)
        h = fit_hmm(x, 1, seed=0, n_init=1)
        assert np.isfinite(g.log_likelihood) and np.isfinite(h.log_likelihood)
        assert expected_durations(h.transmat)[0] == np.inf

    def test_very_short_series_raise(self):
        with pytest.raises(ValueError, match="too short"):
            fit_hmm(np.array([1.0, 2.0, 3.0]), 2)
        with pytest.raises(ValueError, match="too few"):
            fit_gmm(np.array([1.0, 2.0]), 2)
        with pytest.raises(ValueError, match="more observations"):
            fit_pca(np.random.default_rng(0).standard_normal((3, 5)))


class TestBoundaries:
    def test_probability_exactly_at_thresholds(self):
        """Exactly-at-threshold probabilities never flip the state (strict
        inequalities are the documented convention)."""
        p = np.full(10, 0.70)
        assert not hysteresis_regime(p, enter=0.70, exit_=0.30).any()
        p2 = np.full(10, 0.30)
        state = hysteresis_regime(np.concatenate([[0.9], p2]), enter=0.7, exit_=0.3)
        assert state.all()  # entered at 0.9, 0.30 never exits (strict <)
        assert not naive_threshold_regime(np.array([0.5]), threshold=0.5)[0]

    def test_probability_zero_and_one(self):
        p = np.array([0.0, 1.0, 0.0])
        out = hysteresis_regime(p)
        np.testing.assert_array_equal(out, [False, True, False])

    def test_expanding_zscore_constant_prefix(self):
        """Constant prefix then variation: no inf, no lookahead artifact."""
        s = pd.Series(
            [1.0] * 20 + [2.0, 3.0, 1.5, 2.5] * 5,
            index=pd.bdate_range("2020-01-01", periods=40),
        )
        z = expanding_zscore(s, min_periods=5)
        assert not np.isinf(z.to_numpy()).any()
        assert z.iloc[-1] == z.iloc[-1]  # finite tail

    def test_min_duration_filter_run_exactly_at_threshold(self):
        s = np.array([0] * 5 + [1] * 3 + [0] * 5)
        # run of 3 with min_duration=3 survives (>= threshold)
        np.testing.assert_array_equal(min_duration_filter(s, 3), s)
        # run of 3 with min_duration=4 is removed
        np.testing.assert_array_equal(
            min_duration_filter(s, 4), np.zeros(13, dtype=int)
        )


class TestNumericalExtremes:
    def test_hmm_with_extreme_outlier(self):
        """A 50-sigma outlier must not underflow the forward pass."""
        rng = np.random.default_rng(3)
        x = rng.normal(0, 0.01, 500)
        x[250] = 0.5  # 50-sigma day (crash)
        fit = fit_hmm(x, 2, seed=0, n_init=1, max_iter=50)
        probs, ll = fit.filter(x)
        assert np.isfinite(ll)
        assert not np.isnan(probs).any()

    def test_stationary_distribution_near_absorbing(self):
        p = np.array([[1.0 - 1e-9, 1e-9], [1e-9, 1.0 - 1e-9]])
        pi = stationary_distribution(p)
        np.testing.assert_allclose(pi @ p, pi, atol=1e-12)
        assert pi.sum() == pytest.approx(1.0, abs=1e-12)
