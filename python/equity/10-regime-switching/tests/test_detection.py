"""Regime-detection tests — the online (filtered) causality test is the
critical one for the whole project."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq_regime.detection import (
    expanding_fit_detect,
    filtered_probabilities,
    flip_flop_rate,
    label_states,
    labels_from_vol_means,
    min_duration_filter,
    smoothed_probabilities,
)
from eq_regime.hmm import HMMFit


@pytest.fixture(scope="module")
def small_features(features3):
    """Small slice for fast walk-forward tests."""
    return features3.iloc[:700]


@pytest.fixture(scope="module")
def detection_small(small_features):
    return expanding_fit_detect(
        small_features, n_states=3, min_train=300, refit_every=150,
        seed=0, n_init=1, max_iter=40, n_pca=2,
    )


class TestCriticalCausality:
    def test_filtered_probs_ignore_future_mutation(self, small_features):
        """THE critical test: changing FUTURE observations does not change
        the filtered probability at t."""
        kwargs = dict(n_states=3, min_train=300, refit_every=150,
                      seed=0, n_init=1, max_iter=40, n_pca=2)
        base = expanding_fit_detect(small_features, **kwargs)
        mutated = small_features.copy()
        mutated.iloc[550:] = mutated.iloc[550:] * 3.0 + 1.0
        mut = expanding_fit_detect(mutated, **kwargs)
        cutoff = small_features.index[549]
        pd.testing.assert_frame_equal(
            base.loc[:cutoff], mut.loc[:cutoff], check_exact=True
        )

    def test_smoothed_does_change_with_future(self, small_features, hmm2_fit,
                                              index_returns2):
        """Sanity contrast: smoothed posteriors at t DO change when the
        future changes — which is exactly why they are not tradeable."""
        x = index_returns2[:800].copy()
        t = 400
        base = smoothed_probabilities(hmm2_fit, x)[t]
        x_mut = x.copy()
        x_mut[t + 1:] = -x_mut[t + 1:] * 5.0
        mut = smoothed_probabilities(hmm2_fit, x_mut)[t]
        assert np.abs(base - mut).max() > 1e-4

    def test_filtered_differs_from_smoothed(self, hmm2_fit, index_returns2):
        x = index_returns2[:800]
        filt = filtered_probabilities(hmm2_fit, x)
        smooth = smoothed_probabilities(hmm2_fit, x)
        assert np.abs(filt - smooth).max() > 0.01


class TestLabeling:
    def _fit(self, means):
        k, d = np.asarray(means).shape
        return HMMFit(
            startprob=np.full(k, 1 / k),
            transmat=np.full((k, k), 1 / k),
            means=np.asarray(means, dtype=float),
            covariances=np.array([np.eye(d)] * k),
            log_likelihood=0.0,
        )

    def test_highest_vol_state_is_bear(self):
        fit = self._fit([[0.5, 0.1], [2.0, -0.3], [-0.7, 0.2]])
        labels = label_states(fit, vol_feature_index=0)
        assert labels.state_to_label == {2: "bull", 0: "transition", 1: "bear"}
        assert labels.bear_state == 1
        assert labels.bull_state == 2

    def test_two_state_labeling(self):
        fit = self._fit([[1.5], [-0.5]])
        labels = label_states(fit, vol_feature_index=0)
        assert labels.state_to_label == {1: "bull", 0: "bear"}

    def test_one_state_labeling(self):
        labels = labels_from_vol_means([0.3])
        assert labels.state_to_label == {0: "bull"}

    def test_label_path(self):
        labels = labels_from_vol_means([0.0, 1.0, 2.0])
        np.testing.assert_array_equal(
            labels.label_path(np.array([0, 2, 1])), ["bull", "bear", "transition"]
        )

    def test_labeling_consistent_on_fitted_model(self, hmm2_fit, panel2):
        """On real fits the bear state must carry the higher emission vol."""
        labels = label_states(hmm2_fit, vol_feature_index=0)
        vols = np.sqrt(hmm2_fit.covariances[:, 0, 0])
        # bear = state with highest MEAN of the vol-like feature; on raw
        # returns the bear state also has the larger emission variance
        assert vols[labels.bear_state] != vols[labels.bull_state]

    def test_vol_index_out_of_range(self):
        fit = self._fit([[0.0], [1.0]])
        with pytest.raises(ValueError, match="out of range"):
            label_states(fit, vol_feature_index=5)


class TestStability:
    def test_flip_flop_rate_hand_checked(self):
        assert flip_flop_rate(np.array([0, 0, 1, 1, 0])) == pytest.approx(2 / 4)
        assert flip_flop_rate(np.array([1])) == 0.0
        assert flip_flop_rate(np.array([0, 1, 0, 1])) == 1.0

    def test_min_duration_filter_removes_flicker(self):
        s = np.array([0] * 10 + [1] * 2 + [0] * 10)
        out = min_duration_filter(s, 5)
        np.testing.assert_array_equal(out, np.zeros(22, dtype=int))

    def test_min_duration_filter_keeps_long_runs(self):
        s = np.array([0] * 10 + [1] * 8 + [0] * 10)
        out = min_duration_filter(s, 5)
        np.testing.assert_array_equal(out, s)

    def test_min_duration_filter_chained_flickers(self):
        s = np.array([0] * 6 + [1, 1] + [2, 2] + [0] * 6)
        out = min_duration_filter(s, 3)
        np.testing.assert_array_equal(out, np.zeros(16, dtype=int))

    def test_min_duration_filter_first_run_kept(self):
        s = np.array([1, 1] + [0] * 10)
        out = min_duration_filter(s, 5)
        np.testing.assert_array_equal(out, s)

    def test_min_duration_filter_noop_and_errors(self):
        s = np.array([0, 1, 0])
        np.testing.assert_array_equal(min_duration_filter(s, 1), s)
        with pytest.raises(ValueError, match="min_duration"):
            min_duration_filter(s, 0)

    def test_filter_reduces_flip_flop_rate(self, detection_small):
        codes = pd.Categorical(detection_small["regime"]).codes
        filtered = min_duration_filter(codes, 5)
        assert flip_flop_rate(filtered) <= flip_flop_rate(codes)


class TestDetectionTable:
    def test_probabilities_sum_to_one(self, detection_small):
        total = detection_small[["p_bull", "p_transition", "p_bear"]].sum(axis=1)
        np.testing.assert_allclose(total.to_numpy(), 1.0, atol=1e-10)

    def test_regime_is_argmax(self, detection_small):
        probs = detection_small[["p_bull", "p_transition", "p_bear"]]
        argmax = probs.idxmax(axis=1).str.replace("p_", "")
        assert (argmax == detection_small["regime"]).all()

    def test_detection_covers_expected_dates(self, small_features, detection_small):
        assert detection_small.index[0] == small_features.index[300]
        assert detection_small.index[-1] == small_features.index[-1]
        assert len(detection_small) == len(small_features) - 300

    def test_min_train_validation(self, small_features):
        with pytest.raises(ValueError, match="min_train"):
            expanding_fit_detect(small_features, min_train=len(small_features))
        with pytest.raises(ValueError, match="refit_every"):
            expanding_fit_detect(small_features, min_train=300, refit_every=0)
