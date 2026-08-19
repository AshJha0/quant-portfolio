"""Single-regime data, label switching, and degenerate transition matrices.

These are the domain-specific edge cases required by CONVENTIONS.md item 6
for a regime-switching project. The recurring theme is that a *degenerate*
fit must be detectable from the model output rather than pass as a
confident-looking set of regimes.
"""

import numpy as np
import pytest

from eq_regime.data import make_gbm_panel
from eq_regime.detection import (flip_flop_rate, label_states,
                                 labels_from_vol_means, min_duration_filter)
from eq_regime.gmm import fit_gmm, match_permutation
from eq_regime.hmm import (expected_durations, fit_hmm, forward_backward,
                           forward_filter, stationary_distribution, viterbi)


# ---------------------------------------------------------------------------
# Single-regime (null) data
# ---------------------------------------------------------------------------

def test_single_regime_data_fitted_with_two_states_stays_finite():
    """Fitting K=2 to data that truly has one regime must not produce NaN
    parameters or a degenerate covariance — the reg_covar ridge is what
    keeps this well posed."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 2)) * 0.01
    fit = fit_hmm(x, 2, seed=0, n_init=2)
    assert np.isfinite(fit.transmat).all()
    assert np.isfinite(fit.means).all()
    assert np.isfinite(fit.covariances).all()
    assert np.isfinite(fit.log_likelihood)
    assert np.allclose(fit.transmat.sum(axis=1), 1.0, atol=1e-12)
    for k in range(2):
        assert np.linalg.eigvalsh(fit.covariances[k]).min() > 0.0


def test_single_regime_states_are_economically_indistinguishable():
    """On one-regime data the two fitted state means must be close relative
    to the pooled dispersion — the honest signature of 'there is no second
    regime here'."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((800, 1)) * 0.01
    fit = fit_hmm(x, 2, seed=1, n_init=3)
    separation = abs(fit.means[0, 0] - fit.means[1, 0])
    assert separation < 3.0 * float(x.std())


def test_gbm_null_panel_regimes_flip_more_than_true_regime_panel():
    """The null: on a no-regime GBM panel the decoded path should be far less
    persistent than on genuine regime data (this is the diagnostic a desk
    uses to reject a spurious regime model)."""
    from eq_regime.data import make_regime_panel
    from eq_regime.features import build_features

    null = build_features(make_gbm_panel(n_assets=5, n_days=1200, seed=11).prices)
    real = build_features(make_regime_panel(n_states=2, n_assets=5, n_days=1200,
                                            seed=3).prices)
    f_null = fit_hmm(null.to_numpy(), 2, seed=0, n_init=2)
    f_real = fit_hmm(real.to_numpy(), 2, seed=0, n_init=2)
    ff_null = flip_flop_rate(f_null.viterbi(null.to_numpy()))
    ff_real = flip_flop_rate(f_real.viterbi(real.to_numpy()))
    assert ff_null > ff_real


def test_k1_hmm_is_a_single_gaussian():
    rng = np.random.default_rng(2)
    x = rng.normal(0.001, 0.02, size=(400, 1))
    fit = fit_hmm(x, 1, seed=0, n_init=1)
    assert fit.transmat.shape == (1, 1)
    assert fit.transmat[0, 0] == pytest.approx(1.0)
    assert fit.means[0, 0] == pytest.approx(float(x.mean()), rel=1e-6)
    # A single absorbing state: stationary mass 1, infinite expected duration.
    assert stationary_distribution(fit.transmat) == pytest.approx([1.0])
    assert np.isinf(expected_durations(fit.transmat)[0])


# ---------------------------------------------------------------------------
# Label switching
# ---------------------------------------------------------------------------

def test_label_switching_does_not_change_economic_labels():
    """Raw state indices are arbitrary; permuting them must leave the
    vol-sorted economic labels pointing at the same states."""
    vol_means = np.array([0.10, 0.35, 0.20])
    lab = labels_from_vol_means(vol_means)
    assert lab.state_to_label[0] == "bull"      # lowest vol
    assert lab.state_to_label[1] == "bear"      # highest vol
    assert lab.state_to_label[2] == "transition"
    # Permute the state ordering; the *labels of the same vol levels* survive.
    perm = [2, 0, 1]
    lab_p = labels_from_vol_means(vol_means[perm])
    assert lab_p.state_to_label[perm.index(0)] == "bull"
    assert lab_p.state_to_label[perm.index(1)] == "bear"
    assert lab_p.bear_state == perm.index(1)
    assert lab_p.bull_state == perm.index(0)


def test_label_path_is_invariant_under_state_relabelling():
    """The decoded *economic* path must be identical whichever way EM happened
    to number the states."""
    vol_means = np.array([0.08, 0.40])
    path = np.array([0, 0, 1, 1, 0, 1, 0])
    lab = labels_from_vol_means(vol_means)
    labelled = lab.label_path(path)
    # Now relabel: swap state ids and swap the vol means to match.
    lab_swapped = labels_from_vol_means(vol_means[::-1])
    labelled_swapped = lab_swapped.label_path(1 - path)
    assert list(labelled) == list(labelled_swapped)


def test_match_permutation_recovers_planted_relabelling():
    """Exhaustive-search matcher must undo an arbitrary permutation exactly."""
    true_means = np.array([[0.0, 1.0], [5.0, 5.0], [-3.0, 2.0]])
    for perm in ((2, 0, 1), (1, 2, 0), (0, 2, 1), (0, 1, 2)):
        est = true_means[list(perm)]
        # est[j] == true_means[perm[j]]; the matcher returns p with
        # est[p[i]] ~ true_means[i], i.e. p is the inverse of perm.
        p = match_permutation(true_means, est)
        for i in range(3):
            assert np.allclose(est[p[i]], true_means[i])


def test_match_permutation_shape_mismatch_raises():
    with pytest.raises(ValueError, match="identical shapes"):
        match_permutation(np.zeros((3, 2)), np.zeros((2, 2)))


def test_labels_require_at_least_one_state():
    with pytest.raises(ValueError, match="at least one state"):
        labels_from_vol_means(np.array([]))


def test_label_states_rejects_out_of_range_feature_index():
    rng = np.random.default_rng(5)
    fit = fit_hmm(rng.standard_normal((300, 2)) * 0.01, 2, seed=0, n_init=1)
    with pytest.raises(ValueError, match="out of range"):
        label_states(fit, vol_feature_index=5)
    with pytest.raises(ValueError, match="out of range"):
        label_states(fit, vol_feature_index=-1)


# ---------------------------------------------------------------------------
# Degenerate transition matrices
# ---------------------------------------------------------------------------

def test_absorbing_state_stationary_and_duration():
    P = np.array([[1.0, 0.0], [0.2, 0.8]])
    pi = stationary_distribution(P)
    assert pi == pytest.approx([1.0, 0.0], abs=1e-12)
    assert np.allclose(pi @ P, pi, atol=1e-12)
    d = expected_durations(P)
    assert np.isinf(d[0])
    assert d[1] == pytest.approx(5.0, rel=1e-12)


def test_identity_transition_matrix_is_stationary_but_not_unique():
    """P = I is reducible: every distribution is stationary. The solver
    returns the minimum-norm (uniform) one; the invariance identity must
    still hold exactly, and infinite durations flag the degeneracy."""
    P = np.eye(3)
    pi = stationary_distribution(P)
    assert np.allclose(pi @ P, pi, atol=1e-14)
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(np.isinf(expected_durations(P)))


def test_stationary_distribution_invariance_on_random_chains():
    """Property: pi P = pi and sum(pi) = 1 for random row-stochastic chains."""
    rng = np.random.default_rng(9)
    for k in (2, 3, 5):
        for _ in range(20):
            P = rng.uniform(0.01, 1.0, size=(k, k))
            P /= P.sum(axis=1, keepdims=True)
            pi = stationary_distribution(P)
            assert pi.sum() == pytest.approx(1.0, abs=1e-12)
            assert np.all(pi >= -1e-15)
            assert np.allclose(pi @ P, pi, atol=1e-12)


def test_stationary_distribution_rejects_invalid_matrices():
    with pytest.raises(ValueError, match="rows must sum to 1"):
        stationary_distribution(np.array([[0.5, 0.4], [0.2, 0.8]]))
    with pytest.raises(ValueError, match="square"):
        stationary_distribution(np.array([[0.5, 0.5, 0.0], [0.2, 0.8, 0.0]]))
    with pytest.raises(ValueError, match="NaN or Inf"):
        stationary_distribution(np.array([[np.nan, 0.0], [0.2, 0.8]]))
    with pytest.raises(ValueError, match="non-negative"):
        stationary_distribution(np.array([[1.5, -0.5], [0.2, 0.8]]))


def test_expected_durations_rejects_bad_shapes():
    with pytest.raises(ValueError, match="square"):
        expected_durations(np.array([[0.5, 0.5, 0.0]]))
    with pytest.raises(ValueError, match="NaN or Inf"):
        expected_durations(np.array([[np.inf, 0.0], [0.2, 0.8]]))


def test_fitted_transition_matrix_is_row_stochastic_and_strictly_positive():
    """Property over several seeds: EM must never emit a row that fails to
    normalise or a hard zero (the 1e-12 clip keeps Viterbi logs finite)."""
    rng = np.random.default_rng(13)
    for seed in range(4):
        x = np.vstack([rng.normal(0, 0.005, (200, 1)),
                       rng.normal(0, 0.03, (200, 1))])
        fit = fit_hmm(x, 2, seed=seed, n_init=2)
        assert np.allclose(fit.transmat.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(fit.transmat > 0.0)
        assert np.all(fit.transmat <= 1.0)
        # Viterbi must therefore run without -inf propagation.
        path = viterbi(x, fit.startprob, fit.transmat, fit.means, fit.covariances)
        assert set(np.unique(path)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Probability invariants
# ---------------------------------------------------------------------------

def test_filtered_and_smoothed_probabilities_are_valid_distributions():
    rng = np.random.default_rng(17)
    x = np.vstack([rng.normal(0, 0.005, (150, 1)), rng.normal(0, 0.03, (150, 1))])
    fit = fit_hmm(x, 2, seed=0, n_init=2)
    filt, ll = forward_filter(x, fit.startprob, fit.transmat, fit.means,
                              fit.covariances)
    gamma, xi_sum, ll2 = forward_backward(x, fit.startprob, fit.transmat,
                                          fit.means, fit.covariances)
    for probs in (filt, gamma):
        assert np.all(probs >= -1e-12) and np.all(probs <= 1.0 + 1e-12)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-10)
    # Both passes must report the same likelihood.
    assert ll == pytest.approx(ll2, rel=1e-12)
    # Expected transition counts total T-1.
    assert xi_sum.sum() == pytest.approx(len(x) - 1, rel=1e-8)


def test_filtered_probabilities_are_causal():
    """Row t of the filter must not change when the future is altered — the
    property that makes filtered probs tradeable and smoothed probs not."""
    rng = np.random.default_rng(21)
    # Deliberately OVERLAPPING regimes: with well-separated states the
    # posteriors saturate at 0/1 and smoothing has nothing left to add, which
    # would make the contrast below vacuous.
    x = np.vstack([rng.normal(0.0, 0.010, (100, 1)),
                   rng.normal(0.0, 0.016, (100, 1))])
    fit = fit_hmm(x, 2, seed=0, n_init=2)
    args = (fit.startprob, fit.transmat, fit.means, fit.covariances)
    full, _ = forward_filter(x, *args)
    truncated, _ = forward_filter(x[:120], *args)
    # Causality: every filtered row up to the truncation point is bit-stable.
    assert np.allclose(full[:120], truncated, atol=1e-12)
    filter_drift = float(np.abs(full[:120] - truncated).max())
    # The smoother, by contrast, does depend on the future.
    g_full, _, _ = forward_backward(x, *args)
    g_trunc, _, _ = forward_backward(x[:120], *args)
    smoother_drift = float(np.abs(g_full[:120] - g_trunc).max())
    assert smoother_drift > 0.0
    # Orders of magnitude apart: the filter is future-independent to machine
    # precision, the smoother is not.
    assert smoother_drift > 1e4 * max(filter_drift, np.finfo(float).eps)
    # And the last smoothed row of a series always equals its filtered row
    # (beta_T = 1), which is the boundary identity behind the distinction.
    assert np.allclose(g_trunc[-1], truncated[-1], atol=1e-12)


def test_em_log_likelihood_is_monotone_non_decreasing():
    rng = np.random.default_rng(23)
    x = np.vstack([rng.normal(0, 0.005, (200, 1)), rng.normal(0, 0.03, (200, 1))])
    fit = fit_hmm(x, 2, seed=0, n_init=1, max_iter=60)
    hist = np.array(fit.log_likelihood_history)
    assert len(hist) > 2
    assert np.all(np.diff(hist) >= -1e-9)


# ---------------------------------------------------------------------------
# NaN / Inf rejection
# ---------------------------------------------------------------------------

def test_hmm_rejects_non_finite_observations():
    """Regression: an isnan-only check let Inf through, and every sufficient
    statistic came back NaN with only RuntimeWarnings to show for it."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 2)) * 0.01
    x_inf = x.copy()
    x_inf[3, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        fit_hmm(x_inf, 2, seed=0)
    x_nan = x.copy()
    x_nan[7, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        fit_hmm(x_nan, 2, seed=0)


def test_gmm_rejects_non_finite_observations():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 2)) * 0.01
    x_inf = x.copy()
    x_inf[5, 1] = -np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        fit_gmm(x_inf, 2, seed=0)


def test_min_duration_filter_on_degenerate_paths():
    """Constant path and alternating path: the filter must be idempotent on
    the former and fully smooth the latter."""
    const = np.zeros(50, dtype=int)
    assert np.array_equal(min_duration_filter(const, 5), const)
    alt = np.arange(50) % 2
    smoothed = min_duration_filter(alt, 5)
    assert flip_flop_rate(smoothed) < flip_flop_rate(alt)
    with pytest.raises(ValueError, match="min_duration"):
        min_duration_filter(const, 0)
