"""HMM-from-scratch tests: forward-backward, Baum-Welch, Viterbi,
stationary distribution, durations, and the hmmlearn cross-check."""

from __future__ import annotations

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM as HLGaussianHMM

from eq_regime.gmm import match_permutation
from eq_regime.hmm import (
    expected_durations,
    fit_hmm,
    forward_backward,
    forward_filter,
    stationary_distribution,
    viterbi,
)


def _fit_wellsep(wellsep_2state):
    x, *_ = wellsep_2state
    return fit_hmm(x, 2, seed=0, n_init=2, max_iter=150, tol=1e-8)


@pytest.fixture(scope="module")
def wellsep_fit(wellsep_2state):
    return _fit_wellsep(wellsep_2state)


def test_filtered_probabilities_sum_to_one(wellsep_2state, wellsep_fit):
    x, *_ = wellsep_2state
    probs, ll = wellsep_fit.filter(x)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    assert np.isfinite(ll)


def test_smoothed_posteriors_sum_to_one(wellsep_2state, wellsep_fit):
    x, *_ = wellsep_2state
    gamma = wellsep_fit.smooth(x)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-12)


def test_baum_welch_loglik_monotone(wellsep_2state):
    x, *_ = wellsep_2state
    fit = fit_hmm(x[:1500], 2, seed=0, n_init=1, max_iter=100, tol=1e-12)
    hist = np.array(fit.log_likelihood_history)
    assert len(hist) > 5
    assert np.all(np.diff(hist) >= -1e-7), "Baum-Welch log-likelihood decreased"


def test_transition_rows_sum_to_one(wellsep_fit):
    np.testing.assert_allclose(wellsep_fit.transmat.sum(axis=1), 1.0, atol=1e-12)
    assert (wellsep_fit.transmat >= 0).all()


def test_recovery_of_known_parameters(wellsep_2state, wellsep_fit):
    """Transition matrix and state params recovered, permutation-matched."""
    x, states, transition, means, sigmas = wellsep_2state
    perm = match_permutation(means[:, None], wellsep_fit.means)
    est_means = np.array([wellsep_fit.means[perm[i], 0] for i in range(2)])
    est_sig = np.array([np.sqrt(wellsep_fit.covariances[perm[i], 0, 0]) for i in range(2)])
    est_trans = wellsep_fit.transmat[np.ix_(perm, perm)]
    np.testing.assert_allclose(est_means, means, atol=0.10)
    np.testing.assert_allclose(est_sig, sigmas, atol=0.08)
    np.testing.assert_allclose(np.diag(est_trans), np.diag(transition), atol=0.02)


def test_viterbi_accuracy_above_90pct(wellsep_2state, wellsep_fit):
    x, states, _, means, _ = wellsep_2state
    perm = match_permutation(means[:, None], wellsep_fit.means)
    path = wellsep_fit.viterbi(x)
    # relabel estimated path into true-state indexing
    inv = np.empty(2, dtype=int)
    for i in range(2):
        inv[perm[i]] = i
    acc = np.mean(inv[path] == states)
    assert acc > 0.90


def test_expected_durations_identity():
    p = np.array([[0.98, 0.02], [0.10, 0.90]])
    d = expected_durations(p)
    np.testing.assert_allclose(d, [1 / 0.02, 1 / 0.10], rtol=1e-12)


def test_stationary_distribution_identity(wellsep_fit):
    pi = stationary_distribution(wellsep_fit.transmat)
    np.testing.assert_allclose(pi @ wellsep_fit.transmat, pi, atol=1e-12)
    assert pi.sum() == pytest.approx(1.0, abs=1e-12)
    assert (pi >= 0).all()


def test_stationary_distribution_hand_checked():
    p = np.array([[0.9, 0.1], [0.3, 0.7]])
    pi = stationary_distribution(p)
    np.testing.assert_allclose(pi, [0.75, 0.25], atol=1e-12)


def test_hmmlearn_same_params_same_loglik(wellsep_2state, wellsep_fit):
    """Loading OUR parameters into hmmlearn reproduces our log-likelihood."""
    x, *_ = wellsep_2state
    hl = HLGaussianHMM(n_components=2, covariance_type="full", init_params="", params="")
    hl.startprob_ = wellsep_fit.startprob
    hl.transmat_ = wellsep_fit.transmat
    hl.means_ = wellsep_fit.means
    hl.covars_ = wellsep_fit.covariances
    ll_hl = hl.score(x[:, None] if x.ndim == 1 else x)
    ll_us = wellsep_fit.score(x)
    assert ll_us == pytest.approx(ll_hl, abs=1e-6)


def test_hmmlearn_independent_fit_crosscheck(wellsep_2state, wellsep_fit):
    """An independently EM-fitted hmmlearn model lands on the same optimum
    (per-observation log-likelihood within 1e-3)."""
    x, *_ = wellsep_2state
    hl = HLGaussianHMM(
        n_components=2, covariance_type="full", n_iter=200, tol=1e-8, random_state=0
    ).fit(x[:, None])
    ll_hl = hl.score(x[:, None]) / len(x)
    ll_us = wellsep_fit.score(x) / len(x)
    assert ll_us == pytest.approx(ll_hl, abs=1e-3)


def test_forward_filter_is_causal(wellsep_2state, wellsep_fit):
    """Mutating future observations leaves filtered probs at t unchanged."""
    x, *_ = wellsep_2state
    t = 500
    f = wellsep_fit
    base, _ = forward_filter(x[: t + 1], f.startprob, f.transmat, f.means, f.covariances)
    mutated = x.copy()
    mutated[t + 1 :] += 100.0
    full, _ = forward_filter(mutated, f.startprob, f.transmat, f.means, f.covariances)
    np.testing.assert_array_equal(base[t], full[t])


def test_smoothed_uses_future(wellsep_2state, wellsep_fit):
    """Contrast: the smoothed posterior at t DOES change with the future."""
    # Overlapping emissions (means +-0.5, sigma 1): the posterior at t is
    # genuinely uncertain, so future evidence moves it materially.
    rng = np.random.default_rng(9)
    startprob = np.array([0.5, 0.5])
    transmat = np.array([[0.95, 0.05], [0.05, 0.95]])
    means = np.array([[-0.5], [0.5]])
    covs = np.array([[[1.0]], [[1.0]]])
    x = rng.standard_normal(200)
    t = 100
    gamma_base, _, _ = forward_backward(x, startprob, transmat, means, covs)
    mutated = x.copy()
    mutated[t + 1 : t + 20] += 3.0
    gamma_mut, _, _ = forward_backward(mutated, startprob, transmat, means, covs)
    assert np.abs(gamma_base[t] - gamma_mut[t]).max() > 1e-2


def test_viterbi_hand_checked():
    """Two nearly deterministic states: the path follows the observations."""
    startprob = np.array([0.5, 0.5])
    transmat = np.array([[0.9, 0.1], [0.1, 0.9]])
    means = np.array([[0.0], [10.0]])
    covs = np.array([[[1.0]], [[1.0]]])
    x = np.array([0.1, -0.2, 0.0, 10.2, 9.8, 10.0, 0.05])[:, None]
    path = viterbi(x, startprob, transmat, means, covs)
    np.testing.assert_array_equal(path, [0, 0, 0, 1, 1, 1, 0])


def test_multivariate_fit_runs(features3):
    x = features3.to_numpy()[:400]
    fit = fit_hmm(x, 2, seed=0, n_init=1, max_iter=30)
    assert fit.means.shape == (2, x.shape[1])
    probs, _ = fit.filter(x)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)


def test_k1_fit(wellsep_2state):
    x, *_ = wellsep_2state
    fit = fit_hmm(x[:500], 1, seed=0, n_init=1)
    assert fit.transmat.shape == (1, 1)
    assert fit.transmat[0, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(expected_durations(fit.transmat), [np.inf])


def test_validation_errors():
    x = np.random.default_rng(0).standard_normal(100)
    with pytest.raises(ValueError, match="n_states"):
        fit_hmm(x, 0)
    with pytest.raises(ValueError, match="too short"):
        fit_hmm(x[:5], 2)
    with pytest.raises(ValueError, match="rows must sum"):
        stationary_distribution(np.array([[0.5, 0.4], [0.2, 0.8]]))
    xn = x.copy()
    xn[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_hmm(xn, 2)
