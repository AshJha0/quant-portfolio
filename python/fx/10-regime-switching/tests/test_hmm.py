"""HMM tests: forward-backward identities, Baum-Welch, Viterbi, hmmlearn."""

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from fx_regime import (
    HMMModel,
    expected_durations,
    filtered_probabilities,
    fit_hmm,
    hmm_bic,
    log_backward,
    log_forward,
    match_states,
    smoothed_probabilities,
    stationary_distribution,
    viterbi,
    TRANSITION_2,
)
from fx_regime.gmm import gaussian_logpdf


def _known_model(sep=4.0):
    """Well-separated 2-state, 2-D Gaussian HMM."""
    return HMMModel(
        startprob=np.array([0.6, 0.4]),
        transmat=np.array([[0.95, 0.05], [0.10, 0.90]]),
        means=np.array([[-sep / 2, 0.0], [sep / 2, 1.0]]),
        covs=np.array([np.eye(2) * 0.25, np.eye(2) * 0.25]),
    )


def _sample(model, T, seed=0):
    rng = np.random.default_rng(seed)
    k = model.k
    cdf = np.cumsum(model.transmat, axis=1)
    s = int(rng.choice(k, p=model.startprob))
    states = np.empty(T, dtype=int)
    X = np.empty((T, model.means.shape[1]))
    chols = [np.linalg.cholesky(c) for c in model.covs]
    for t in range(T):
        states[t] = s
        X[t] = model.means[s] + chols[s] @ rng.standard_normal(2)
        s = int(np.searchsorted(cdf[s], rng.random(), side="right"))
        s = min(s, k - 1)
    return X, states


def test_filtered_probabilities_rows_sum_to_one():
    model = _known_model()
    X, _ = _sample(model, 300)
    filt = filtered_probabilities(model, X)
    assert np.allclose(filt.sum(axis=1), 1.0, atol=1e-10)
    assert (filt >= 0).all()


def test_smoothed_posteriors_rows_sum_to_one():
    model = _known_model()
    X, _ = _sample(model, 300)
    gamma = smoothed_probabilities(model, X)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-10)


def test_forward_backward_loglik_consistency():
    """Same likelihood from the forward and the backward recursion."""
    model = _known_model()
    X, _ = _sample(model, 200)
    _, ll_fwd = log_forward(model, X)
    log_beta = log_backward(model, X)
    logb = model.emission_logprob(X)
    with np.errstate(divide="ignore"):
        ll_bwd = logsumexp(np.log(model.startprob) + logb[0] + log_beta[0])
    assert np.isclose(ll_fwd, ll_bwd, atol=1e-8)


def test_baum_welch_monotone():
    model = _known_model()
    X, _ = _sample(model, 500)
    fit = fit_hmm(X, 2, seed=0, n_init=1, max_iter=50)
    path = np.array(fit.log_likelihood_path)
    assert (np.diff(path) >= -1e-6).all()


def test_transition_and_state_recovery_permutation_matched():
    model = _known_model()
    X, _ = _sample(model, 3000, seed=1)
    fit = fit_hmm(X, 2, seed=0, n_init=2)
    perm = match_states(model.means, fit.means)
    inv = np.argsort(perm)
    means_aligned = fit.means[inv]
    A_aligned = fit.transmat[np.ix_(inv, inv)]
    assert np.abs(means_aligned - model.means).max() < 0.1
    assert np.abs(A_aligned - model.transmat).max() < 0.03


def test_viterbi_accuracy_separated_case():
    model = _known_model(sep=4.0)
    X, states = _sample(model, 1000, seed=2)
    fit = fit_hmm(X, 2, seed=0, n_init=2)
    path = viterbi(fit, X)
    perm = match_states(model.means, fit.means)
    acc = (np.array([perm[s] for s in path]) == states).mean()
    assert acc > 0.90


def test_viterbi_matches_brute_force():
    model = _known_model(sep=1.0)
    X, _ = _sample(model, 7, seed=3)
    logb = model.emission_logprob(X)
    with np.errstate(divide="ignore"):
        log_pi, log_A = np.log(model.startprob), np.log(model.transmat)
    best, best_lp = None, -np.inf
    for path in itertools.product(range(2), repeat=7):
        lp = log_pi[path[0]] + logb[0, path[0]]
        for t in range(1, 7):
            lp += log_A[path[t - 1], path[t]] + logb[t, path[t]]
        if lp > best_lp:
            best, best_lp = path, lp
    assert tuple(viterbi(model, X)) == best


def test_expected_durations_formula():
    A = np.array([[0.9, 0.1], [0.2, 0.8]])
    d = expected_durations(A)
    assert np.allclose(d, [10.0, 5.0])


def test_stationary_distribution_identity():
    for P in (TRANSITION_2, np.array([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])):
        pi = stationary_distribution(P)
        assert np.allclose(pi @ P, pi, atol=1e-10)
        assert np.isclose(pi.sum(), 1.0)
        assert (pi >= 0).all()
    # hand check for TRANSITION_2: balance pi1*0.01 = pi2*0.05
    pi = stationary_distribution(TRANSITION_2)
    assert np.allclose(pi, [5.0 / 6.0, 1.0 / 6.0], atol=1e-10)


def test_filtered_differs_from_smoothed():
    model = _known_model(sep=1.5)
    X, _ = _sample(model, 300, seed=4)
    filt = filtered_probabilities(model, X)
    smth = smoothed_probabilities(model, X)
    assert not np.allclose(filt, smth, atol=1e-3)
    # ... but they agree at the last time step
    assert np.allclose(filt[-1], smth[-1], atol=1e-10)


def test_filtered_is_causal_mutation():
    """CRITICAL: filtered probs at t unchanged by future perturbation."""
    model = _known_model()
    X, _ = _sample(model, 400, seed=5)
    filt1 = filtered_probabilities(model, X)
    X2 = X.copy()
    X2[250:] += 10.0
    filt2 = filtered_probabilities(model, X2)
    assert np.array_equal(filt1[:250], filt2[:250])
    assert not np.allclose(filt1[250:], filt2[250:])


def test_hmmlearn_cross_check_fixed_model():
    """Likelihood, posteriors and Viterbi vs hmmlearn, identical params."""
    from hmmlearn.hmm import GaussianHMM

    model = _known_model()
    X, _ = _sample(model, 500, seed=6)
    ref = GaussianHMM(n_components=2, covariance_type="full", init_params="")
    ref.startprob_ = model.startprob
    ref.transmat_ = model.transmat
    ref.means_ = model.means
    ref.covars_ = model.covs
    _, ll = log_forward(model, X)
    assert np.isclose(ref.score(X), ll, atol=1e-6)
    assert np.allclose(
        ref.predict_proba(X), smoothed_probabilities(model, X), atol=1e-8
    )
    assert np.array_equal(ref.predict(X), viterbi(model, X))


def test_hmmlearn_cross_check_fitted():
    """Both EMs find the same optimum on well-separated data."""
    from hmmlearn.hmm import GaussianHMM

    model = _known_model()
    X, _ = _sample(model, 1500, seed=7)
    ours = fit_hmm(X, 2, seed=0, n_init=2)
    ref = GaussianHMM(
        n_components=2, covariance_type="full", n_iter=200, random_state=0
    ).fit(X)
    perm = match_states(ref.means_, ours.means)
    inv = np.argsort(perm)
    assert np.abs(ours.means[inv] - ref.means_).max() < 0.05
    assert np.abs(ours.transmat[np.ix_(inv, inv)] - ref.transmat_).max() < 0.02
    _, ll_ours = log_forward(ours, X)
    assert abs(ll_ours - ref.score(X)) / abs(ref.score(X)) < 1e-3


def test_k_equals_one():
    X = np.random.default_rng(8).standard_normal((100, 2))
    fit = fit_hmm(X, 1, seed=0)
    ll_manual = gaussian_logpdf(X, fit.means[0], fit.covs[0]).sum()
    assert np.isclose(fit.log_likelihood, ll_manual, atol=1e-8)
    assert fit.transmat.shape == (1, 1)


def test_hmm_bic_penalises_extra_states():
    model = _known_model()
    X, _ = _sample(model, 800, seed=9)
    m2 = fit_hmm(X, 2, seed=0, n_init=2)
    m3 = fit_hmm(X, 3, seed=0, n_init=2)
    assert hmm_bic(m2, X) < hmm_bic(m3, X)


def test_seed_determinism():
    model = _known_model()
    X, _ = _sample(model, 400, seed=10)
    a = fit_hmm(X, 2, seed=3, n_init=2)
    b = fit_hmm(X, 2, seed=3, n_init=2)
    assert np.allclose(a.transmat, b.transmat)
    assert a.log_likelihood == b.log_likelihood


def test_short_series_and_bad_inputs_raise():
    X = np.zeros((5, 2))
    with pytest.raises(ValueError):
        fit_hmm(X, 2)
    with pytest.raises(ValueError):
        fit_hmm(np.zeros((100, 2)), 0)
    bad = _known_model()
    bad.transmat = np.array([[0.5, 0.4], [0.1, 0.9]])
    with pytest.raises(ValueError):
        log_forward(bad, np.zeros((10, 2)))
    with pytest.raises(ValueError):
        stationary_distribution(np.array([[0.5, 0.4], [0.5, 0.5]]))
