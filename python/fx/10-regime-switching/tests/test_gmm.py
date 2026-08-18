"""GMM tests: EM monotonicity, recovery, BIC, sklearn cross-check."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from fx_regime import fit_gmm, gaussian_logpdf, generate_null_gbm_panel, select_k_bic
from fx_regime.hmm import match_states


def _two_clusters(n=400, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n // 2, 2)) * 0.5 + np.array([-3.0, 0.0])
    b = rng.standard_normal((n // 2, 2)) * 0.5 + np.array([3.0, 1.0])
    return np.vstack([a, b])


def test_gaussian_logpdf_matches_scipy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    mean = np.array([0.1, -0.2, 0.3])
    A = rng.standard_normal((3, 3))
    cov = A @ A.T + np.eye(3)
    ours = gaussian_logpdf(X, mean, cov)
    ref = multivariate_normal(mean, cov).logpdf(X)
    assert np.allclose(ours, ref, atol=1e-10)


def test_responsibilities_sum_to_one():
    model = fit_gmm(_two_clusters(), 2, seed=0)
    resp = model.predict_proba(_two_clusters())
    assert np.allclose(resp.sum(axis=1), 1.0, atol=1e-10)
    assert (resp >= 0).all()


def test_loglik_monotone_non_decreasing():
    model = fit_gmm(_two_clusters(), 2, seed=0, n_init=1)
    path = np.array(model.log_likelihood_path)
    assert (np.diff(path) >= -1e-8).all()


def test_parameter_recovery_two_clusters():
    model = fit_gmm(_two_clusters(2000, seed=1), 2, seed=0)
    true_means = np.array([[-3.0, 0.0], [3.0, 1.0]])
    perm = match_states(true_means, model.means)
    aligned = model.means[np.argsort(perm)]
    assert np.abs(aligned - true_means).max() < 0.1
    assert np.abs(model.weights - 0.5).max() < 0.05


def test_bic_selects_true_k():
    X = _two_clusters(1000, seed=2)
    best_k, bics = select_k_bic(X, k_max=3, seed=0)
    assert best_k == 2
    assert bics[2] < bics[1] and bics[2] < bics[3]


def test_bic_prefers_one_state_on_null_data():
    """Null-GBM guard: no regimes -> BIC must not invent them."""
    panel = generate_null_gbm_panel(700, seed=0)
    X = panel.returns[["AUD", "JPY", "EUR"]].to_numpy()
    best_k, _ = select_k_bic(X, k_max=3, seed=0)
    assert best_k == 1


def test_aic_bic_formulas_hand_check():
    X = _two_clusters(200, seed=3)
    model = fit_gmm(X, 2, seed=0)
    ll = float(model.score_samples(X).sum())
    p = 2
    n_params = (2 - 1) + 2 * p + 2 * p * (p + 1) // 2
    assert np.isclose(model.bic(X), -2 * ll + n_params * np.log(len(X)), atol=1e-9)
    assert np.isclose(model.aic(X), -2 * ll + 2 * n_params, atol=1e-9)


def test_sklearn_cross_check_fixed_params():
    """score_samples must match sklearn exactly for identical parameters."""
    X = _two_clusters(300, seed=4)
    model = fit_gmm(X, 2, seed=0)
    sk = GaussianMixture(n_components=2, covariance_type="full")
    sk.weights_ = model.weights
    sk.means_ = model.means
    sk.covariances_ = model.covs
    prec_chol = np.empty_like(model.covs)
    for j in range(2):
        L = np.linalg.cholesky(model.covs[j])
        prec_chol[j] = np.linalg.solve(L, np.eye(2)).T
    sk.precisions_cholesky_ = prec_chol
    assert np.allclose(sk.score_samples(X), model.score_samples(X), atol=1e-8)


def test_sklearn_cross_check_fitted_loglik():
    X = _two_clusters(800, seed=5)
    ours = fit_gmm(X, 2, seed=0)
    sk = GaussianMixture(n_components=2, n_init=3, random_state=0).fit(X)
    ll_ours = ours.score_samples(X).mean()
    ll_sk = sk.score(X)
    assert abs(ll_ours - ll_sk) < 0.01


def test_seed_determinism():
    X = _two_clusters(300, seed=6)
    a = fit_gmm(X, 2, seed=1)
    b = fit_gmm(X, 2, seed=1)
    assert np.allclose(a.means, b.means)
    assert a.log_likelihood == b.log_likelihood


def test_k_equals_one():
    X = _two_clusters(200, seed=7)
    model = fit_gmm(X, 1, seed=0)
    assert np.allclose(model.means[0], X.mean(axis=0), atol=1e-8)
    assert np.allclose(model.weights, [1.0])
    # covariance ~ MLE sample covariance (ddof=0) up to reg_covar
    assert np.allclose(
        model.covs[0], np.cov(X.T, ddof=0) + 1e-6 * np.eye(2), atol=1e-6
    )


def test_degenerate_data_regularised():
    """Near-singular cluster (pegged dimension) must not crash EM."""
    rng = np.random.default_rng(8)
    X = np.hstack([rng.standard_normal((200, 1)), np.zeros((200, 1))])
    model = fit_gmm(X, 2, seed=0, reg_covar=1e-6)
    assert np.isfinite(model.log_likelihood)


def test_invalid_inputs_raise():
    X = _two_clusters(50)
    with pytest.raises(ValueError):
        fit_gmm(X, 0)
    with pytest.raises(ValueError):
        fit_gmm(X[:1], 2)
    with pytest.raises(ValueError):
        select_k_bic(X, k_max=0)
