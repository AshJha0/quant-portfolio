"""GMM-from-scratch tests: EM monotonicity, recovery, BIC, sklearn cross-check."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as SkGMM

from eq_regime.gmm import aic, bic, fit_gmm, gmm_log_likelihood, match_permutation, select_k_bic


@pytest.fixture(scope="module")
def mix2d():
    """Well-separated 2-component 2-D mixture with known parameters."""
    rng = np.random.default_rng(0)
    n1, n2 = 400, 600
    m1, m2 = np.array([-3.0, 0.0]), np.array([3.0, 2.0])
    c1 = np.array([[1.0, 0.3], [0.3, 0.5]])
    c2 = np.array([[0.6, -0.2], [-0.2, 1.2]])
    x = np.vstack(
        [rng.multivariate_normal(m1, c1, n1), rng.multivariate_normal(m2, c2, n2)]
    )
    rng.shuffle(x)
    return x, np.array([m1, m2]), np.array([c1, c2]), np.array([0.4, 0.6])


@pytest.fixture(scope="module")
def mix3_1d():
    """Clear 3-component 1-D mixture."""
    rng = np.random.default_rng(1)
    x = np.concatenate(
        [
            rng.normal(-5.0, 0.6, 300),
            rng.normal(0.0, 0.6, 400),
            rng.normal(5.0, 0.6, 300),
        ]
    )
    rng.shuffle(x)
    return x


def test_loglik_monotone_every_iteration(mix2d):
    x, *_ = mix2d
    fit = fit_gmm(x, 2, seed=0, n_init=1, tol=1e-10, max_iter=200)
    hist = np.array(fit.log_likelihood_history)
    assert len(hist) > 3
    assert np.all(np.diff(hist) >= -1e-8), "EM log-likelihood decreased"


def test_parameter_recovery_permutation_matched(mix2d):
    x, true_means, true_covs, true_w = mix2d
    fit = fit_gmm(x, 2, seed=0)
    perm = match_permutation(true_means, fit.means)
    for i in range(2):
        np.testing.assert_allclose(fit.means[perm[i]], true_means[i], atol=0.15)
        np.testing.assert_allclose(fit.covariances[perm[i]], true_covs[i], atol=0.25)
        assert fit.weights[perm[i]] == pytest.approx(true_w[i], abs=0.05)


def test_bic_selects_true_k_2(mix2d):
    x, *_ = mix2d
    best_k, scores = select_k_bic(x, k_range=(1, 2, 3, 4), seed=0, n_init=2)
    assert best_k == 2
    assert scores[2] < scores[1] and scores[2] < scores[4]


def test_bic_selects_true_k_3(mix3_1d):
    best_k, scores = select_k_bic(mix3_1d, k_range=(1, 2, 3, 4, 5), seed=0, n_init=2)
    assert best_k == 3


def test_sklearn_loglik_crosscheck(mix2d):
    """Converged per-observation log-likelihood agrees with sklearn."""
    x, *_ = mix2d
    ours = fit_gmm(x, 2, seed=0, tol=1e-9)
    sk = SkGMM(2, covariance_type="full", n_init=3, random_state=0, tol=1e-9,
               reg_covar=1e-6, max_iter=500).fit(x)
    ll_sk = sk.score(x)              # mean per-sample
    ll_us = ours.log_likelihood / len(x)
    assert ll_us == pytest.approx(ll_sk, abs=1e-4)


def test_sklearn_same_params_same_loglik(mix2d):
    """Scoring OUR parameters with sklearn's machinery matches exactly."""
    x, *_ = mix2d
    ours = fit_gmm(x, 2, seed=0)
    manual = gmm_log_likelihood(x, ours.weights, ours.means, ours.covariances)
    # independent evaluation with scipy
    dens = sum(
        w * multivariate_normal(m, c).pdf(x)
        for w, m, c in zip(ours.weights, ours.means, ours.covariances)
    )
    assert manual == pytest.approx(np.log(dens).sum(), abs=1e-8)


def test_responsibilities_and_weights_normalised(mix2d):
    x, *_ = mix2d
    fit = fit_gmm(x, 2, seed=0)
    assert fit.weights.sum() == pytest.approx(1.0, abs=1e-12)
    resp = np.exp(fit.log_responsibilities(x))
    np.testing.assert_allclose(resp.sum(axis=1), 1.0, atol=1e-12)


def test_degenerate_singleton_cluster_regularized():
    """One far outlier: reg_covar must keep the singleton component PD."""
    rng = np.random.default_rng(2)
    x = np.vstack([rng.normal(0, 1, (200, 2)), np.array([[50.0, 50.0]])])
    fit = fit_gmm(x, 2, seed=0, n_init=2, reg_covar=1e-6)
    assert np.isfinite(fit.log_likelihood)
    for c in fit.covariances:
        np.linalg.cholesky(c)  # PD after regularization


def test_k1_matches_gaussian_mle():
    rng = np.random.default_rng(3)
    x = rng.multivariate_normal([1.0, -1.0], [[2.0, 0.5], [0.5, 1.0]], 500)
    fit = fit_gmm(x, 1, seed=0, reg_covar=0.0)
    mean = x.mean(0)
    cov = np.cov(x.T, ddof=0)
    ll = multivariate_normal(mean, cov).logpdf(x).sum()
    assert fit.log_likelihood == pytest.approx(ll, rel=1e-8)
    np.testing.assert_allclose(fit.means[0], mean, atol=1e-8)


def test_aic_bic_formulas(mix2d):
    x, *_ = mix2d
    fit = fit_gmm(x, 2, seed=0)
    d = 2
    expected_params = (2 - 1) + 2 * d + 2 * d * (d + 1) // 2
    assert fit.n_params == expected_params
    assert bic(fit, len(x)) == pytest.approx(
        -2 * fit.log_likelihood + expected_params * np.log(len(x))
    )
    assert aic(fit, len(x)) == pytest.approx(-2 * fit.log_likelihood + 2 * expected_params)


def test_match_permutation():
    true = np.array([[0.0, 0.0], [5.0, 5.0], [-5.0, 5.0]])
    est = np.array([[5.1, 4.9], [-4.8, 5.2], [0.1, -0.1]])
    assert match_permutation(true, est) == (2, 0, 1)


def test_validation_errors():
    x = np.random.default_rng(0).standard_normal((50, 2))
    with pytest.raises(ValueError, match="n_components"):
        fit_gmm(x, 0)
    with pytest.raises(ValueError, match="too few"):
        fit_gmm(x[:3], 2)
    xn = x.copy()
    xn[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_gmm(xn, 2)
    with pytest.raises(ValueError, match="n_init"):
        fit_gmm(x, 2, n_init=0)
