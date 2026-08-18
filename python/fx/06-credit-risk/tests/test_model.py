"""IRLS logistic: recovery, sklearn cross-check, separation, scorecard, ratings."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fx_credit.data.synthetic import generate_logistic_data
from fx_credit.model import (
    RATING_BANDS,
    RATING_ORDER,
    assign_rating,
    fit_logistic_irls,
    pd_from_score,
    predict_pd,
    rating_midpoint_pd,
    score_from_pd,
)

TRUE_BETA = np.array([0.8, -0.5, 0.3])
TRUE_INTERCEPT = -2.0


@pytest.fixture(scope="module")
def fitted():
    X, y = generate_logistic_data(TRUE_BETA, TRUE_INTERCEPT, n=40_000, seed=3)
    return X, y, fit_logistic_irls(X, y)


def test_irls_recovers_true_coefficients(fitted):
    _, _, fit = fitted
    est = np.r_[fit.intercept, fit.coef]
    true = np.r_[TRUE_INTERCEPT, TRUE_BETA]
    # within 4 standard errors and absolutely close
    assert np.all(np.abs(est - true) < 4.0 * fit.se)
    assert np.max(np.abs(est - true)) < 0.08


def test_irls_converged_flag(fitted):
    _, _, fit = fitted
    assert fit.converged and fit.n_iter < 30


def test_irls_matches_sklearn_1e6(fitted):
    X, y, fit = fitted
    sk = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000).fit(X, y)
    assert np.max(np.abs(fit.coef - sk.coef_.ravel())) < 1e-6
    assert abs(fit.intercept - sk.intercept_[0]) < 1e-6


def test_irls_standard_errors_positive_and_sane(fitted):
    _, _, fit = fitted
    assert np.all(fit.se > 0)
    assert np.all(fit.se < 0.05)  # n=40k, well-conditioned
    assert np.allclose(fit.se, np.sqrt(np.diag(fit.cov)))


def test_intercept_only_analytic():
    """Intercept-only MLE: logit(ybar), SE = 1/sqrt(n p (1-p)) — exact identities."""
    y = np.r_[np.ones(30), np.zeros(70)]
    X = np.empty((100, 0))
    fit = fit_logistic_irls(X, y)
    p = 0.3
    assert fit.intercept == pytest.approx(np.log(p / (1 - p)), abs=1e-8)
    assert fit.se[0] == pytest.approx(1.0 / np.sqrt(100 * p * (1 - p)), rel=1e-6)


def test_predict_pd_range_and_monotonicity(fitted):
    X, _, fit = fitted
    p = predict_pd(fit, X)
    assert np.all((p > 0) & (p < 1))
    # increasing the positive-coefficient feature raises PD
    x0 = np.zeros((1, 3))
    x1 = x0.copy()
    x1[0, 0] = 1.0
    assert predict_pd(fit, x1)[0] > predict_pd(fit, x0)[0]


def test_separation_raises():
    x = np.r_[np.linspace(-2, -0.1, 50), np.linspace(0.1, 2, 50)].reshape(-1, 1)
    y = np.r_[np.zeros(50), np.ones(50)]
    with pytest.raises(ValueError, match="separation"):
        fit_logistic_irls(x, y)


def test_separation_handled_with_ridge():
    x = np.r_[np.linspace(-2, -0.1, 50), np.linspace(0.1, 2, 50)].reshape(-1, 1)
    y = np.r_[np.zeros(50), np.ones(50)]
    fit = fit_logistic_irls(x, y, ridge=1e-2)
    assert np.isfinite(fit.coef[0]) and fit.coef[0] > 0


def test_collinear_features_raise():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 1))
    X = np.hstack([x, x])  # exactly collinear
    y = (rng.random(200) < 0.3).astype(float)
    with pytest.raises(ValueError):
        fit_logistic_irls(X, y)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="binary"):
        fit_logistic_irls(np.zeros((3, 1)), np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="matching"):
        fit_logistic_irls(np.zeros((3, 1)), np.array([0.0, 1.0]))


# --- scorecard scaling -----------------------------------------------------

def test_score_at_base_odds_is_base_score():
    pd_at_base = 1.0 / (1.0 + 50.0)  # odds_good = 50
    assert score_from_pd(pd_at_base) == pytest.approx(600.0, abs=1e-10)


def test_pdo_identity_exact():
    """Doubling good odds adds exactly PDO points."""
    p1 = 1.0 / (1.0 + 50.0)
    p2 = 1.0 / (1.0 + 100.0)
    assert score_from_pd(p2) - score_from_pd(p1) == pytest.approx(20.0, abs=1e-10)


def test_score_pd_round_trip():
    p = np.array([1e-4, 1e-3, 0.01, 0.05, 0.2, 0.5, 0.9])
    assert np.max(np.abs(pd_from_score(score_from_pd(p)) - p)) < 1e-12


def test_score_monotone_decreasing_in_pd():
    p = np.linspace(1e-4, 0.5, 200)
    s = score_from_pd(p)
    assert np.all(np.diff(s) < 0)


# --- rating bands ----------------------------------------------------------

def test_rating_band_midpoints_monotone():
    mids = [RATING_BANDS[r][1] for r in RATING_ORDER]
    assert np.all(np.diff(mids) > 0)


def test_rating_band_uppers_monotone_and_cover():
    uppers = [RATING_BANDS[r][0] for r in RATING_ORDER]
    assert np.all(np.diff(uppers) > 0)
    assert uppers[-1] > 1.0  # covers all of [0, 1]


def test_midpoints_inside_bands():
    prev_upper = 0.0
    for r in RATING_ORDER:
        upper, mid = RATING_BANDS[r]
        assert prev_upper <= mid < upper
        prev_upper = upper


def test_assign_rating_monotone_in_pd():
    pds = [1e-5, 5e-4, 2e-3, 5e-3, 0.02, 0.05, 0.15, 0.5]
    idx = [RATING_ORDER.index(assign_rating(p)) for p in pds]
    assert idx == sorted(idx)
    assert assign_rating(1e-5) == "AAA" and assign_rating(0.5) == "C"


def test_assign_rating_band_edges():
    assert assign_rating(0.0) == "AAA"
    assert assign_rating(1.0) == "C"
    assert assign_rating(0.0299) == "BB"
    assert assign_rating(0.0300) == "B"


def test_assign_rating_invalid_raises():
    with pytest.raises(ValueError, match="pd_1y"):
        assign_rating(1.5)


def test_rating_midpoint_lookup():
    assert rating_midpoint_pd("BB") == 0.02
    with pytest.raises(ValueError, match="unknown rating"):
        rating_midpoint_pd("ZZ")


def test_band_mean_pd_monotone_on_grid():
    """Average model PD within each assigned band is monotone across bands."""
    pds = np.geomspace(1e-5, 0.6, 500)
    letters = [assign_rating(p) for p in pds]
    means = {}
    for letter, p in zip(letters, pds):
        means.setdefault(letter, []).append(p)
    ordered = [np.mean(means[r]) for r in RATING_ORDER if r in means]
    assert np.all(np.diff(ordered) > 0)
