"""Tests for the from-scratch IRLS logistic regression and scorecard scaling:
coefficient recovery, sklearn cross-check, standard errors, separation,
PDO property, monotone score, stepwise selection."""

import numpy as np
import pandas as pd
import pytest

from eq_credit.model import (
    ScorecardScaling,
    SeparationWarning,
    crosscheck_sklearn,
    fit_logistic,
    scorecard_points_table,
    stepwise_select,
)


def _simulate(n: int, beta: np.ndarray, b0: float, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(beta)))
    p = 1.0 / (1.0 + np.exp(-(b0 + X @ beta)))
    y = (rng.uniform(size=n) < p).astype(float)
    return X, y


def test_irls_recovers_true_coefficients() -> None:
    beta = np.array([0.8, -0.5, 0.0])
    X, y = _simulate(200_000, beta, b0=-2.0, seed=3)
    fit = fit_logistic(X, y)
    assert fit.converged
    np.testing.assert_allclose(fit.coef, beta, atol=0.03)
    assert fit.intercept == pytest.approx(-2.0, abs=0.03)


def test_matches_sklearn_to_1e6() -> None:
    beta = np.array([0.7, -1.2])
    X, y = _simulate(5_000, beta, b0=-1.0, seed=5)
    fit = fit_logistic(X, y)
    assert crosscheck_sklearn(X, y, fit) < 1e-6


def test_standard_errors_match_analytic_fisher() -> None:
    # SEs must equal sqrt(diag((X'WX)^{-1})) at the MLE — computed
    # independently here (statsmodels-style analytic covariance).
    beta = np.array([0.5, -0.8])
    X, y = _simulate(20_000, beta, b0=-1.5, seed=8)
    fit = fit_logistic(X, y)
    Xd = np.column_stack([np.ones(len(y)), X])
    eta = Xd @ np.concatenate([[fit.intercept], fit.coef])
    p = 1.0 / (1.0 + np.exp(-eta))
    W = p * (1 - p)
    cov = np.linalg.inv((Xd * W[:, None]).T @ Xd)
    np.testing.assert_allclose(fit.se, np.sqrt(np.diag(cov)), rtol=1e-8)


def test_wald_z_and_pvalues_consistent() -> None:
    X, y = _simulate(20_000, np.array([0.6]), b0=-2.0, seed=9)
    fit = fit_logistic(X, y)
    assert fit.z[1] == pytest.approx(fit.coef[0] / fit.se[1])
    assert fit.p_values[1] < 1e-6  # strong true effect
    assert 0 <= fit.p_values[1] <= 1


def test_insignificant_noise_coefficient() -> None:
    X, y = _simulate(20_000, np.array([0.8, 0.0]), b0=-2.0, seed=10)
    fit = fit_logistic(X, y)
    assert fit.p_values[2] > 0.01  # noise feature not significant


def test_separation_detected_and_warned() -> None:
    # Perfectly separable data: y = 1 iff x > 0.
    x = np.linspace(-1, 1, 200)
    y = (x > 0).astype(float)
    with pytest.warns(SeparationWarning):
        fit = fit_logistic(x[:, None], y)
    assert np.isfinite(fit.coef).all()  # did not blow up to inf/nan


def test_separation_ridge_regularizes() -> None:
    x = np.linspace(-1, 1, 200)
    y = (x > 0).astype(float)
    fit = fit_logistic(x[:, None], y, ridge=1.0)
    assert fit.converged
    assert np.abs(fit.coef[0]) < 50


def test_zero_defaults_raises_informative() -> None:
    X = np.random.default_rng(0).standard_normal((100, 2))
    with pytest.raises(ValueError, match="zero defaults"):
        fit_logistic(X, np.zeros(100))


def test_all_defaults_raises() -> None:
    X = np.random.default_rng(0).standard_normal((100, 2))
    with pytest.raises(ValueError, match="no goods"):
        fit_logistic(X, np.ones(100))


def test_non_binary_target_raises() -> None:
    with pytest.raises(ValueError, match="binary"):
        fit_logistic(np.ones((3, 1)), np.array([0.0, 0.5, 1.0]))


def test_empty_design_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_logistic(np.empty((0, 2)), np.empty(0))


def test_predict_proba_in_unit_interval() -> None:
    X, y = _simulate(2_000, np.array([1.0]), b0=-1.0, seed=11)
    fit = fit_logistic(X, y)
    p = fit.predict_proba(X)
    assert ((p > 0) & (p < 1)).all()


def test_summary_table_shape_and_names() -> None:
    X = pd.DataFrame(
        np.random.default_rng(1).standard_normal((3_000, 2)), columns=["a", "b"]
    )
    y = (np.random.default_rng(2).uniform(size=3_000) < 0.3).astype(float)
    fit = fit_logistic(X, y)
    s = fit.summary()
    assert list(s["term"]) == ["intercept", "a", "b"]
    assert set(s.columns) == {"term", "coef", "se", "z", "p_value"}


# ------------------------------------------------------------------ scorecard
def test_pdo_property_holds_exactly() -> None:
    sc = ScorecardScaling(base_score=600, base_odds=50, pdo=20)
    # Base anchor: PD at odds 50:1 (good:bad) -> PD = 1/51.
    assert sc.score_from_pd(1 / 51) == pytest.approx(600.0, abs=1e-10)
    # Doubling the odds adds exactly PDO points: odds 100:1 -> PD = 1/101.
    assert sc.score_from_pd(1 / 101) == pytest.approx(620.0, abs=1e-10)
    # And again: odds 200:1.
    assert sc.score_from_pd(1 / 201) == pytest.approx(640.0, abs=1e-10)
    # Halving the odds subtracts PDO: odds 25:1.
    assert sc.score_from_pd(1 / 26) == pytest.approx(580.0, abs=1e-10)


def test_score_monotone_decreasing_in_pd() -> None:
    sc = ScorecardScaling()
    pds = np.linspace(0.001, 0.5, 100)
    scores = sc.score_from_pd(pds)
    assert np.all(np.diff(scores) < 0)


def test_score_pd_round_trip() -> None:
    sc = ScorecardScaling()
    pds = np.array([0.001, 0.02, 0.10, 0.40])
    np.testing.assert_allclose(sc.pd_from_score(sc.score_from_pd(pds)), pds, rtol=1e-12)


def test_pd_zero_one_clamped_finite() -> None:
    sc = ScorecardScaling()
    s0 = sc.score_from_pd(0.0)
    s1 = sc.score_from_pd(1.0)
    assert np.isfinite(s0) and np.isfinite(s1)
    assert s0 > s1  # PD=0 is the best possible score


def test_stepwise_selects_signal_drops_noise() -> None:
    rng = np.random.default_rng(42)
    n = 20_000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    p = 1 / (1 + np.exp(-(-2.5 + 0.9 * x1 - 0.7 * x2)))
    y = (rng.uniform(size=n) < p).astype(float)
    woe_df = pd.DataFrame({"x1": x1, "x2": x2, "noise": noise})
    ivs = {"x1": 0.4, "x2": 0.3, "noise": 0.001}  # noise below iv_min
    sel = stepwise_select(woe_df, y, ivs)
    assert set(sel) == {"x1", "x2"}


def test_stepwise_excludes_suspicious_iv() -> None:
    rng = np.random.default_rng(1)
    n = 5_000
    x1 = rng.standard_normal(n)
    p = 1 / (1 + np.exp(-(-2.0 + x1)))
    y = (rng.uniform(size=n) < p).astype(float)
    leaky = y + 0.01 * rng.standard_normal(n)
    woe_df = pd.DataFrame({"x1": x1, "leaky": leaky})
    sel = stepwise_select(woe_df, y, {"x1": 0.3, "leaky": 5.0})
    assert "leaky" not in sel


def test_points_table_sums_to_score() -> None:
    # Build a tiny scorecard by hand and check the points identity.
    from eq_credit.data.synthetic import generate_loan_book
    from eq_credit.woe import WOETransformer

    df = generate_loan_book(20_000, seed=33)
    wt = WOETransformer(["leverage", "roa"]).fit(df, "default")
    woe = wt.transform(df)
    fit = fit_logistic(woe, df["default"], feature_names=list(woe.columns))
    sc = ScorecardScaling()
    pts = scorecard_points_table(fit, wt.binnings_, sc)
    # For one obligor: sum of its bin points == scaled score from its PD.
    row = df.iloc[100]
    woe_row = woe.iloc[100]
    pd_hat = fit.predict_proba(woe_row.to_numpy()[None, :])[0]
    expected_score = sc.score_from_pd(pd_hat)
    total = 0.0
    for feat, w in [("leverage", woe_row["woe_leverage"]), ("roa", woe_row["woe_roa"])]:
        sub = pts[pts["feature"] == feat]
        match = sub.iloc[(sub["woe"] - w).abs().argmin()]
        total += match["points"]
    assert total == pytest.approx(expected_score, abs=1e-6)
