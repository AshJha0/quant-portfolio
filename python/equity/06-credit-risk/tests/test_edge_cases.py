"""Edge cases per the documentation contract: zero defaults, single feature,
all-missing feature, PD boundary clamps, empty portfolio, degenerate inputs."""

import numpy as np
import pandas as pd
import pytest

from eq_credit.data.synthetic import generate_loan_book
from eq_credit.model import ScorecardScaling, fit_logistic
from eq_credit.portfolio_risk import (
    el_by_bucket,
    expected_loss,
    simulate_portfolio_losses,
)
from eq_credit.validation import psi, roc_auc
from eq_credit.woe import WOETransformer, fit_numeric_binning


def test_zero_defaults_raises_everywhere() -> None:
    n = 200
    x = np.random.default_rng(0).standard_normal(n)
    y0 = pd.Series(np.zeros(n, dtype=int))
    with pytest.raises(ValueError, match="zero defaults"):
        fit_numeric_binning(pd.Series(x), y0, "x")
    with pytest.raises(ValueError, match="zero defaults"):
        fit_logistic(x[:, None], y0.to_numpy())
    with pytest.raises(ValueError, match="single class"):
        roc_auc(y0.to_numpy(), x)


def test_single_feature_scorecard_end_to_end() -> None:
    df = generate_loan_book(8_000, seed=41, missing=False, outliers=False)
    wt = WOETransformer(["leverage"]).fit(df, "default")
    woe = wt.transform(df)
    fit = fit_logistic(woe, df["default"], feature_names=list(woe.columns))
    assert fit.converged
    assert fit.coef[0] < 0  # higher WOE (safer) -> lower PD
    auc = roc_auc(df["default"].to_numpy(), fit.predict_proba(woe))
    assert auc > 0.6


def test_all_missing_feature_raises() -> None:
    y = pd.Series([0] * 90 + [1] * 10)
    x = pd.Series([np.nan] * 100)
    with pytest.raises(ValueError, match="entirely missing"):
        fit_numeric_binning(x, y, "ghost")


def test_constant_feature_single_bin() -> None:
    rng = np.random.default_rng(2)
    y = pd.Series((rng.uniform(size=500) < 0.1).astype(int))
    x = pd.Series(np.ones(500))
    fb = fit_numeric_binning(x, y, "const")
    assert fb.iv == pytest.approx(0.0, abs=1e-12)  # no information


def test_pd_boundary_clamps_in_score() -> None:
    sc = ScorecardScaling()
    scores = sc.score_from_pd(np.array([0.0, 1e-12, 0.5, 1.0 - 1e-12, 1.0]))
    assert np.isfinite(scores).all()
    assert np.all(np.diff(scores) <= 0)  # still ordered
    # Round trip at the clamp is stable.
    assert 0.0 <= sc.pd_from_score(scores[0]) <= 1.0


def test_empty_portfolio_raises() -> None:
    with pytest.raises(ValueError, match="empty portfolio"):
        expected_loss(np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="empty portfolio"):
        el_by_bucket(
            pd.DataFrame(columns=["rating", "pd", "lgd", "ead"]), "rating"
        )


def test_simulation_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="rho"):
        simulate_portfolio_losses(0.02, 0.5, 1.0, 0.0, 100, seed=1, n_loans=10)
    with pytest.raises(ValueError, match="n_sims"):
        simulate_portfolio_losses(0.02, 0.5, 1.0, 0.2, 0, seed=1, n_loans=10)
    with pytest.raises(ValueError, match="strictly"):
        simulate_portfolio_losses(0.0, 0.5, 1.0, 0.2, 100, seed=1, n_loans=10)


def test_psi_empty_sample_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        psi(np.array([]), np.array([1.0]))


def test_tiny_sample_logit_still_finite() -> None:
    # 30 obs, 3 defaults: noisy but must return finite estimates and SEs.
    rng = np.random.default_rng(5)
    x = rng.standard_normal(30)
    y = np.zeros(30)
    y[:3] = 1
    fit = fit_logistic(x[:, None], y)
    assert np.isfinite(fit.coef).all() and np.isfinite(fit.se).all()


def test_extreme_but_valid_basel_inputs() -> None:
    from eq_credit.portfolio_risk import basel_k

    # LGD = 0 => no capital; PD at floor with tiny LGD stays finite.
    assert basel_k(0.01, 0.0, 2.5)[0] == 0.0
    assert np.isfinite(basel_k(0.0003, 0.01, 1.0)[0])
    # Very high PD: formula turns over but stays non-negative and finite.
    k = basel_k(np.linspace(0.5, 0.9999, 50), 0.45, 2.5)
    assert np.isfinite(k).all() and (k >= 0).all()
