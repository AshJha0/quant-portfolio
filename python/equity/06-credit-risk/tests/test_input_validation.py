"""NaN/Inf rejection and parameter-domain validation added in the hardening
pass: every public entry point must refuse non-finite inputs with an
informative ValueError instead of silently propagating garbage."""

import numpy as np
import pytest

from eq_credit.model import ScorecardScaling, fit_logistic
from eq_credit.portfolio_risk import (
    asset_correlation,
    basel_k,
    economic_capital,
    expected_loss,
    simulate_portfolio_losses,
)
from eq_credit.validation import brier_score, hosmer_lemeshow
from eq_credit.woe import woe_iv_from_counts


def test_fit_logistic_rejects_nan_and_inf_features() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 2))
    y = (rng.uniform(size=50) < 0.3).astype(float)
    X_nan = X.copy()
    X_nan[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        fit_logistic(X_nan, y)
    X_inf = X.copy()
    X_inf[7, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        fit_logistic(X_inf, y)


def test_expected_loss_rejects_nan_pd() -> None:
    # NaN passes naive range checks (NaN < 0 is False) — must be caught.
    with pytest.raises(ValueError, match="finite"):
        expected_loss(np.array([0.02, np.nan]), 0.4, 100.0)
    with pytest.raises(ValueError, match="finite"):
        expected_loss(0.02, np.nan, 100.0)
    with pytest.raises(ValueError, match="finite"):
        expected_loss(0.02, 0.4, np.inf)


def test_basel_k_rejects_nan_inputs() -> None:
    with pytest.raises(ValueError, match="PD"):
        basel_k(np.nan, 0.45)
    with pytest.raises(ValueError, match="LGD"):
        basel_k(0.01, np.nan)
    with pytest.raises(ValueError, match="PD"):
        asset_correlation(np.array([0.01, np.nan]))


def test_simulation_zero_total_ead_raises() -> None:
    with pytest.raises(ValueError, match="total EAD"):
        simulate_portfolio_losses(0.02, 0.5, 0.0, 0.2, 100, seed=1, n_loans=10)


def test_economic_capital_invalid_quantile_raises() -> None:
    losses = np.linspace(0.0, 0.1, 100)
    with pytest.raises(ValueError, match="q must"):
        economic_capital(losses, el_rate=0.01, q=1.5)
    with pytest.raises(ValueError, match="q must"):
        economic_capital(losses, el_rate=0.01, q=0.0)


def test_scorecard_scaling_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="pdo"):
        ScorecardScaling(pdo=0.0)
    with pytest.raises(ValueError, match="pdo"):
        ScorecardScaling(pdo=-20.0)
    with pytest.raises(ValueError, match="base_odds"):
        ScorecardScaling(base_odds=0.0)
    with pytest.raises(ValueError, match="base_score"):
        ScorecardScaling(base_score=np.nan)


def test_woe_counts_reject_negative_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        woe_iv_from_counts(np.array([10.0, -1.0]), np.array([2.0, 3.0]))
    with pytest.raises(ValueError, match="finite"):
        woe_iv_from_counts(np.array([10.0, np.nan]), np.array([2.0, 3.0]))


def test_brier_and_hl_reject_out_of_range_pd() -> None:
    y = np.array([0, 1, 0, 1] * 5, dtype=float)
    bad_pd = np.full(20, 0.5)
    bad_pd[0] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score(y, bad_pd)
    nan_pd = np.full(20, 0.5)
    nan_pd[3] = np.nan
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        hosmer_lemeshow(y, nan_pd, n_groups=4)


def test_hl_tiny_sample_raises_informative() -> None:
    y = np.array([0.0, 1.0, 0.0])
    p = np.array([0.1, 0.6, 0.2])
    with pytest.raises(ValueError, match="n_groups"):
        hosmer_lemeshow(y, p, n_groups=10)
