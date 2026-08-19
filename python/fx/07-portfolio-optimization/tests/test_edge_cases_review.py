"""Edge-case and property tests added in the review pass (project 07).

Focus, per the FX portfolio-construction domain:

* degenerate correlation matrices (singular, perfectly correlated, near-rank-1),
* pegged-pair zero-volatility assets in the universe,
* infeasible constraint sets (gross < net, unreachable box),
* single-asset and two-row universes,
* NaN/Inf rejection at every public entry point.
"""

import numpy as np
import pandas as pd
import pytest

from fx_port import (
    empirical_cvar,
    empirical_var,
    erc_weights,
    is_psd,
    lw_shrinkage,
    max_utility,
    min_cvar,
    min_variance_slsqp,
    min_variance_weights,
    optimal_hedge_ratios,
    psd_repair,
    risk_contributions,
    sample_cov,
    sharpe_ratio,
    annualized_vol,
    max_drawdown,
    tangency_weights,
    vol_target,
)

COLS = ["EUR", "JPY", "AUD"]


def _diag(*v):
    return pd.DataFrame(np.diag(v), index=COLS[: len(v)], columns=COLS[: len(v)])


# ---------------------------------------------------------------------------
# Degenerate correlation matrices
# ---------------------------------------------------------------------------

def test_perfectly_correlated_pair_is_singular_and_reports_it():
    """Two currencies with correlation 1.0 give a singular Sigma."""
    s = pd.DataFrame(
        [[1e-4, 1e-4], [1e-4, 1e-4]], index=["EUR", "CHF"], columns=["EUR", "CHF"]
    )
    assert not is_psd(s, tol=-1e-16) or np.linalg.matrix_rank(s.to_numpy()) == 1
    with pytest.raises(ValueError, match="singular"):
        min_variance_weights(s)


def test_psd_repair_makes_singular_matrix_invertible():
    """A floored repair restores invertibility and keeps the matrix PSD."""
    s = pd.DataFrame(
        [[1e-4, 1e-4], [1e-4, 1e-4]], index=["EUR", "CHF"], columns=["EUR", "CHF"]
    )
    fixed = psd_repair(s, min_eig=1e-10)
    assert is_psd(fixed)
    assert np.linalg.eigvalsh(fixed.to_numpy()).min() >= 1e-10 - 1e-18
    w = min_variance_weights(fixed)
    assert w.sum() == pytest.approx(1.0)


def test_psd_repair_projects_indefinite_correlation_matrix():
    """A correlation matrix with a negative eigenvalue (bad estimate) is repaired."""
    bad = pd.DataFrame(
        [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]],
        index=COLS,
        columns=COLS,
    )
    assert np.linalg.eigvalsh(bad.to_numpy()).min() < 0
    fixed = psd_repair(bad, min_eig=0.0)
    assert is_psd(fixed)
    # nearest-PSD projection keeps the diagonal close and stays symmetric
    assert np.allclose(fixed.to_numpy(), fixed.to_numpy().T, atol=1e-14)


def test_psd_repair_is_idempotent_on_already_psd_matrix():
    s = _diag(1e-4, 2e-4, 1.5e-4)
    once = psd_repair(s, min_eig=0.0)
    twice = psd_repair(once, min_eig=0.0)
    assert np.allclose(once.to_numpy(), twice.to_numpy(), atol=1e-15)
    assert np.allclose(once.to_numpy(), s.to_numpy(), atol=1e-15)


# ---------------------------------------------------------------------------
# Pegged pair: zero-vol asset in the universe
# ---------------------------------------------------------------------------

def test_pegged_zero_vol_asset_makes_sigma_singular():
    """A hard-pegged currency (zero variance row/col) is not invertible."""
    s = _diag(1e-4, 2e-4, 0.0)
    with pytest.raises(ValueError, match="singular"):
        min_variance_weights(s)


def test_erc_rejects_pegged_zero_vol_asset_with_reason():
    s = _diag(1e-4, 2e-4, 0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        erc_weights(s)


def test_pegged_asset_dominates_min_variance_when_floored():
    """After flooring, the near-zero-vol peg takes almost the whole min-var book."""
    s = psd_repair(_diag(1e-4, 2e-4, 1e-12), min_eig=1e-14)
    w = min_variance_weights(s)
    assert w.iloc[2] > 0.99
    assert w.sum() == pytest.approx(1.0)


def test_vol_target_rejects_zero_variance_portfolio():
    s = _diag(0.0, 0.0)
    w = pd.Series([0.5, 0.5], index=COLS[:2])
    with pytest.raises(ValueError, match="zero-variance"):
        vol_target(w, s, target_vol=0.10)


def test_risk_contributions_undefined_at_zero_variance():
    s = _diag(0.0, 0.0)
    with pytest.raises(ValueError, match="variance is zero"):
        risk_contributions(pd.Series([0.5, 0.5], index=COLS[:2]), s)


# ---------------------------------------------------------------------------
# Infeasible constraints
# ---------------------------------------------------------------------------

def test_gross_below_net_budget_is_rejected():
    mu = pd.Series([0.001, 0.002, 0.0005], index=COLS)
    s = _diag(1e-4, 2e-4, 1.5e-4)
    with pytest.raises(ValueError, match="infeasible"):
        max_utility(mu, s, sum_to=1.0, gross_limit=0.5)


def test_box_cap_unreachable_for_net_budget_is_rejected():
    """3 currencies capped at 20% each cannot sum to 1.0 -> explicit error."""
    mu = pd.Series([0.001, 0.002, 0.0005], index=COLS)
    s = _diag(1e-4, 2e-4, 1.5e-4)
    with pytest.raises(ValueError, match="infeasible box"):
        max_utility(mu, s, sum_to=1.0, bounds=(0.0, 0.2))
    with pytest.raises(ValueError, match="infeasible box"):
        min_variance_slsqp(s, sum_to=1.0, bounds=(0.0, 0.2))


def test_box_floor_above_net_budget_is_rejected():
    mu = pd.Series([0.001, 0.002, 0.0005], index=COLS)
    s = _diag(1e-4, 2e-4, 1.5e-4)
    with pytest.raises(ValueError, match="infeasible box"):
        max_utility(mu, s, sum_to=0.0, bounds=(0.5, 1.0))


def test_inverted_box_is_rejected():
    s = _diag(1e-4, 2e-4, 1.5e-4)
    with pytest.raises(ValueError, match="lower bound"):
        min_variance_slsqp(s, sum_to=1.0, bounds=(0.6, 0.1))


def test_feasible_box_at_the_boundary_still_solves():
    """Exactly-attainable box (3 x 1/3 = 1.0) must NOT be rejected."""
    s = _diag(1e-4, 2e-4, 1.5e-4)
    res = min_variance_slsqp(s, sum_to=1.0, bounds=(0.0, 1.0 / 3.0))
    assert res.success
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(res.weights.to_numpy() <= 1.0 / 3.0 + 1e-8)


def test_cvar_lp_infeasible_return_floor_raises():
    rng = np.random.default_rng(0)
    sc = pd.DataFrame(rng.normal(0.0, 0.01, (250, 3)), columns=COLS)
    with pytest.raises(ValueError, match="infeasible|failed"):
        min_cvar(sc, sum_to=1.0, gross_limit=1.0, return_floor=10.0)


def test_no_long_tangency_portfolio_raises():
    """All-negative excess means: the normalised tangency formula is meaningless."""
    mu = pd.Series([-0.001, -0.002, -0.0005], index=COLS)
    s = _diag(1e-4, 2e-4, 1.5e-4)
    with pytest.raises(ValueError, match="no long tangency"):
        tangency_weights(mu, s)


# ---------------------------------------------------------------------------
# Tiny universes / samples
# ---------------------------------------------------------------------------

def test_single_asset_min_variance_is_trivially_one():
    s = pd.DataFrame([[1e-4]], index=["EUR"], columns=["EUR"])
    w = min_variance_weights(s)
    assert w.iloc[0] == pytest.approx(1.0)


def test_single_asset_erc_is_one():
    s = pd.DataFrame([[1e-4]], index=["EUR"], columns=["EUR"])
    assert erc_weights(s).iloc[0] == pytest.approx(1.0)


def test_empty_sigma_rejected():
    empty = pd.DataFrame(np.zeros((0, 0)))
    with pytest.raises(ValueError, match="empty"):
        min_variance_weights(empty)


def test_two_row_sample_cov_works_and_three_needed_for_factor():
    rng = np.random.default_rng(3)
    r = pd.DataFrame(rng.normal(0, 0.01, (2, 3)), columns=COLS)
    cov = sample_cov(r)
    assert cov.shape == (3, 3)
    with pytest.raises(ValueError, match="at least"):
        sample_cov(r.iloc[:1])


def test_vol_metrics_need_two_observations():
    with pytest.raises(ValueError):
        annualized_vol(pd.Series([0.01]))


def test_constant_return_series_has_zero_vol_and_undefined_sharpe():
    r = pd.Series(np.full(50, 0.0004))
    assert annualized_vol(r) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="Sharpe undefined"):
        sharpe_ratio(r)
    assert max_drawdown(r) == pytest.approx(0.0)


def test_lw_shrinkage_intensity_in_unit_interval_on_short_sample():
    """Ledoit-Wolf delta stays in [0,1] even when T < N (rank-deficient S)."""
    rng = np.random.default_rng(11)
    r = pd.DataFrame(rng.normal(0, 0.01, (4, 3)), columns=COLS)
    sigma, delta = lw_shrinkage(r)
    assert 0.0 <= delta <= 1.0
    assert is_psd(sigma, tol=1e-12)


# ---------------------------------------------------------------------------
# NaN / Inf rejection
# ---------------------------------------------------------------------------

def test_min_variance_rejects_nan_sigma():
    s = _diag(1e-4, np.nan, 1e-4)
    with pytest.raises(ValueError, match="NaN/Inf"):
        min_variance_weights(s)


def test_max_utility_rejects_nan_mu():
    mu = pd.Series([0.001, np.nan, 0.0005], index=COLS)
    s = _diag(1e-4, 2e-4, 1.5e-4)
    with pytest.raises(ValueError, match="NaN/Inf"):
        max_utility(mu, s)


def test_psd_repair_and_is_psd_reject_nan():
    s = _diag(1e-4, np.nan, 1e-4)
    with pytest.raises(ValueError, match="NaN/Inf"):
        psd_repair(s)
    with pytest.raises(ValueError, match="NaN/Inf"):
        is_psd(s)


def test_covariance_estimators_reject_inf_panel():
    rng = np.random.default_rng(5)
    r = pd.DataFrame(rng.normal(0, 0.01, (60, 3)), columns=COLS)
    r.iloc[7, 1] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        sample_cov(r)


def test_metrics_reject_nan_returns():
    r = pd.Series([0.01, np.nan, -0.02, 0.005])
    for fn in (annualized_vol, max_drawdown, sharpe_ratio):
        with pytest.raises(ValueError, match="NaN/Inf"):
            fn(r)


def test_cvar_var_reject_nonfinite_scenarios():
    with pytest.raises(ValueError, match="NaN/Inf"):
        empirical_cvar(np.array([0.01, np.inf, -0.02]))
    with pytest.raises(ValueError, match="NaN/Inf"):
        empirical_var(np.array([0.01, np.nan, -0.02]))


def test_min_cvar_rejects_nan_scenarios():
    rng = np.random.default_rng(2)
    sc = pd.DataFrame(rng.normal(0, 0.01, (100, 3)), columns=COLS)
    sc.iloc[5, 1] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        min_cvar(sc)


def test_hedging_rejects_nan_fx_panel():
    rng = np.random.default_rng(1)
    idx = pd.RangeIndex(80)
    fx = pd.DataFrame(rng.normal(0, 0.01, (80, 2)), index=idx, columns=["JPY", "EUR"])
    u = pd.Series(rng.normal(0, 0.02, 80), index=idx)
    exp = pd.Series([0.3, 0.4], index=["JPY", "EUR"])
    fx.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        optimal_hedge_ratios(u, fx, exp)


def test_erc_rejects_nan_sigma_with_clear_message():
    s = _diag(1e-4, np.nan, 1e-4)
    with pytest.raises(ValueError, match="NaN/Inf"):
        erc_weights(s)


def test_risk_contributions_reject_nan():
    s = _diag(1e-4, np.nan, 1e-4)
    with pytest.raises(ValueError, match="finite"):
        risk_contributions(pd.Series([0.3, 0.3, 0.4], index=COLS), s)
