"""Tests for eq_port.mvo: closed forms, SLSQP cross-checks, frontier
properties and input validation."""

import numpy as np
import pytest

from eq_port.data import generate_panel
from eq_port.mvo import (
    efficient_frontier,
    max_sharpe_constrained,
    min_variance_constrained,
    min_variance_weights,
    portfolio_return,
    portfolio_vol,
    tangency_weights,
    target_return_portfolio,
    target_risk_portfolio,
)

PANEL = generate_panel(n_assets=6, n_periods=100, seed=11)
MU = PANEL.true_mean
COV = PANEL.true_cov


# --------------------------------------------------------------- closed forms

def test_min_variance_matches_formula():
    inv = np.linalg.inv(COV)
    ones = np.ones(len(MU))
    expected = inv @ ones / (ones @ inv @ ones)
    np.testing.assert_allclose(min_variance_weights(COV), expected, atol=1e-12)


def test_min_variance_sums_to_one():
    assert min_variance_weights(COV).sum() == pytest.approx(1.0, abs=1e-12)


def test_min_variance_is_global_min_over_random_portfolios():
    w_mv = min_variance_weights(COV)
    v_mv = portfolio_vol(w_mv, COV)
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = rng.normal(size=len(MU))
        w /= w.sum()
        assert portfolio_vol(w, COV) >= v_mv - 1e-15


def test_tangency_matches_formula():
    rf = 1e-5
    inv = np.linalg.inv(COV)
    ex = MU - rf
    expected = inv @ ex / (np.ones(len(MU)) @ inv @ ex)
    np.testing.assert_allclose(tangency_weights(MU, COV, rf=rf), expected, atol=1e-12)


def test_tangency_maximizes_sharpe_over_random_portfolios():
    w_tan = tangency_weights(MU, COV)
    sr_tan = portfolio_return(w_tan, MU) / portfolio_vol(w_tan, COV)
    rng = np.random.default_rng(1)
    for _ in range(50):
        w = np.abs(rng.normal(size=len(MU)))
        w /= w.sum()
        assert portfolio_return(w, MU) / portfolio_vol(w, COV) <= sr_tan + 1e-12


def test_tangency_negative_denominator_raises():
    with pytest.raises(ValueError, match="tangency"):
        tangency_weights(-np.abs(MU), COV, rf=0.0)


# ------------------------------------------------------- SLSQP vs closed form

def test_slsqp_min_variance_matches_closed_form():
    w_num = min_variance_constrained(COV, bounds=None)
    np.testing.assert_allclose(w_num, min_variance_weights(COV), atol=1e-6)


def test_slsqp_max_sharpe_matches_closed_form():
    w_num = max_sharpe_constrained(MU, COV, bounds=None)
    np.testing.assert_allclose(w_num, tangency_weights(MU, COV), atol=1e-5)


def test_long_only_constraint_binds_on_terrible_asset():
    mu = MU.copy()
    mu[2] = -0.05  # catastrophic expected return
    w = max_sharpe_constrained(mu, COV, bounds=(0.0, 1.0))
    assert w[2] == pytest.approx(0.0, abs=1e-8)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(w >= -1e-10)


def test_long_only_min_variance_within_bounds():
    w = min_variance_constrained(COV, bounds=(0.0, 1.0))
    assert np.all(w >= -1e-10) and np.all(w <= 1 + 1e-10)
    assert w.sum() == pytest.approx(1.0, abs=1e-10)


def test_box_bounds_respected():
    w = min_variance_constrained(COV, bounds=(0.05, 0.30))
    assert np.all(w >= 0.05 - 1e-9) and np.all(w <= 0.30 + 1e-9)


# -------------------------------------------------------------- target return

def test_target_return_is_achieved_exactly():
    tgt = float(MU.mean()) * 1.1
    w = target_return_portfolio(MU, COV, tgt, bounds=None)
    assert portfolio_return(w, MU) == pytest.approx(tgt, abs=1e-10)
    assert w.sum() == pytest.approx(1.0, abs=1e-10)


def test_target_return_variance_is_minimal_under_perturbation():
    """Perturbing the solution inside the feasible set must not lower variance."""
    tgt = float(MU.mean()) * 1.1
    w = target_return_portfolio(MU, COV, tgt, bounds=None)
    v = portfolio_vol(w, COV) ** 2
    rng = np.random.default_rng(5)
    ones = np.ones(len(MU))
    basis = np.stack([ones, MU])  # constraint normals
    for _ in range(30):
        d = rng.normal(size=len(MU))
        # project out constraint-violating components
        coef, *_ = np.linalg.lstsq(basis.T, d, rcond=None)
        d = d - basis.T @ coef
        d /= np.linalg.norm(d)
        for eps in (1e-4, 1e-3):
            wp = w + eps * d
            assert wp @ COV @ wp >= v - 1e-16


# ------------------------------------------------------------------- frontier

def test_frontier_vol_monotone_above_min_var():
    fr = efficient_frontier(MU, COV, n_points=30)
    assert np.all(np.diff(fr.returns) > 0)
    assert np.all(np.diff(fr.vols) >= -1e-14)


def test_frontier_starts_at_min_variance():
    fr = efficient_frontier(MU, COV, n_points=10)
    np.testing.assert_allclose(fr.weights[0], min_variance_weights(COV), atol=1e-10)


def test_two_fund_theorem():
    """Any analytic frontier portfolio is an affine combo of two others."""
    fr = efficient_frontier(MU, COV, n_points=7)
    w1, w2 = fr.weights[1], fr.weights[5]
    r1, r2 = fr.returns[1], fr.returns[5]
    for k in (0, 2, 3, 4, 6):
        alpha = (fr.returns[k] - r2) / (r1 - r2)
        combo = alpha * w1 + (1 - alpha) * w2
        np.testing.assert_allclose(combo, fr.weights[k], atol=1e-10)


def test_constrained_frontier_dominated_by_unconstrained():
    fr_c = efficient_frontier(MU, COV, n_points=8, bounds=(0.0, 1.0))
    for r, v in zip(fr_c.returns, fr_c.vols):
        w_u = target_return_portfolio(MU, COV, float(r), bounds=None)
        assert portfolio_vol(w_u, COV) <= v + 1e-9


def test_constrained_frontier_weights_feasible():
    fr = efficient_frontier(MU, COV, n_points=8, bounds=(0.0, 1.0))
    assert np.all(fr.weights >= -1e-8) and np.all(fr.weights <= 1 + 1e-8)
    np.testing.assert_allclose(fr.weights.sum(axis=1), 1.0, atol=1e-8)


def test_frontier_degenerate_mu_raises():
    with pytest.raises(ValueError, match="degenerate"):
        efficient_frontier(np.full(len(MU), 0.01), COV)


def test_frontier_npoints_validation():
    with pytest.raises(ValueError, match="n_points"):
        efficient_frontier(MU, COV, n_points=1)


# ---------------------------------------------------------------- target risk

def test_target_risk_hits_vol_cap():
    fr = efficient_frontier(MU, COV, n_points=5)
    v_target = float(fr.vols[3])  # above min-var vol: cap binds
    w = target_risk_portfolio(MU, COV, v_target, bounds=None)
    assert portfolio_vol(w, COV) == pytest.approx(v_target, rel=1e-6)
    # on the frontier: same return as the target-return portfolio there
    assert portfolio_return(w, MU) == pytest.approx(float(fr.returns[3]), rel=1e-5)


def test_target_risk_validates():
    with pytest.raises(ValueError, match="target_vol"):
        target_risk_portfolio(MU, COV, -0.1)


# ----------------------------------------------------------------- validation

def test_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="mismatch"):
        tangency_weights(MU[:3], COV)


def test_non_square_cov_raises():
    with pytest.raises(ValueError, match="square"):
        min_variance_weights(np.zeros((3, 4)))


def test_non_symmetric_cov_raises():
    a = COV.copy()
    a[0, 1] += 1.0
    with pytest.raises(ValueError, match="symmetric"):
        min_variance_weights(a)


def test_non_psd_cov_raises_informatively():
    a = np.diag([1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="positive definite"):
        min_variance_weights(a)


def test_nan_inputs_raise():
    bad = COV.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        min_variance_weights(bad)
