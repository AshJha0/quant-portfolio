"""Tests for eq_port.risk_parity: ERC solver, risk contributions, naive
parity and vol targeting."""

import numpy as np
import pytest

from eq_port.data import generate_panel
from eq_port.risk_parity import (
    erc_weights,
    inverse_vol_weights,
    risk_contributions,
    vol_target_overlay,
)

COV = generate_panel(n_assets=7, n_periods=100, seed=21).true_cov


def _const_corr_cov(vols: np.ndarray, rho: float) -> np.ndarray:
    n = len(vols)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    return corr * np.outer(vols, vols)


# --------------------------------------------------------- risk contributions

def test_rc_euler_identity_exact():
    rng = np.random.default_rng(2)
    w = rng.dirichlet(np.ones(7))
    rc = risk_contributions(w, COV)
    assert rc.sum() == pytest.approx(float(w @ COV @ w), abs=1e-18)


def test_rc_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="mismatch"):
        risk_contributions(np.ones(3), COV)


# ------------------------------------------------------------------------ ERC

def test_erc_contributions_all_equal():
    w = erc_weights(COV)
    rc = risk_contributions(w, COV)
    assert (rc.max() - rc.min()) / rc.mean() < 1e-8


def test_erc_weights_positive_and_sum_to_one():
    w = erc_weights(COV)
    assert np.all(w > 0)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_erc_equal_vol_uncorrelated_is_equal_weight():
    cov = np.eye(5) * 0.04
    np.testing.assert_allclose(erc_weights(cov), np.full(5, 0.2), atol=1e-10)


def test_erc_uncorrelated_equals_inverse_vol():
    cov = np.diag([0.01, 0.04, 0.09, 0.16])
    np.testing.assert_allclose(erc_weights(cov), inverse_vol_weights(cov), atol=1e-10)


def test_erc_constant_correlation_equals_inverse_vol():
    cov = _const_corr_cov(np.array([0.10, 0.18, 0.25, 0.32]), rho=0.5)
    np.testing.assert_allclose(erc_weights(cov), inverse_vol_weights(cov), atol=1e-9)


def test_erc_scale_invariance():
    np.testing.assert_allclose(erc_weights(COV), erc_weights(COV * 252.0), atol=1e-10)


def test_erc_custom_budgets_respected():
    b = np.array([4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    w = erc_weights(COV, budget=b)
    rc = risk_contributions(w, COV)
    np.testing.assert_allclose(rc / rc.sum(), b / b.sum(), atol=1e-9)


def test_erc_single_asset():
    np.testing.assert_allclose(erc_weights(np.array([[0.04]])), [1.0])


def test_erc_zero_vol_asset_raises_informatively():
    cov = np.diag([0.04, 0.0, 0.02])
    with pytest.raises(ValueError, match="zero-vol"):
        erc_weights(cov)
    with pytest.raises(ValueError, match="zero-vol"):
        inverse_vol_weights(cov)


def test_erc_bad_budget_raises():
    with pytest.raises(ValueError, match="positive"):
        erc_weights(COV, budget=np.array([1, 1, 1, 1, 1, 1, -1.0]))
    with pytest.raises(ValueError, match="entries"):
        erc_weights(COV, budget=np.ones(3))


def test_inverse_vol_sums_to_one_and_orders_by_vol():
    cov = np.diag([0.01, 0.04])
    w = inverse_vol_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1]  # lower vol gets more weight


# --------------------------------------------------------------- vol targeting

def test_vol_target_hits_target_exactly():
    w = erc_weights(COV)
    target = 0.10
    lev = vol_target_overlay(w, COV, target, periods_per_year=252.0)
    vol = np.sqrt(lev @ COV @ lev * 252.0)
    assert vol == pytest.approx(target, abs=1e-14)


def test_vol_target_leverage_cap_binds():
    w = erc_weights(COV)
    lev = vol_target_overlay(w, COV, 5.0, periods_per_year=252.0, max_leverage=1.5)
    np.testing.assert_allclose(lev, 1.5 * w, atol=1e-15)


def test_vol_target_direction_preserved():
    w = erc_weights(COV)
    lev = vol_target_overlay(w, COV, 0.10)
    np.testing.assert_allclose(lev / lev.sum(), w, atol=1e-12)


def test_vol_target_validates():
    w = np.full(7, 1 / 7)
    with pytest.raises(ValueError, match="target_vol"):
        vol_target_overlay(w, COV, 0.0)
    with pytest.raises(ValueError, match="zero ex-ante"):
        vol_target_overlay(np.zeros(7), COV, 0.1)
    with pytest.raises(ValueError, match="mismatch"):
        vol_target_overlay(np.ones(3), COV, 0.1)
