"""Edge cases across the stack: single asset, two-asset closed forms,
singular/perfectly-correlated covariances, zero-vol assets, negative
means under long-only, and T < N estimation windows."""

import numpy as np
import pandas as pd
import pytest

from eq_port.covariance import (
    condition_number,
    is_psd,
    ledoit_wolf_cc,
    psd_repair,
    sample_cov,
)
from eq_port.data import generate_panel
from eq_port.mvo import (
    max_sharpe_constrained,
    min_variance_constrained,
    min_variance_weights,
    tangency_weights,
)
from eq_port.risk_parity import erc_weights, inverse_vol_weights
from eq_port.metrics import annualized_return, max_drawdown, sharpe_ratio

RNG = np.random.default_rng(99)


# --------------------------------------------------------------- single asset

def test_single_asset_degenerates_gracefully():
    cov = np.array([[0.04]])
    mu = np.array([0.01])
    np.testing.assert_allclose(min_variance_weights(cov), [1.0])
    np.testing.assert_allclose(tangency_weights(mu, cov), [1.0])
    np.testing.assert_allclose(erc_weights(cov), [1.0])
    np.testing.assert_allclose(inverse_vol_weights(cov), [1.0])
    np.testing.assert_allclose(min_variance_constrained(cov, bounds=(0, 1)), [1.0],
                               atol=1e-10)


def test_single_asset_lw_and_metrics():
    x = RNG.normal(0.001, 0.02, size=(100, 1))
    res = ledoit_wolf_cc(x)
    assert res.intensity == 0.0 and res.cov.shape == (1, 1)
    r = x.ravel()
    assert np.isfinite(sharpe_ratio(r))
    assert 0.0 <= max_drawdown(r) < 1.0
    assert np.isfinite(annualized_return(r))


# ----------------------------------------------------------------- two assets

def test_two_asset_min_variance_closed_form():
    s1, s2, rho = 0.2, 0.3, 0.4
    c = rho * s1 * s2
    cov = np.array([[s1**2, c], [c, s2**2]])
    w1 = (s2**2 - c) / (s1**2 + s2**2 - 2 * c)
    np.testing.assert_allclose(min_variance_weights(cov), [w1, 1 - w1], atol=1e-14)


def test_two_asset_tangency_closed_form():
    mu = np.array([0.08, 0.05])
    s1, s2, rho = 0.2, 0.15, 0.3
    c = rho * s1 * s2
    cov = np.array([[s1**2, c], [c, s2**2]])
    # w proportional to Sigma^{-1} mu, hand-inverted 2x2
    det = s1**2 * s2**2 - c**2
    raw = np.array(
        [(s2**2 * mu[0] - c * mu[1]) / det, (s1**2 * mu[1] - c * mu[0]) / det]
    )
    np.testing.assert_allclose(tangency_weights(mu, cov), raw / raw.sum(), atol=1e-14)


# ------------------------------------------------ singular / perfect correlation

def test_perfectly_correlated_assets_raise_then_repair():
    vols = np.array([0.1, 0.2, 0.3])
    cov = np.outer(vols, vols)  # rank 1: correlation exactly 1
    with pytest.raises(ValueError, match="positive definite"):
        min_variance_weights(cov)
    repaired = psd_repair(cov, eps=1e-8)
    w = min_variance_weights(repaired)  # now solvable
    assert w.sum() == pytest.approx(1.0, abs=1e-10)


def test_perfectly_correlated_erc_still_solves_after_repair():
    vols = np.array([0.1, 0.2])
    repaired = psd_repair(np.outer(vols, vols), eps=1e-8)
    w = erc_weights(repaired)
    assert np.all(w > 0) and w.sum() == pytest.approx(1.0, abs=1e-12)


# -------------------------------------------------------------- zero-vol asset

def test_zero_vol_asset_repaired_min_variance_prefers_it():
    """A (near) riskless asset should absorb nearly all min-var weight."""
    cov = np.diag([0.04, 0.09, 0.0])
    repaired = psd_repair(cov, eps=1e-10)
    w = min_variance_weights(repaired)
    assert w[2] > 0.99


def test_zero_vol_asset_lw_still_works():
    x = RNG.normal(0, 0.01, size=(50, 3))
    x[:, 1] = 0.0  # constant (zero-vol) asset
    res = ledoit_wolf_cc(x)
    assert 0.0 <= res.intensity <= 1.0
    assert np.all(np.isfinite(res.cov))


# ------------------------------------------- negative means under long-only

def test_all_negative_means_long_only_max_sharpe_still_feasible():
    cov = generate_panel(n_assets=4, n_periods=50, seed=1).true_cov
    mu = np.array([-0.01, -0.02, -0.005, -0.03])
    w = max_sharpe_constrained(mu, cov, bounds=(0.0, 1.0))
    assert np.all(w >= -1e-9) and w.sum() == pytest.approx(1.0, abs=1e-8)
    with pytest.raises(ValueError, match="tangency"):
        tangency_weights(mu, cov)  # unconstrained closed form must refuse


def test_one_negative_mean_asset_gets_zero_long_only():
    cov = generate_panel(n_assets=4, n_periods=50, seed=2).true_cov
    mu = np.array([0.0004, 0.0005, -0.05, 0.0003])
    w = max_sharpe_constrained(mu, cov, bounds=(0.0, 1.0))
    assert w[2] == pytest.approx(0.0, abs=1e-8)


# ----------------------------------------------------- T < N estimation window

def test_short_window_singular_sample_but_lw_invertible():
    x = RNG.normal(0.0, 0.01, size=(5, 8))  # T=5 < N=8
    s = sample_cov(x)
    assert condition_number(s) == np.inf  # sample is singular
    res = ledoit_wolf_cc(x)
    assert np.isfinite(condition_number(res.cov))
    assert is_psd(res.cov)
    w = min_variance_constrained(psd_repair(res.cov), bounds=(0.0, 1.0))
    assert w.sum() == pytest.approx(1.0, abs=1e-8)


# ------------------------------------------------------------ input hygiene

def test_nan_returns_rejected_everywhere():
    x = RNG.normal(0, 0.01, size=(50, 3))
    x[10, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sample_cov(x)
    with pytest.raises(ValueError, match="NaN"):
        ledoit_wolf_cc(x)


def test_generate_panel_validates():
    with pytest.raises(ValueError, match="n_assets"):
        generate_panel(n_assets=0)
    with pytest.raises(ValueError, match="n_periods"):
        generate_panel(n_periods=0)


def test_generate_panel_regimes_raise_correlations():
    p = generate_panel(n_assets=6, n_periods=1000, seed=5, regimes=True)
    assert p.crisis_mask.sum() > 0
    def avg_corr(c):
        d = np.sqrt(np.diag(c))
        corr = c / np.outer(d, d)
        return (corr.sum() - len(c)) / (len(c) * (len(c) - 1))
    assert avg_corr(p.crisis_cov) > avg_corr(p.true_cov)


def test_generate_panel_moments_match_sample():
    """Long-sample check: sample moments approach the stated true moments."""
    p = generate_panel(n_assets=4, n_periods=120_000, seed=8)
    x = p.returns.to_numpy()
    se_mean = np.sqrt(np.diag(p.true_cov) / x.shape[0])
    assert np.all(np.abs(x.mean(axis=0) - p.true_mean) < 4 * se_mean)
    s = sample_cov(x)
    np.testing.assert_allclose(s, p.true_cov, rtol=0.05, atol=1e-7)


def test_generate_panel_deterministic_given_seed():
    a = generate_panel(n_assets=3, n_periods=50, seed=42).returns
    b = generate_panel(n_assets=3, n_periods=50, seed=42).returns
    pd.testing.assert_frame_equal(a, b)
