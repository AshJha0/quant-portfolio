"""Tests for eq_port.covariance: sample/EWMA/Ledoit-Wolf/single-factor
estimators, PSD repair and conditioning diagnostics."""

import numpy as np
import pytest

from eq_port.covariance import (
    condition_number,
    ewma_cov,
    is_psd,
    ledoit_wolf_cc,
    psd_repair,
    sample_cov,
    single_factor_cov,
)

RNG = np.random.default_rng(42)


@pytest.fixture()
def panel() -> np.ndarray:
    return RNG.normal(0.0003, 0.012, size=(300, 5))


# --------------------------------------------------------------------- sample

def test_sample_cov_matches_numpy(panel):
    np.testing.assert_allclose(sample_cov(panel), np.cov(panel.T, ddof=1), atol=1e-16)


def test_sample_cov_ddof0_matches_numpy(panel):
    np.testing.assert_allclose(
        sample_cov(panel, ddof=0), np.cov(panel.T, ddof=0), atol=1e-16
    )


def test_sample_cov_too_few_obs_raises():
    with pytest.raises(ValueError, match="observations"):
        sample_cov(np.zeros((1, 3)))


# ----------------------------------------------------------------------- EWMA

def test_ewma_recursion_identity(panel):
    """S_T must equal lam * S_{T-1} + (1-lam) r_T r_T' exactly."""
    lam = 0.94
    s_prev = ewma_cov(panel[:-1], lam=lam)
    s_full = ewma_cov(panel, lam=lam)
    expected = lam * s_prev + (1 - lam) * np.outer(panel[-1], panel[-1])
    np.testing.assert_allclose(s_full, expected, atol=1e-18)


def test_ewma_single_observation_is_outer_product():
    r = np.array([[0.01, -0.02]])
    np.testing.assert_allclose(ewma_cov(r), np.outer(r[0], r[0]))


def test_ewma_is_psd(panel):
    assert is_psd(ewma_cov(panel))


def test_ewma_invalid_lambda_raises():
    for lam in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="lam"):
            ewma_cov(np.zeros((5, 2)), lam=lam)


# ---------------------------------------------------------------- Ledoit-Wolf

def test_lw_intensity_in_unit_interval():
    for seed in range(6):
        x = np.random.default_rng(seed).normal(0, 0.01, size=(40, 10))
        assert 0.0 <= ledoit_wolf_cc(x).intensity <= 1.0


def test_lw_improves_conditioning_when_n_close_to_t():
    x = RNG.normal(0, 0.01, size=(12, 10))  # T barely above N
    res = ledoit_wolf_cc(x)
    assert res.intensity > 0.0
    assert condition_number(res.cov) < condition_number(res.sample)


def test_lw_works_when_sample_is_singular():
    x = RNG.normal(0, 0.01, size=(6, 10))  # T < N: sample cov singular
    res = ledoit_wolf_cc(x)
    assert condition_number(res.sample) == np.inf
    assert 0.0 < res.intensity <= 1.0
    assert np.isfinite(condition_number(res.cov))
    assert is_psd(res.cov)


def test_lw_degenerate_single_asset_returns_sample():
    x = RNG.normal(0, 0.01, size=(50, 1))
    res = ledoit_wolf_cc(x)
    assert res.intensity == 0.0
    np.testing.assert_allclose(res.cov, res.sample)


def test_lw_shrunk_is_stated_convex_combination(panel):
    res = ledoit_wolf_cc(panel)
    expected = res.intensity * res.target + (1 - res.intensity) * res.sample
    np.testing.assert_allclose(res.cov, expected, atol=1e-18)


def test_lw_preserves_variances(panel):
    """Target and sample share the diagonal, so shrinkage keeps variances."""
    res = ledoit_wolf_cc(panel)
    np.testing.assert_allclose(np.diag(res.cov), np.diag(res.sample), atol=1e-18)


def test_lw_target_is_constant_correlation(panel):
    res = ledoit_wolf_cc(panel)
    d = np.sqrt(np.diag(res.target))
    corr = res.target / np.outer(d, d)
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    np.testing.assert_allclose(off, off[0], atol=1e-12)


def test_lw_shrinks_more_with_less_data():
    # factor-structured data: the truth differs from the constant-correlation
    # target, so the intensity decays toward 0 as T grows (delta ~ kappa/T)
    from eq_port.data import generate_panel

    x = generate_panel(n_assets=8, n_periods=4000, seed=3).returns.to_numpy()
    short = ledoit_wolf_cc(x[:40]).intensity
    long = ledoit_wolf_cc(x).intensity
    assert short > long


# -------------------------------------------------------------- single-factor

def test_single_factor_recovers_true_one_factor_cov():
    rng = np.random.default_rng(7)
    t, n = 60_000, 4
    beta = np.array([0.8, 1.0, 1.2, 0.9])
    m = rng.normal(0, 0.01, size=t)
    eps = rng.normal(0, 0.005, size=(t, n))
    x = np.outer(m, beta) + eps
    est = single_factor_cov(x, market=m)
    true = 0.01**2 * np.outer(beta, beta) + np.eye(n) * 0.005**2
    np.testing.assert_allclose(est, true, atol=5e-6)


def test_single_factor_is_psd(panel):
    assert is_psd(single_factor_cov(panel))


def test_single_factor_validates(panel):
    with pytest.raises(ValueError, match="market"):
        single_factor_cov(panel, market=np.zeros(10))
    with pytest.raises(ValueError, match="zero variance"):
        single_factor_cov(panel, market=np.zeros(panel.shape[0]))


# ----------------------------------------------------------------- PSD repair

def test_psd_repair_fixes_indefinite_matrix():
    a = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, 0.9], [0.7, 0.9, 0.6]])
    assert not is_psd(a)
    fixed = psd_repair(a)
    assert is_psd(fixed)
    assert np.allclose(fixed, fixed.T)


def test_psd_repair_nearly_preserves_psd_input(panel):
    s = sample_cov(panel)
    np.testing.assert_allclose(psd_repair(s), s, rtol=1e-8, atol=1e-12)


def test_psd_repair_makes_singular_invertible():
    ones = np.ones((4, 4))  # rank-1, perfectly correlated
    fixed = psd_repair(ones, eps=1e-8)
    assert np.isfinite(condition_number(fixed))
    np.linalg.cholesky(fixed)  # must not raise


def test_psd_repair_validates_shape():
    with pytest.raises(ValueError, match="square"):
        psd_repair(np.zeros((2, 3)))


# ---------------------------------------------------------------- diagnostics

def test_condition_number_identity_is_one():
    assert condition_number(np.eye(5)) == pytest.approx(1.0)


def test_condition_number_singular_is_inf():
    assert condition_number(np.ones((3, 3))) == np.inf


def test_is_psd_detects_negative_eigenvalue():
    a = np.diag([1.0, -0.1])
    assert not is_psd(a)
    assert is_psd(np.diag([1.0, 0.0]))
