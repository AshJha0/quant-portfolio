"""Tests for covariance estimators: sample, EWMA, Ledoit-Wolf, one-factor, PSD."""

import numpy as np
import pandas as pd
import pytest

from fx_port import (
    ewma_cov,
    is_psd,
    lw_shrinkage,
    one_factor_cov,
    psd_repair,
    sample_cov,
    total_log_returns,
)
from fx_port.data import make_panel


@pytest.fixture(scope="module")
def returns():
    panel = make_panel(seed=5, n_days=1000)
    return total_log_returns(panel.spots, panel.rates).total


def test_sample_cov_matches_numpy(returns):
    s = sample_cov(returns)
    expected = np.cov(returns.to_numpy().T, ddof=1)
    assert np.allclose(s.to_numpy(), expected, atol=1e-18)


def test_sample_cov_too_short_raises(returns):
    with pytest.raises(ValueError, match="at least"):
        sample_cov(returns.iloc[:1])


def test_sample_cov_nan_raises(returns):
    bad = returns.copy()
    bad.iloc[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sample_cov(bad)


def test_ewma_recursion_hand_computed():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.DataFrame({"A": [0.01, -0.02, 0.005, 0.01], "B": [0.0, 0.01, -0.01, 0.02]},
                     index=idx)
    lam = 0.9
    x = r.to_numpy()
    s = x[:2].T @ x[:2] / 2  # init window = 2
    for t in (2, 3):
        s = lam * s + (1 - lam) * np.outer(x[t], x[t])
    got = ewma_cov(r, lam=lam, init_window=2)
    assert np.allclose(got.to_numpy(), s, atol=1e-18)


def test_ewma_symmetric_psd(returns):
    s = ewma_cov(returns)
    assert np.allclose(s, s.T)
    assert is_psd(s)


def test_ewma_invalid_lambda(returns):
    with pytest.raises(ValueError, match="lam"):
        ewma_cov(returns, lam=1.0)


def test_lw_intensity_in_unit_interval(returns):
    _, delta = lw_shrinkage(returns)
    assert 0.0 <= delta <= 1.0
    _, delta_short = lw_shrinkage(returns.iloc[:30])
    assert 0.0 <= delta_short <= 1.0
    # shorter sample = noisier S = heavier shrinkage
    assert delta_short > delta


def test_lw_improves_conditioning(returns):
    short = returns.iloc[:24]  # T = 2N: sample cov is ill-conditioned
    s = sample_cov(short, ddof=0)
    lw, delta = lw_shrinkage(short)
    assert delta > 0
    cond_s = np.linalg.cond(s.to_numpy())
    cond_lw = np.linalg.cond(lw.to_numpy())
    assert cond_lw < cond_s


def test_lw_is_convex_combination(returns):
    x = returns.iloc[:100]
    lw, delta = lw_shrinkage(x)
    s = sample_cov(x, ddof=0).to_numpy()
    m = np.trace(s) / len(s)
    expected = delta * m * np.eye(len(s)) + (1 - delta) * s
    assert np.allclose(lw.to_numpy(), expected, atol=1e-18)


def test_one_factor_recovers_riskonoff_signs(returns):
    model = one_factor_cov(returns)
    for ccy in ("AUD", "NZD", "MXN", "BRL"):
        assert model.loadings[ccy] > 0, f"{ccy} should be risk-on (+)"
    for ccy in ("JPY", "CHF"):
        assert model.loadings[ccy] < 0, f"{ccy} should be safe haven (-)"


def test_one_factor_orientation_and_psd(returns):
    model = one_factor_cov(returns)
    ew = returns.sub(returns.mean()).mean(axis=1)
    assert np.corrcoef(model.factor, ew)[0, 1] > 0  # risk-on = dollar down
    assert is_psd(model.cov)
    assert (model.resid_var >= 0).all()
    assert model.factor_var > 0


def test_one_factor_explains_riskon_correlation(returns):
    model = one_factor_cov(returns)
    # implied AUD-NZD correlation positive, AUD-JPY negative
    c = model.cov
    corr = c.loc["AUD", "NZD"] / np.sqrt(c.loc["AUD", "AUD"] * c.loc["NZD", "NZD"])
    assert corr > 0.05
    corr_jpy = c.loc["AUD", "JPY"] / np.sqrt(c.loc["AUD", "AUD"] * c.loc["JPY", "JPY"])
    assert corr_jpy < 0


def test_psd_repair_fixes_indefinite():
    bad = pd.DataFrame(
        [[1.0, 0.99, 0.0], [0.99, 1.0, 0.99], [0.0, 0.99, 1.0]],
        index=list("ABC"), columns=list("ABC"),
    )
    assert not is_psd(bad, tol=1e-12)
    fixed = psd_repair(bad)
    assert is_psd(fixed)
    # Higham projection: PSD input passes through unchanged
    good = pd.DataFrame(np.eye(3), index=list("ABC"), columns=list("ABC"))
    assert np.allclose(psd_repair(good), good, atol=1e-14)


def test_psd_repair_floor_makes_invertible():
    panel = make_panel(seed=3, n_days=300, include_peg=True)
    ret = total_log_returns(panel.spots, panel.rates).total
    s = sample_cov(ret)
    floored = psd_repair(s, min_eig=1e-10)
    assert np.linalg.eigvalsh(floored.to_numpy()).min() >= 1e-10 * (1 - 1e-6)
    np.linalg.solve(floored.to_numpy(), np.ones(len(floored)))  # must not raise


def test_psd_repair_negative_floor_raises(returns):
    with pytest.raises(ValueError, match="min_eig"):
        psd_repair(sample_cov(returns), min_eig=-1.0)


def test_lw_labels_preserved(returns):
    lw, _ = lw_shrinkage(returns)
    assert list(lw.index) == list(returns.columns)
    assert list(lw.columns) == list(returns.columns)
