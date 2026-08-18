"""Hand-checked tests for performance and downside metrics."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fx_port import (
    annualized_return,
    annualized_vol,
    excess_kurtosis,
    max_drawdown,
    sharpe_ratio,
    sharpe_se_lo,
    skewness,
    sortino_ratio,
    style_attribution,
    summary,
)


@pytest.fixture(scope="module")
def r():
    rng = np.random.default_rng(21)
    return pd.Series(0.005 * rng.standard_normal(600) + 0.0003)


def test_sharpe_hand_computed():
    x = np.array([0.01, -0.01, 0.02, 0.0])
    expected = x.mean() / x.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(x) == pytest.approx(expected, rel=1e-14)


def test_sharpe_with_rf():
    x = np.array([0.01, 0.02, 0.015, 0.005])
    rf = 0.005
    expected = (x - rf).mean() / (x - rf).std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(x, rf=rf) == pytest.approx(expected, rel=1e-14)


def test_sharpe_se_lo_formula(r):
    sr_daily = r.mean() / r.std(ddof=1)
    expected = np.sqrt((1 + 0.5 * sr_daily**2) / len(r)) * np.sqrt(252)
    assert sharpe_se_lo(r) == pytest.approx(expected, rel=1e-14)
    # SE shrinks with sample size
    assert sharpe_se_lo(r) < sharpe_se_lo(r.iloc[:100])


def test_sortino_hand_computed():
    x = np.array([0.02, -0.01, 0.01, -0.02])
    downside = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    expected = x.mean() / downside * np.sqrt(252)
    assert sortino_ratio(x) == pytest.approx(expected, rel=1e-14)


def test_max_drawdown_hand_computed():
    # curve: 1 -> 1.1 -> 0.88 -> 0.99 ; peak 1.1, trough 0.88 -> MDD = 20%
    x = np.log(np.array([1.1, 0.8, 1.125]))
    assert max_drawdown(x, log_returns=True) == pytest.approx(0.2, rel=1e-12)
    xs = np.array([0.1, -0.2, 0.1])  # simple returns version
    curve = np.cumprod(1 + xs)
    peak = np.maximum.accumulate(curve)
    assert max_drawdown(xs, log_returns=False) == pytest.approx(
        (1 - curve / peak).max(), rel=1e-14
    )


def test_skew_kurtosis_match_scipy(r):
    assert skewness(r) == pytest.approx(float(stats.skew(r)), rel=1e-10)
    assert excess_kurtosis(r) == pytest.approx(
        float(stats.kurtosis(r)), rel=1e-10
    )


def test_annualization(r):
    assert annualized_return(r) == pytest.approx(r.mean() * 252, rel=1e-14)
    assert annualized_vol(r) == pytest.approx(r.std(ddof=1) * np.sqrt(252), rel=1e-14)


def test_summary_keys_and_consistency(r):
    s = summary(r)
    expected_keys = {
        "ann_return", "ann_vol", "sharpe", "sharpe_se", "sortino",
        "max_drawdown", "skew", "excess_kurtosis", "var", "cvar",
    }
    assert set(s) == expected_keys
    assert s["sharpe"] == pytest.approx(sharpe_ratio(r), rel=1e-14)
    assert s["cvar"] >= s["var"]  # CVaR dominates VaR by construction


def test_zero_vol_raises():
    flat = np.zeros(10)
    with pytest.raises(ValueError, match="olatility|variance"):
        sharpe_ratio(flat)
    with pytest.raises(ValueError, match="variance"):
        skewness(flat)
    with pytest.raises(ValueError, match="downside|Sortino"):
        sortino_ratio(np.full(10, 0.01))


def test_empty_series_raises():
    with pytest.raises(ValueError, match="empty"):
        sharpe_ratio(np.array([]))


def test_attribution_static_weights_sums():
    idx = pd.bdate_range("2020-01-01", periods=5)
    styles = pd.DataFrame(
        {"carry": [0.01, -0.01, 0.0, 0.02, 0.01],
         "value": [0.0, 0.01, -0.01, 0.0, 0.005]}, index=idx
    )
    w = pd.Series({"carry": 0.6, "value": 0.4})
    attr = style_attribution(styles, w)
    assert np.allclose(attr["total"], attr[["carry", "value"]].sum(axis=1))
    assert np.allclose(attr["total"], styles @ w)


def test_attribution_panel_weights():
    idx = pd.bdate_range("2020-01-01", periods=4)
    styles = pd.DataFrame({"a": [0.01, 0.02, -0.01, 0.0],
                           "b": [0.0, -0.01, 0.01, 0.02]}, index=idx)
    w = pd.DataFrame({"a": [0.5, 0.6, 0.7, 0.8], "b": [0.5, 0.4, 0.3, 0.2]},
                     index=idx)
    attr = style_attribution(styles, w)
    assert np.allclose(attr["total"], (styles * w).sum(axis=1))
    with pytest.raises(ValueError, match="index"):
        style_attribution(styles, w.iloc[:-1])
