"""Tests for eq_port.metrics: hand-checked values on tiny arrays plus
structural properties."""

import numpy as np
import pytest

from eq_port.metrics import (
    annualized_return,
    annualized_vol,
    calmar_ratio,
    diversification_ratio,
    effective_n,
    max_drawdown,
    realized_risk_contributions,
    sharpe_lo,
    sharpe_ratio,
    sortino_ratio,
    summary_table,
)
from eq_port.risk_parity import risk_contributions


# ------------------------------------------------------------ hand-checked

def test_sharpe_hand_checked():
    r = np.array([0.01, 0.03])
    expected = 0.02 / np.std(r, ddof=1) * np.sqrt(252.0)
    assert sharpe_ratio(r) == pytest.approx(expected, abs=1e-14)


def test_sharpe_with_rf_hand_checked():
    r = np.array([0.02, 0.04, 0.03])
    rf = 0.01
    ex = r - rf
    expected = ex.mean() / np.std(ex, ddof=1) * np.sqrt(252.0)
    assert sharpe_ratio(r, rf=rf) == pytest.approx(expected, abs=1e-13)


def test_sortino_hand_checked():
    r = np.array([0.02, -0.01, 0.03])
    dd = np.sqrt((0.0 + 0.01**2 + 0.0) / 3.0)
    expected = r.mean() / dd * np.sqrt(252.0)
    assert sortino_ratio(r) == pytest.approx(expected, abs=1e-12)


def test_sortino_no_downside_is_inf():
    assert sortino_ratio(np.array([0.01, 0.02])) == np.inf


def test_max_drawdown_hand_checked():
    r = np.array([0.1, -0.5, 0.2])
    # wealth: 1 -> 1.1 -> 0.55 -> 0.66 ; peak 1.1 ; trough 0.55
    assert max_drawdown(r) == pytest.approx(1.0 - 0.55 / 1.1, abs=1e-14)


def test_max_drawdown_zero_for_monotone_gains():
    assert max_drawdown(np.array([0.01, 0.02, 0.005])) == 0.0


def test_max_drawdown_counts_initial_loss():
    assert max_drawdown(np.array([-0.10, 0.02])) == pytest.approx(0.10, abs=1e-14)


def test_annualized_return_geometric():
    r = np.full(252, 0.001)
    assert annualized_return(r) == pytest.approx(1.001**252 - 1.0, rel=1e-12)


def test_annualized_vol_hand_checked():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    assert annualized_vol(r) == pytest.approx(np.std(r, ddof=1) * np.sqrt(252.0))


def test_calmar_is_return_over_drawdown():
    r = np.array([0.1, -0.5, 0.2, 0.05, 0.05])
    assert calmar_ratio(r) == pytest.approx(annualized_return(r) / max_drawdown(r))


# ------------------------------------------------------------------ Lo Sharpe

def test_lo_sharpe_equals_iid_sharpe_with_zero_lags():
    r = np.random.default_rng(0).normal(0.001, 0.01, size=500)
    res = sharpe_lo(r, n_lags=0)
    assert res.sharpe_lo == pytest.approx(res.sharpe, abs=1e-12)
    assert res.sharpe == pytest.approx(sharpe_ratio(r), abs=1e-12)
    assert res.se > 0


def test_lo_sharpe_penalizes_positive_autocorrelation():
    rng = np.random.default_rng(1)
    eps = rng.normal(0, 0.01, size=3000)
    r = np.empty_like(eps)
    r[0] = eps[0]
    for t in range(1, len(eps)):  # AR(1) with phi = 0.4 (smoothed returns)
        r[t] = 0.4 * r[t - 1] + eps[t]
    r += 0.0005
    res = sharpe_lo(r)
    assert res.sharpe_lo < res.sharpe  # sqrt-time annualisation overstates SR
    assert res.se > 0


def test_lo_sharpe_validates():
    with pytest.raises(ValueError, match="observations"):
        sharpe_lo(np.array([0.01, 0.02]))


# ------------------------------------------------------------- concentration

def test_effective_n_of_equal_weight_is_n():
    for n in (1, 4, 25):
        assert effective_n(np.full(n, 1.0 / n)) == pytest.approx(float(n), abs=1e-12)


def test_effective_n_single_asset_is_one():
    assert effective_n(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_diversification_ratio_at_least_one():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(5, 5))
    cov = a @ a.T + 5 * np.eye(5)
    for _ in range(20):
        w = rng.dirichlet(np.ones(5))
        assert diversification_ratio(w, cov) >= 1.0 - 1e-12


def test_diversification_ratio_one_for_perfect_correlation():
    vols = np.array([0.1, 0.2])
    cov = np.outer(vols, vols)  # correlation = 1
    assert diversification_ratio(np.array([0.5, 0.5]), cov) == pytest.approx(1.0)


def test_diversification_ratio_rejects_short_positions():
    with pytest.raises(ValueError, match="long-only"):
        diversification_ratio(np.array([1.5, -0.5]), np.eye(2))


# ------------------------------------------------- realized risk contributions

def test_realized_rc_sums_to_one_and_matches_sample_cov():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.01, size=(300, 4))
    w = np.array([0.4, 0.3, 0.2, 0.1])
    rc = realized_risk_contributions(w, x)
    assert rc.sum() == pytest.approx(1.0, abs=1e-12)
    s = np.cov(x.T, ddof=1)
    expected = risk_contributions(w, s)
    np.testing.assert_allclose(rc, expected / expected.sum(), atol=1e-12)


def test_realized_rc_validates():
    with pytest.raises(ValueError, match="mismatch"):
        realized_risk_contributions(np.ones(3), np.zeros((10, 4)))


# --------------------------------------------------------------------- table

def test_summary_table_structure():
    rng = np.random.default_rng(5)
    tbl = summary_table(
        {"A": rng.normal(0.001, 0.01, 300), "B": rng.normal(0.0, 0.02, 300)}
    )
    assert list(tbl.index) == ["A", "B"]
    for col in ("AnnRet", "AnnVol", "Sharpe", "SharpeSE_Lo", "Sortino", "MaxDD", "Calmar"):
        assert col in tbl.columns
