"""Portfolio layer: closed-form oracles, an independent library cross-check,
and hand-computed walk-forward P&L."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qp_bt.portfolio import (
    ledoit_wolf,
    mean_variance,
    pca_cov,
    portfolio_backtest,
    risk_contributions,
    risk_parity,
    sample_cov,
    vol_target,
)

RNG = np.random.default_rng(20261103)
A = RNG.normal(size=(4, 4))
S4 = A @ A.T / 4 + 0.05 * np.eye(4)
MU4 = np.array([0.03, 0.01, -0.02, 0.015])


def test_ledoit_wolf_matches_sklearn():
    sk = pytest.importorskip("sklearn.covariance")
    x = RNG.normal(size=(40, 6)) @ RNG.normal(size=(6, 6))
    ours, d = ledoit_wolf(x)
    theirs, d_sk = sk.ledoit_wolf(x)
    assert d == pytest.approx(d_sk, abs=1e-12)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_ledoit_wolf_intensity_in_unit_interval_and_pd():
    x = RNG.normal(size=(8, 20))  # N > T: sample cov is singular
    s, d = ledoit_wolf(x)
    assert 0 <= d <= 1
    assert np.linalg.eigvalsh(s).min() > 0


def test_pca_full_rank_recovers_sample_and_keeps_diagonal():
    x = RNG.normal(size=(100, 5))
    np.testing.assert_allclose(pca_cov(x, 5), sample_cov(x), atol=1e-12)
    c1 = pca_cov(x, 1)
    np.testing.assert_allclose(np.diag(c1), np.diag(sample_cov(x)), atol=1e-12)
    assert np.linalg.matrix_rank(c1 - np.diag(np.diag(c1))) <= 5
    with pytest.raises(ValueError):
        pca_cov(x, 0)


def test_mvo_unconstrained_is_markowitz_closed_form():
    w = mean_variance(MU4, S4, gamma=3.0)
    np.testing.assert_allclose(w, np.linalg.solve(S4, MU4) / 3.0, atol=1e-14)


def test_mvo_slsqp_path_agrees_with_closed_form_when_constraints_slack():
    w = mean_variance(MU4, S4, gamma=3.0, lower=-100, upper=100)
    np.testing.assert_allclose(w, np.linalg.solve(S4, MU4) / 3.0, atol=1e-6)  # SLSQP precision


def test_mvo_budget_long_only_and_gross():
    w = mean_variance(MU4, S4, gamma=5.0, lower=0.0, budget=1.0)
    assert w.min() >= -1e-9 and w.sum() == pytest.approx(1.0, abs=1e-9)
    g = mean_variance(MU4, S4, gamma=0.1, gross=1.5)
    assert np.abs(g).sum() <= 1.5 + 1e-6


def test_turnover_penalty_creates_no_trade_region():
    w_prev = np.linalg.solve(S4, MU4)  # optimum for gamma = 1
    # tiny forecast change: with a cost above the gradient shift, no trade
    w = mean_variance(MU4 * 1.001, S4, gamma=1.0, w_prev=w_prev, turnover_cost=1e-3)
    np.testing.assert_allclose(w, w_prev, atol=1e-7)
    # without the cost the book moves
    w0 = mean_variance(MU4 * 1.001, S4, gamma=1.0, w_prev=w_prev, turnover_cost=0.0, lower=-1e3)
    assert np.abs(w0 - w_prev).sum() > 1e-4


def test_risk_parity_diagonal_is_inverse_vol():
    s = np.diag([0.04, 0.01, 0.09])
    w = risk_parity(s)
    inv = 1 / np.sqrt(np.diag(s))
    np.testing.assert_allclose(w, inv / inv.sum(), atol=1e-12)


def test_risk_parity_equalises_contributions():
    w = risk_parity(S4)
    rc = risk_contributions(w, S4)
    np.testing.assert_allclose(rc, rc.mean(), rtol=1e-9)
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()


def test_vol_target_and_leverage_cap():
    w = np.array([0.5, 0.5, 0.0, 0.0])
    v = vol_target(w, S4, 0.1)
    assert np.sqrt(v @ S4 @ v) == pytest.approx(0.1)
    capped = vol_target(w, S4, 1e3, max_leverage=2.0)
    np.testing.assert_allclose(capped, 2 * w)


def test_portfolio_backtest_hand_computed():
    idx = pd.bdate_range("2026-01-05", periods=5)
    r = pd.DataFrame({"a": [0.01, 0.02, -0.01, 0.03, 0.00], "b": [0.00, -0.01, 0.02, 0.01, 0.05]}, index=idx)
    calls = []

    def fn(hist, w_prev):
        calls.append(len(hist))
        return np.array([1.0, 0.0]) if len(hist) == 2 else np.array([0.5, 0.5])

    res = portfolio_backtest(r, fn, est_window=2, rebalance_every=2, cost_bps=10)
    assert calls == [2, 4]  # decided on rows 1 and 3 from history only
    # row 2 holds [1, 0]; row 3 holds [1, 0]; row 4 holds [.5, .5]
    np.testing.assert_allclose(res.gross.to_numpy(), [0, 0, -0.01, 0.03, 0.025])
    np.testing.assert_allclose(res.cost.to_numpy(), [0, 0, 1e-3, 0, 1e-3])
    np.testing.assert_allclose(res.net, res.gross - res.cost, atol=0)


def test_portfolio_backtest_no_lookahead():
    r = pd.DataFrame(RNG.normal(0, 0.01, size=(80, 3)))

    def fn(hist, w_prev):
        return risk_parity(ledoit_wolf(hist.to_numpy())[0])

    a = portfolio_backtest(r, fn, est_window=30, rebalance_every=5)
    r2 = r.copy()
    r2.iloc[51:] *= -7
    b = portfolio_backtest(r2, fn, est_window=30, rebalance_every=5)
    np.testing.assert_array_equal(a.net.iloc[:51], b.net.iloc[:51])


def test_bad_inputs():
    with pytest.raises(ValueError):
        sample_cov(np.array([[np.nan, 1.0], [1.0, 2.0]]))
    with pytest.raises(ValueError):
        mean_variance(MU4, S4 + np.triu(np.ones((4, 4)), 1))
    with pytest.raises(ValueError):
        portfolio_backtest(pd.DataFrame({"a": [0.0, 0.1]}), lambda h, w: w, est_window=2)
