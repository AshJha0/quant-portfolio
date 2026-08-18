"""Metrics: hand-computed Sharpe/Sortino/MDD, Lo SE behaviour, turnover."""

import numpy as np
import pandas as pd
import pytest

from fx_pairs.metrics import (
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sharpe_se_lo,
    sortino_ratio,
    turnover,
)


class TestSharpe:
    def test_hand_computed(self):
        r = np.array([0.01, -0.005, 0.02, 0.0, 0.01])
        expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
        assert sharpe_ratio(r) == pytest.approx(expected, rel=1e-12)

    def test_zero_vol_is_nan(self):
        assert np.isnan(sharpe_ratio(np.zeros(100)))
        assert np.isnan(sharpe_ratio(np.array([0.01])))

    def test_lo_se_iid_matches_classical(self):
        """For iid returns the Lo SE collapses to sqrt((1+SR^2/2)/T)."""
        rng = np.random.default_rng(0)
        r = 0.001 + 0.01 * rng.standard_normal(5000)
        sr_ann, se = sharpe_se_lo(r, q=0)
        sr = r.mean() / r.std(ddof=1)
        classical = np.sqrt((1 + 0.5 * sr**2) / len(r)) * np.sqrt(252)
        assert se == pytest.approx(classical, rel=1e-10)

    def test_lo_se_inflates_for_autocorrelated_pnl(self):
        """Positively autocorrelated P&L (typical of slow mean-reversion books)
        must widen the Sharpe confidence band."""
        rng = np.random.default_rng(1)
        eps = rng.standard_normal(5000)
        r = np.empty(5000)
        r[0] = eps[0]
        for t in range(1, 5000):
            r[t] = 0.6 * r[t - 1] + eps[t]
        r = 0.0002 + 0.01 * r
        _, se_iid = sharpe_se_lo(r, q=0)
        _, se_lo = sharpe_se_lo(r, q=20)
        assert se_lo > 1.5 * se_iid

    def test_short_sample_nan(self):
        sr, se = sharpe_se_lo(np.ones(5))
        assert np.isnan(sr) and np.isnan(se)


class TestSortino:
    def test_hand_computed(self):
        r = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
        assert sortino_ratio(r) == pytest.approx(
            r.mean() / downside * np.sqrt(252), rel=1e-12)

    def test_no_downside_is_inf(self):
        assert sortino_ratio(np.array([0.01, 0.02, 0.005])) == np.inf


class TestDrawdown:
    def test_hand_computed(self):
        pnl = np.array([1.0, 1.0, -3.0, 1.0, -1.0])
        # equity: 1, 2, -1, 0, -1 ; peak: 1, 2, 2, 2, 2 ; max dd = 3
        assert max_drawdown(pnl) == pytest.approx(3.0, abs=1e-15)

    def test_monotone_gains_have_zero_drawdown(self):
        assert max_drawdown(np.ones(10)) == 0.0

    def test_empty_is_zero(self):
        assert max_drawdown(np.array([])) == 0.0


class TestHitRateTurnover:
    def test_hit_rate(self):
        assert hit_rate([1.0, -1.0, 2.0, 3.0]) == pytest.approx(0.75)
        assert np.isnan(hit_rate([]))

    def test_turnover_hand_computed(self):
        pos = pd.Series([0.0, 1.0, 1.0, 0.0, -1.0])
        # |dpos| = 0,1,0,1,1 -> mean 0.6 -> annualised
        assert turnover(pos) == pytest.approx(0.6 * 252, rel=1e-12)

    def test_flat_book_zero_turnover(self):
        assert turnover(np.zeros(100)) == 0.0
