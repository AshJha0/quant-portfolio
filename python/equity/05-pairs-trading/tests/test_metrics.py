"""Hand-checked performance metrics and the Lo-adjusted Sharpe SE."""

import numpy as np
import pandas as pd
import pytest

from eq_pairs.metrics import (
    avg_holding_period,
    cost_drag,
    drawdown_series,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sharpe_se,
    sortino_ratio,
    turnover,
)


class TestSharpe:
    def test_hand_computed(self):
        r = np.array([0.01, 0.02, 0.03])
        # mean = 0.02, std (ddof=1) = 0.01
        assert sharpe_ratio(r) == pytest.approx(2.0 * np.sqrt(252.0), abs=1e-12)

    def test_zero_vol_is_nan(self):
        assert np.isnan(sharpe_ratio(np.full(10, 0.001)))

    def test_se_iid_formula(self):
        r = np.array([0.01, 0.02, 0.03, 0.00, 0.01, -0.01])
        sr_p = r.mean() / r.std(ddof=1)
        expected = np.sqrt((1.0 + 0.5 * sr_p**2) / len(r)) * np.sqrt(252.0)
        assert sharpe_se(r) == pytest.approx(expected, abs=1e-12)

    def test_lo_se_exceeds_iid_under_positive_autocorr(self):
        """AR(1) daily P&L (the mean-reversion book signature) must widen
        the Sharpe standard error under the Lo adjustment."""
        rng = np.random.default_rng(80)
        n, phi = 1500, 0.6
        e = rng.standard_normal(n)
        r = np.empty(n)
        r[0] = e[0]
        for t in range(1, n):
            r[t] = phi * r[t - 1] + e[t]
        r = 0.001 + 0.01 * r / r.std()
        se_iid = sharpe_se(r, lo_adjust=False)
        se_lo = sharpe_se(r, lo_adjust=True, q=20)
        assert se_lo > 1.3 * se_iid

    def test_lo_se_close_to_iid_for_white_noise(self):
        rng = np.random.default_rng(81)
        r = 0.001 + 0.01 * rng.standard_normal(2000)
        ratio = sharpe_se(r, lo_adjust=True, q=10) / sharpe_se(r, lo_adjust=False)
        assert 0.85 < ratio < 1.15

    def test_lo_q_validation(self):
        with pytest.raises(ValueError, match="q must be"):
            sharpe_se(np.array([0.01, 0.02, 0.03]), lo_adjust=True, q=10)

    def test_input_validation(self):
        with pytest.raises(ValueError, match="NaN"):
            sharpe_ratio(np.array([0.01, np.nan]))
        with pytest.raises(ValueError, match="at least"):
            sharpe_ratio(np.array([0.01]))


class TestSortino:
    def test_hand_computed(self):
        r = np.array([0.02, -0.01, 0.03, -0.02])
        downside = np.sqrt((0.01**2 + 0.02**2) / 4.0)
        expected = r.mean() / downside * np.sqrt(252.0)
        assert sortino_ratio(r) == pytest.approx(expected, abs=1e-12)

    def test_no_downside_is_nan(self):
        assert np.isnan(sortino_ratio(np.array([0.01, 0.02, 0.03])))

    def test_mar_shifts_downside(self):
        r = np.array([0.02, 0.005, 0.03, 0.01])
        assert np.isnan(sortino_ratio(r, mar=0.0))
        assert np.isfinite(sortino_ratio(r, mar=0.01))


class TestDrawdown:
    def test_hand_computed_mdd(self):
        equity = np.array([100.0, 120.0, 90.0, 130.0, 80.0])
        assert max_drawdown(equity) == pytest.approx(50.0, abs=1e-12)

    def test_drawdown_series_values(self):
        equity = np.array([100.0, 120.0, 90.0, 130.0, 80.0])
        np.testing.assert_allclose(
            drawdown_series(equity), [0.0, 0.0, -30.0, 0.0, -50.0], atol=1e-12
        )

    def test_monotone_equity_zero_mdd(self):
        assert max_drawdown(np.linspace(100, 200, 50)) == pytest.approx(0.0, abs=1e-12)


class TestTradeStats:
    def test_hit_rate(self):
        assert hit_rate([1.0, -2.0, 3.0, 0.5]) == pytest.approx(0.75)
        assert hit_rate([1.0, -1.0, 0.0]) == pytest.approx(1.0 / 3.0)

    def test_hit_rate_empty_nan(self):
        assert np.isnan(hit_rate([]))

    def test_avg_holding(self):
        assert avg_holding_period([2, 4, 6]) == pytest.approx(4.0)
        assert np.isnan(avg_holding_period([]))


class TestTurnoverAndCosts:
    def test_turnover_hand_computed(self):
        # $12mm traded over 252 days on $1mm capital -> 12x / year
        assert turnover(12_000_000.0, 1_000_000.0, 252) == pytest.approx(12.0)

    def test_cost_drag_hand_computed(self):
        # $5,000 of costs over half a year on $1mm -> 1%/yr drag
        assert cost_drag(5_000.0, 1_000_000.0, 126) == pytest.approx(0.01)

    def test_validations(self):
        with pytest.raises(ValueError, match="capital"):
            turnover(1.0, 0.0, 252)
        with pytest.raises(ValueError, match="n_days"):
            turnover(1.0, 1.0, 0)
        with pytest.raises(ValueError, match="capital"):
            cost_drag(1.0, -5.0, 252)
        with pytest.raises(ValueError, match="n_days"):
            cost_drag(1.0, 1.0, 0)
