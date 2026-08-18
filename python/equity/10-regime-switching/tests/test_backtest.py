"""Backtest tests: exact ledger, benchmarks, and the full-pipeline
no-lookahead mutation test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq_regime.backtest import (
    ma_timing_weights,
    run_ledger,
    summary_stats,
    walk_forward_backtest,
)
from eq_regime.data import make_regime_panel

BT_KWARGS = dict(
    n_states=2, min_train=300, refit_every=200, cost_bps=5.0, seed=0,
    n_pca=2, detect_kwargs=dict(n_init=1, max_iter=40),
)


@pytest.fixture(scope="module")
def bt_panel():
    return make_regime_panel(n_states=2, n_assets=5, n_days=1000, seed=21)


@pytest.fixture(scope="module")
def bt_result(bt_panel):
    return walk_forward_backtest(bt_panel.prices, **BT_KWARGS)


class TestLedgerExact:
    def test_hand_computed_scenario(self):
        """3 weights, 4 returns, 10bps: every ledger cell checked by hand."""
        idx = pd.bdate_range("2021-01-01", periods=5)
        rets = pd.Series([0.0, 0.01, -0.02, 0.03, 0.005], index=idx)
        weights = pd.Series([1.0, 0.5, 0.5], index=idx[:3])
        led = run_ledger(weights, rets, cost_bps=10.0)
        c = 10.0 / 1e4
        # day 2: w=1 earns r=0.01, cost = c*|1-0| (entry)
        # day 3: w=0.5 earns r=-0.02, cost = c*0.5
        # day 4: w=0.5 earns r=0.03, cost = 0
        exp_net = np.array([1.0 * 0.01 - c * 1.0, 0.5 * -0.02 - c * 0.5, 0.5 * 0.03])
        np.testing.assert_allclose(led["net_ret"].to_numpy(), exp_net, atol=1e-15)
        np.testing.assert_allclose(
            led["equity"].to_numpy(), np.cumprod(1 + exp_net), atol=1e-15
        )
        np.testing.assert_array_equal(led.index, idx[1:4])
        np.testing.assert_allclose(led["gross_ret"], [0.01, -0.01, 0.015])
        np.testing.assert_allclose(led["cost"], [c, 0.5 * c, 0.0])

    def test_zero_cost_equity_is_weighted_compounding(self):
        idx = pd.bdate_range("2021-01-01", periods=4)
        rets = pd.Series([0.0, 0.02, 0.02, 0.02], index=idx)
        weights = pd.Series([1.0, 1.0, 1.0], index=idx[:3])
        led = run_ledger(weights, rets, cost_bps=0.0)
        assert led["equity"].iloc[-1] == pytest.approx(1.02**3)

    def test_last_weight_without_next_return_dropped(self):
        idx = pd.bdate_range("2021-01-01", periods=3)
        rets = pd.Series([0.0, 0.01, 0.01], index=idx)
        weights = pd.Series([1.0, 1.0, 1.0], index=idx)  # last has no next day
        led = run_ledger(weights, rets, cost_bps=0.0)
        assert len(led) == 2

    def test_validation(self):
        idx = pd.bdate_range("2021-01-01", periods=3)
        rets = pd.Series([0.0, 0.01, 0.01], index=idx)
        w = pd.Series([1.0, 1.0], index=idx[:2])
        with pytest.raises(ValueError, match="cost_bps"):
            run_ledger(w, rets, cost_bps=-1.0)
        with pytest.raises(ValueError, match="at least 2"):
            run_ledger(w.iloc[:1], rets)
        w_bad = pd.Series([1.0, 1.0], index=pd.bdate_range("1999-01-01", periods=2))
        with pytest.raises(ValueError, match="not contained"):
            run_ledger(w_bad, rets)


class TestNoLookahead:
    def test_full_pipeline_mutation(self, bt_panel):
        """Mutating future prices leaves all earlier ledger rows identical:
        features, PCA, HMM fits, filtered probs, weights and P&L at t only
        ever see data up to t."""
        base = walk_forward_backtest(bt_panel.prices, **BT_KWARGS)
        mutated_prices = bt_panel.prices.copy()
        mutated_prices.iloc[800:] *= 0.5  # 50% crash in the future
        mut = walk_forward_backtest(mutated_prices, **BT_KWARGS)
        cutoff = bt_panel.prices.index[798]
        pd.testing.assert_frame_equal(
            base.ledger.loc[:cutoff], mut.ledger.loc[:cutoff], check_exact=True
        )
        pd.testing.assert_frame_equal(
            base.detection.loc[:cutoff], mut.detection.loc[:cutoff], check_exact=True
        )


class TestBenchmarksAndStats:
    def test_buy_and_hold_equity(self, bt_panel, bt_result):
        """B&H ledger equals compounded index returns net of one entry cost."""
        led = bt_result.benchmark
        prices = bt_panel.prices
        simple = np.exp(np.log(prices / prices.shift(1)).iloc[1:]).mean(axis=1) - 1.0
        np.testing.assert_allclose(
            led["gross_ret"].to_numpy(), simple.loc[led.index].to_numpy(), atol=1e-14
        )
        ratio = led["equity"].to_numpy() / np.cumprod(1 + led["net_ret"].to_numpy())
        np.testing.assert_allclose(ratio, 1.0, atol=1e-12)
        assert (led["weight"] == 1.0).all()
        assert led["cost"].iloc[0] == pytest.approx(5e-4)
        assert (led["cost"].iloc[1:] == 0).all()

    def test_ma_rule_is_causal_and_binary(self, bt_panel):
        w = ma_timing_weights(bt_panel.prices, ma_window=50)
        assert set(np.unique(w.dropna())) <= {0.0, 1.0}
        up = pd.DataFrame(
            {"a": np.linspace(100, 300, 300), "b": np.linspace(100, 300, 300)},
            index=pd.bdate_range("2020-01-01", periods=300),
        )
        wu = ma_timing_weights(up, ma_window=50)
        assert (wu.iloc[50:] == 1.0).all()

    def test_summary_stats_constructed(self):
        idx = pd.bdate_range("2021-01-01", periods=252)
        net = np.full(252, 0.001)
        led = pd.DataFrame(
            {"net_ret": net, "equity": np.cumprod(1 + net), "cost": 0.0},
            index=idx,
        )
        s = summary_stats(led)
        assert s["cagr"] == pytest.approx(1.001**252 - 1, rel=1e-10)
        assert s["max_drawdown"] == 0.0
        assert s["ann_vol"] == pytest.approx(0.0, abs=1e-12)

    def test_ledgers_share_dates(self, bt_result):
        assert bt_result.ledger.index.equals(bt_result.benchmark.index)
        assert bt_result.ledger.index.equals(bt_result.ma_rule.index)

    def test_strategy_derisks_in_bear(self, bt_result):
        """Average weight in detected bear regimes below bull regimes."""
        det = bt_result.detection
        w = bt_result.ledger["weight"]
        regime = det["regime"].reindex(w.index).shift(1).dropna()
        w = w.loc[regime.index]
        if (regime == "bear").any() and (regime == "bull").any():
            assert w[regime == "bear"].mean() < w[regime == "bull"].mean()
