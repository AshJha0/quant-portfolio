"""Walk-forward: window hygiene, frozen parameters, gating behaviour."""

import numpy as np
import pandas as pd
import pytest

from eq_pairs.backtest import (
    WalkForwardWindow,
    ZERO_COSTS,
    align_pair,
    backtest_pair,
    walk_forward_pair,
    walk_forward_portfolio,
    walk_forward_windows,
)
from eq_pairs.data import cointegrated_pair, correlated_random_walks, mixed_panel
from eq_pairs.signals import SignalRules, generate_signals, time_stop_bars
from eq_pairs.spread import compute_spread


class TestWindows:
    def test_formation_and_trading_never_overlap(self):
        wins = walk_forward_windows(1000, formation=252, trading=63)
        assert len(wins) > 0
        for w in wins:
            assert w.formation_end < w.trading_start  # the core guarantee
            assert w.formation_end - w.formation_start + 1 == 252
            assert w.trading_end - w.trading_start + 1 == 63

    def test_default_step_gives_contiguous_trading_windows(self):
        wins = walk_forward_windows(1000, formation=252, trading=63)
        for a, b in zip(wins[:-1], wins[1:]):
            assert b.trading_start == a.trading_end + 1  # no gap, no overlap

    def test_window_count(self):
        wins = walk_forward_windows(252 + 63 * 4, formation=252, trading=63)
        assert len(wins) == 4

    def test_malformed_window_raises(self):
        with pytest.raises(ValueError, match="malformed"):
            WalkForwardWindow(0, 100, 50, 150)  # trading starts inside formation

    def test_parameter_validation(self):
        with pytest.raises(ValueError, match="formation"):
            walk_forward_windows(500, formation=10, trading=63)
        with pytest.raises(ValueError, match="trading"):
            walk_forward_windows(500, formation=252, trading=1)
        with pytest.raises(ValueError, match="step"):
            walk_forward_windows(500, formation=252, trading=63, step=0)

    def test_too_short_sample_no_windows(self):
        assert walk_forward_windows(200, formation=252, trading=63) == []


class TestWalkForwardPair:
    def _pair(self):
        # fast mean reversion (half-life ~ 4.6 days) so the EG gate has
        # power on a one-year formation window
        df, truth = cointegrated_pair(
            n=1000, beta=1.4, kappa=0.15, sigma=1.0, seed=90
        )
        return df, truth

    def test_runs_and_reports_windows(self):
        df, _ = self._pair()
        res, windows = walk_forward_pair(
            df["Y"], df["X"], formation=252, trading=63, costs=ZERO_COSTS
        )
        assert len(windows) == len(walk_forward_windows(1000, 252, 63))
        assert windows["traded"].any()
        assert res is not None
        # every window row respects formation < trading ordering
        assert (windows["formation_end"] < windows["trading_start"]).all()

    def test_parameters_frozen_during_trading_window(self):
        """Reconstruct a traded window's P&L from the recorded FORMATION
        parameters only; it must match the walk-forward output exactly.
        If any parameter were re-estimated inside the trading window the
        reconstruction would diverge."""
        df, _ = self._pair()
        res, windows = walk_forward_pair(
            df["Y"], df["X"], formation=252, trading=63, costs=ZERO_COSTS
        )
        y, x = align_pair(df["Y"], df["X"])
        traded = windows[windows["traded"]]
        assert len(traded) > 0
        row = traded.iloc[0]
        y_t = y.loc[row["trading_start"] : row["trading_end"]]
        x_t = x.loc[row["trading_start"] : row["trading_end"]]
        s = compute_spread(y_t, x_t, row["beta"], row["alpha"])
        z = (s - row["mu"]) / row["stat_std"]
        rules = SignalRules(max_holding=time_stop_bars(row["half_life"], k=3.0))
        target = generate_signals(z, rules)["position"]
        recon = backtest_pair(
            y_t, x_t, target, beta=row["beta"], costs=ZERO_COSTS
        )
        got = res.daily.loc[row["trading_start"] : row["trading_end"], "net_pnl"]
        np.testing.assert_allclose(
            got.to_numpy(), recon.daily["net_pnl"].to_numpy(), atol=1e-9
        )

    def test_frozen_beta_recovers_truth(self):
        df, truth = self._pair()
        _, windows = walk_forward_pair(
            df["Y"], df["X"], formation=252, trading=63, costs=ZERO_COSTS
        )
        betas = windows.loc[windows["traded"], "beta"]
        assert len(betas) > 0
        assert np.median(np.abs(betas - truth.beta)) < 0.15

    def test_cointegrated_pair_profitable_without_costs(self):
        df, _ = self._pair()
        res, _ = walk_forward_pair(
            df["Y"], df["X"], formation=252, trading=63, costs=ZERO_COSTS
        )
        assert res is not None and res.net_pnl > 0.0

    def test_trap_pair_mostly_gated_out(self):
        df, _ = correlated_random_walks(n=1000, rho=0.92, seed=91)
        res, windows = walk_forward_pair(
            df["A"], df["B"], formation=252, trading=63, costs=ZERO_COSTS
        )
        assert windows["traded"].mean() <= 0.5  # EG gate blocks most windows

    def test_positions_closed_at_each_window_end(self):
        df, _ = self._pair()
        res, windows = walk_forward_pair(
            df["Y"], df["X"], formation=252, trading=63, costs=ZERO_COSTS
        )
        for _, row in windows[windows["traded"]].iterrows():
            end = row["trading_end"]
            if end in res.daily.index:
                assert res.daily.loc[end, "position"] == 0

    def test_sample_too_short_raises(self):
        df, _ = cointegrated_pair(n=100, seed=92)
        with pytest.raises(ValueError, match="too short"):
            walk_forward_pair(df["Y"], df["X"], formation=252, trading=63)


class TestWalkForwardPortfolio:
    def test_funnel_counts_and_aggregation(self):
        prices, truth = mixed_panel(n=800, seed=93)
        from eq_pairs.universe import candidate_pairs

        pairs = candidate_pairs(list(prices.columns), truth.sectors)
        port, records = walk_forward_portfolio(
            prices, pairs, formation=252, trading=63, costs=ZERO_COSTS,
            max_pairs=5,
        )
        assert len(records) > 0
        assert (records["n_corr_survivors"] <= records["n_candidates"]).all()
        assert (records["n_traded"] <= records["n_corr_survivors"]).all()
        assert port is not None
        assert {"net_pnl", "n_positions"} <= set(port.daily.columns)
        att = port.attribution()
        assert att["net_pnl"].sum() == pytest.approx(port.net_pnl, abs=1e-6)

    def test_too_short_raises(self):
        prices, truth = mixed_panel(n=100, seed=94)
        with pytest.raises(ValueError, match="too short"):
            walk_forward_portfolio(prices, [("CO0_Y", "CO0_X")], formation=252, trading=63)
