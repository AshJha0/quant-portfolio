"""Walk-forward: window hygiene, frozen parameters, gating behaviour."""

import numpy as np
import pytest
from eq_pairs.backtest import (
    ZERO_COSTS,
    WalkForwardWindow,
    walk_forward_pair,
    walk_forward_portfolio,
    walk_forward_windows,
)
from eq_pairs.data import cointegrated_pair, correlated_random_walks, mixed_panel


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
        for a, b in zip(wins[:-1], wins[1:], strict=False):
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

    def test_partial_last_window_is_kept_on_request(self):
        n = 252 + 63 * 4 + 51
        full = walk_forward_windows(n, formation=252, trading=63)
        part = walk_forward_windows(n, formation=252, trading=63, allow_partial=True)
        assert len(full) == 4 and len(part) == 5
        assert part[-1].trading_end == n - 1
        assert part[-1].trading_start == full[-1].trading_end + 1

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
        """Mutation test. Scramble the prices inside one trading window: the
        parameters recorded for that window (fitted on its formation window)
        must not move, and the P&L of every earlier window must not move.
        A re-estimate inside the trading window, or any use of future data,
        would change one of them."""
        df, _ = self._pair()
        res, windows = walk_forward_pair(
            df["Y"], df["X"], formation=252, trading=63, costs=ZERO_COSTS
        )
        traded = windows[windows["traded"]]
        assert len(traded) >= 2
        row = traded.iloc[1]
        mutated = df.copy()
        rng = np.random.default_rng(0)
        mask = (mutated.index >= row["trading_start"]) & (mutated.index <= row["trading_end"])
        mutated.loc[mask, "Y"] *= np.exp(rng.normal(0.0, 0.05, mask.sum()))
        res2, windows2 = walk_forward_pair(
            mutated["Y"], mutated["X"], formation=252, trading=63, costs=ZERO_COSTS
        )
        k = row.name
        for col in ("beta", "alpha", "mu", "stat_std", "half_life"):
            assert windows2.loc[k, col] == windows.loc[k, col]
        before = res.daily.index < row["trading_start"]
        np.testing.assert_array_equal(
            res.daily.loc[before, "net_pnl"].to_numpy(),
            res2.daily.loc[res2.daily.index < row["trading_start"], "net_pnl"].to_numpy(),
        )
        # and the mutation did reach the trading window
        win = slice(row["trading_start"], row["trading_end"])
        assert not np.allclose(res.daily.loc[win, "net_pnl"].to_numpy(),
                               res2.daily.loc[win, "net_pnl"].to_numpy())

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
        _res, windows = walk_forward_pair(
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
        prices, _truth = mixed_panel(n=100, seed=94)
        with pytest.raises(ValueError, match="too short"):
            walk_forward_portfolio(prices, [("CO0_Y", "CO0_X")], formation=252, trading=63)


class TestDynamicHedgeAndScreens:
    """The rolling-OLS and RLS hedge ratios and the SSD screen are wired into
    walk_forward_portfolio and reported by the pipeline."""

    def _panel(self):
        from eq_pairs.universe import candidate_pairs

        prices, truth = mixed_panel(n=900, seed=95)
        return prices, candidate_pairs(list(prices.columns), truth.sectors)

    @pytest.mark.parametrize("hedge", ["rolling", "rls"])
    def test_dynamic_hedge_is_causal(self, hedge):
        """Scrambling the last 40 days of the sample must not change any
        position decided before them."""
        prices, pairs = self._panel()
        port, _ = walk_forward_portfolio(prices, pairs, formation=252, trading=63,
                                         costs=ZERO_COSTS, max_pairs=4, hedge=hedge)
        cut = prices.index[-40]
        mutated = prices.copy()
        rng = np.random.default_rng(1)
        mutated.loc[cut:] *= np.exp(rng.normal(0, 0.05, mutated.loc[cut:].shape))
        port2, _ = walk_forward_portfolio(mutated, pairs, formation=252, trading=63,
                                          costs=ZERO_COSTS, max_pairs=4, hedge=hedge)
        assert port is not None and port2 is not None
        for a, b in zip(port.pairs, port2.pairs, strict=False):
            assert a.name == b.name
            before = a.daily.index < cut
            np.testing.assert_array_equal(a.daily.loc[before, "position"],
                                          b.daily.loc[b.daily.index < cut, "position"])

    def test_hedge_variants_trade_differently(self):
        prices, pairs = self._panel()
        res = {
            h: walk_forward_portfolio(prices, pairs, formation=252, trading=63,
                                      costs=ZERO_COSTS, max_pairs=4, hedge=h)[0]
            for h in ("static", "rolling", "rls")
        }
        pnl = {h: r.net_pnl for h, r in res.items()}
        assert len({round(v, 6) for v in pnl.values()}) == 3

    def test_ssd_screen_keeps_top_n(self):
        prices, pairs = self._panel()
        _, rec = walk_forward_portfolio(prices, pairs, formation=252, trading=63,
                                        costs=ZERO_COSTS, screen="ssd", ssd_top_n=5)
        assert (rec["n_corr_survivors"] == 5).all()

    def test_bad_options_raise(self):
        prices, pairs = self._panel()
        with pytest.raises(ValueError, match="hedge"):
            walk_forward_portfolio(prices, pairs, hedge="kalman")
        with pytest.raises(ValueError, match="screen"):
            walk_forward_portfolio(prices, pairs, screen="cluster")
        with pytest.raises(ValueError, match="hedge_window"):
            walk_forward_portfolio(prices, pairs, hedge="rolling", hedge_window=5)
