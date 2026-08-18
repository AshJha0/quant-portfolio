"""Signals: z-score, state machine, vol targeting, carry-aware entry filter."""

import numpy as np
import pandas as pd
import pytest

from fx_pairs.data import synthetic as syn
from fx_pairs.signals import (
    carry_entry_veto,
    generate_positions,
    vol_target_scale,
    zscore,
)


class TestZScore:
    def test_rolling_matches_pandas(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.standard_normal(300))
        z = zscore(s, window=50)
        m = s.rolling(50).mean()
        sd = s.rolling(50).std(ddof=1)
        assert np.allclose(z.dropna().values, ((s - m) / sd).dropna().values,
                          atol=1e-12)
        assert z.iloc[:49].isna().all()

    def test_frozen_stats_formula(self):
        s = pd.Series([1.0, 2.0, 3.0])
        z = zscore(s, mu=2.0, sigma=0.5)
        assert np.allclose(z.values, [-2.0, 0.0, 2.0], atol=1e-15)

    def test_invalid_args_raise(self):
        s = pd.Series(np.arange(10.0))
        with pytest.raises(ValueError):
            zscore(s)
        with pytest.raises(ValueError):
            zscore(s, window=50, mu=0.0, sigma=1.0)
        with pytest.raises(ValueError):
            zscore(s, mu=0.0, sigma=0.0)


class TestStateMachine:
    def test_long_entry_and_exit(self):
        z = pd.Series([0.0, -2.5, -1.0, -0.4, 0.0])
        pos, trades = generate_positions(z, entry=2.0, exit_=0.5)
        assert list(pos) == [0, 1, 1, 0, 0]
        assert len(trades) == 1
        assert trades[0].side == 1 and trades[0].exit_reason == "exit"
        assert (trades[0].entry, trades[0].exit) == (1, 3)

    def test_short_entry_and_exit(self):
        z = pd.Series([0.0, 2.5, 1.0, 0.3, 0.0])
        pos, trades = generate_positions(z, entry=2.0, exit_=0.5)
        assert list(pos) == [0, -1, -1, 0, 0]
        assert trades[0].side == -1

    def test_exit_on_overshoot_through_mean(self):
        z = pd.Series([0.0, -2.5, 0.8, 0.9])
        pos, trades = generate_positions(z, entry=2.0, exit_=0.5)
        assert list(pos) == [0, 1, 0, 0]
        assert trades[0].exit_reason == "exit"

    def test_stop_loss(self):
        z = pd.Series([0.0, -2.5, -3.0, -4.5, -1.0])
        pos, trades = generate_positions(z, entry=2.0, exit_=0.5, stop=4.0)
        assert list(pos) == [0, 1, 1, 0, 0]
        assert trades[0].exit_reason == "stop"

    def test_time_stop(self):
        z = pd.Series([0.0, -2.5, -1.5, -1.5, -1.5, -1.5])
        pos, trades = generate_positions(z, entry=2.0, exit_=0.5, stop=None,
                                         max_holding=2)
        assert list(pos) == [0, 1, 1, 0, 0, 0]
        assert trades[0].exit_reason == "time"

    def test_open_position_closed_as_eod(self):
        z = pd.Series([0.0, -2.5, -1.5])
        _, trades = generate_positions(z, entry=2.0, exit_=0.5)
        assert trades[-1].exit_reason == "eod"

    def test_nan_warmup_forces_flat(self):
        z = pd.Series([np.nan, np.nan, -3.0, -0.1])
        pos, _ = generate_positions(z, entry=2.0, exit_=0.5)
        assert list(pos) == [0, 0, 1, 0]

    def test_never_flips_without_flat_bar(self):
        """Property: positions never jump -1 <-> +1 directly."""
        rng = np.random.default_rng(1)
        z = pd.Series(3.5 * np.sin(np.arange(500) / 3.0)
                      + rng.standard_normal(500))
        pos, _ = generate_positions(z, entry=2.0, exit_=0.5, stop=4.0)
        jumps = np.abs(np.diff(pos))
        assert jumps.max() <= 1.0

    def test_entry_permission_masks_gate_entries_only(self):
        z = pd.Series([0.0, -2.5, -2.5, -0.1])
        allow_long = np.array([True, False, True, True])
        pos, trades = generate_positions(z, entry=2.0, exit_=0.5,
                                         allow_long=allow_long)
        # entry vetoed at t=1, taken at t=2
        assert list(pos) == [0, 0, 1, 0]

    def test_invalid_thresholds_raise(self):
        z = pd.Series(np.zeros(10))
        with pytest.raises(ValueError):
            generate_positions(z, entry=1.0, exit_=1.5)
        with pytest.raises(ValueError):
            generate_positions(z, entry=2.0, exit_=0.5, stop=1.5)
        with pytest.raises(ValueError):
            generate_positions(z, entry=2.0, exit_=0.5, max_holding=0)


class TestVolTarget:
    def test_scale_hits_target_on_constant_vol(self):
        s = pd.Series(syn.simulate_ou(2000, 20.0, 0.0, 0.05, seed=5))
        scale = vol_target_scale(s, target_vol=0.10, lookback=63)
        realised_ann = s.diff().std() * np.sqrt(252)
        assert scale.dropna().median() == pytest.approx(0.10 / realised_ann,
                                                        rel=0.15)

    def test_scale_is_capped(self):
        s = pd.Series(np.concatenate([syn.simulate_ou(200, 20, 0, 0.05, seed=6),
                                      np.zeros(200)]))
        s.iloc[200:] = s.iloc[199]  # vol collapses to zero (peg)
        scale = vol_target_scale(s, target_vol=0.10, lookback=20,
                                 max_leverage=10.0)
        assert scale.max() <= 10.0 + 1e-12

    def test_invalid_args_raise(self):
        s = pd.Series(np.arange(100.0))
        with pytest.raises(ValueError):
            vol_target_scale(s, target_vol=0.0)
        with pytest.raises(ValueError):
            vol_target_scale(s, lookback=1)


class TestCarryFilter:
    def test_adverse_carry_vetoes_marginal_entry(self):
        """Short-spread entries are vetoed when the position would pay more
        carry over the expected holding than the reversion could earn."""
        z = pd.Series([2.1])  # marginal short signal
        # long-spread carry +40bp/day (huge): a SHORT pays it
        allow_long, allow_short = carry_entry_veto(
            z, sigma_spread=0.01, carry_per_day=0.004, half_life=10.0,
            entry=2.0, exit_=0.5)
        assert allow_long[0]
        assert not allow_short[0]

    def test_favourable_carry_never_vetoed(self):
        z = pd.Series([-2.1, 2.1])
        allow_long, allow_short = carry_entry_veto(
            z, sigma_spread=0.01, carry_per_day=0.0, half_life=10.0)
        assert allow_long.all() and allow_short.all()

    def test_deep_signal_survives_moderate_drag(self):
        """A very stretched entry (large expected reversion) is kept even with
        adverse carry; the marginal one is skipped."""
        z = pd.Series([2.05, 6.0])
        _, allow_short = carry_entry_veto(
            z, sigma_spread=0.01, carry_per_day=2e-3, half_life=10.0,
            entry=2.0, exit_=0.5)
        assert not allow_short[0]
        assert allow_short[1]

    def test_filtered_trades_are_subset(self):
        """The carry filter only removes trades, never adds or alters others'
        entries."""
        rng = np.random.default_rng(7)
        z = pd.Series(2.8 * np.sin(np.arange(400) / 5.0)
                      + 0.5 * rng.standard_normal(400))
        allow_long, allow_short = carry_entry_veto(
            z, sigma_spread=0.01, carry_per_day=3e-4, half_life=15.0)
        pos_all, trades_all = generate_positions(z, entry=2.0, exit_=0.5)
        pos_f, trades_f = generate_positions(z, entry=2.0, exit_=0.5,
                                             allow_long=allow_long,
                                             allow_short=allow_short)
        assert len(trades_f) <= len(trades_all)
        # every long trade in the filtered set exists at an allowed entry bar
        for tr in trades_f:
            if tr.side == 1:
                assert allow_long[tr.entry]
            else:
                assert allow_short[tr.entry]

    def test_infinite_half_life_disables_veto(self):
        z = pd.Series([2.5, -2.5])
        allow_long, allow_short = carry_entry_veto(
            z, sigma_spread=0.01, carry_per_day=1.0, half_life=np.inf)
        assert allow_long.all() and allow_short.all()

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            carry_entry_veto(pd.Series([1.0]), sigma_spread=0.0,
                             carry_per_day=0.0, half_life=10.0)
