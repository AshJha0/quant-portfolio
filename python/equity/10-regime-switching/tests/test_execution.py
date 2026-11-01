"""Execution convention, financing, baselines and causal filters.

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as hst

from eq_regime.backtest import (
    ExecutionConfig,
    random_timing_weights,
    run_ledger,
)
from eq_regime.risk import regimes_known_at_decision
from eq_regime.strategy import build_weights, causal_min_duration


def _rets(values):
    return pd.Series(values, index=pd.bdate_range("2021-01-01", periods=len(values)))


class TestFillPrice:
    def test_hand_computed_next_fill(self):
        """Fill at p_t + f (p_{t+1} - p_t); every cell by hand, f = 0.5."""
        rets = _rets([0.0, 0.01, -0.02, 0.03])
        w = pd.Series([1.0, 0.5], index=rets.index[:2])
        led = run_ledger(w, rets, cost_bps=0.0, borrow_spread_bps=0.0)
        level = np.cumprod(1.0 + rets.to_numpy())  # index level p_t
        for i, d in enumerate(led.index):
            pos = rets.index.get_loc(d)
            p_t, p_next = level[pos - 1], level[pos]
            p_fill = p_t * (1.0 + led["fill_ret"].iloc[i])
            assert p_fill == pytest.approx(p_t + 0.5 * (p_next - p_t), rel=1e-14)
        b1 = 1.01 / 1.005 - 1.0
        assert led["gross_ret"].iloc[0] == pytest.approx(b1, rel=1e-14)
        a2, b2 = -0.01, 0.98 / 0.99 - 1.0
        exp2 = (1.0 + 1.0 * a2) * (1.0 + 0.5 * b2) - 1.0
        assert led["gross_ret"].iloc[1] == pytest.approx(exp2, rel=1e-14)
        assert list(led["prev_weight"]) == [0.0, 1.0]

    def test_constant_weight_earns_index_return(self):
        rets = _rets([0.0, 0.01, -0.02, 0.03, 0.004])
        w = pd.Series(1.0, index=rets.index[:4])
        led = run_ledger(w, rets, cost_bps=0.0, borrow_spread_bps=0.0)
        np.testing.assert_allclose(led["gross_ret"].iloc[1:], rets.iloc[2:], rtol=1e-13)

    def test_perfect_foresight_earns_nothing_at_next_close(self):
        """s_t = sign(r_{t+1}). With f = 1 the position decided at t is filled at
        the close of t+1 and cannot see r_{t+1}: mean gross P&L is zero within
        4 SE. A same-close fill earns |r| every day, far outside that band, so
        this test fails on any engine that fills at the close of t."""
        rng = np.random.default_rng(0)
        r = _rets(rng.normal(0.0, 0.01, 3000))
        sig = pd.Series(np.sign(r.shift(-1)).fillna(0.0).to_numpy(), index=r.index)
        next_close = run_ledger(sig, r, cost_bps=0.0, borrow_spread_bps=0.0,
                                execution=ExecutionConfig("next", fill_fraction=1.0))
        g = next_close["gross_ret"].to_numpy()
        assert abs(g.mean()) < 4.0 * g.std(ddof=1) / np.sqrt(g.size)
        same = run_ledger(sig, r, cost_bps=0.0, borrow_spread_bps=0.0,
                          execution=ExecutionConfig("close_auction", auction_cost_bp=0.0))
        gs = same["gross_ret"].to_numpy()
        assert gs.mean() > 20.0 * gs.std(ddof=1) / np.sqrt(gs.size)

    def test_default_fill_keeps_only_the_post_fill_part(self):
        """Default f = 0.5: the foresight signal earns about half of |r|."""
        rng = np.random.default_rng(1)
        r = _rets(rng.normal(0.0, 0.01, 3000))
        sig = pd.Series(np.sign(r.shift(-1)).fillna(0.0).to_numpy(), index=r.index)
        g = run_ledger(sig, r, cost_bps=0.0, borrow_spread_bps=0.0)["gross_ret"]
        full = r.abs().iloc[1:].mean()
        assert 0.4 * full < g.mean() < 0.6 * full


class TestCloseAuction:
    def test_charges_exactly_the_auction_cost(self):
        rets = _rets([0.0, 0.01, -0.02, 0.03])
        w = pd.Series([1.0, 0.25, 0.75], index=rets.index[:3])
        ex = ExecutionConfig("close_auction", auction_cost_bp=12.0)
        led = run_ledger(w, rets, cost_bps=5.0, borrow_spread_bps=0.0, execution=ex)
        dw = np.array([1.0, 0.75, 0.5])
        np.testing.assert_allclose(led["cost"], (5.0 + 12.0) / 1e4 * dw, rtol=1e-14)
        np.testing.assert_allclose(led["gross_ret"], w.to_numpy() * rets.iloc[1:].to_numpy(),
                                   rtol=1e-14)
        nxt = run_ledger(w, rets, cost_bps=5.0, borrow_spread_bps=0.0)
        np.testing.assert_allclose(led["cost"] - nxt["cost"], 12.0 / 1e4 * dw, rtol=1e-12)


class TestFinancing:
    def test_leverage_pays_and_cash_earns(self):
        rets = _rets([0.0] * 4)
        w = pd.Series([1.5, 1.5, 0.5], index=rets.index[:3])
        led = run_ledger(w, rets, cost_bps=0.0, cash_rate=0.02, borrow_spread_bps=50.0,
                         execution=ExecutionConfig("next", fill_fraction=0.5))
        borrow, cash = 0.025 / 252, 0.02 / 252
        # day 1: time-weighted weight 0.75 -> 0.25 in cash
        # day 2: 1.5 all day -> 0.5 borrowed
        # day 3: 0.5 * 1.5 + 0.5 * 0.5 = 1.0 -> nothing
        np.testing.assert_allclose(led["financing"], [0.25 * cash, -0.5 * borrow, 0.0],
                                   rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(led["net_ret"], led["financing"], rtol=1e-12, atol=1e-18)

    def test_unlevered_book_pays_nothing(self):
        rets = _rets([0.0, 0.01, 0.02])
        w = pd.Series([1.0, 1.0], index=rets.index[:2])
        led = run_ledger(w, rets, cost_bps=0.0, borrow_spread_bps=50.0,
                         execution=ExecutionConfig("close_auction", auction_cost_bp=0.0))
        assert (led["financing"] == 0.0).all()


class TestValidation:
    @pytest.mark.parametrize("f", [0.0, -0.1, 1.01, float("nan"), float("inf")])
    def test_fill_fraction(self, f):
        with pytest.raises(ValueError, match="fill_fraction"):
            ExecutionConfig("next", fill_fraction=f)

    def test_mode_and_auction(self):
        with pytest.raises(ValueError, match="mode"):
            ExecutionConfig("same_close")
        with pytest.raises(ValueError, match="auction_cost_bp"):
            ExecutionConfig("close_auction", auction_cost_bp=float("nan"))
        with pytest.raises(ValueError, match="auction_cost_bp"):
            ExecutionConfig("close_auction", auction_cost_bp=-1.0)

    def test_ledger_inputs(self):
        rets = _rets([0.0, 0.01, 0.02])
        w = pd.Series([1.0, 1.0], index=rets.index[:2])
        for kw in (dict(cash_rate=float("nan")), dict(borrow_spread_bps=-1.0),
                   dict(cost_bps=float("inf"))):
            with pytest.raises(ValueError):
                run_ledger(w, rets, **kw)
        with pytest.raises(ValueError, match="NaN"):
            run_ledger(pd.Series([1.0, np.nan], index=rets.index[:2]), rets)


class TestRandomTiming:
    def test_equal_exposure_and_same_distribution(self):
        w = pd.Series(np.r_[np.ones(60), np.zeros(40)], index=pd.RangeIndex(100))
        ref = pd.Series(np.full(100, 0.8))
        rt = random_timing_weights(w, seed=3, reference=ref)
        assert rt.mean() == pytest.approx(0.8, rel=1e-12)
        assert sorted(random_timing_weights(w, seed=3)) == sorted(w)
        assert not rt.equals(w)


class TestCausalMinDuration:
    def test_short_runs_never_reach_output(self):
        s = np.array([0, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0])
        out = causal_min_duration(s, 3)
        assert out.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]

    @settings(max_examples=60, deadline=None)
    @given(hst.lists(hst.integers(0, 2), min_size=2, max_size=60), hst.integers(1, 6),
           hst.integers(0, 59))
    def test_is_causal(self, states, k, cut):
        """Output up to t is unchanged by any change to states after t."""
        s = np.array(states)
        cut = cut % len(s)
        mutated = s.copy()
        mutated[cut + 1:] = (mutated[cut + 1:] + 1) % 3
        np.testing.assert_array_equal(causal_min_duration(s, k)[: cut + 1],
                                      causal_min_duration(mutated, k)[: cut + 1])

    def test_rejects_bad_duration(self):
        with pytest.raises(ValueError):
            causal_min_duration(np.zeros(3), 0)

    def test_wired_into_build_weights(self):
        """A two-day bear flicker moves weights without the filter, not with it."""
        idx = pd.bdate_range("2021-01-01", periods=12)
        p_bear = np.full(12, 0.05)
        p_bear[5:7] = 0.95
        regime = np.where(p_bear > 0.5, "bear", "bull")
        det = pd.DataFrame({"p_bear": p_bear, "regime": regime}, index=idx)
        lr = pd.Series(0.0, index=idx)
        raw = build_weights(det, lr, target_vol=None)
        filt = build_weights(det, lr, target_vol=None, min_duration=3)
        assert raw.iloc[5] == 0.0
        assert (filt == 1.0).all()


class TestAttributionLag:
    def test_pnl_day_gets_the_regime_of_the_decision_day(self):
        """The P&L of day t is labelled with the regime detected at t-1; a
        regime first detected on the last day labels nothing (no bfill)."""
        idx = pd.bdate_range("2021-01-01", periods=6)
        reg = pd.Series(["bull", "bull", "bear", "bull", "bull", "bear"], index=idx)
        out = regimes_known_at_decision(reg, idx)
        assert out.index.equals(idx[1:])
        assert out.loc[idx[3]] == "bear" and out.loc[idx[2]] == "bull"
        assert (out == "bear").sum() == 1
