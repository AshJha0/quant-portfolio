"""End-to-end scenario tests: SNB floor-break P&L shape, carry sign flip,
triangular null case through the full funnel, funnel selectivity."""

import numpy as np
import pandas as pd
import pytest

import fx_pairs as fp
from fx_pairs.data import synthetic as syn


def _frozen_signal_run(p1, p2, form=252, entry=2.0, exit_=0.5, stop=None,
                       **bt_kwargs):
    """Formation-frozen z-score strategy (as a desk would run it live)."""
    eg = fp.engle_granger(np.log(p1.values[:form]), np.log(p2.values[:form]))
    sp = fp.log_spread(p1, p2, eg.beta, eg.alpha)
    mu = float(sp.iloc[:form].mean())
    sig = float(sp.iloc[:form].std())
    z = fp.zscore(sp, mu=mu, sigma=sig)
    pos, trades = fp.generate_positions(z, entry=entry, exit_=exit_, stop=stop)
    pos[:form] = 0.0
    res = fp.run_backtest(p1, p2, pos, eg.beta, trades=trades, **bt_kwargs)
    return res, eg


@pytest.fixture(scope="module")
def run():
    p1, p2, meta = syn.make_floor_then_break(seed=3)
    res, _ = _frozen_signal_run(p1, p2, form=252, stop=None,
                                pip_spread_1=1.2, pip_spread_2=1.0)
    return res, meta


@pytest.fixture(scope="module")
def runs():
    p1, p2, meta = syn.make_carry_flip_pair(seed=4)
    kwargs = dict(form=252, entry=1.5, exit_=0.25, stop=None,
                  pip_spread_1=1.0, pip_spread_2=0.5)
    p_spot, eg = _frozen_signal_run(p1, p2, **kwargs)
    p_carry, _ = _frozen_signal_run(p1, p2, rates=meta["rates"], **kwargs)
    return p_spot, p_carry


class TestFloorBreakScenario:
    """SNB 2011-2015 shape: steady gains under the floor, then one day that
    wipes out years of P&L (and then some)."""

    def test_profitable_before_break(self, run):
        res, meta = run
        bi = meta["break_idx"]
        assert res.total_pnl.iloc[:bi].sum() > 0.0

    def test_steady_gains_shape(self, run):
        """Pre-break equity: high hit-rate grind — most 63-day blocks are up."""
        res, meta = run
        bi = meta["break_idx"]
        pre = res.total_pnl.iloc[252:bi]
        blocks = [pre.iloc[i:i + 63].sum() for i in range(0, len(pre) - 63, 63)]
        assert np.mean(np.array(blocks) > 0) >= 0.7

    def test_long_the_peg_into_the_break(self, run):
        """The crowded trade: z pinned below -entry at the floor => long."""
        res, meta = run
        assert res.positions.iloc[meta["break_idx"] - 1] == 1.0

    def test_single_day_loss_exceeds_cumulative_gains(self, run):
        res, meta = run
        bi = meta["break_idx"]
        cum_gains = res.total_pnl.iloc[:bi].sum()
        break_day = res.total_pnl.iloc[bi]
        assert break_day < 0.0
        assert -break_day > cum_gains
        # the break day is roughly the engineered -15% gap on a unit position
        assert break_day == pytest.approx(meta["jump"], abs=0.02)

    def test_stop_cannot_prevent_the_gap_loss(self):
        """A z-stop triggers only AFTER the gap prints: same break-day loss."""
        p1, p2, meta = syn.make_floor_then_break(seed=3)
        # stop above the dip level so it is not hit pre-break
        res, _ = _frozen_signal_run(p1, p2, form=252, stop=25.0,
                                    pip_spread_1=1.2, pip_spread_2=1.0)
        bi = meta["break_idx"]
        assert res.total_pnl.iloc[bi] == pytest.approx(meta["jump"], abs=0.02)


class TestCarryFlipScenario:
    """A persistent 7% deposit-rate differential: the spot-only backtest loses,
    the carry-inclusive backtest of the SAME trades makes money."""

    def test_spot_only_pnl_negative(self, runs):
        p_spot, _ = runs
        assert p_spot.total_pnl.sum() < 0.0

    def test_carry_inclusive_pnl_positive(self, runs):
        _, p_carry = runs
        assert p_carry.total_pnl.sum() > 0.0

    def test_carry_component_is_the_difference(self, runs):
        p_spot, p_carry = runs
        dec = p_carry.decomposition()
        assert dec["carry"] > 0.0
        assert dec["total"] == pytest.approx(
            p_spot.total_pnl.sum() + dec["carry"], abs=1e-12)

    def test_carry_decomposition_reported(self, runs):
        _, p_carry = runs
        summ = fp.summarize(p_carry)
        assert summ["carry_pnl"] > abs(summ["spot_pnl"])


class TestTriangularNullCase:
    """Full-funnel null test: a synthetic cross and its USD-leg replication
    pass the correlation screen with corr ~ 1, then the cointegration stage
    must declare the spread degenerate — not tradable."""

    def test_funnel_flags_triangular_pair(self):
        legs, _ = syn.make_two_block_panel(n=600, seed=2)
        audjpy = fp.make_cross(legs, "AUD", "JPY")
        synth = fp.make_cross(legs, "AUD", "USD") * fp.make_cross(legs, "USD", "JPY")
        prices = pd.DataFrame({"AUDJPY": audjpy, "SYNTH": synth})
        screen = fp.correlation_screen(prices, min_abs_corr=0.5)
        assert len(screen) == 1
        assert screen.iloc[0]["corr"] == pytest.approx(1.0, abs=1e-12)
        eg = fp.engle_granger(np.log(audjpy.values), np.log(synth.values))
        assert eg.degenerate and not eg.cointegrated

    def test_degenerate_spread_has_no_tradable_zscore(self):
        legs, _ = syn.make_two_block_panel(n=600, seed=2)
        tri = fp.triangular_spread(legs, "AUD", "USD", "JPY")
        with pytest.raises(ValueError, match="zero variance"):
            fp.fit_ou_ols(tri.values)


class TestFunnelSelectivity:
    def test_planted_cointegrated_pair_beats_spurious(self):
        """Selection funnel: among a planted cointegrated pair and a merely
        correlated pair, only the planted one survives the EG stage."""
        p1, p2, _ = syn.make_cointegrated_pair(n=1000, beta=1.0, kappa=20.0,
                                               sigma_ou=0.05, seed=31)
        w1, w2 = syn.make_correlated_walks(n=1000, rho=0.9, seed=32)
        eg_good = fp.engle_granger(np.log(p1.values), np.log(p2.values))
        eg_bad = fp.engle_granger(np.log(w1.values), np.log(w2.values))
        assert eg_good.cointegrated and not eg_bad.cointegrated
        assert eg_good.stat < eg_bad.stat

    def test_walk_forward_on_carry_flip_pair_decomposition(self):
        """Walk-forward on the carry-flip pair: the carry ledger, not the spot
        leg, is what keeps the strategy alive."""
        p1, p2, meta = syn.make_carry_flip_pair(seed=4)
        wf = fp.walk_forward_backtest(p1, p2, formation=252, trading=63,
                                      entry=1.5, exit_=0.25, stop=None,
                                      require_coint=False,
                                      pip_spread_1=1.0, pip_spread_2=0.5,
                                      rates=meta["rates"])
        dec = wf.result.decomposition()
        assert dec["carry"] > 0.0
        assert dec["total"] > dec["total"] - dec["carry"]  # carry strictly helps
