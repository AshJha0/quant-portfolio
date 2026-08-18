"""Backtest engine: hand ledger, decomposition identities, costs, no-lookahead,
walk-forward integrity, edge cases."""

import numpy as np
import pandas as pd
import pytest

import fx_pairs as fp
from fx_pairs.carry import carry_accrual
from fx_pairs.data import synthetic as syn


@pytest.fixture(scope="module")
def coint_pair():
    p1, p2, truth = syn.make_cointegrated_pair(n=1500, beta=1.0, kappa=20.0,
                                               sigma_ou=0.05, seed=9)
    return p1, p2, truth


def _signal_positions(p1, p2, form=252, window=126, entry=2.0, exit_=0.5,
                      stop=4.0):
    eg = fp.engle_granger(np.log(p1.values[:form]), np.log(p2.values[:form]))
    sp = fp.log_spread(p1, p2, eg.beta, eg.alpha)
    z = fp.zscore(sp, window=window)
    pos, trades = fp.generate_positions(z, entry=entry, exit_=exit_, stop=stop)
    return pos, trades, eg


class TestHandLedger:
    def test_three_trade_ledger_exact(self):
        """Every booked number checked against a hand-built ledger:
        3 position changes, log-return spot P&L, linear carry with lagged
        rates and ACT/365 gaps, half-spread costs on both legs."""
        idx = pd.bdate_range("2020-01-06", periods=6)  # Mon..Fri + Mon
        p1 = pd.Series([1.2000, 1.2100, 1.1900, 1.2050, 1.1980, 1.2020], index=idx)
        p2 = pd.Series([0.7500, 0.7550, 0.7480, 0.7530, 0.7490, 0.7510], index=idx)
        pos = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0, 0.0], index=idx)
        beta = 0.8
        rates = {"rb1": 0.05, "rq1": 0.01, "rb2": 0.02, "rq2": 0.01}
        res = fp.run_backtest(p1, p2, pos, beta, pip_spread_1=1.0,
                              pip_spread_2=2.0, rates=rates,
                              carry_method="linear", notional=1.0)

        lp1, lp2 = np.log(p1.values), np.log(p2.values)
        dts = np.array([0, 1, 1, 1, 1, 3]) / 365.0  # Fri->Mon = 3 days
        n1 = pos.values
        n2 = pos.values * beta
        spot_exp = np.zeros(6)
        carry_exp = np.zeros(6)
        cost_exp = np.zeros(6)
        for t in range(1, 6):
            spot_exp[t] = n1[t - 1] * (lp1[t] - lp1[t - 1]) \
                - n2[t - 1] * (lp2[t] - lp2[t - 1])
            carry_exp[t] = n1[t - 1] * (0.05 - 0.01) * dts[t] \
                - n2[t - 1] * (0.02 - 0.01) * dts[t]
        for t in range(6):
            prev1 = n1[t - 1] if t > 0 else 0.0
            prev2 = n2[t - 1] if t > 0 else 0.0
            hs1 = 0.5 * 1.0 * 1e-4 / p1.values[t]
            hs2 = 0.5 * 2.0 * 1e-4 / p2.values[t]
            cost_exp[t] = -(abs(n1[t] - prev1) * hs1 + abs(n2[t] - prev2) * hs2)

        assert np.allclose(res.spot_pnl.values, spot_exp, atol=1e-15)
        assert np.allclose(res.carry_pnl.values, carry_exp, atol=1e-18)
        assert np.allclose(res.cost_pnl.values, cost_exp, atol=1e-18)
        assert np.allclose(res.total_pnl.values,
                           spot_exp + carry_exp + cost_exp, atol=1e-15)
        # three position changes -> exactly three cost hits
        assert int((res.cost_pnl < 0).sum()) == 3


class TestIdentities:
    def test_total_equals_spot_plus_carry_plus_cost(self, coint_pair):
        p1, p2, _ = coint_pair
        pos, trades, eg = _signal_positions(p1, p2)
        rates = {"rb1": 0.04, "rq1": 0.01, "rb2": 0.02, "rq2": 0.01}
        res = fp.run_backtest(p1, p2, pos, eg.beta, pip_spread_1=0.7,
                              pip_spread_2=1.0, rates=rates)
        lhs = res.total_pnl.values
        rhs = (res.spot_pnl + res.carry_pnl + res.cost_pnl).values
        assert np.allclose(lhs, rhs, atol=1e-18, rtol=0)
        assert res.cost_pnl.max() <= 0.0

    def test_carry_backtest_equals_spot_backtest_plus_carry_ledger(self, coint_pair):
        """Identity: switching carry on changes nothing except adding the
        independently computed carry ledger."""
        p1, p2, _ = coint_pair
        pos, _, eg = _signal_positions(p1, p2)
        rates = {"rb1": 0.06, "rq1": 0.01, "rb2": 0.02, "rq2": 0.015}
        res_no = fp.run_backtest(p1, p2, pos, eg.beta, pip_spread_1=0.7,
                                 pip_spread_2=1.0)
        res_c = fp.run_backtest(p1, p2, pos, eg.beta, pip_spread_1=0.7,
                                pip_spread_2=1.0, rates=rates)
        accr1 = carry_accrual(0.06, 0.01, p1.index)
        accr2 = carry_accrual(0.02, 0.015, p1.index)
        ledger = np.zeros(len(p1))
        ledger[1:] = pos[:-1] * accr1[1:] - pos[:-1] * eg.beta * accr2[1:]
        assert np.allclose(res_c.total_pnl.values,
                           res_no.total_pnl.values + ledger, atol=1e-18)
        assert np.allclose(res_c.spot_pnl.values, res_no.spot_pnl.values,
                           atol=0, rtol=0)
        assert np.allclose(res_c.cost_pnl.values, res_no.cost_pnl.values,
                           atol=0, rtol=0)


class TestCosts:
    def test_costs_scale_with_notional(self, coint_pair):
        p1, p2, _ = coint_pair
        pos, _, eg = _signal_positions(p1, p2)
        r1 = fp.run_backtest(p1, p2, pos, eg.beta, pip_spread_1=1.0,
                             pip_spread_2=1.0, notional=1.0)
        r5 = fp.run_backtest(p1, p2, pos, eg.beta, pip_spread_1=1.0,
                             pip_spread_2=1.0, notional=5.0)
        assert np.allclose(r5.cost_pnl.values, 5.0 * r1.cost_pnl.values,
                           atol=1e-18)

    def test_costs_scale_with_pip_spread(self, coint_pair):
        p1, p2, _ = coint_pair
        pos, _, eg = _signal_positions(p1, p2)
        r1 = fp.run_backtest(p1, p2, pos, eg.beta, pip_spread_1=1.0,
                             pip_spread_2=1.0)
        r2 = fp.run_backtest(p1, p2, pos, eg.beta, pip_spread_1=2.0,
                             pip_spread_2=2.0)
        assert np.allclose(r2.cost_pnl.values, 2.0 * r1.cost_pnl.values,
                           atol=1e-18)

    def test_hedge_leg_cost_scales_with_beta(self, coint_pair):
        p1, p2, _ = coint_pair
        pos, _, eg = _signal_positions(p1, p2)
        # cost only on leg 2: compare beta and 2*beta
        ra = fp.run_backtest(p1, p2, pos, 1.0, pip_spread_1=0.0,
                             pip_spread_2=1.0)
        rb = fp.run_backtest(p1, p2, pos, 2.0, pip_spread_1=0.0,
                             pip_spread_2=1.0)
        assert np.allclose(rb.cost_pnl.values, 2.0 * ra.cost_pnl.values,
                           atol=1e-18)

    def test_em_costs_kill_a_strategy_majors_survive(self, coint_pair):
        """Cost sensitivity: identical signal path, major spreads (<1 pip)
        vs EM spreads (30-60 pips): the EM version flips to a loss."""
        p1, p2, _ = coint_pair
        pos, _, eg = _signal_positions(p1, p2)
        major = fp.run_backtest(p1, p2, pos, eg.beta, pair1="AUDUSD",
                                pair2="NZDUSD", pip_spread_1=0.7,
                                pip_spread_2=1.0)
        em = fp.run_backtest(p1, p2, pos, eg.beta, pair1="USDZAR",
                             pair2="USDMXN", pip_spread_1=60.0,
                             pip_spread_2=30.0)
        assert major.total_pnl.sum() > 0.0
        assert em.total_pnl.sum() < 0.0
        # identical gross spot P&L, only costs differ
        assert np.allclose(major.spot_pnl.values, em.spot_pnl.values)


class TestNoLookahead:
    def test_perturbing_future_prices_leaves_past_pnl_unchanged(self, coint_pair):
        """Detector: rebuild signals+backtest on prices perturbed after day k;
        P&L through k must be bit-identical."""
        p1, p2, _ = coint_pair
        k = 800

        def full_run(pa, pb):
            pos, _, eg = _signal_positions(pa, pb)
            rates = {"rb1": 0.04, "rq1": 0.01, "rb2": 0.02, "rq2": 0.01}
            return fp.run_backtest(pa, pb, pos, eg.beta, pip_spread_1=0.7,
                                   pip_spread_2=1.0, rates=rates)

        base = full_run(p1, p2)
        p1_pert = p1.copy()
        p2_pert = p2.copy()
        p1_pert.iloc[k + 1:] *= 1.5  # violent future shock
        p2_pert.iloc[k + 1:] *= 0.7
        pert = full_run(p1_pert, p2_pert)
        assert np.array_equal(base.total_pnl.values[: k + 1],
                              pert.total_pnl.values[: k + 1])
        assert np.array_equal(base.positions.values[: k + 1],
                              pert.positions.values[: k + 1])


class TestWalkForward:
    def test_window_integrity(self, coint_pair):
        p1, p2, _ = coint_pair
        wf = fp.walk_forward_backtest(p1, p2, formation=252, trading=63,
                                      pip_spread_1=0.7, pip_spread_2=1.0)
        prev_end = None
        for w in wf.windows:
            assert w.f1 - w.f0 == 252          # formation length
            assert w.f1 == w.t0                # formation ends where trading starts
            assert w.t1 > w.t0                 # non-empty trading window
            if prev_end is not None:
                assert w.t0 == prev_end        # trading windows tile, no gaps
            prev_end = w.t1
        assert wf.windows[0].t0 == 252
        assert wf.windows[-1].t1 == len(p1)

    def test_flat_at_window_boundaries(self, coint_pair):
        p1, p2, _ = coint_pair
        wf = fp.walk_forward_backtest(p1, p2, formation=252, trading=63,
                                      pip_spread_1=0.7, pip_spread_2=1.0)
        for w in wf.windows:
            assert wf.positions.iloc[w.t1 - 1] == 0.0

    def test_skips_noncointegrated_windows(self):
        p1, p2 = syn.make_correlated_walks(n=800, rho=0.9, seed=100)
        wf = fp.walk_forward_backtest(p1, p2, formation=252, trading=63,
                                      require_coint=True, coint_level="5%",
                                      pip_spread_1=1.0, pip_spread_2=1.0)
        skipped = [w for w in wf.windows if not w.traded]
        assert len(skipped) > 0
        for w in skipped:
            assert np.all(wf.positions.iloc[w.t0:w.t1] == 0.0)

    def test_sample_too_short_raises(self, coint_pair):
        p1, p2, _ = coint_pair
        with pytest.raises(ValueError):
            fp.walk_forward_backtest(p1.iloc[:200], p2.iloc[:200],
                                     formation=252, trading=63)


class TestEdgeCases:
    def test_zero_trade_backtest_is_all_zero(self, coint_pair):
        p1, p2, _ = coint_pair
        pos = np.zeros(len(p1))
        res = fp.run_backtest(p1, p2, pos, 1.0, pip_spread_1=1.0,
                              pip_spread_2=1.0,
                              rates={"rb1": 0.05, "rq1": 0.01,
                                     "rb2": 0.02, "rq2": 0.01})
        assert np.all(res.total_pnl.values == 0.0)
        assert np.all(res.equity.values == 0.0)
        summ = fp.summarize(res)
        assert np.isnan(summ["sharpe"])
        assert summ["n_trades"] == 0

    def test_missing_days_handled(self, coint_pair):
        """Gappy calendar (holidays / missing data): engine runs, carry
        accrues over the actual calendar gaps."""
        p1, p2, _ = coint_pair
        keep = np.ones(len(p1), dtype=bool)
        keep[np.arange(50, len(p1), 97)] = False  # knock out scattered days
        p1g, p2g = p1[keep], p2[keep]
        pos, _, eg = _signal_positions(p1g, p2g)
        res = fp.run_backtest(p1g, p2g, pos, eg.beta, pip_spread_1=0.7,
                              pip_spread_2=1.0,
                              rates={"rb1": 0.05, "rq1": 0.01,
                                     "rb2": 0.01, "rq2": 0.01})
        assert np.isfinite(res.total_pnl.values).all()
        # always-long unit carry over the gapped index accrues calendar time
        led = fp.carry_ledger(pd.Series(1.0, index=p1g.index), 0.05, 0.01,
                              method="linear")
        total_days = (p1g.index[-1] - p1g.index[0]).days
        assert led.sum() == pytest.approx(0.04 * total_days / 365.0, rel=1e-12)

    def test_input_validation(self, coint_pair):
        p1, p2, _ = coint_pair
        pos = np.zeros(len(p1))
        with pytest.raises(ValueError, match="index"):
            fp.run_backtest(p1, p2.iloc[:-1], pos, 1.0)
        with pytest.raises(ValueError, match="length"):
            fp.run_backtest(p1, p2, pos[:-1], 1.0)
        with pytest.raises(ValueError, match="pip"):
            fp.run_backtest(p1, p2, pos, 1.0, pip_spread_1=-1.0)
        with pytest.raises(ValueError, match="notional"):
            fp.run_backtest(p1, p2, pos, 1.0, notional=0.0)
        with pytest.raises(ValueError, match="rates"):
            fp.run_backtest(p1, p2, pos, 1.0, rates={"rb1": 0.05})
        bad = pos.copy()
        bad[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            fp.run_backtest(p1, p2, bad, 1.0)
