"""Funnel, carry veto, vol targeting and RLS hedge are used by reported
results (examples/run_pipeline.py); these tests pin their behaviour."""

import fx_pairs as fp
import numpy as np
import pandas as pd
import pytest
from fx_pairs.data import synthetic as syn

PAIRS = ["AUDUSD", "NZDUSD", "USDCAD", "USDJPY", "USDCHF", "AUDNZD", "AUDJPY", "CHFJPY"]


def _universe(seed, order=("SEKUSD", "NOKUSD")):
    legs, _ = syn.make_two_block_panel(n=1500, seed=seed)
    uni = pd.DataFrame({p: fp.market_price_from_legs(legs, p) for p in PAIRS})
    p1, p2, _ = syn.make_cointegrated_pair(n=1500, beta=1.0, kappa=20.0,
                                           sigma_ou=0.05, seed=seed)
    planted = {"SEKUSD": p1.values, "NOKUSD": p2.values}
    for name in order:
        uni[name] = planted[name]
    return uni


def _found(res):
    return any({r.pair_1, r.pair_2} == {"SEKUSD", "NOKUSD"} and r.passed
               for r in res.itertuples())


def test_funnel_contains_and_finds_the_planted_pair():
    # 20 seeds; the planted half-life is 8.7 days over 1500 days, so EG power
    # is essentially 1. Allow one miss.
    hits = [_found(fp.cointegration_funnel(_universe(s))) for s in range(20)]
    assert sum(hits) >= 19


def test_funnel_orientation_does_not_use_ground_truth():
    # swapping the planted columns flips the regression direction; the pair is
    # still found, and the screen output follows column order only
    res_a = fp.cointegration_funnel(_universe(3, ("SEKUSD", "NOKUSD")))
    res_b = fp.cointegration_funnel(_universe(3, ("NOKUSD", "SEKUSD")))
    assert _found(res_a) and _found(res_b)
    row_b = res_b[(res_b.pair_1 == "NOKUSD") & (res_b.pair_2 == "SEKUSD")]
    assert len(row_b) == 1


def test_funnel_rejects_bad_level():
    with pytest.raises(ValueError):
        fp.cointegration_funnel(_universe(1), level="7%")


def test_carry_veto_book_never_pays_net_carry():
    p1, p2, meta = syn.make_carry_flip_pair(seed=4)
    form = 252
    eg = fp.engle_granger(np.log(p1.values[:form]), np.log(p2.values[:form]))
    sp = fp.log_spread(p1, p2, eg.beta, eg.alpha)
    mu, sd = float(sp.iloc[:form].mean()), float(sp.iloc[:form].std())
    z = fp.zscore(sp, mu=mu, sigma=sd)
    hl = fp.fit_ou_ols(sp.values[:form]).half_life
    al, ash = fp.carry_entry_veto(z, sigma_spread=sd,
                                  carry_per_day=float(meta["carry_per_day_long"]),
                                  half_life=hl, entry=1.5, exit_=0.25,
                                  holding_multiple=6.0)
    assert al.all()            # long spread earns carry: never vetoed
    assert not ash.all()       # some short entries are vetoed
    pos, _ = fp.generate_positions(z, entry=1.5, exit_=0.25, stop=None,
                                   allow_long=al, allow_short=ash)
    pos[:form] = 0.0
    zf = z.to_numpy()
    entries = np.flatnonzero((np.diff(np.concatenate([[0.0], pos])) < 0) & (pos < 0))
    # every short entry that survived the veto clears the carry hurdle
    z_star = 0.25 + 6.0 * hl * meta["carry_per_day_long"] / sd
    assert all(zf[i] >= z_star - 1e-12 for i in entries)
    res = fp.run_backtest(p1, p2, pos, eg.beta, rates=meta["rates"])
    assert res.carry_pnl.sum() >= 0.0


def test_hedged_vol_scale_hits_target_and_is_causal():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2018-01-01", periods=1500)
    p2 = pd.Series(np.exp(np.cumsum(0.006 * rng.standard_normal(1500))), index=idx)
    p1 = pd.Series(np.exp(np.log(p2.values) + np.cumsum(0.004 * rng.standard_normal(1500))),
                   index=idx)
    sc = fp.hedged_vol_scale(p1, p2, 1.0, target_vol=0.10, lookback=63)
    assert (sc.iloc[:63] == 0).all()
    pos = np.ones(1500) * sc.to_numpy()
    res = fp.run_backtest(p1, p2, pos, 1.0, pip_spread_1=0.0, pip_spread_2=0.0)
    realised = res.total_pnl.iloc[100:].std(ddof=1) * np.sqrt(252)
    # constant 0.4%/day spread vol: the target is hit up to sampling error of
    # the 63-day vol estimate (about 1/sqrt(126) = 9% per window)
    assert realised == pytest.approx(0.10, rel=0.08)
    # causal: perturbing prices after k leaves the scale through k unchanged
    k = 800
    p1b = p1.copy()
    p1b.iloc[k + 1:] *= 1.3
    sc_b = fp.hedged_vol_scale(p1b, p2, 1.0, target_vol=0.10, lookback=63)
    pd.testing.assert_series_equal(sc.iloc[:k + 1], sc_b.iloc[:k + 1])


def test_rls_hedge_backtest_has_no_lookahead():
    p1, p2, _ = syn.make_cointegrated_pair(n=800, seed=9)
    beta = fp.RLSHedge(lam=0.995).fit_path(p2, p1)["beta"]
    z = fp.zscore(fp.log_spread(p1, p2, 1.0, 0.0), window=60)
    pos, _ = fp.generate_positions(z)
    res = fp.run_backtest(p1, p2, pos, beta, pip_spread_1=0.7, pip_spread_2=1.0)
    k = 500
    p1b = p1.copy()
    p1b.iloc[k + 1:] *= 1.05
    beta_b = fp.RLSHedge(lam=0.995).fit_path(p2, p1b)["beta"]
    np.testing.assert_array_equal(beta.values[:k + 1], beta_b.values[:k + 1])
    z_b = fp.zscore(fp.log_spread(p1b, p2, 1.0, 0.0), window=60)
    pos_b, _ = fp.generate_positions(z_b)
    res_b = fp.run_backtest(p1b, p2, pos_b, beta_b, pip_spread_1=0.7, pip_spread_2=1.0)
    np.testing.assert_array_equal(res.total_pnl.values[:k + 1],
                                  res_b.total_pnl.values[:k + 1])
