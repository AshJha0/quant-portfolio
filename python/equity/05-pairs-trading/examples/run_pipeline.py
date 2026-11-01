"""End-to-end pairs-trading pipeline on synthetic data.

Every number in README.md and docs/VALIDATION.md comes from this script.
All data is synthetic: the generator plants cointegrated pairs with known
half-lives, and the edge reported here is that planted mean reversion.

Sections
--------
1. One seeded panel (seed 7), shown step by step: selection funnel, OU fit,
   in-sample backtest, walk-forward portfolio with per-pair attribution,
   execution-timing and impact sensitivity, regime-break case study.
2. Trial log on seed 7: every walk-forward configuration evaluated
   (3 hedge methods x 3 entry thresholds), with the deflated Sharpe ratio of
   the headline configuration against that trial count and the probability
   of backtest overfitting (CSCV) across the nine.
3. Multi-seed distribution: the headline numbers over ``--seeds`` seeds
   (default 30) as mean [95% CI].

Usage: python examples/run_pipeline.py [--seeds N]
The script prints only; it writes no files.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# The fits are thousands of tiny least-squares problems; BLAS threading only
# adds contention. Set before numpy is imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from eq_pairs.backtest import (
    ZERO_COSTS,
    CostModel,
    ExecutionConfig,
    backtest_pair,
    backtest_portfolio,
    walk_forward_portfolio,
)
from eq_pairs.cointegration import engle_granger
from eq_pairs.data import mixed_panel, regime_break_pair
from eq_pairs.metrics import summary
from eq_pairs.research_stats import (
    TrialLog,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    pbo_cscv,
    seed_distribution,
    sharpe_ratio,
)
from eq_pairs.signals import SignalRules, generate_signals, time_stop_bars, zscore_ou
from eq_pairs.spread import compute_spread, fit_ou_mle, fit_ou_ols
from eq_pairs.universe import candidate_pairs, correlation_screen

GROSS = 1_000_000.0  # gross $ per pair position
MAX_PAIRS = 6  # most pairs the walk-forward book may hold at once
CAPITAL = GROSS * MAX_PAIRS  # capital base for every return figure: $6mm
COSTS = CostModel(
    cost_bps=5.0,
    slippage_bps=2.0,
    borrow_bps=50.0,
    impact_coef=0.7,  # square-root law, mid-range of published 0.5-1
    adv_dollars=20_000_000.0,  # assumed ADV per name (synthetic data has no volume)
    financing_bps=50.0,  # long-leg financing spread
    short_rebate_bps=0.0,  # no rebate on short proceeds (conservative)
)
EXEC = ExecutionConfig()  # next mode, f = 0.5
MIN_CORR = 0.60
EG_LEVEL = "5%"
N_PANEL = 1500
SHOW_SEED = 7
ENTRY_GRID = (1.5, 2.0, 2.5)
HEDGES = ("static", "rolling", "rls")


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def wf(prices, cands, **kw):
    args = dict(formation=252, trading=63, max_pairs=MAX_PAIRS, min_corr=MIN_CORR,
                costs=COSTS, gross_per_pair=GROSS, execution=EXEC, allow_partial=True)
    args.update(kw)
    return walk_forward_portfolio(prices, cands, **args)


def wf_summary(port):
    if port is None:
        return None
    ledger = pd.concat([p.ledger for p in port.pairs], ignore_index=True)
    trades = pd.concat([p.trades for p in port.pairs], ignore_index=True)
    return summary(port.daily, trades, ledger, CAPITAL)


def returns_of(port, index):
    if port is None:
        return pd.Series(0.0, index=index)
    return (port.daily["net_pnl"] / CAPITAL).reindex(index).fillna(0.0)


def funnel(prices, truth):
    """Correlation screen then Engle-Granger. Each regression is oriented in
    candidate order (sorted tickers); ground truth is used only to label rows."""
    cands = candidate_pairs(list(prices.columns), truth.sectors)
    corr_surv = correlation_screen(prices, cands, min_corr=MIN_CORR)
    rows, survivors = [], []
    for a, b in corr_surv.index:
        eg = engle_granger(prices[a].to_numpy(), prices[b].to_numpy())
        t = truth.pairs.get((a, b)) or truth.pairs.get((b, a))
        rows.append({
            "pair": f"{a}/{b}",
            "ret_corr": round(float(corr_surv.loc[[(a, b)], "corr"].iloc[0]), 3),
            "eg_adf_stat": round(eg.stat, 2),
            "eg_crit_5%": round(eg.crit["5%"], 2),
            "cointegrated@5%": eg.cointegrated(EG_LEVEL),
            "truth": t.kind if t is not None else "none",
        })
        if eg.cointegrated(EG_LEVEL):
            survivors.append((a, b, eg))
    return cands, corr_surv, pd.DataFrame(rows).set_index("pair"), survivors


def in_sample(prices, survivors, costs=COSTS):
    targets, betas = {}, {}
    for a, b, eg in survivors:
        ou = fit_ou_ols(eg.resid)
        if not ou.mean_reverting:
            continue
        s = compute_spread(prices[a], prices[b], eg.beta, eg.alpha)
        rules = SignalRules(max_holding=time_stop_bars(ou.half_life, k=3.0))
        targets[(a, b)] = generate_signals(zscore_ou(s, ou), rules)["position"]
        betas[(a, b)] = eg.beta
    if not targets:
        return None
    return backtest_portfolio(prices, targets, betas, costs=costs, gross_per_pair=GROSS,
                              execution=EXEC)


def regime_break(seed):
    brk, bt = regime_break_pair(n=N_PANEL, break_frac=0.5, seed=seed)
    k = bt.break_index
    y, x = brk["Y"], brk["X"]
    eg = engle_granger(y.iloc[:k].to_numpy(), x.iloc[:k].to_numpy())
    ou = fit_ou_ols(eg.resid)
    z = zscore_ou(compute_spread(y, x, eg.beta, eg.alpha), ou)
    rules = SignalRules(max_holding=time_stop_bars(ou.half_life, k=3.0))
    res = backtest_pair(y, x, generate_signals(z, rules)["position"], beta=eg.beta,
                        costs=COSTS, gross=GROSS, name="BRK", execution=EXEC)
    naive_rules = SignalRules(entry_z=2.0, exit_z=0.0, stop_z=100.0, max_holding=None)
    naive = backtest_pair(y, x, generate_signals(z, naive_rules)["position"], beta=eg.beta,
                          costs=COSTS, gross=GROSS, name="BRK-naive", execution=EXEC)
    d = y.index[k]
    return {
        "eg": eg, "ou": ou, "truth": bt, "res": res, "break_date": d,
        "pre": res.daily.loc[:d, "net_pnl"].sum(),
        "post": res.daily.loc[d:, "net_pnl"].sum(),
        "naive_post": naive.daily.loc[d:, "net_pnl"].sum(),
    }


def show_seed(seed: int) -> None:
    hr(f"1) SYNTHETIC PANEL (seed={seed})")
    prices, truth = mixed_panel(n=N_PANEL, seed=seed)
    print(f"panel: {prices.shape[1]} names x {prices.shape[0]} days, "
          f"{prices.index[0].date()} .. {prices.index[-1].date()}")
    print(f"truth: {len(truth.cointegrated_pairs())} planted cointegrated pairs, "
          f"{len(truth.trap_pairs())} correlated-random-walk traps, "
          f"{len(truth.break_pairs())} regime-break pair, 4 idiosyncratic names")
    print(f"capital base for returns: ${CAPITAL:,.0f} (= ${GROSS:,.0f} gross per pair x "
          f"{MAX_PAIRS} pair slots)")

    hr("2) SELECTION FUNNEL (orientation from ticker order, truth only labels rows)")
    cands, corr_surv, tab, survivors = funnel(prices, truth)
    print(tab.to_string())
    print(f"\nfunnel: {len(cands)} same-sector candidates -> {len(corr_surv)} pass "
          f"correlation screen (rho >= {MIN_CORR}) -> {len(survivors)} pass "
          f"Engle-Granger at {EG_LEVEL}")
    traps = tab[tab["truth"] == "correlated_rw"]
    print(f"traps through the correlation screen: {len(traps)}; rejected by EG: "
          f"{int((~traps['cointegrated@5%']).sum())}")

    hr("3) OU FIT ON SURVIVORS (truth shown in the regression's orientation)")
    rows = []
    for a, b, eg in survivors:
        ou, ou_mle = fit_ou_ols(eg.resid), fit_ou_mle(eg.resid)
        t, flip = truth.pairs.get((a, b)), False
        if t is None and (b, a) in truth.pairs:
            t, flip = truth.pairs[(b, a)], True
        rows.append({
            "pair": f"{a}/{b}",
            "beta_true": round((1.0 / t.beta) if flip else t.beta, 3) if t else np.nan,
            "beta_est": round(eg.beta, 3),
            "HL_true_d": round(t.half_life, 1) if t else np.nan,
            "HL_ols_d": round(ou.half_life, 1),
            "HL_mle_d": round(ou_mle.half_life, 1),
        })
    print(pd.DataFrame(rows).set_index("pair").to_string())
    print("(beta_true for a reversed regression is 1/beta of the planted one, "
          "a first-order approximation)")

    hr("4) IN-SAMPLE BACKTEST (pairs selected and fitted on the full sample)")
    port = in_sample(prices, survivors)
    free = in_sample(prices, survivors, costs=ZERO_COSTS)
    ledger = pd.concat([p.ledger for p in port.pairs], ignore_index=True)
    trades = pd.concat([p.trades for p in port.pairs], ignore_index=True)
    m_free = summary(free.daily, trades, ledger, CAPITAL)
    m_cost = summary(port.daily, trades, ledger, CAPITAL)
    print(pd.DataFrame({"no_costs": m_free, "with_costs": m_cost}).round(4).to_string())
    print("\nper-pair attribution (with costs):")
    print(port.attribution().round(0).to_string())

    hr("5) WALK-FORWARD PORTFOLIO (formation 252d -> trading 63d, out of sample)")
    wf_port, rec = wf(prices, cands)
    print(rec.to_string(index=False))
    m_wf = wf_summary(wf_port)
    print("\nwalk-forward metrics (with costs):")
    print(pd.Series(m_wf).round(4).to_string())
    att = wf_port.attribution()
    print("\nper-pair attribution:")
    print(att.round(0).to_string())
    top = att["net_pnl"].iloc[0]
    print(f"\nlargest single-pair share of net P&L: {att.index[0]} "
          f"{top / att['net_pnl'].sum():.0%}")
    r = returns_of(wf_port, prices.index)
    r = r[r.index >= prices.index[252]]  # trading days only (after the first formation)
    pt, lo, hi = block_bootstrap_ci(r.to_numpy(), sharpe_ratio, n_boot=2000, seed=0)
    print(f"walk-forward Sharpe {pt:.2f}, circular block bootstrap 95% CI "
          f"[{lo:.2f}, {hi:.2f}] (block {int(np.ceil(len(r) ** (1 / 3)))} days)")

    hr("6) EXECUTION TIMING AND IMPACT (walk-forward, same seed)")
    rows = []
    for label, ex in [("next, f=0.5 (default)", EXEC),
                      ("next, f=1.0 (next close)", ExecutionConfig(fill_fraction=1.0)),
                      ("close auction, 10 bp", ExecutionConfig(mode="close_auction"))]:
        m = wf_summary(wf(prices, cands, execution=ex)[0])
        rows.append({"execution": label, "sharpe": m["sharpe"], "ann_return": m["ann_return"],
                     "cost_drag": m["cost_drag"]})
    print(pd.DataFrame(rows).set_index("execution").round(4).to_string())
    rows = []
    for y_coef in (0.0, 0.3, 0.7, 1.0):
        for g in (GROSS, 10 * GROSS):
            c = CostModel(**{**COSTS.__dict__, "impact_coef": y_coef})
            p_, _ = walk_forward_portfolio(prices, cands, formation=252, trading=63,
                                           max_pairs=MAX_PAIRS, min_corr=MIN_CORR, costs=c,
                                           gross_per_pair=g, execution=EXEC,
                                           allow_partial=True)
            ledger = pd.concat([p.ledger for p in p_.pairs], ignore_index=True)
            trades = pd.concat([p.trades for p in p_.pairs], ignore_index=True)
            m = summary(p_.daily, trades, ledger, g * MAX_PAIRS)
            rows.append({"impact_Y": y_coef, "gross_per_pair_$mm": g / 1e6,
                         "pct_ADV_per_leg": g / 2 / COSTS.adv_dollars,
                         "sharpe": m["sharpe"], "ann_return": m["ann_return"]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    hr("7) REGIME BREAK (pre-break parameters traded through the break)")
    rb = regime_break(42)
    print(f"pre-break fit: EG stat {rb['eg'].stat:.2f} (5% crit {rb['eg'].crit['5%']:.2f}); "
          f"half-life {rb['ou'].half_life:.1f}d (true {rb['truth'].half_life:.1f}d)")
    print(f"net P&L before break:                ${rb['pre']:>12,.0f}")
    print(f"net P&L after break, with stops:     ${rb['post']:>12,.0f}")
    print(f"net P&L after break, without stops:  ${rb['naive_post']:>12,.0f}")


def trial_log(seed: int) -> None:
    hr(f"8) TRIAL LOG, DEFLATED SHARPE AND PBO (seed={seed})")
    prices, truth = mixed_panel(n=N_PANEL, seed=seed)
    cands = candidate_pairs(list(prices.columns), truth.sectors)
    log = TrialLog()
    cols = {}
    for h in HEDGES:
        for ez in ENTRY_GRID:
            port, _ = wf(prices, cands, hedge=h, rule_overrides={"entry_z": ez})
            r = returns_of(port, prices.index)
            live = r[r.index >= prices.index[252]]
            cols[f"{h}/z{ez}"] = live.to_numpy()
            log.log(f"{h}/z{ez}", float(live.mean() / live.std(ddof=1)), hedge=h, entry_z=ez)
    perf = np.column_stack(list(cols.values()))
    tab = pd.DataFrame({"config": list(cols),
                        "ann_sharpe": [sharpe_ratio(c) for c in cols.values()]})
    print(tab.round(3).to_string(index=False))
    head = cols["static/z2.0"]
    dsr = deflated_sharpe_ratio(head, log.n_trials, trial_sharpes=log.sharpes())
    print(f"\ntrials logged: {log.n_trials}")
    print(f"headline config static/z2.0: deflated Sharpe ratio {dsr:.3f} "
          f"(probability the true Sharpe exceeds the best of {log.n_trials} null trials)")
    print(f"PBO (CSCV, 10 splits) across the {log.n_trials} configs: "
          f"{pbo_cscv(perf, n_splits=10):.3f}")


def multi_seed(n_seeds: int) -> None:
    hr(f"9) MULTI-SEED DISTRIBUTION ({n_seeds} seeds, synthetic panels)")
    rec: dict[str, list[float]] = {}

    def add(k, v):
        rec.setdefault(k, []).append(float(v))

    for seed in range(n_seeds):
        prices, truth = mixed_panel(n=N_PANEL, seed=1000 + seed)
        cands, _, tab, survivors = funnel(prices, truth)
        planted = set(truth.cointegrated_pairs())
        found = {(a, b) for a, b, _ in survivors} | {(b, a) for a, b, _ in survivors}
        add("planted pairs passing EG (of 4)", len(planted & found))
        traps = tab[tab["truth"] == "correlated_rw"]
        add("trap pairs passing EG", int(traps["cointegrated@5%"].sum()))
        port = in_sample(prices, survivors)
        if port is not None:
            m = summary(port.daily, pd.DataFrame(columns=["pnl", "bars_held"]),
                        pd.concat([p.ledger for p in port.pairs], ignore_index=True), CAPITAL)
            add("in-sample Sharpe", m["sharpe"])
        for label, ex in [("", EXEC), (" (f=1.0)", ExecutionConfig(fill_fraction=1.0)),
                          (" (close auction)", ExecutionConfig(mode="close_auction"))]:
            m = wf_summary(wf(prices, cands, execution=ex)[0])
            if m is None:
                continue
            add(f"walk-forward Sharpe{label}", m["sharpe"])
            if not label:
                add("walk-forward ann. return on capital", m["ann_return"])
                add("walk-forward ann. vol", m["ann_vol"])
                add("walk-forward max drawdown ($)", m["max_drawdown"])
                add("walk-forward cost drag (per yr)", m["cost_drag"])
                add("walk-forward trades", m["n_trades"])
        rb = regime_break(2000 + seed)
        add("break pair: pre-break net P&L ($)", rb["pre"])
        add("break pair: post-break net P&L, stops ($)", rb["post"])
        add("break pair: post-break net P&L, no stops ($)", rb["naive_post"])
    rows = []
    for k, v in rec.items():
        s = seed_distribution(v)
        rows.append({"metric": k, "n": s.n, "mean": s.mean, "ci95_low": s.ci_low,
                     "ci95_high": s.ci_high, "sd": s.sd})
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(pd.DataFrame(rows).set_index("metric").to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    if args.seeds < 2:
        raise SystemExit("--seeds must be at least 2")
    t0 = time.time()
    pd.set_option("display.width", 140)
    show_seed(SHOW_SEED)
    trial_log(SHOW_SEED)
    multi_seed(args.seeds)
    hr("DONE")
    print(f"total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
