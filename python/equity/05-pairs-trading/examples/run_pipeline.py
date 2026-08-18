"""End-to-end pairs-trading pipeline on the synthetic mixed panel.

Reproduces every number quoted in README.md and docs/VALIDATION.md:

1. Generate a seeded panel mixing truly cointegrated pairs, the classic
   correlated-but-NOT-cointegrated trap pairs, one regime-break pair, and
   idiosyncratic names.
2. Selection funnel: same-sector candidates -> correlation screen (returns)
   -> Engle-Granger test with MacKinnon EG critical values. Shows the trap:
   high-correlation random-walk pairs sail through the correlation screen
   and are rejected by the residual ADF.
3. OU fit on survivors: recover the known kappa / half-life.
4. In-sample backtest of the survivors with and without transaction costs.
5. Walk-forward portfolio (re-selection each formation window).
6. Regime-break case study: the pair is traded on pre-break parameters and
   loses money after cointegration dies.

Runs offline in well under 90 seconds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from eq_pairs.backtest import (
    CostModel,
    ZERO_COSTS,
    backtest_pair,
    backtest_portfolio,
    walk_forward_portfolio,
)
from eq_pairs.cointegration import engle_granger
from eq_pairs.data import mixed_panel, regime_break_pair
from eq_pairs.metrics import summary
from eq_pairs.signals import SignalRules, generate_signals, time_stop_bars, zscore_ou
from eq_pairs.spread import compute_spread, fit_ou_mle, fit_ou_ols
from eq_pairs.universe import candidate_pairs, correlation_screen

GROSS = 1_000_000.0  # gross $ per pair position
COSTS = CostModel(cost_bps=5.0, slippage_bps=2.0, borrow_bps=50.0)
MIN_CORR = 0.60
EG_LEVEL = "5%"


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    t0 = time.time()
    pd.set_option("display.width", 120)

    # ------------------------------------------------------------------ 1
    hr("1) SYNTHETIC PANEL (seed=7)")
    prices, truth = mixed_panel(n=1500, seed=7)
    print(f"panel: {prices.shape[1]} names x {prices.shape[0]} days, "
          f"{prices.index[0].date()} .. {prices.index[-1].date()}")
    print(f"truth: {len(truth.cointegrated_pairs())} cointegrated pairs, "
          f"{len(truth.trap_pairs())} correlated-random-walk traps, "
          f"{len(truth.break_pairs())} regime-break pair, 4 idiosyncratic names")

    # ------------------------------------------------------------------ 2
    hr("2) SELECTION FUNNEL: correlation screen vs cointegration test")
    cands = candidate_pairs(list(prices.columns), truth.sectors)
    corr_surv = correlation_screen(prices, cands, min_corr=MIN_CORR)
    rows = []
    eg_survivors = []
    for a, b in corr_surv.index:
        # orient the regression in the truth order (y on x) where known, so
        # the recovered beta is directly comparable to the planted one
        if (b, a) in truth.pairs:
            a, b = b, a
        eg = engle_granger(prices[a].to_numpy(), prices[b].to_numpy())
        kind = truth.pairs.get((a, b))
        rows.append(
            {
                "pair": f"{a}/{b}",
                "ret_corr": round(
                    corr_surv.loc[[(a, b) if (a, b) in corr_surv.index else (b, a)], "corr"].iloc[0],
                    3,
                ),
                "eg_adf_stat": round(eg.stat, 2),
                "eg_crit_5%": round(eg.crit["5%"], 2),
                "cointegrated@5%": eg.cointegrated(EG_LEVEL),
                "truth": kind.kind if kind is not None else "none",
            }
        )
        if eg.cointegrated(EG_LEVEL):
            eg_survivors.append((a, b, eg))
    funnel = pd.DataFrame(rows).set_index("pair")
    print(funnel.to_string())
    print(f"\nfunnel: {len(cands)} same-sector candidates "
          f"-> {len(corr_surv)} pass correlation screen (|rho| >= {MIN_CORR}) "
          f"-> {len(eg_survivors)} pass Engle-Granger at {EG_LEVEL}")
    traps_in = [r for r in rows if r["truth"] == "correlated_rw"]
    traps_rejected = [r for r in traps_in if not r["cointegrated@5%"]]
    print(f"THE TRAP: {len(traps_in)} correlated-random-walk pairs passed the "
          f"correlation screen; {len(traps_rejected)} of them were rejected by "
          f"the residual ADF test.")

    # ------------------------------------------------------------------ 3
    hr("3) OU FIT ON SURVIVORS: recovering the known half-life")
    rows = []
    fits = {}
    for a, b, eg in eg_survivors:
        ou = fit_ou_ols(eg.resid)
        ou_mle = fit_ou_mle(eg.resid)
        fits[(a, b)] = (eg, ou)
        t = truth.pairs.get((a, b))
        rows.append(
            {
                "pair": f"{a}/{b}",
                "beta_true": round(t.beta, 3) if t else np.nan,
                "beta_est": round(eg.beta, 3),
                "kappa_true": round(t.kappa, 3) if t else np.nan,
                "kappa_ols": round(ou.kappa, 3),
                "kappa_mle": round(ou_mle.kappa, 3),
                "HL_true_d": round(t.half_life, 1) if t else np.nan,
                "HL_est_d": round(ou.half_life, 1),
            }
        )
    print(pd.DataFrame(rows).set_index("pair").to_string())

    # ------------------------------------------------------------------ 4
    hr("4) IN-SAMPLE BACKTEST OF SURVIVORS: with vs without costs")
    targets, betas = {}, {}
    for (a, b), (eg, ou) in fits.items():
        s = compute_spread(prices[a], prices[b], eg.beta, eg.alpha)
        z = zscore_ou(s, ou)
        rules = SignalRules(max_holding=time_stop_bars(ou.half_life, k=3.0))
        targets[(a, b)] = generate_signals(z, rules)["position"]
        betas[(a, b)] = eg.beta
    port_free = backtest_portfolio(prices, targets, betas, costs=ZERO_COSTS,
                                   gross_per_pair=GROSS)
    port_cost = backtest_portfolio(prices, targets, betas, costs=COSTS,
                                   gross_per_pair=GROSS)
    capital = GROSS * len(targets)
    ledger_all = pd.concat([p.ledger for p in port_cost.pairs], ignore_index=True)
    trades_all = pd.concat([p.trades for p in port_cost.pairs], ignore_index=True)
    m_free = summary(port_free.daily, trades_all, ledger_all, capital)
    m_cost = summary(port_cost.daily, trades_all, ledger_all, capital)
    tab = pd.DataFrame({"no_costs": m_free, "with_costs": m_cost})
    print(tab.round(4).to_string())
    print(f"\ncost drag: net P&L falls from ${m_free['total_net_pnl']:,.0f} to "
          f"${m_cost['total_net_pnl']:,.0f} "
          f"(-${m_free['total_net_pnl'] - m_cost['total_net_pnl']:,.0f}, "
          f"{m_cost['cost_drag'] * 1e4:.0f} bps/yr on capital)")

    hr("4b) PER-PAIR ATTRIBUTION (with costs)")
    print(port_cost.attribution().round(0).to_string())

    # ------------------------------------------------------------------ 5
    hr("5) WALK-FORWARD PORTFOLIO (formation 252d -> trading 63d)")
    wf_port, wf_records = walk_forward_portfolio(
        prices, cands, formation=252, trading=63, max_pairs=6,
        min_corr=MIN_CORR, costs=COSTS, gross_per_pair=GROSS,
    )
    print(wf_records.to_string(index=False))
    if wf_port is not None:
        wf_ledger = pd.concat([p.ledger for p in wf_port.pairs], ignore_index=True)
        wf_trades = pd.concat([p.trades for p in wf_port.pairs], ignore_index=True)
        m_wf = summary(wf_port.daily, wf_trades, wf_ledger, capital)
        print("\nwalk-forward metrics (out-of-sample, with costs):")
        print(pd.Series(m_wf).round(4).to_string())
        print("\nwalk-forward per-pair attribution:")
        print(wf_port.attribution().round(0).to_string())

    # ------------------------------------------------------------------ 6
    hr("6) REGIME BREAK: trading on pre-break parameters loses after the break")
    brk, brk_truth = regime_break_pair(n=1500, break_frac=0.5, seed=42)
    k = brk_truth.break_index
    y, x = brk["Y"], brk["X"]
    eg = engle_granger(y.iloc[:k].to_numpy(), x.iloc[:k].to_numpy())
    ou = fit_ou_ols(eg.resid)
    print(f"pre-break fit (first {k} days): EG stat {eg.stat:.2f} "
          f"(5% crit {eg.crit['5%']:.2f}) -> cointegrated: {eg.cointegrated()}; "
          f"half-life {ou.half_life:.1f}d (true {brk_truth.half_life:.1f}d)")
    s = compute_spread(y, x, eg.beta, eg.alpha)
    z = zscore_ou(s, ou)
    rules = SignalRules(max_holding=time_stop_bars(ou.half_life, k=3.0))
    target = generate_signals(z, rules)["position"]
    res = backtest_pair(y, x, target, beta=eg.beta, costs=COSTS, gross=GROSS,
                        name="BRK")
    break_date = y.index[k]
    pre = res.daily.loc[: break_date, "net_pnl"].sum()
    post = res.daily.loc[break_date:, "net_pnl"].sum()
    print(f"net P&L before break: ${pre:>12,.0f}")
    print(f"net P&L after break:  ${post:>12,.0f}   <- convergence trades stop converging")
    print(f"trades entered post-break (stops + re-entry arming limit the "
          f"bleeding): {(res.trades['entry_date'] >= break_date).sum()}")
    # same book WITHOUT stops: ride the diverging spread and keep re-entering
    naive_rules = SignalRules(entry_z=2.0, exit_z=0.0, stop_z=100.0, max_holding=None)
    naive_target = generate_signals(z, naive_rules)["position"]
    naive = backtest_pair(y, x, naive_target, beta=eg.beta, costs=COSTS,
                          gross=GROSS, name="BRK-naive")
    naive_post = naive.daily.loc[break_date:, "net_pnl"].sum()
    print(f"same book WITHOUT stop-loss/time-stop, post-break: "
          f"${naive_post:>12,.0f}   <- what the kill-switch is for")

    hr("DONE")
    print(f"total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
