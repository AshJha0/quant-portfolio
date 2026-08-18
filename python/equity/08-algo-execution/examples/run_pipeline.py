"""End-to-end demo: daily alpha pipeline + intraday execution pipeline.

Run from the project root:  python examples/run_pipeline.py   (< 120 s)

Sections
--------
1. Alpha layer: planted-alpha panel -> features -> ICs with Newey-West
   t-stats -> decay table -> decile monotonicity -> long-short backtest
   gross vs net of costs -> deflated Sharpe (N tried signals).
2. Capacity: cost-adjusted Sharpe vs AUM.
3. Execution layer: 5%-ADV parent order via TWAP / VWAP / POV / AC on the
   simulator, slippage vs benchmarks over 200 seeded replications, AC
   efficient frontier, TCA report for one order.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import eq_algo as ea

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

T0 = time.time()
SEED = 42


def hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# 1. Alpha layer
# ---------------------------------------------------------------------------
hdr("1. ALPHA LAYER — planted-alpha panel (150 stocks x 1250 days)")
panel = ea.generate_daily_panel(n_stocks=150, n_days=1250, seed=SEED)
prices, volumes = panel.prices, panel.volumes
fwd1 = ea.forward_returns(prices, 1)

features = {
    "mom_12_1": ea.momentum(prices, 252, 21),
    "mom_6_1": ea.momentum(prices, 126, 21),
    "reversal_1m": ea.short_term_reversal(prices, 21),
    "vol_63d": -ea.realized_vol(prices, 63),          # low-vol tilt
    "ma_20_100": ea.ma_crossover(prices, 20, 100),
    "rsi_14": -ea.rsi(prices, 14),                    # overbought -> fade
    "turnover_z": ea.turnover_zscore(volumes, 63),
}

print("\nFeature ICs (1-day forward, Spearman), Newey-West t-stats:")
rows = {}
for name, f in features.items():
    ic = ea.information_coefficient(ea.winsorize(f, 0.01, 0.99), fwd1)
    s = ea.ic_summary(ic, lags=5)
    rows[name] = {k: s[k] for k in ("mean_ic", "ic_std", "tstat_nw", "icir_annual", "n_obs")}
ic_table = pd.DataFrame(rows).T
print(ic_table.to_string())

hdr("Signal combination + decay")
combined = ea.combine_ic_weighted(
    {k: features[k] for k in ("mom_12_1", "mom_6_1", "reversal_1m")}, fwd1)
decay = ea.signal_decay(combined, prices, horizons=[1, 2, 3, 5, 10, 15, 20])
print("\nIC decay of the combined signal (mean IC vs horizon):")
print(decay.to_string())

dec = ea.decile_portfolios(combined, fwd1, n_quantiles=10)
dec_means = dec.mean() * 1e4
print("\nDecile mean 1-day forward returns (bps), Q1=low ... Q10=high:")
print(dec_means.to_string())
print(f"Monotonicity (Spearman rho of decile means): "
      f"{ea.quantile_monotonicity(dec_means[[f'Q{i}' for i in range(1, 11)]]):.3f}")

hdr("Long-short backtest — gross vs net of costs")
AUM = 200e6
cfg_net = ea.BacktestConfig(gross_exposure=2.0, max_weight=0.04, linear_cost_bps=5.0,
                            impact_coef=0.1, aum=AUM, rebalance_band=0.25)
res = ea.run_backtest(prices, combined, cfg_net, volumes=volumes)
live = res.ledger.iloc[300:]  # after signal warm-up
gross = ea.perf_summary(live["gross_ret"], live["turnover"])
net = ea.perf_summary(live["net_ret"], live["turnover"])
print(f"\nAUM ${AUM/1e6:.0f}m, 5 bps linear + sqrt impact, 0.25-z rebalance band")
summary = pd.DataFrame({"gross": gross, "net": net})
print(summary.to_string())
drag = (live["gross_ret"] - live["net_ret"]).mean() * 252 * 1e4
print(f"Annualised cost drag: {drag:.0f} bps  "
      f"(mean daily turnover {live['turnover'].mean():.2%})")

hdr("Deflated Sharpe — guarding against N tried signals")
print("\nMarginal single-feature strategy (MA 20/100 crossover, net of 5 bps):")
res_ma = ea.run_backtest(prices, ea.cs_zscore(features["ma_20_100"]),
                         ea.BacktestConfig(linear_cost_bps=5.0))
ma_net = res_ma.ledger["net_ret"].iloc[300:]
for n_trials in (1, 7, 45):
    d = ea.deflated_sharpe_ratio(ma_net, n_trials=n_trials)
    print(f"N={n_trials:3d} tried:  SR_annual={d['sr_annual']:.2f}  "
          f"E[max noise SR]={d['sr_benchmark']*np.sqrt(252):.2f} (ann)  "
          f"PSR0={d['psr0']:.4f}  DSR={d['dsr']:.4f}")
print("(An SR that survives PSR vs 0 can fail once you admit having tried "
      "N variants.)")

# ---------------------------------------------------------------------------
# 2. Capacity
# ---------------------------------------------------------------------------
hdr("2. CAPACITY — cost-adjusted Sharpe vs AUM")
adv_med = float(panel.adv_dollars.median())
cap = ea.capacity_curve(live["gross_ret"], float(live["turnover"].mean()),
                        aum_grid=[10e6, 50e6, 100e6, 250e6, 500e6, 1e9, 2e9, 5e9],
                        adv_dollars=adv_med, n_names=30, sigma_daily=0.02,
                        linear_cost_bps=5.0, impact_coef=0.1)
cap["daily_cost_drag"] = cap["daily_cost_drag"] * 1e4
cap = cap.rename(columns={"daily_cost_drag": "drag_bps_per_day"})
cap.index = [f"${a/1e6:,.0f}m" for a in cap.index]
print(f"\nMedian name ADV ${adv_med/1e6:.1f}m; turnover spread over ~30 names/day")
print(cap.to_string())

# ---------------------------------------------------------------------------
# 3. Execution layer
# ---------------------------------------------------------------------------
hdr("3. EXECUTION — 5% ADV parent order, TWAP / VWAP / POV / AC")
icfg = ea.IntradayConfig(mid0=100.0, day_volume=1_000_000, n_buckets=26,
                         sigma_daily=0.02, spread_bps=5.0, temp_coef=1.0,
                         perm_coef=0.5, vol_noise=0.2)
mkt = ea.IntradayMarket(icfg)
X = 0.05 * icfg.day_volume  # 50,000 shares = 5% ADV

# AC calibrated to the simulator: sigma in currency/sqrt(day); eta such that
# eta * (X/T) matches the sqrt-law cost at TWAP participation; gamma equals
# the simulator's permanent move per share (perm_coef*sigma*mid0/V_day).
acp = ea.ACParams(total_shares=X, n_slices=icfg.n_buckets,
                  sigma=icfg.sigma_daily * icfg.mid0, eta=2.0e-6, gamma=1e-6,
                  epsilon=icfg.mid0 * icfg.spread_bps * 1e-4 / 2)
lam_mid = 5e-6
schedules = {
    "TWAP": ea.twap_schedule(X, icfg.n_buckets),
    "VWAP": ea.vwap_schedule(X, icfg.profile),
    "POV 10%": ea.pov_schedule(X, icfg.profile * icfg.day_volume, 0.10),
    f"AC lam={lam_mid:g}": ea.ac_trades(acp, lam_mid),
    "Aggressive (2 bkts)": np.r_[np.full(2, X / 2), np.zeros(icfg.n_buckets - 2)],
}
tab = ea.evaluate_schedules(mkt, schedules, side=1, n_reps=200, seed=SEED)
print("\nSlippage vs benchmarks, 200 seeded replications (bps, +=cost):")
print(tab.to_string())

hdr("Almgren-Chriss efficient frontier (3 risk aversions)")
front = ea.efficient_frontier(acp, [1e-6, 5e-6, 5e-5])
front["exp_cost_bps"] = front["expected_cost"] / (X * icfg.mid0) * 1e4
front["std_bps"] = front["std"] / (X * icfg.mid0) * 1e4
front.index = [f"{l:.0e}" for l in front.index]
print(front[["kappa", "exp_cost_bps", "std_bps"]].to_string())
print("(lambda up -> expected cost up, variance down: the risk/cost trade-off)")

hdr("TCA report — one TWAP order (decision price 99.80, arrival 100.00)")
res1 = mkt.execute(schedules["TWAP"], side=1, seed=1, decision_price=99.80)
rep = ea.tca_report(res1)
print(f"\nfilled {rep.filled_qty:,.0f}/{rep.parent_qty:,.0f} @ avg "
      f"{rep.avg_fill_price:.4f}; final mid {rep.final_price:.4f}")
print(pd.Series(rep.bps()).to_string())
print("\nPer-share attribution (currency), TOTAL row qty-weighted:")
print(ea.slippage_attribution(res1).loc[["TOTAL"]].to_string())

agg = ea.aggregate_tca([ea.tca_report(mkt.execute(schedules['TWAP'], side=1,
                                                  seed=1000 + i, decision_price=99.80))
                        for i in range(50)])
print("\nAggregate TCA over 50 TWAP orders (bps):")
print(agg.to_string())

print(f"\nDone in {time.time() - T0:.1f}s")
