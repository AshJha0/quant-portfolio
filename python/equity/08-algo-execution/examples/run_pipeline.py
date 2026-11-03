"""End-to-end demo on synthetic data: daily alpha pipeline + intraday execution.

Run from the project root:  python examples/run_pipeline.py
(175-247 s measured; writes nothing to disk).

Everything here runs on the seeded synthetic generator. The generator plants
next-day returns on z(momentum 12-1) and z(reversal 21d); the features the
book combines are those same regressors. The edge below is therefore a
property of the generator, not evidence of a tradable signal. No real-data
result exists yet.

Sections
--------
1. Alpha layer, one seed in detail: feature ICs with Newey-West t-stats,
   decay, deciles, every configuration logged in a TrialLog, PBO (CSCV)
   across those configurations, deflated Sharpe of the headline book at the
   logged trial count, Lo/Newey-West SE and a block-bootstrap CI.
2. Alpha layer across N_SEEDS seeds: mean [95% CI] of every headline.
3. Capacity: full backtest re-run per AUM at impact coefficients Y = 0.3,
   0.7, 1.0 with trailing per-name ADV.
4. Execution: deterministic expected implementation shortfall per schedule,
   Monte Carlo std with standard errors, AC frontier, TCA with a modelled
   decision-to-arrival move.
"""

from __future__ import annotations

import time

import eq_algo as ea
import numpy as np
import pandas as pd
from eq_algo.research_stats import (
    TrialLog,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    lo_sharpe_se,
    pbo_cscv,
    seed_distribution,
    sharpe_ratio,
)

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

T0 = time.time()
SEED = 42
N_SEEDS = 20
N_STOCKS, N_DAYS, WARMUP = 150, 1250, 300
AUM = 200e6
Y = 0.7
EXEC = ea.ExecutionConfig()                       # fill at p_t + 0.5 (p_{t+1} - p_t)
AUCTION = ea.ExecutionConfig(mode="close_auction", auction_cost_bp=10.0)
CFG = ea.BacktestConfig(gross_exposure=2.0, max_weight=0.04, linear_cost_bps=5.0,
                        impact_coef=Y, aum=AUM, rebalance_band=0.25,
                        borrow_bps_annual=50.0)


def hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build_features(prices, volumes):
    return {
        "mom_12_1": ea.momentum(prices, 252, 21),
        "mom_6_1": ea.momentum(prices, 126, 21),
        "reversal_1m": ea.short_term_reversal(prices, 21),
        "vol_63d": -ea.realized_vol(prices, 63),
        "ma_20_100": ea.ma_crossover(prices, 20, 100),
        "rsi_14": -ea.rsi(prices, 14),
        "turnover_z": ea.turnover_zscore(volumes, 63),
    }


COMBO = ("mom_12_1", "mom_6_1", "reversal_1m")


def one_seed(seed: int) -> dict[str, float]:
    panel = ea.generate_daily_panel(n_stocks=N_STOCKS, n_days=N_DAYS, seed=seed)
    prices, volumes = panel.prices, panel.volumes
    fwd1 = ea.forward_returns(prices, 1)
    feats = build_features(prices, volumes)
    ic = ea.ic_summary(ea.information_coefficient(ea.winsorize(feats["mom_12_1"], 0.01, 0.99),
                                                  fwd1), lags=5)
    combined = ea.combine_ic_weighted({k: feats[k] for k in COMBO}, fwd1)
    led = ea.run_backtest(prices, combined, CFG, volumes=volumes,
                          execution=EXEC).ledger.iloc[WARMUP:]
    auc = ea.run_backtest(prices, combined, CFG, volumes=volumes,
                          execution=AUCTION).ledger.iloc[WARMUP:]
    return {
        "mom_12_1 IC": ic["mean_ic"],
        "gross Sharpe": sharpe_ratio(led["gross_ret"]),
        "net Sharpe": sharpe_ratio(led["net_ret"]),
        "net Sharpe, close auction": sharpe_ratio(auc["net_ret"]),
        "gross return (% p.a.)": 100 * led["gross_ret"].mean() * 252,
        "net return (% p.a.)": 100 * led["net_ret"].mean() * 252,
        "cost drag (bp p.a.)": 1e4 * (led["cost"] + led["borrow"]).mean() * 252,
        "daily turnover (%)": 100 * led["turnover"].mean(),
    }


# ---------------------------------------------------------------------------
# 1. Alpha layer, one seed in detail
# ---------------------------------------------------------------------------
hdr(f"1. ALPHA LAYER, seed {SEED}: synthetic planted-alpha panel "
    f"({N_STOCKS} stocks x {N_DAYS} days)")
panel = ea.generate_daily_panel(n_stocks=N_STOCKS, n_days=N_DAYS, seed=SEED)
prices, volumes = panel.prices, panel.volumes
fwd1 = ea.forward_returns(prices, 1)
features = build_features(prices, volumes)

print("\nFeature ICs (1-day forward, Spearman), Newey-West t-stats (synthetic):")
rows = {}
for name, f in features.items():
    ic = ea.information_coefficient(ea.winsorize(f, 0.01, 0.99), fwd1)
    s = ea.ic_summary(ic, lags=5)
    rows[name] = {k: s[k] for k in ("mean_ic", "ic_std", "tstat_nw", "icir_annual", "n_obs")}
print(pd.DataFrame(rows).T.to_string())

combined = ea.combine_ic_weighted({k: features[k] for k in COMBO}, fwd1)
decay = ea.signal_decay(combined, prices, horizons=[1, 2, 3, 5, 10, 15, 20])
print("\nIC decay of the combined signal (mean IC vs horizon):")
print(decay.to_string())
dec = ea.decile_portfolios(combined, fwd1, n_quantiles=10)
dec_means = dec.mean() * 1e4
print("\nDecile mean 1-day forward returns (bps), Q1=low ... Q10=high:")
print(dec_means.to_string())
print(f"Monotonicity (Spearman rho of decile means): "
      f"{ea.quantile_monotonicity(dec_means[[f'Q{i}' for i in range(1, 11)]]):.3f}")

hdr("Trial log: every configuration evaluated on this panel")
log = TrialLog()
net_by_cfg = {}
candidates = {k: ea.cs_zscore(v) for k, v in features.items()}
candidates["combined_ic_weighted"] = combined
for name, sig in candidates.items():
    led = ea.run_backtest(prices, sig, CFG, volumes=volumes, execution=EXEC).ledger.iloc[WARMUP:]
    r = led["net_ret"]
    net_by_cfg[name] = r.to_numpy()
    log.log(name, float(r.mean() / r.std(ddof=1)), kind="signal")
# execution variants of the headline book are trials too
for name, ex in (("combined, close auction", AUCTION),
                 ("combined, next close", ea.ExecutionConfig(fill_fraction=1.0))):
    r = ea.run_backtest(prices, combined, CFG, volumes=volumes,
                        execution=ex).ledger.iloc[WARMUP:]["net_ret"]
    log.log(name, float(r.mean() / r.std(ddof=1)), kind="execution")
trial_tab = pd.DataFrame(log.trials).set_index("name")
trial_tab["sharpe_annual"] = trial_tab["sharpe"] * np.sqrt(252)
print(trial_tab[["kind", "sharpe_annual"]].to_string())
print(f"Logged trials: {log.n_trials}")
pbo = pbo_cscv(np.column_stack(list(net_by_cfg.values())), n_splits=16)
print(f"PBO (CSCV, 16 blocks, {len(net_by_cfg)} signal configurations): {pbo:.3f}")

hdr("Headline book: combined signal, net of costs (synthetic, seed 42)")
res = ea.run_backtest(prices, combined, CFG, volumes=volumes, execution=EXEC)
live = res.ledger.iloc[WARMUP:]
net = live["net_ret"].to_numpy()
summary = pd.DataFrame({"gross": ea.perf_summary(live["gross_ret"], live["turnover"]),
                        "net": ea.perf_summary(live["net_ret"], live["turnover"])})
print(f"\nAUM ${AUM/1e6:.0f}m, 5 bp linear + sqrt impact Y={Y}, 50 bp/yr borrow, "
      f"0.25-z band, fill at p_t + 0.5 (p_t+1 - p_t)")
print(summary.to_string())
sr = sharpe_ratio(net)
se = lo_sharpe_se(net)
_, lo, hi = block_bootstrap_ci(net, sharpe_ratio, n_boot=2000, seed=SEED)
dsr_null = deflated_sharpe_ratio(net, log.n_trials)
dsr_trials = deflated_sharpe_ratio(net, log.n_trials, trial_sharpes=log.sharpes())
print(f"Net Sharpe {sr:.2f}, Lo/Newey-West SE {se:.2f}, "
      f"block-bootstrap 95% CI [{lo:.2f}, {hi:.2f}]")
print(f"Deflated Sharpe at {log.n_trials} logged trials: {dsr_null:.4f} "
      f"(null spread 1/(n-1)); {dsr_trials:.4f} (spread of the logged trial Sharpes)")

# ---------------------------------------------------------------------------
# 2. Across seeds
# ---------------------------------------------------------------------------
hdr(f"2. HEADLINES ACROSS {N_SEEDS} SEEDS (synthetic): mean [95% CI] (sd)")
per_seed = pd.DataFrame([one_seed(s) for s in range(N_SEEDS)])
dist = {k: seed_distribution(per_seed[k].to_numpy()) for k in per_seed.columns}
print(pd.DataFrame({k: {"mean": d.mean, "ci_low": d.ci_low, "ci_high": d.ci_high,
                        "sd": d.sd, "min": d.minimum, "max": d.maximum}
                    for k, d in dist.items()}).T.to_string())

# ---------------------------------------------------------------------------
# 3. Capacity
# ---------------------------------------------------------------------------
hdr("3. CAPACITY: full backtest per AUM and impact coefficient Y (synthetic, seed 42)")
cap = ea.capacity_backtest(prices, combined, volumes, CFG,
                           aum_grid=[10e6, 100e6, 1e9, 5e9], impact_coefs=(0.3, 0.7, 1.0),
                           execution=EXEC, start=WARMUP)
cap = cap.reset_index()
cap["aum"] = [f"${a/1e6:,.0f}m" for a in cap["aum"]]
cap["ann_net_return"] *= 100
print(cap.rename(columns={"ann_net_return": "net_ret_%pa"}).to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Execution layer
# ---------------------------------------------------------------------------
hdr("4. EXECUTION: 5% ADV parent order, TWAP / VWAP / POV / AC (synthetic simulator)")
icfg = ea.IntradayConfig(mid0=100.0, day_volume=1_000_000, n_buckets=26,
                         sigma_daily=0.02, spread_bps=5.0, temp_coef=1.0,
                         perm_coef=0.5, vol_noise=0.2)
mkt = ea.IntradayMarket(icfg)
X = 0.05 * icfg.day_volume
acp = ea.ACParams(total_shares=X, n_slices=icfg.n_buckets,
                  sigma=icfg.sigma_daily * icfg.mid0, eta=2.0e-6, gamma=1e-6,
                  epsilon=icfg.mid0 * icfg.spread_bps * 1e-4 / 2)
lam_mid = 5e-6
# W3: the optimum under the simulator's own square-root law, built from the
# forecast volume curve: cost_k = c_k * n_k^1.5 with c_k = S * Y * sigma / V_k^0.5
vol_fc = icfg.profile * icfg.day_volume
c_pl = icfg.mid0 * icfg.temp_coef * icfg.sigma_daily / np.sqrt(vol_fc)
x_pl = ea.power_law_trajectory(X, c_pl, 0.5, lam_mid,
                               (icfg.mid0 * icfg.sigma_daily) ** 2 / icfg.n_buckets)
schedules = {
    "TWAP": ea.twap_schedule(X, icfg.n_buckets),
    "VWAP": ea.vwap_schedule(X, icfg.profile),
    "POV 10%": ea.pov_schedule(X, forecast_volumes=vol_fc, participation=0.10),
    f"AC lam={lam_mid:g}": ea.ac_trades(acp, lam_mid),
    f"Power law lam={lam_mid:g}": -np.diff(x_pl),
    "Aggressive (2 bkts)": np.r_[np.full(2, X / 2), np.zeros(icfg.n_buckets - 2)],
}
N_REPS = 500
tab = ea.evaluate_schedules(mkt, schedules, side=1, n_reps=N_REPS, seed=SEED)
print(f"\nModel-implied IS from the zero-noise simulator; Monte Carlo over {N_REPS} "
      f"replications for the spread (bps, + = cost):")
print(tab[["model_implied_is_bps", "std_is_bps", "mean_is_bps", "se_mean_is_bps"]]
      .sort_values("model_implied_is_bps").to_string())

hdr("Both optimisers under both impact laws (zero-noise simulator)")
print(ea.compare_impact_laws(icfg, X, lam_mid).round(4).to_string())

hdr("Almgren-Chriss efficient frontier (3 risk aversions)")
front = ea.efficient_frontier(acp, [1e-6, 5e-6, 5e-5])
front["exp_cost_bps"] = front["expected_cost"] / (X * icfg.mid0) * 1e4
front["std_bps"] = front["std"] / (X * icfg.mid0) * 1e4
front.index = [f"{l:.0e}" for l in front.index]
print(front[["kappa", "exp_cost_bps", "std_bps"]].to_string())

hdr("TCA: TWAP orders, decision at the prior close, release at the next session")
# The decision price is the prior close; the arrival mid follows it by a
# seeded move with sd sigma_daily * sqrt(0.5), the same half-day gap as the
# daily fill convention.
rng = np.random.default_rng(SEED)


def tca_one(seed):
    p_dec = icfg.mid0 / (1.0 + icfg.sigma_daily * np.sqrt(0.5) * rng.standard_normal())
    return ea.tca_report(mkt.execute(schedules["TWAP"], side=1, seed=seed, decision_price=p_dec))


rep = tca_one(1)
print(f"\nOne order: decision {rep.decision_price:.4f}, arrival {rep.arrival_price:.4f}")
print(pd.Series(rep.bps()).to_string())
agg = ea.aggregate_tca([tca_one(1000 + i) for i in range(200)])
print("\nAggregate TCA over 200 TWAP orders (bps):")
print(agg.to_string())

print(f"\nDone in {time.time() - T0:.1f}s")
