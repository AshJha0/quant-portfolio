"""End-to-end demo: portfolio -> VaR/ES (all methods) -> backtest -> Basel
traffic light -> stress & reverse stress.

Run from the project root:  python examples/run_pipeline.py   (< 120 s)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eq_var as ev
from eq_var.data import demo_covariance, demo_portfolio, simulate_garch_returns

pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

t0 = time.time()
rng = np.random.default_rng(42)

# ----------------------------------------------------------------------- #
# 1. Portfolio & factor mapping
# ----------------------------------------------------------------------- #
pf = demo_portfolio()
cov = demo_covariance()
expos = pf.delta_exposures()

print("=" * 78)
print("EQUITY VaR / ES ENGINE — demo pipeline")
print("=" * 78)
print(f"\nPortfolio value: ${pf.value():,.0f}")
print("Dollar exposures (per unit factor move):")
for name, e in zip(pf.factor_names, expos):
    print(f"  {name:8s} {e:>14,.0f}")

# ----------------------------------------------------------------------- #
# 2. VaR & ES, all methods, 95/99, 1d & 10d
# ----------------------------------------------------------------------- #
# Historical scenarios: 1000 days of fat-tailed (t) factor returns
hist_returns = ev.simulate_factor_returns(cov, 1000, dist="t", df=6, seed=rng)
hist_pnl = pf.pnl(hist_returns, method="full")

rows = []
for alpha in (0.05, 0.01):
    mc_pnl_n = ev.monte_carlo_pnl(pf, cov, 100_000, dist="normal", seed=7)
    mc_pnl_t = ev.monte_carlo_pnl(pf, cov, 100_000, dist="t", df=6, seed=7)
    var_1d = {
        "historical": ev.historical_var(hist_pnl, alpha),
        "age_weighted": ev.age_weighted_var(hist_pnl, alpha, lam=0.98),
        "FHS": ev.filtered_historical_var(hist_pnl, alpha),
        "parametric_normal": ev.parametric_var(expos, cov, alpha, dist="normal"),
        "parametric_t6": ev.parametric_var(expos, cov, alpha, dist="t", df=6),
        "MC_normal": -float(np.quantile(mc_pnl_n, alpha)),
        "MC_t6": -float(np.quantile(mc_pnl_t, alpha)),
    }
    es_1d = {
        "historical": ev.expected_shortfall(hist_pnl, alpha),
        "parametric_normal": ev.parametric_es(expos, cov, alpha, dist="normal"),
        "parametric_t6": ev.parametric_es(expos, cov, alpha, dist="t", df=6),
        "MC_normal": ev.expected_shortfall(mc_pnl_n, alpha),
        "MC_t6": ev.expected_shortfall(mc_pnl_t, alpha),
    }
    for m, v in var_1d.items():
        rows.append(
            {
                "method": m,
                "conf": f"{100 * (1 - alpha):.0f}%",
                "VaR_1d": v,
                "VaR_10d_sqrt": ev.scale_var_sqrt_time(v, 10),
                "ES_1d": es_1d.get(m, np.nan),
            }
        )
var_table = pd.DataFrame(rows)
print("\n--- VaR / ES by method (1d and sqrt-time 10d, $) ---")
print(var_table.to_string(index=False))

# 10d via overlapping windows (alternative to sqrt-time)
ov = ev.overlapping_horizon_pnl(hist_pnl, 10)
print(
    f"\n10d 99% VaR: sqrt-time {ev.scale_var_sqrt_time(ev.historical_var(hist_pnl, 0.01), 10):,.0f}"
    f" vs overlapping-window {ev.historical_var(ov, 0.01):,.0f}"
    "  (overlapping windows are serially dependent — wide error bars)"
)

# MC standard errors
mc_pnl = ev.monte_carlo_pnl(pf, cov, 100_000, dist="t", df=6, seed=7)
se = ev.var_standard_error_bootstrap(mc_pnl, 0.01, n_boot=200, seed=1)
lo, hi = ev.var_confidence_interval(mc_pnl, 0.01)
print(f"\nMC t6 99% VaR = {-np.quantile(mc_pnl, 0.01):,.0f}  (bootstrap SE {se:,.0f}, 95% CI [{lo:,.0f}, {hi:,.0f}])")

print(
    "\nWhy the methods disagree: the t(6) factor model has excess kurtosis, so"
    "\nfat-tail-aware methods (historical on t data, parametric-t, MC-t) put more"
    "\nmass beyond the 99% point than the normal ones; the gap widens from 95%"
    "\nto 99% because tail-shape differences grow with confidence level."
)

# ----------------------------------------------------------------------- #
# 3. 500-day backtest on GARCH data: parametric-normal vs HS vs FHS
# ----------------------------------------------------------------------- #
print("\n--- 500-day 99% VaR backtest on GARCH(1,1)-t factor data ---")
n_bt, window = 500, 250
garch_rets = simulate_garch_returns(
    n_bt + window, cov, alpha_g=0.13, beta_g=0.85, df=5.0, seed=5
)
garch_pnl = pf.pnl(garch_rets, method="full")


def var_param_unconditional(hist: np.ndarray, alpha: float) -> float:
    from scipy.stats import norm

    return float(-norm.ppf(alpha) * np.std(hist, ddof=1))


methods = {
    "parametric_normal": var_param_unconditional,
    "historical": ev.historical_var,
    "FHS": ev.filtered_historical_var,
}
results = []
for name, fn in methods.items():
    bt = ev.rolling_var_backtest(garch_pnl, fn, window=window, alpha=0.01, name=name)
    results.append(bt.summary())
bt_table = pd.DataFrame(results)
print(bt_table.to_string(index=False))
print(
    "\nReading: unconditional-normal parametric VaR ignores vol clustering and"
    "\nfat tails -> too many exceptions, Kupiec rejects; FHS rescales to the"
    "\ncurrent vol regime and passes both coverage and independence."
)

fhs_bt = ev.rolling_var_backtest(garch_pnl, ev.filtered_historical_var, window, 0.01, "FHS")
print("\nException clustering table (FHS):")
print(ev.exception_cluster_table(fhs_bt.exceptions).to_string(index=False))

# ----------------------------------------------------------------------- #
# 4. Basel traffic light
# ----------------------------------------------------------------------- #
print("\n--- Basel traffic light (250-day 99% VaR window) ---")
for name, fn in methods.items():
    bt = ev.rolling_var_backtest(garch_pnl[: window + 250], fn, window=window, alpha=0.01, name=name)
    tl = ev.basel_traffic_light(bt.n_exceptions)
    print(
        f"  {name:18s} exceptions={bt.n_exceptions:2d}/250  zone={tl['zone']:6s}"
        f"  capital multiplier k={tl['multiplier']:.2f}"
    )

# ES backtest (Acerbi-Szekely Z2) at 97.5% on the last 250 days, FHS-style ES
tail = garch_pnl[-250:]
var975 = ev.historical_var(garch_pnl[:750], 0.025)
es975 = ev.expected_shortfall(garch_pnl[:750], 0.025)
z2 = ev.acerbi_szekely_z2(tail, var975, es975, alpha=0.025)
print(f"\nES 97.5% backtest (Acerbi-Szekely): Z2={z2['z2']:.3f}, exceptions={z2['n_exceptions']}, reject={z2['reject']}")

# ----------------------------------------------------------------------- #
# 5. Stress testing
# ----------------------------------------------------------------------- #
print("\n--- Stress scenarios (P&L $, full revaluation vs delta-gamma) ---")
print(ev.scenario_table(pf).drop(columns="description").to_string(index=False))

print("\n--- SPX sensitivity ladder (full reval) ---")
print(ev.sensitivity_ladder(pf, "SPX").to_string(index=False))

rs = ev.reverse_stress_delta(expos, cov, radius=3.0)
rsg = ev.reverse_stress_delta_gamma(expos, pf.gamma_matrix(), cov, radius=3.0, seed=5)
print("\n--- Reverse stress (worst joint 3-sigma move) ---")
print("factor shocks (delta closed form):")
for name, s in zip(pf.factor_names, rs["shock"]):
    print(f"  {name:8s} {s:+.4f}")
print(f"worst-case loss (delta): ${rs['loss']:,.0f}")
print(f"worst-case loss (delta-gamma, numeric): ${rsg['loss']:,.0f} (long gamma trims the tail)")

print(f"\nPipeline completed in {time.time() - t0:.1f}s")
