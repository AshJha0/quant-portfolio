"""End-to-end FX pairs pipeline on seeded synthetic data (offline, < 90 s).

Stages:
 1. Universe: USD-leg panel with risk-on/risk-off two-block structure,
    market pairs and crosses, candidate enumeration + correlation screen.
 2. Triangular null case: the no-arbitrage identity must be flagged
    degenerate by the cointegration machinery.
 3. Cointegration funnel: Engle-Granger on screened candidates (spurious
    correlation-only pairs must be rejected; the planted cointegrated pair
    must survive).
 4. OU spread modelling: parameter recovery vs ground truth, OLS vs MLE, RLS.
 5. Backtests: with/without costs and carry; EM-cost sensitivity; the
    carry-flip pair where ignoring carry flips the sign of P&L.
 6. SNB-style floor-then-break case study.
 7. Walk-forward with formation/trading windows and full metrics.

Run:  python examples/run_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fx_pairs as fp  # noqa: E402
from fx_pairs.data import synthetic as syn  # noqa: E402

T0 = time.time()


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show_summary(name: str, summ: dict) -> None:
    print(f"  {name}")
    print(f"    total P&L {summ['total_pnl']:+.4f}  "
          f"(spot {summ['spot_pnl']:+.4f}, carry {summ['carry_pnl']:+.4f}, "
          f"costs {summ['cost_pnl']:+.4f})")
    print(f"    Sharpe {summ['sharpe']:+.2f} (Lo SE {summ['sharpe_se_lo']:.2f})"
          f"  Sortino {summ['sortino']:+.2f}  MDD {summ['max_drawdown']:.4f}")
    print(f"    trades {summ['n_trades']:.0f}  hit rate {summ['hit_rate']:.2f}"
          f"  turnover {summ['turnover']:.0f}x/yr")


# ---------------------------------------------------------------- 1. universe
hr("1. UNIVERSE: USD legs, crosses, correlation screen")
legs, regime = syn.make_two_block_panel(n=1500, seed=2)
pair_names = ["AUDUSD", "NZDUSD", "USDCAD", "USDJPY", "USDCHF",
              "AUDNZD", "AUDJPY", "CHFJPY"]
prices = pd.DataFrame({p: fp.market_price_from_legs(legs, p) for p in pair_names})

# a hard peg joins the universe and must be screened out with a warning
import warnings  # noqa: E402
prices["PEGUSD"] = 3.6725
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    screen = fp.correlation_screen(prices, min_abs_corr=0.60)
print(f"  currencies: {list(legs.columns)}; instruments: {len(prices.columns)}"
      f" -> candidate pairs: {len(fp.enumerate_candidate_pairs(prices.columns))}")
print(f"  warning raised: {caught[0].message}")
print(f"  pairs passing |corr| >= 0.60 screen: {len(screen)}")
print(screen.head(6).to_string(index=False,
                               formatters={"corr": "{:.3f}".format}))
calm = np.log(prices[["AUDUSD", "USDJPY"]]).diff()[regime.values == 0].corr()
off = np.log(prices[["AUDUSD", "USDJPY"]]).diff()[regime.values == 1].corr()
print(f"  risk regime check, corr(AUDUSD, USDJPY): calm "
      f"{calm.iloc[0, 1]:+.2f} vs risk-off {off.iloc[0, 1]:+.2f}")

# ------------------------------------------------------- 2. triangular null case
hr("2. TRIANGULAR NULL CASE (no-arbitrage identity)")
tri = fp.triangular_spread(legs, "AUD", "USD", "JPY")
print(f"  spread log(AUDUSD)+log(USDJPY)-log(AUDJPY): "
      f"max|.| = {np.max(np.abs(tri.values)):.2e}, std = {np.std(tri.values):.2e}")
synth = fp.make_cross(legs, "AUD", "USD") * fp.make_cross(legs, "USD", "JPY")
eg_tri = fp.engle_granger(np.log(prices["AUDJPY"].values), np.log(synth.values))
print(f"  Engle-Granger on AUDJPY vs synthetic replication: degenerate="
      f"{eg_tri.degenerate}, cointegrated={eg_tri.cointegrated}")
print("  -> correctly refused: an exact identity is not a tradable spread "
      "(any deviation is inside the bid-ask at daily frequency).")

# --------------------------------------------------------- 3. cointegration funnel
hr("3. SELECTION FUNNEL: correlation screen -> Engle-Granger")
p1c, p2c, truth = syn.make_cointegrated_pair(n=1500, beta=1.0, kappa=20.0,
                                             sigma_ou=0.05, seed=9)
candidates = [(a, b) for a, b in screen[["pair_1", "pair_2"]].itertuples(index=False)]
survivors = []
for a, b in candidates:
    eg = fp.engle_granger(np.log(prices[a].values), np.log(prices[b].values))
    if eg.cointegrated and not eg.degenerate:
        survivors.append((a, b, eg.stat))
eg_planted = fp.engle_granger(np.log(p1c.values), np.log(p2c.values))
print(f"  screened candidates: {len(candidates)}; "
      f"EG 5% survivors among factor-correlated (non-cointegrated) pairs: "
      f"{len(survivors)}")
print(f"  planted cointegrated pair (AUDUSD-vs-NZDUSD-like): "
      f"stat {eg_planted.stat:.2f} vs 5% cv {eg_planted.crit_values['5%']:.2f}"
      f" -> cointegrated={eg_planted.cointegrated}")
print(f"  hedge-ratio recovery: beta_hat {eg_planted.beta:.3f} "
      f"(true {truth['beta']:.1f}), alpha_hat {eg_planted.alpha:+.4f}")

# ------------------------------------------------------------- 4. OU modelling
hr("4. OU SPREAD MODEL: recovery, OLS vs MLE, half-life, RLS")
s_true = fp.log_spread(p1c, p2c, truth["beta"], truth["alpha"])
ols = fp.fit_ou_ols(s_true.values)
mle = fp.fit_ou_mle(s_true.values)
print(f"  true:  kappa {truth['kappa']:.1f}  sigma {truth['sigma_ou']:.3f}  "
      f"half-life {truth['half_life_days']:.1f} bd")
print(f"  OLS :  kappa {ols.kappa:.1f}  sigma {ols.sigma:.3f}  "
      f"half-life {ols.half_life:.1f} bd  theta {ols.theta:+.4f}")
print(f"  MLE :  kappa {mle.kappa:.1f}  sigma {mle.sigma:.3f}  "
      f"half-life {mle.half_life:.1f} bd")
rls = fp.RLSHedge(lam=0.995)
rls_path = rls.fit_path(p2c, p1c)
print(f"  RLS hedge ratio: final beta {rls_path['beta'].iloc[-1]:.3f} "
      f"(EG static {eg_planted.beta:.3f})")

# ---------------------------------------------------------------- 5. backtests
hr("5. BACKTESTS: costs and carry matter")
form = 252
sp = fp.log_spread(p1c, p2c, eg_planted.beta, eg_planted.alpha)
z = fp.zscore(sp, window=126)
pos, trades = fp.generate_positions(z, entry=2.0, exit_=0.5, stop=4.0)

flat_rates = {"rb1": 0.03, "rq1": 0.01, "rb2": 0.02, "rq2": 0.01}
runs = {
    "frictionless (no costs, no carry)": dict(pip_spread_1=0.0, pip_spread_2=0.0),
    "major costs (0.7 / 1.0 pips)": dict(pip_spread_1=0.7, pip_spread_2=1.0),
    "major costs + carry": dict(pip_spread_1=0.7, pip_spread_2=1.0,
                                rates=flat_rates),
    "EM costs (60 / 30 pips)": dict(pair1="USDZAR", pair2="USDMXN",
                                    pip_spread_1=60.0, pip_spread_2=30.0),
}
summaries = {}
for name, kw in runs.items():
    res = fp.run_backtest(p1c, p2c, pos, eg_planted.beta, trades=trades, **kw)
    summaries[name] = fp.summarize(res)
    show_summary(name, summaries[name])
print("  -> same signal path: profitable at major spreads, unprofitable at EM"
      " spreads.")

hr("5b. CARRY-FLIP PAIR: ignoring carry flips the sign of P&L")
p1f, p2f, meta = syn.make_carry_flip_pair(seed=4)
eg_f = fp.engle_granger(np.log(p1f.values[:form]), np.log(p2f.values[:form]))
sp_f = fp.log_spread(p1f, p2f, eg_f.beta, eg_f.alpha)
mu_f, sig_f = float(sp_f.iloc[:form].mean()), float(sp_f.iloc[:form].std())
z_f = fp.zscore(sp_f, mu=mu_f, sigma=sig_f)
pos_f, trades_f = fp.generate_positions(z_f, entry=1.5, exit_=0.25, stop=None)
pos_f[:form] = 0.0
r_spot = fp.run_backtest(p1f, p2f, pos_f, eg_f.beta, trades=trades_f,
                         pip_spread_1=1.0, pip_spread_2=0.5)
r_carry = fp.run_backtest(p1f, p2f, pos_f, eg_f.beta, trades=trades_f,
                          pip_spread_1=1.0, pip_spread_2=0.5,
                          rates=meta["rates"])
show_summary("spot-only backtest (WRONG for FX)", fp.summarize(r_spot))
show_summary("carry-inclusive backtest", fp.summarize(r_carry))
print(f"  persistent differential: base leg {meta['rates']['rb1']:.0%} vs "
      f"hedge leg {meta['rates']['rb2']:.0%} -> carry "
      f"{fp.summarize(r_carry)['carry_pnl']:+.4f} flips the sign of the book.")

# carry-aware entry filter: long entries earn the differential (never vetoed);
# the mirrored short-spread book would PAY 7%/yr and gets vetoed wholesale.
hl_f = fp.fit_ou_ols(sp_f.values[:form]).half_life
# conservative desk assumption: against a persistent drift, realised holds run
# far past the OU half-life (the time-stop-free trades in 5b held for months)
hold_mult = 6.0
allow_l, allow_s = fp.carry_entry_veto(
    z_f, sigma_spread=sig_f,
    carry_per_day=float(meta["carry_per_day_long"]), half_life=hl_f,
    entry=1.5, exit_=0.25, holding_multiple=hold_mult)
long_all = int((z_f <= -1.5).sum())
long_kept = int(((z_f <= -1.5) & pd.Series(allow_l, index=z_f.index)).sum())
short_all = int((z_f >= 1.5).sum())
short_kept = int(((z_f >= 1.5) & pd.Series(allow_s, index=z_f.index)).sum())
z_star = 0.25 + hold_mult * hl_f * meta["carry_per_day_long"] / sig_f
print(f"  carry-aware filter (hold ~ {hold_mult:.0f} half-lives = "
      f"{hold_mult * hl_f:.0f} days): long entry bars kept "
      f"{long_kept}/{long_all} (earn carry, never vetoed); short entry bars "
      f"kept {short_kept}/{short_all} -- a short pays "
      f"{365 * meta['carry_per_day_long']:.1%}/yr, so entries below "
      f"|z| ~ {z_star:.2f} are skipped")

# ------------------------------------------------------------ 6. floor & break
hr("6. SNB-STYLE FLOOR-THEN-BREAK CASE STUDY")
p1s, p2s, meta_s = syn.make_floor_then_break(seed=3)
bi = meta_s["break_idx"]
eg_s = fp.engle_granger(np.log(p1s.values[:form]), np.log(p2s.values[:form]))
sp_s = fp.log_spread(p1s, p2s, eg_s.beta, eg_s.alpha)
mu_s, sig_s = float(sp_s.iloc[:form].mean()), float(sp_s.iloc[:form].std())
z_s = fp.zscore(sp_s, mu=mu_s, sigma=sig_s)
pos_s, trades_s = fp.generate_positions(z_s, entry=2.0, exit_=0.5, stop=None)
pos_s[:form] = 0.0
r_s = fp.run_backtest(p1s, p2s, pos_s, eg_s.beta, trades=trades_s,
                      pip_spread_1=1.2, pip_spread_2=1.0)
cum_pre = float(r_s.total_pnl.iloc[:bi].sum())
break_day = float(r_s.total_pnl.iloc[bi])
post = float(r_s.total_pnl.sum())
print(f"  formation scan on the pegged period: EG stat {eg_s.stat:.2f} "
      f"(cointegrated={eg_s.cointegrated}) -- looks like a PERFECT reverter")
print(f"  pre-break: cumulative P&L {cum_pre:+.4f} over {bi} days "
      f"(hit rate {fp.summarize(r_s)['hit_rate']:.2f})")
print(f"  position on break eve: {r_s.positions.iloc[bi - 1]:+.0f} "
      f"(long the pegged pair, with the crowd)")
print(f"  BREAK DAY P&L: {break_day:+.4f}  -- one day erases "
      f"{abs(break_day) / max(cum_pre, 1e-12):.1f}x the accumulated gains")
print(f"  full-sample P&L: {post:+.4f}; a z-stop cannot help: the market gaps"
      " straight through it.")

# --------------------------------------------------------------- 7. walk-forward
hr("7. WALK-FORWARD (252d formation / 63d trading)")
wf = fp.walk_forward_backtest(p1c, p2c, formation=252, trading=63,
                              entry=2.0, exit_=0.5, stop=4.0,
                              require_coint=True, coint_level="10%",
                              pip_spread_1=0.7, pip_spread_2=1.0,
                              rates=flat_rates)
traded = sum(w.traded for w in wf.windows)
print(f"  windows: {len(wf.windows)} ({traded} traded, "
      f"{len(wf.windows) - traded} skipped by the formation EG gate)")
wf_summ = fp.summarize(wf.result)
show_summary("walk-forward, costs + carry", wf_summ)
print(f"  mean formation beta: "
      f"{np.mean([w.beta for w in wf.windows if w.traded]):.3f}")

print(f"\nPipeline completed in {time.time() - T0:.1f}s")
