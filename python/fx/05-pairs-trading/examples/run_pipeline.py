"""End-to-end FX pairs pipeline on seeded synthetic data (offline).

Every result is synthetic: the planted pair is cointegrated by construction,
so any profit is a property of the generator, not evidence of an edge in real
FX markets. Sections 1-7 walk through one seed; section 8 repeats the
headline experiments over many seeds and reports means with 95% confidence
intervals, a deflated Sharpe ratio at the logged trial count and PBO.

Execution: signals use data up to the close of day t; orders fill at a
modelled t+1 price p_t + f (p_{t+1} - p_t), f = 0.5 (ExecutionConfig()).

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

Run:  python examples/run_pipeline.py [--seeds N]
The script prints to stdout and writes no files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fx_pairs as fp  # noqa: E402
from fx_pairs.data import synthetic as syn  # noqa: E402

from fx_pairs import research_stats as rs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=60,
                help="number of seeds for the section 8 distributions")
ARGS = ap.parse_args()
if ARGS.seeds < 2:
    raise SystemExit("--seeds must be >= 2")

T0 = time.time()
TRIALS = rs.TrialLog()  # every strategy configuration evaluated on planted data


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
hr("3. SELECTION FUNNEL: planted pair hidden in the universe")
PLANTED = ("SEKUSD", "NOKUSD")  # labels for the synthetic planted pair


def run_funnel(legs_panel, p1x, p2x, level="5%"):
    """fp.cointegration_funnel on the market pairs plus the planted pair."""
    uni = pd.DataFrame({p: fp.market_price_from_legs(legs_panel, p)
                        for p in pair_names})
    uni[PLANTED[0]] = p1x.values
    uni[PLANTED[1]] = p2x.values
    res = fp.cointegration_funnel(uni, min_abs_corr=0.60, level=level)
    found = [(r.pair_1, r.pair_2, r.eg_stat) for r in res.itertuples() if r.passed]
    return len(res), found


p1c, p2c, truth = syn.make_cointegrated_pair(n=1500, beta=1.0, kappa=20.0,
                                             sigma_ou=0.05, seed=9)
n_cand, survivors = run_funnel(legs, p1c, p2c)
planted_found = any({a_, b_} == set(PLANTED) for a_, b_, _ in survivors)
print(f"  instruments: {len(pair_names) + 2} (8 market pairs + planted pair "
      f"{PLANTED[0]}/{PLANTED[1]}); screened candidates: {n_cand}")
print(f"  EG 5% survivors: {[(a_, b_, round(st, 2)) for a_, b_, st in survivors]}")
print(f"  planted pair found: {planted_found}; false positives: "
      f"{sum({a_, b_} != set(PLANTED) for a_, b_, _ in survivors)} "
      f"(EG tests run: {n_cand}; about {0.05 * n_cand:.1f} false positives "
      "are expected at 5% under the null)")
eg_planted = fp.engle_granger(np.log(p1c.values), np.log(p2c.values))
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
rls_path = fp.RLSHedge(lam=0.995).fit_path(p2c, p1c)
print(f"  RLS hedge ratio (lam 0.995): final beta {rls_path['beta'].iloc[-1]:.3f} "
      f"(EG static {eg_planted.beta:.3f}); used as a causal hedge in section 5")

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
TRIALS.log("in-sample: full-sample EG hedge, rolling z(126), 2.0/0.5/4.0",
           summaries["major costs (0.7 / 1.0 pips)"]["sharpe"] / np.sqrt(252))
res_rls = fp.run_backtest(p1c, p2c, pos, rls_path["beta"], trades=trades,
                          pip_spread_1=0.7, pip_spread_2=1.0)
show_summary("major costs, RLS hedge ratio (causal, lam 0.995)",
             fp.summarize(res_rls))
TRIALS.log("in-sample signal, RLS hedge", fp.summarize(res_rls)["sharpe"] / np.sqrt(252))
res_auc = fp.run_backtest(p1c, p2c, pos, eg_planted.beta, trades=trades,
                          pip_spread_1=0.7, pip_spread_2=1.0,
                          execution=fp.ExecutionConfig("close_auction"))
show_summary("major costs, close-auction fill (+10 bp)", fp.summarize(res_auc))
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
pos_v, trades_v = fp.generate_positions(z_f, entry=1.5, exit_=0.25, stop=None,
                                        allow_long=allow_l, allow_short=allow_s)
pos_v[:form] = 0.0
r_veto = fp.run_backtest(p1f, p2f, pos_v, eg_f.beta, trades=trades_v,
                         pip_spread_1=1.0, pip_spread_2=0.5, rates=meta["rates"])
show_summary("carry-inclusive backtest with the carry entry veto",
             fp.summarize(r_veto))
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
TRIALS.log("walk-forward 252/63, entry 2.0 exit 0.5 stop 4.0",
           wf_summ["sharpe"] / np.sqrt(252))
scale = fp.hedged_vol_scale(p1c, p2c, wf.result.beta, target_vol=0.10)
wf_vt = fp.run_backtest(p1c, p2c, wf.positions * scale, wf.result.beta,
                        trades=wf.result.trades, pip_spread_1=0.7, pip_spread_2=1.0, rates=flat_rates)
show_summary("walk-forward, vol-targeted to 10% (63d realised spread vol)",
             fp.summarize(wf_vt))
TRIALS.log("walk-forward + vol target 10%",
           fp.summarize(wf_vt)["sharpe"] / np.sqrt(252))



# ------------------------------------------------------- 8. many seeds + stats
hr(f"8. DISTRIBUTIONS OVER {ARGS.seeds} SEEDS (synthetic), DSR, PBO")
GRID = [(e, x) for e in (1.5, 2.0, 2.5) for x in (0.0, 0.5)]
WF_KW = dict(formation=252, trading=63, stop=4.0, require_coint=True,
             coint_level="10%", pip_spread_1=0.7, pip_spread_2=1.0,
             rates=flat_rates)


def sr(res):
    s_ = fp.summarize(res)["sharpe"]
    return float(s_) if np.isfinite(s_) else 0.0


def one_seed(seed):
    out = {}
    lg, _ = syn.make_two_block_panel(n=1500, seed=seed)
    q1, q2, _ = syn.make_cointegrated_pair(n=1500, beta=1.0, kappa=20.0,
                                           sigma_ou=0.05, seed=seed)
    _, surv = run_funnel(lg, q1, q2)
    out["funnel_found"] = float(any({a_, b_} == set(PLANTED) for a_, b_, _ in surv))
    out["funnel_false_pos"] = float(sum({a_, b_} != set(PLANTED) for a_, b_, _ in surv))
    e = fp.engle_granger(np.log(q1.values), np.log(q2.values))
    zz = fp.zscore(fp.log_spread(q1, q2, e.beta, e.alpha), window=126)
    pp, tt = fp.generate_positions(zz, entry=2.0, exit_=0.5, stop=4.0)
    for key, kw in (("is_frictionless", dict(pip_spread_1=0.0, pip_spread_2=0.0)),
                    ("is_major", dict(pip_spread_1=0.7, pip_spread_2=1.0)),
                    ("is_em", dict(pair1="USDZAR", pair2="USDMXN",
                                   pip_spread_1=60.0, pip_spread_2=30.0))):
        out[key] = sr(fp.run_backtest(q1, q2, pp, e.beta, trades=tt, **kw))
    grid_returns = []
    for ent, ext in GRID:
        w = fp.walk_forward_backtest(q1, q2, entry=ent, exit_=ext, **WF_KW)
        grid_returns.append(w.result.total_pnl.to_numpy())
        if (ent, ext) == (2.0, 0.5):
            out["wf"] = sr(w.result)
            out["wf_windows_traded"] = float(np.mean([x.traded for x in w.windows]))
            sc = fp.hedged_vol_scale(q1, q2, w.result.beta, target_vol=0.10)
            out["wf_voltarget"] = sr(fp.run_backtest(
                q1, q2, w.positions * sc, w.result.beta, pip_spread_1=0.7,
                pip_spread_2=1.0, rates=flat_rates))
            ws = fp.walk_forward_backtest(
                q1, q2, entry=ent, exit_=ext,
                execution=fp.ExecutionConfig("close_auction", auction_cost_bp=0.0),
                **WF_KW)
            out["wf_same_close"] = sr(ws.result)
    out["pbo"] = rs.pbo_cscv(np.column_stack(grid_returns)[252:], n_splits=10)
    f1, f2, m = syn.make_carry_flip_pair(seed=seed)
    ef = fp.engle_granger(np.log(f1.values[:form]), np.log(f2.values[:form]))
    spf = fp.log_spread(f1, f2, ef.beta, ef.alpha)
    mf, sf = float(spf.iloc[:form].mean()), float(spf.iloc[:form].std())
    zf = fp.zscore(spf, mu=mf, sigma=sf)
    pf, tf = fp.generate_positions(zf, entry=1.5, exit_=0.25, stop=None)
    pf[:form] = 0.0
    out["flip_spot_only"] = fp.summarize(fp.run_backtest(
        f1, f2, pf, ef.beta, trades=tf, pip_spread_1=1.0, pip_spread_2=0.5))["total_pnl"]
    out["flip_with_carry"] = fp.summarize(fp.run_backtest(
        f1, f2, pf, ef.beta, trades=tf, pip_spread_1=1.0, pip_spread_2=0.5,
        rates=m["rates"]))["total_pnl"]
    hl = fp.fit_ou_ols(spf.values[:form]).half_life
    al, ash = fp.carry_entry_veto(zf, sigma_spread=sf,
                                  carry_per_day=float(m["carry_per_day_long"]),
                                  half_life=hl, entry=1.5, exit_=0.25,
                                  holding_multiple=6.0)
    pv, tv = fp.generate_positions(zf, entry=1.5, exit_=0.25, stop=None,
                                   allow_long=al, allow_short=ash)
    pv[:form] = 0.0
    out["flip_veto"] = fp.summarize(fp.run_backtest(
        f1, f2, pv, ef.beta, trades=tv, pip_spread_1=1.0, pip_spread_2=0.5,
        rates=m["rates"]))["total_pnl"]
    return out


rows = [one_seed(1000 + k) for k in range(ARGS.seeds)]
LABELS = [
    ("funnel_found", "funnel finds the planted pair (share of seeds)"),
    ("funnel_false_pos", "funnel false positives per seed"),
    ("is_frictionless", "in-sample Sharpe, frictionless"),
    ("is_major", "in-sample Sharpe, major costs"),
    ("is_em", "in-sample Sharpe, EM costs"),
    ("wf", "walk-forward Sharpe, costs + carry"),
    ("wf_same_close", "walk-forward Sharpe, same-close fill, no auction cost"),
    ("wf_voltarget", "walk-forward Sharpe, vol-targeted 10%"),
    ("wf_windows_traded", "share of walk-forward windows passing the EG gate"),
    ("pbo", "PBO over the 6-config entry/exit grid"),
    ("flip_spot_only", "carry-flip pair P&L, spot only"),
    ("flip_with_carry", "carry-flip pair P&L, with carry"),
    ("flip_veto", "carry-flip pair P&L, with carry and entry veto"),
]
print(f"  synthetic, seeds 1000..{999 + ARGS.seeds}; mean [95% CI] (sd)")
print(f"  {'quantity':58s} mean [95% CI] (sd)")
for key, label in LABELS:
    d = rs.seed_distribution([r_[key] for r_ in rows])
    print(f"  {label:58s} {d.mean:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}] "
          f"({d.sd:.3f})")

# deflated Sharpe of the reference walk-forward (seed 9) at the logged count
for ent, ext in GRID:
    if (ent, ext) == (2.0, 0.5):
        continue
    w = fp.walk_forward_backtest(p1c, p2c, entry=ent, exit_=ext, **WF_KW)
    TRIALS.log(f"walk-forward grid entry {ent} exit {ext}", sr(w.result) / np.sqrt(252))
ref = wf.result.total_pnl.to_numpy()[252:]
dsr = rs.deflated_sharpe_ratio(ref, TRIALS.n_trials, trial_sharpes=TRIALS.sharpes())
pt, lo, hi = rs.block_bootstrap_ci(ref, lambda x: rs.sharpe_ratio(x), n_boot=2000,
                                   block_length=20, seed=0)
print(f"\n  reference seed 9 walk-forward: Sharpe {pt:.2f}, block-bootstrap 95% CI "
      f"[{lo:.2f}, {hi:.2f}] (20-day blocks), Lo/NW SE "
      f"{rs.lo_sharpe_se(ref):.2f}")
print(f"  trials logged on the seed-9 planted pair: {TRIALS.n_trials}")
for t_ in TRIALS.trials:
    print(f"    - {t_['name']}")
print(f"  deflated Sharpe ratio at {TRIALS.n_trials} trials: {dsr:.3f}")

print(f"\nPipeline completed in {time.time() - T0:.1f}s")
