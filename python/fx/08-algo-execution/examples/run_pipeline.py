"""End-to-end FX algo-trading and execution pipeline demo.

Every number is computed on synthetic data. The intraday edge is planted by
the generator (hourly AR(1) increments, phi = 0.25); no real-data result
exists yet.

Runs, in order (all seeded, all offline, nothing written to disk):

1. Signal layer: feature ICs on the planted alpha vs pure noise, then the
   session-filtered vol-targeted backtest over N_SEEDS seeds under the
   shared execution convention (fill at a modelled t+1 price), with a
   close-auction variant, EM spreads, a deflated Sharpe at the logged trial
   count and PBO over the weight grid.
2. Execution: 500mm EURUSD parent order, TWAP vs liquidity-weighted vs POV
   vs piecewise Almgren-Chriss (uncalibrated eta and lambda), full day and
   London-only, over N_REP seeded replications.
3. Venues: last-look dealer stream vs firm-liquidity ECN for uninformed and
   informed flow.
4. WM/R fix: fix-targeting schedule vs TWAP tracking error to an
   independent fix print (intra-bucket market observations from a separate
   RNG), and the self-impact of the order on the print.
5. TCA: exact implementation-shortfall decomposition of one order.

Run from the project root:  python examples/run_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fx_algo import (  # noqa: E402
    BARS_PER_YEAR,
    EURUSD,
    BacktestConfig,
    ExecutionConfig,
    FirmVenue,
    IntradayBacktester,
    LastLookVenue,
    MarketSimulator,
    build_bars,
    carry_gate,
    combine_signals,
    decompose_implementation_shortfall,
    eta_from_depth,
    feature_matrix,
    fix_print,
    fix_schedule,
    fix_self_impact_pips,
    generate_daily_panel,
    generate_ticks,
    information_coefficient,
    liquidity_weighted_schedule,
    piecewise_ac_schedule,
    pov_schedule,
    session_filter,
    slippage_vs_benchmark,
    twap_benchmark,
    twap_schedule,
    venue_comparison,
    vol_target_positions,
)
from fx_algo.research_stats import (  # noqa: E402
    TrialLog,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    pbo_cscv,
    seed_distribution,
    sharpe_ratio,
)

N_REP = 200
N_SEEDS = 30
PARENT_MM = 500.0
WEIGHT_GRID = (1.0, 0.7, 0.5, 0.3, 0.0)  # weight on mom_1, the rest on mom_4
EM_SPREADS = {"asia": 30.0, "london": 12.0, "overlap": 10.0, "ny": 8.0, "late": 40.0}


def hr(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def ci(values, digits: int) -> str:
    s = seed_distribution(values)
    f = f"{{:+.{digits}f}}"
    return f"{f.format(s.mean)} [{f.format(s.ci_low)}, {f.format(s.ci_high)}]"


# ----------------------------------------------------------------- signals
def strategy_positions(fm: pd.DataFrame, bars: pd.DataFrame, w1: float) -> pd.Series:
    sig = combine_signals(fm, {"mom_1": w1, "mom_4": 1.0 - w1})
    pos = vol_target_positions(sig, bars["close"].pct_change().fillna(0.0), 0.10)
    pos = session_filter(pos, bars["hour"], ("london", "overlap", "ny"))
    return carry_gate(pos, bars["hour"], fm["carry"])


def signal_demo() -> None:
    hr("1. SIGNAL LAYER: planted intraday alpha (synthetic), session-filtered backtest")
    n_days = 60
    bars = build_bars(generate_ticks(n_days=n_days, phi=0.25, seed=0), 1.0)
    fm = feature_matrix(bars, generate_daily_panel(n_days, r_base=0.03, r_quote=0.05, seed=0))
    fwd = bars["close"].pct_change().shift(-1)

    print(f"\nFeature ICs vs next 1h return (seed 0, {n_days} days, planted phi=0.25):")
    print(f"{'feature':<14}{'IC':>9}{'t-stat':>9}")
    for c in ("mom_1", "mom_4", "reversion", "breakout"):
        ic, t = information_coefficient(fm[c], fwd)
        print(f"{c:<14}{ic:>9.3f}{t:>9.2f}")
    noise_bars = build_bars(generate_ticks(n_days=n_days, phi=0.0, seed=99), 1.0)
    ic0, t0 = information_coefficient(
        feature_matrix(noise_bars)["mom_1"], noise_bars["close"].pct_change().shift(-1)
    )
    print(f"{'mom_1 (noise)':<14}{ic0:>9.3f}{t0:>9.2f}   <- no alpha planted")

    cfg = BacktestConfig(
        pip_size=EURUSD.pip_size,
        spread_pips_by_session=EURUSD.spread_pips,
        r_base=0.03,
        r_quote=0.05,
    )
    em_cfg = BacktestConfig(pip_size=EURUSD.pip_size, spread_pips_by_session=EM_SPREADS)
    variants = {
        "next, f=0.5 (default)": (cfg, ExecutionConfig()),
        "next, f=1.0": (cfg, ExecutionConfig(fill_fraction=1.0)),
        "close auction, 0.5 bp": (cfg, ExecutionConfig("close_auction", auction_cost_bp=0.5)),
        "close auction, 0 bp": (cfg, ExecutionConfig("close_auction", auction_cost_bp=0.0)),
        "EM spreads, f=0.5": (em_cfg, ExecutionConfig()),
    }
    log = TrialLog()
    rows = {k: {"gross": [], "net": [], "sharpe": [], "trades": []} for k in variants}
    headline = None
    grid_net = None
    for seed in range(N_SEEDS):
        b = build_bars(generate_ticks(n_days=n_days, phi=0.25, seed=seed), 1.0)
        f = feature_matrix(b, generate_daily_panel(n_days, r_base=0.03, r_quote=0.05, seed=seed))
        pos = strategy_positions(f, b, 0.7)
        for k, (c, ex) in variants.items():
            led, sm = IntradayBacktester(c).run(b, pos, execution=ex)
            rows[k]["gross"].append(sm["gross_pips"])
            rows[k]["net"].append(sm["net_pips"])
            rows[k]["sharpe"].append(sm["sharpe_ann"])
            rows[k]["trades"].append(sm["n_trades"])
            if seed == 0:
                net = led["net_pips"].to_numpy()
                log.log(k, sharpe_ratio(net, 1.0), w_mom1=0.7)
                if headline is None:
                    headline = net
        if seed == 0:
            cols = []
            for w1 in WEIGHT_GRID:
                led, _ = IntradayBacktester(cfg).run(b, strategy_positions(f, b, w1))
                cols.append(led["net_pips"].to_numpy())
                if w1 != 0.7:
                    log.log(f"mom_1 weight {w1}", sharpe_ratio(cols[-1], 1.0), w_mom1=w1)
            grid_net = np.column_stack(cols)

    print(f"\nmom_1/mom_4 (0.7/0.3) vol-targeted strategy, EURUSD, {n_days} days per seed.")
    print(f"Synthetic data, {N_SEEDS} seeds, mean [95% CI] across seeds:")
    print(f"{'variant':<24}{'gross pips':>24}{'net pips':>24}{'Sharpe':>24}{'trades':>8}")
    for k, r in rows.items():
        print(f"{k:<24}{ci(r['gross'], 0):>24}{ci(r['net'], 0):>24}"
              f"{ci(r['sharpe'], 2):>24}{np.mean(r['trades']):>8.0f}")

    ann = np.sqrt(BARS_PER_YEAR)
    _, lo, hi = block_bootstrap_ci(headline, lambda x: sharpe_ratio(x, BARS_PER_YEAR), seed=1)
    dsr = deflated_sharpe_ratio(headline, log.n_trials, log.sharpes())
    pbo = pbo_cscv(grid_net, n_splits=8)
    print(f"\nSeed 0, default variant: Sharpe {sharpe_ratio(headline, 1.0) * ann:.2f}, "
          f"block-bootstrap 95% CI [{lo:.2f}, {hi:.2f}] ({BARS_PER_YEAR:.0f} bars/yr)")
    print(f"Trials logged on seed 0: {log.n_trials} ({len(variants)} execution/cost variants"
          f" + {len(WEIGHT_GRID) - 1} other weight sets)")
    print(f"Deflated Sharpe ratio at {log.n_trials} trials: {dsr:.3f}")
    print(f"PBO (CSCV, 8 blocks) over the {len(WEIGHT_GRID)}-point weight grid: {pbo:.2f}")


# --------------------------------------------------------------- execution
def scheduler_demo() -> None:
    hr(f"2. EXECUTION: {PARENT_MM:.0f}mm EURUSD buy, scheduler comparison "
       f"({N_REP} replications, synthetic)")

    def run_grid(label: str, start: float, horizon: float) -> None:
        sim = MarketSimulator(EURUSD, start_hour=start, horizon_hours=horizon, dt_minutes=5.0)
        eta = eta_from_depth(sim.depth_bucket, k_eta=0.02)  # uncalibrated
        scheds = {
            "TWAP": twap_schedule(PARENT_MM, sim.n_buckets),
            "liquidity-weighted": liquidity_weighted_schedule(PARENT_MM, sim.depth_bucket),
            "POV 5% (analog)": pov_schedule(PARENT_MM, sim.depth_bucket, 0.05),
            "piecewise-AC (lam=1e-5)": piecewise_ac_schedule(
                PARENT_MM, eta, sim.sigma_bucket_pips, 1e-5
            ),
        }
        print(f"\n{label}:")
        print(f"{'schedule':<26}{'ctrl cost':>11}{'IS mean [95% CI]':>26}{'IS sd':>9}")
        for name, q in scheds.items():
            ctrl, tot = [], []
            for s in range(N_REP):
                d = decompose_implementation_shortfall(sim.execute(q, FirmVenue(), seed=s))
                ctrl.append(d["spread_temporary"] + d["permanent_impact"])
                tot.append(d["total"])
            print(f"{name:<26}{np.mean(ctrl):>11.3f}{ci(tot, 2):>26}{np.std(tot, ddof=1):>9.2f}")
        print("  ctrl cost = spread + temporary + permanent, deterministic given the")
        print("  schedule. IS mean is ctrl cost plus mean-zero drift noise; its CI is wide.")

    run_grid("Full 24h day (288 x 5-min buckets)", 0.0, 24.0)
    run_grid("London-only window 07:00-16:00 (108 buckets)", 7.0, 9.0)


def venue_demo() -> None:
    hr(f"3. VENUES: last-look dealer stream vs firm ECN ({N_REP} replications, synthetic)")
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    lw = liquidity_weighted_schedule(PARENT_MM, sim.depth_bucket)
    for alpha, label in ((0.0, "uninformed flow (alpha = 0)"),
                         (0.5, "informed flow (alpha = 0.5 pips/bucket)")):
        rows = {"last-look": [], "firm-ecn": []}
        for s in range(N_REP):
            rl = sim.execute(lw, LastLookVenue(), seed=s, alpha_pips_per_bucket=alpha)
            rf = sim.execute(lw, FirmVenue(), seed=s, alpha_pips_per_bucket=alpha)
            vc = venue_comparison({"last-look": rl, "firm-ecn": rf})
            for k in rows:
                rows[k].append(vc[k])
        print(f"\n{label}:")
        print(f"{'venue':<12}{'quoted 1/2-spread':>18}{'effective cost':>16}"
              f"{'reject rate':>13}{'reject cost':>13}")
        for k, vals in rows.items():
            df = pd.DataFrame(vals)
            print(
                f"{k:<12}{df['quoted_half_spread_pips'].mean():>18.3f}"
                f"{df['effective_cost_pips'].mean():>16.3f}"
                f"{df['rejection_rate'].mean():>13.1%}"
                f"{df['rejection_cost_pips'].mean():>13.3f}"
            )
        diff = (pd.DataFrame(rows["last-look"])["effective_cost_pips"]
                - pd.DataFrame(rows["firm-ecn"])["effective_cost_pips"])
        print(f"  paired diff (LL - firm): {ci(diff.to_numpy(), 3)} pips")


def fix_demo() -> None:
    hr(f"4. WM/R 4PM LONDON FIX: fix-targeting vs TWAP ({N_REP} replications, synthetic)")
    sim = MarketSimulator(EURUSD, start_hour=14.0, horizon_hours=3.0, dt_minutes=1.0)
    q_fix = fix_schedule(100.0, sim.times_hours, 1.0)
    q_twap = twap_schedule(100.0, sim.n_buckets)
    names = ("fix-targeting (5-min window)", "TWAP 14:00-17:00")
    te = {k: [] for k in names}
    te_clean = {k: [] for k in names}
    self_imp = []
    for s in range(N_REP):
        for k, q in zip(names, (q_fix, q_twap)):
            r = sim.execute(q, seed=s)
            te[k].append(slippage_vs_benchmark(r, fix_print(r, seed=100_000 + s)))
            te_clean[k].append(
                slippage_vs_benchmark(r, fix_print(r, seed=100_000 + s, include_own_impact=False))
            )
            if k == names[0]:
                self_imp.append(fix_self_impact_pips(r, seed=100_000 + s))
    print("\n100mm EURUSD buy vs the fix print. The print averages 60 market")
    print("observations per minute of the window, drawn from an RNG independent of the")
    print("execution. 'Unimpacted' strips the order's own permanent impact from the path.")
    print(f"{'schedule':<30}{'TE vs print':>24}{'sd':>8}{'TE vs unimpacted':>24}{'sd':>8}")
    for k in names:
        print(f"{k:<30}{ci(te[k], 3):>24}{np.std(te[k], ddof=1):>8.3f}"
              f"{ci(te_clean[k], 3):>24}{np.std(te_clean[k], ddof=1):>8.3f}")
    print(f"Self-impact of the fix order on the print: {ci(self_imp, 3)} pips")


def tca_demo() -> None:
    hr("5. TCA REPORT: 500mm EURUSD, liquidity-weighted, firm ECN (seed 0, synthetic)")
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    lw = liquidity_weighted_schedule(PARENT_MM, sim.depth_bucket)
    r = sim.execute(lw, FirmVenue(), seed=0)
    d = decompose_implementation_shortfall(r)
    print(f"\narrival mid {r.arrival_mid:.5f} | avg fill {r.avg_fill:.5f} | "
          f"interval TWAP {twap_benchmark(r):.5f}")
    print("\nImplementation shortfall vs arrival (pips per unit, exact sum):")
    for k in ("spread_temporary", "permanent_impact", "market_drift", "total"):
        print(f"  {k:<18}{d[k]:>+9.3f}")
    print(f"  slippage vs interval TWAP: "
          f"{slippage_vs_benchmark(r, twap_benchmark(r)):+.3f} pips")


def main() -> None:
    t0 = time.time()
    print("FX ALGORITHMIC TRADING AND EXECUTION PIPELINE (fx_algo)")
    print("All results below are on synthetic data; the signal edge is planted.")
    signal_demo()
    scheduler_demo()
    venue_demo()
    fix_demo()
    tca_demo()
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
