"""End-to-end FX algo-trading & execution pipeline demo.

Runs, in order (all seeded, all offline, < 120 s total):

1. Signal layer  — feature ICs on planted intraday alpha vs pure noise,
   session-filtered vol-targeted backtest, EURUSD vs EM cost regimes.
2. Execution     — 500mm EURUSD parent order: TWAP vs liquidity-weighted
   vs piecewise Almgren-Chriss, over the full 24h day and over a
   London-only window; cost/variance over 200 seeded replications.
3. Venues        — last-look dealer stream vs firm-liquidity ECN, with
   rejection statistics, for uninformed and informed (alpha) flow.
4. WM/R fix      — fix-targeting schedule vs day-TWAP tracking error to
   the 4pm London fix (5-minute window, post-2015 methodology).
5. TCA           — exact implementation-shortfall decomposition report.

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
    BacktestConfig,
    EURUSD,
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
    fix_benchmark,
    fix_schedule,
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

N_REP = 200
PARENT_MM = 500.0
EM_SPREADS = {"asia": 30.0, "london": 12.0, "overlap": 10.0, "ny": 8.0, "late": 40.0}


def hr(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# ----------------------------------------------------------------- signals
def signal_demo() -> None:
    hr("1. SIGNAL LAYER — planted intraday alpha, session-filtered backtest")
    n_days = 60
    ticks = generate_ticks(n_days=n_days, phi=0.25, seed=0)
    bars = build_bars(ticks, 1.0)
    panel = generate_daily_panel(n_days, r_base=0.03, r_quote=0.05, seed=0)
    fm = feature_matrix(bars, panel)
    fwd = bars["close"].pct_change().shift(-1)

    print(f"\nFeature ICs vs next 1h return ({n_days} days, planted phi=0.25):")
    print(f"{'feature':<12}{'IC':>9}{'t-stat':>9}")
    for c in ("mom_1", "mom_4", "reversion", "breakout"):
        ic, t = information_coefficient(fm[c], fwd)
        print(f"{c:<12}{ic:>9.3f}{t:>9.2f}")
    noise_bars = build_bars(generate_ticks(n_days=n_days, phi=0.0, seed=99), 1.0)
    ic0, t0 = information_coefficient(
        feature_matrix(noise_bars)["mom_1"], noise_bars["close"].pct_change().shift(-1)
    )
    print(f"{'mom_1 (noise)':<12}{ic0:>8.3f}{t0:>9.2f}   <- no alpha planted")
    print("  (the planted world is momentum-driven, so the reversion and breakout")
    print("   factors show zero/negative IC here — honest feature attribution)")

    sig = combine_signals(fm, {"mom_1": 0.7, "mom_4": 0.3})
    pos = vol_target_positions(sig, bars["close"].pct_change().fillna(0.0), 0.10)
    pos = session_filter(pos, bars["hour"], ("london", "overlap", "ny"))
    pos = carry_gate(pos, bars["hour"], fm["carry"])
    cfg = BacktestConfig(
        pip_size=EURUSD.pip_size,
        spread_pips_by_session=EURUSD.spread_pips,
        r_base=0.03,
        r_quote=0.05,
    )
    _, s = IntradayBacktester(cfg).run(bars, pos)
    print(f"\nSession-filtered vol-targeted backtest, EURUSD costs ({n_days} days):")
    print(
        f"  gross {s['gross_pips']:+8.1f} pips | costs {s['cost_pips']:6.1f} | "
        f"carry {s['carry_pips']:+6.2f} | net {s['net_pips']:+8.1f} pips "
        f"| Sharpe {s['sharpe_ann']:.2f} | trades {s['n_trades']}"
    )
    _, s_em = IntradayBacktester(
        BacktestConfig(pip_size=EURUSD.pip_size, spread_pips_by_session=EM_SPREADS)
    ).run(bars, pos)
    print(
        f"Same alpha under EM-style spreads:            "
        f"gross {s_em['gross_pips']:+8.1f} | costs {s_em['cost_pips']:6.1f} | "
        f"net {s_em['net_pips']:+8.1f} pips  <- costs flip profitability"
    )


# --------------------------------------------------------------- execution
def scheduler_demo() -> None:
    hr(f"2. EXECUTION — {PARENT_MM:.0f}mm EURUSD buy: scheduler comparison "
       f"({N_REP} replications)")

    def run_grid(label: str, start: float, horizon: float) -> None:
        sim = MarketSimulator(EURUSD, start_hour=start, horizon_hours=horizon, dt_minutes=5.0)
        eta = eta_from_depth(sim.depth_bucket, k_eta=0.02)
        scheds = {
            "TWAP": twap_schedule(PARENT_MM, sim.n_buckets),
            "liquidity-weighted": liquidity_weighted_schedule(PARENT_MM, sim.depth_bucket),
            "POV 5% (analog)": pov_schedule(PARENT_MM, sim.depth_bucket, 0.05),
            "piecewise-AC (lam=1e-5)": piecewise_ac_schedule(
                PARENT_MM, eta, sim.sigma_bucket_pips, 1e-5
            ),
        }
        print(f"\n{label}:")
        print(f"{'schedule':<26}{'ctrl cost (pips)':>17}{'IS mean':>10}{'IS std':>10}")
        for name, q in scheds.items():
            ctrl, tot = [], []
            for s in range(N_REP):
                d = decompose_implementation_shortfall(sim.execute(q, FirmVenue(), seed=s))
                ctrl.append(d["spread_temporary"] + d["permanent_impact"])
                tot.append(d["total"])
            print(
                f"{name:<26}{np.mean(ctrl):>17.3f}{np.mean(tot):>10.2f}"
                f"{np.std(tot):>10.2f}"
            )
        print("  ctrl cost = spread + temporary + permanent (deterministic given the")
        print("  schedule); IS std is execution risk vs arrival over the replications.")

    run_grid("Full 24h day (288 x 5-min buckets)", 0.0, 24.0)
    run_grid("London-only window 07:00-16:00 (108 buckets)", 7.0, 9.0)


def venue_demo() -> None:
    hr("3. VENUES — last-look dealer stream vs firm-liquidity ECN "
       f"({N_REP} replications)")
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    lw = liquidity_weighted_schedule(PARENT_MM, sim.depth_bucket)
    for alpha, label in ((0.0, "uninformed flow (alpha = 0)"),
                         (0.5, "informed flow (alpha = 0.5 pips/bucket)")):
        rows = {"last-look": [], "firm-ecn": []}
        rej, rejc = [], []
        for s in range(N_REP):
            rl = sim.execute(lw, LastLookVenue(), seed=s, alpha_pips_per_bucket=alpha)
            rf = sim.execute(lw, FirmVenue(), seed=s, alpha_pips_per_bucket=alpha)
            vc = venue_comparison({"last-look": rl, "firm-ecn": rf})
            for k in rows:
                rows[k].append(vc[k])
            rej.append(rl.rejection_rate)
            rejc.append(vc["last-look"]["rejection_cost_pips"])
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
        print(f"  paired diff (LL - firm): {diff.mean():+.3f} pips "
              f"(SE {diff.std(ddof=1) / np.sqrt(N_REP):.4f})")


def fix_demo() -> None:
    hr("4. WM/R 4PM LONDON FIX — fix-targeting vs TWAP tracking error "
       f"({N_REP} replications)")
    sim = MarketSimulator(EURUSD, start_hour=14.0, horizon_hours=3.0, dt_minutes=1.0)
    q_fix = fix_schedule(100.0, sim.times_hours, 1.0)
    q_twap = twap_schedule(100.0, sim.n_buckets)
    te = {"fix-targeting (5-min window)": [], "TWAP 14:00-17:00": []}
    for s in range(N_REP):
        rf = sim.execute(q_fix, seed=s)
        rt = sim.execute(q_twap, seed=s)
        te["fix-targeting (5-min window)"].append(slippage_vs_benchmark(rf, fix_benchmark(rf)))
        te["TWAP 14:00-17:00"].append(slippage_vs_benchmark(rt, fix_benchmark(rt)))
    print(f"\n100mm EURUSD benchmarked to the fix print (window TWAP of mids):")
    print(f"{'schedule':<30}{'TE mean (pips)':>15}{'TE std (pips)':>15}")
    for k, v in te.items():
        print(f"{k:<30}{np.mean(v):>15.3f}{np.std(v):>15.3f}")
    print("  The fix algo pays spread+impact but carries near-zero benchmark risk;")
    print("  the 2013 fix-rigging scandal and the 2015 window reform are discussed")
    print("  in docs/METHODOLOGY.md and docs/VALIDATION.md.")


def tca_demo() -> None:
    hr("5. TCA REPORT — 500mm EURUSD, liquidity-weighted, firm ECN (seed 0)")
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
    print("FX ALGORITHMIC TRADING & EXECUTION PIPELINE (fx_algo)")
    print("OTC market structure: no tape, session liquidity, last-look, WM/R fix")
    signal_demo()
    scheduler_demo()
    venue_demo()
    fix_demo()
    tca_demo()
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
