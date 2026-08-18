"""End-to-end pipeline: data -> pricing -> Greeks -> validation -> hedging.

Reproduces every number quoted in README.md and docs/VALIDATION.md.
Runs offline, fully seeded, in well under 60 seconds:

    PYTHONPATH=src python examples/run_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from eq_options import (
    bs_greeks,
    bs_price,
    compare_models,
    crr_price,
    early_exercise_premium,
    implied_vol,
    mc_convergence_table,
    pnl_std_vs_frequency,
    simulate_delta_hedge,
    tree_convergence_table,
)
from eq_options.data import synthetic_chain

# Reference contract used throughout the report.
S, K, T, R, Q, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.01, 0.20

pd.set_option("display.width", 110)
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    t_start = time.perf_counter()
    print("eq_options pipeline report")
    print(f"reference contract: S={S} K={K} T={T}y r={R} q={Q} sigma={SIGMA}")

    # 1 ---------------------------------------------------------------
    hr("1. Prices across models (European call, then put)")
    for otype in ("call", "put"):
        df = compare_models(S, K, T, R, SIGMA, Q, otype,
                            n_steps=1000, n_paths=200_000, seed=42)
        print(f"\n[{otype}]")
        print(df.to_string())

    amer_put = crr_price(S, K, T, R, SIGMA, Q, "put", "american", 2000)
    prem = early_exercise_premium(S, K, T, R, SIGMA, Q, "put", 2000)
    print(f"\nAmerican put (CRR 2000 steps): {amer_put:.6f}"
          f"  early-exercise premium: {prem:.6f}")

    # 2 ---------------------------------------------------------------
    hr("2. Greeks table (analytic BS, call)")
    g = bs_greeks(S, K, T, R, SIGMA, Q, "call")
    rows = pd.DataFrame([g.as_dict()]).T
    rows.columns = ["value"]
    print(rows.to_string())
    print("\nunits: theta per year, vega per unit vol, rho per unit rate")

    # 3 ---------------------------------------------------------------
    hr("3. Convergence: CRR tree -> BS (European call)")
    print(tree_convergence_table(S, K, T, R, SIGMA, Q, "call").to_string())

    hr("4. Convergence: Monte Carlo -> BS (European call, antithetic+CV)")
    print(mc_convergence_table(S, K, T, R, SIGMA, Q, "call", seed=42).to_string())

    # 5 ---------------------------------------------------------------
    hr("5. Delta-hedging P&L (short 3M ATM call, hedged at true vol)")
    freqs = [4, 16, 64, 256]
    stds = pnl_std_vs_frequency(freqs, 100.0, 100.0, 0.25, 0.02, 0.20,
                                n_paths=4000)
    tbl = pd.DataFrame({
        "n_rebalance": freqs,
        "pnl_std": [stds[n] for n in freqs],
        "std_x_sqrt_n": [stds[n] * n**0.5 for n in freqs],
    }).set_index("n_rebalance")
    print(tbl.to_string())
    print("(std_x_sqrt_n ~ constant confirms the 1/sqrt(N) law)")

    res_true = simulate_delta_hedge(100, 100, 0.25, 0.02, 0.20,
                                    n_rebalance=128, n_paths=8000, seed=17)
    print(f"\nhedged at true vol:      mean P&L = {res_true.mean:+.4f}"
          f"  (SE {res_true.mean_se:.4f}), std = {res_true.std:.4f}")

    res_wrong = simulate_delta_hedge(100, 100, 0.25, 0.02,
                                     sigma_realized=0.15, sigma_hedge=0.25,
                                     n_rebalance=128, n_paths=8000, seed=21)
    print(f"sold@25v, realized 15v:  mean P&L = {res_wrong.mean:+.4f}"
          f"  vs gamma-weighted vol-spread theory {res_wrong.theory_pnl:+.4f}")

    res_tc = simulate_delta_hedge(100, 100, 0.25, 0.02, 0.20, n_rebalance=128,
                                  n_paths=8000, tc_rate=5e-4, seed=17)
    print(f"5bp transaction costs:   mean P&L = {res_tc.mean:+.4f}"
          f"  (cost drag {res_true.mean - res_tc.mean:.4f})")

    # 6 ---------------------------------------------------------------
    hr("6. Implied-vol round trip on a synthetic skewed chain")
    chain = synthetic_chain(S0=100.0, r=0.03, q=0.01, expiries=(0.25, 1.0))
    calls = chain[chain["type"] == "call"].copy()
    calls["iv_recovered"] = [
        implied_vol(row.price, 100.0, row.strike, row.expiry, 0.03, 0.01, "call")
        for row in calls.itertuples()
    ]
    calls["abs_err"] = (calls["iv_recovered"] - calls["iv"]).abs()
    worst = calls["abs_err"].max()
    show = calls[calls["expiry"] == 0.25][["strike", "iv", "iv_recovered", "abs_err"]]
    print(show.to_string(index=False))
    print(f"\nworst |sigma_recovered - sigma_quoted| over "
          f"{len(calls)} quotes: {worst:.2e}")

    elapsed = time.perf_counter() - t_start
    print(f"\npipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
