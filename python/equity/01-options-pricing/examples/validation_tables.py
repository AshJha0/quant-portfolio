"""Reproduce the tables in docs/VALIDATION.md that run_pipeline.py does not print.

Prints to stdout only. Run from the project root::

    PYTHONPATH=src python examples/validation_tables.py

3   Variance reduction: Monte Carlo standard error at 100k paths for each
    combination of antithetic sampling and the control variate (seed 7).
    Discrete-hedging noise: the Derman-Kamal estimate
    sqrt(pi/4) vega sigma / sqrt(N) against the simulated P&L std.

6.1 Smile contradiction: a 3M put chain priced at the synthetic skew vol
    versus a single ATM vol (S=100, r=2%, q=0; synthetic data).
6.3 Dividend modelling: a discrete dividend D=3 paid at t=0.1y, priced
    with the escrowed-dividend model (spot reduced by the PV of the
    dividend when it falls before expiry) versus a continuous yield q=3%
    (S=100, K=100, r=3%, sigma=20%).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eq_options import bs_greeks, bs_price, mc_price, simulate_delta_hedge
from eq_options.data import skew_vol


def smile_table() -> None:
    S, T, r = 100.0, 0.25, 0.02
    F = S * math.exp(r * T)
    atm = skew_vol(F, F, T)
    print("6.1 smile contradiction (3M puts, synthetic skew, S=100, r=2%)")
    print(f"{'K':>5} {'skew IV':>8} {'put@skew':>9} {'put@ATM':>8} {'mispricing':>11}")
    for K in (80.0, 90.0, 110.0):
        iv = skew_vol(K, F, T)
        p_skew = bs_price(S, K, T, r, iv, 0.0, "put")
        p_atm = bs_price(S, K, T, r, atm, 0.0, "put")
        print(f"{K:5.0f} {iv:8.1%} {p_skew:9.4f} {p_atm:8.4f} {p_atm / p_skew - 1:+11.0%}")


def dividend_table() -> None:
    S, K, r, sigma, D, t_div = 100.0, 100.0, 0.03, 0.20, 3.0, 0.1
    print("\n6.3 dividend modelling (ATM call, D=3 at t=0.1y, r=3%, sigma=20%)")
    print(f"{'T':>5} {'escrowed':>9} {'q=3%':>8} {'error':>8}")
    for T in (0.08, 0.12):
        spot = S - D * math.exp(-r * t_div) if t_div < T else S
        correct = bs_price(spot, K, T, r, sigma, 0.0, "call")
        smeared = bs_price(S, K, T, r, sigma, 0.03, "call")
        print(f"{T:5.2f} {correct:9.4f} {smeared:8.4f} {smeared - correct:+8.3f}")


def variance_reduction_table() -> None:
    print("3 variance reduction (S=100, K=100, T=1, r=5%, q=1%, sigma=20%, 100k paths, seed 7)")
    for anti, cv, label in ((False, False, "plain"), (True, False, "antithetic"),
                            (False, True, "control variate"), (True, True, "antithetic + CV")):
        res = mc_price(100, 100, 1.0, 0.05, 0.2, 0.01, "call", 100_000, anti, cv, 7)
        print(f"  {label:<16} SE {res.std_error:.4f}")


def hedging_noise_table() -> None:
    S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.02, 0.20
    vega = bs_greeks(S, K, T, r, sigma, 0.0, "call").vega
    print("\n5 discrete hedging noise (short 3M ATM call, hedged at true vol)")
    print(f"{'N':>5} {'simulated std':>14} {'sqrt(pi/4) vega sigma/sqrt(N)':>31}")
    for n in (4, 16, 64, 256):
        res = simulate_delta_hedge(S, K, T, r, sigma, n_rebalance=n, n_paths=4000, seed=11)
        print(f"{n:5d} {res.std:14.4f} {math.sqrt(math.pi / 4) * vega * sigma / math.sqrt(n):31.4f}")


if __name__ == "__main__":
    variance_reduction_table()
    hedging_noise_table()
    smile_table()
    dividend_table()
