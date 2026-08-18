"""End-to-end FX options pipeline: EURUSD and USDJPY worked examples.

Reproduces the numbers quoted in README.md and docs/VALIDATION.md.
Runtime target < 60 s.  Run from the project root:

    python examples/run_pipeline.py
"""

from __future__ import annotations

import time

import pandas as pd

from fx_options import (analytic_greeks, atm_dns_strike, atm_forward_strike,
                        binomial_convergence_table, binomial_price,
                        cip_forward, compare_models, delta, forward_points,
                        gk_price, hedge_frequency_study, implied_vol,
                        mc_convergence_table, simulate_delta_hedge,
                        strike_from_delta, synthetic_forward_from_options)
from fx_options.data.synthetic import synthetic_vol_quotes

pd.set_option("display.float_format", lambda x: f"{x:,.8f}")

# ----------------------------------------------------------------------
# Market snapshots (stylised but realistic levels).
# EURUSD: premium in USD (quote ccy) -> unadjusted deltas standard.
# USDJPY: pip = 0.01, premium conventionally in USD = BASE ccy ->
#         premium-adjusted deltas are the market standard.
# ----------------------------------------------------------------------
EURUSD = dict(S=1.1000, T=0.5, r_d=0.0425, r_f=0.0290, sigma=0.0825)
USDJPY = dict(S=147.50, T=0.5, r_d=0.0050, r_f=0.0525, sigma=0.1075)


def header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def forwards_section() -> None:
    header("1. FORWARDS (covered interest parity)")
    for name, mkt, pip in (("EURUSD", EURUSD, 1e4), ("USDJPY", USDJPY, 1e2)):
        F = cip_forward(mkt["S"], mkt["T"], mkt["r_d"], mkt["r_f"])
        pts = forward_points(mkt["S"], mkt["T"], mkt["r_d"], mkt["r_f"],
                             pip_factor=pip)
        print(f"{name}: S={mkt['S']:.4f}  F(6m)={F:.6f}  "
              f"forward points={pts:+.2f} pips "
              f"({'premium' if pts > 0 else 'discount'} on base ccy)")
        K = F
        c = gk_price(mkt["S"], K, mkt["T"], mkt["r_d"], mkt["r_f"],
                     mkt["sigma"], "call")
        p = gk_price(mkt["S"], K, mkt["T"], mkt["r_d"], mkt["r_f"],
                     mkt["sigma"], "put")
        F_syn = synthetic_forward_from_options(c, p, K, mkt["T"], mkt["r_d"])
        print(f"        synthetic forward from options: {F_syn:.10f} "
              f"(CIP {F:.10f}, diff {abs(F_syn - F):.2e})")


def pricing_section() -> None:
    header("2. CROSS-MODEL PRICES (ATM-forward strikes)")
    for name, mkt in (("EURUSD", EURUSD), ("USDJPY", USDJPY)):
        K = atm_forward_strike(mkt["S"], mkt["T"], mkt["r_d"], mkt["r_f"])
        print(f"\n{name} 6m ATM-forward call, K={K:.4f}:")
        print(compare_models(mkt["S"], K, mkt["T"], mkt["r_d"], mkt["r_f"],
                             mkt["sigma"], "call", binomial_steps=1000,
                             mc_paths=200_000, mc_seed=42))


def greeks_section() -> None:
    header("3. GREEKS — both rhos, vanna/volga (EURUSD 6m ATMF call)")
    K = atm_forward_strike(EURUSD["S"], EURUSD["T"], EURUSD["r_d"],
                           EURUSD["r_f"])
    g = analytic_greeks(EURUSD["S"], K, EURUSD["T"], EURUSD["r_d"],
                        EURUSD["r_f"], EURUSD["sigma"], "call")
    for field, value in g.as_dict().items():
        print(f"  {field:<14s} {value:+.8f}")
    print("  (rho_domestic > 0, rho_foreign < 0 for a call: higher r_d "
          "lifts the forward, higher r_f is carry lost)")


def delta_conventions_section() -> None:
    header("4. DELTA CONVENTIONS — same option, four deltas")
    for name, mkt in (("EURUSD", EURUSD), ("USDJPY", USDJPY)):
        K = atm_forward_strike(mkt["S"], mkt["T"], mkt["r_d"], mkt["r_f"])
        rows = {conv: delta(mkt["S"], K, mkt["T"], mkt["r_d"], mkt["r_f"],
                            mkt["sigma"], "call", conv)
                for conv in ("spot", "forward", "spot_pa", "forward_pa")}
        std = "spot/forward (premium in quote ccy)" if name == "EURUSD" \
            else "PREMIUM-ADJUSTED (premium paid in USD = base ccy)"
        print(f"\n{name} ATMF call (market standard: {std}):")
        for conv, v in rows.items():
            print(f"  {conv:<12s} {v:+.6f}")
        k25 = strike_from_delta(0.25, mkt["S"], mkt["T"], mkt["r_d"],
                                mkt["r_f"], mkt["sigma"], "call",
                                "forward_pa" if name == "USDJPY" else "spot")
        conv_used = "forward_pa" if name == "USDJPY" else "spot"
        rt = delta(mkt["S"], k25, mkt["T"], mkt["r_d"], mkt["r_f"],
                   mkt["sigma"], "call", conv_used)
        print(f"  25d call strike ({conv_used}): K={k25:.4f} "
              f"(round-trip delta {rt:.10f})")
        k_atmf = atm_forward_strike(mkt["S"], mkt["T"], mkt["r_d"],
                                    mkt["r_f"])
        k_dns = atm_dns_strike(mkt["S"], mkt["T"], mkt["r_d"], mkt["r_f"],
                               mkt["sigma"], conv_used)
        print(f"  ATM-forward K={k_atmf:.4f}  vs  ATM-DNS K={k_dns:.4f} "
              f"({conv_used} convention)")


def implied_vol_section() -> None:
    header("5. IMPLIED VOL ROUND-TRIP")
    worst = 0.0
    for mkt in (EURUSD, USDJPY):
        for k_mult in (0.93, 1.0, 1.07):
            for ot in ("call", "put"):
                K = mkt["S"] * k_mult
                px = gk_price(mkt["S"], K, mkt["T"], mkt["r_d"], mkt["r_f"],
                              mkt["sigma"], ot)
                iv = implied_vol(px, mkt["S"], K, mkt["T"], mkt["r_d"],
                                 mkt["r_f"], ot)
                worst = max(worst, abs(iv - mkt["sigma"]))
    print(f"12 options (2 pairs x 3 strikes x call/put): "
          f"max |sigma_implied - sigma_true| = {worst:.2e}")


def convergence_section() -> None:
    header("6. CONVERGENCE — binomial and Monte Carlo vs GK (EURUSD)")
    K = atm_forward_strike(EURUSD["S"], EURUSD["T"], EURUSD["r_d"],
                           EURUSD["r_f"])
    args = (EURUSD["S"], K, EURUSD["T"], EURUSD["r_d"], EURUSD["r_f"],
            EURUSD["sigma"], "call")
    print("\nBinomial (CRR) -> GK:")
    print(binomial_convergence_table(*args))
    print("\nMonte Carlo (antithetic + control variate) -> GK:")
    print(mc_convergence_table(*args))
    am = binomial_price(*args[:-1], "call", steps=1000, exercise="american")
    eu = binomial_price(*args[:-1], "call", steps=1000, exercise="european")
    print(f"\nAmerican vs European call (r_f < r_d here): "
          f"premium {am - eu:.2e}")
    jm = USDJPY
    kj = atm_forward_strike(jm["S"], jm["T"], jm["r_d"], jm["r_f"])
    amj = binomial_price(jm["S"], kj, jm["T"], jm["r_d"], jm["r_f"],
                         jm["sigma"], "call", steps=1000,
                         exercise="american")
    euj = binomial_price(jm["S"], kj, jm["T"], jm["r_d"], jm["r_f"],
                         jm["sigma"], "call", steps=1000,
                         exercise="european")
    print(f"USDJPY American call (r_f = 5.25% >> r_d = 0.50%): early-"
          f"exercise premium {amj - euj:.6f} JPY per USD ({100 * (amj / euj - 1):.2f}%)")


def hedging_section() -> None:
    header("7. DELTA HEDGING (EURUSD 6m ATMF call, short, 2000 paths)")
    K = atm_forward_strike(EURUSD["S"], EURUSD["T"], EURUSD["r_d"],
                           EURUSD["r_f"])
    rows = hedge_frequency_study(
        EURUSD["S"], K, EURUSD["T"], EURUSD["r_d"], EURUSD["r_f"],
        EURUSD["sigma"], "call", frequencies=(4, 12, 50, 100, 250),
        n_paths=2000, rng=7)
    print(pd.DataFrame(rows).set_index("n_rebalances"))
    res_wrong = simulate_delta_hedge(
        EURUSD["S"], K, EURUSD["T"], EURUSD["r_d"], EURUSD["r_f"],
        EURUSD["sigma"], "call", sigma_hedge=EURUSD["sigma"] + 0.02,
        n_rebalances=100, n_paths=2000, rng=7)
    print(f"\nHedging at wrong vol (10.25% vs true 8.25%): mean P&L "
          f"{res_wrong.mean_pnl:+.6f} (sold too rich -> positive), "
          f"std {res_wrong.std_pnl:.6f}")
    res_tc = simulate_delta_hedge(
        EURUSD["S"], K, EURUSD["T"], EURUSD["r_d"], EURUSD["r_f"],
        EURUSD["sigma"], "call", n_rebalances=100, n_paths=2000, rng=7,
        transaction_cost_pips=1.0)
    print(f"With 1-pip half-spread, 100 rebalances: mean P&L "
          f"{res_tc.mean_pnl:+.6f}, avg costs {res_tc.total_transaction_costs:.6f} "
          f"USD per EUR notional")


def vol_quotes_section() -> None:
    header("8. SYNTHETIC VOL QUOTES (RR/BF terminology — smile is project 9)")
    for q in synthetic_vol_quotes(base_atm=0.0825, skew=-0.0125,
                                  smile=0.0030, rng=11):
        print(f"  {q.tenor_years * 12:4.1f}m  ATM={q.atm:.4f}  "
              f"25dRR={q.rr25:+.4f}  25dBF={q.bf25:.4f}  "
              f"10dRR={q.rr10:+.4f}  10dBF={q.bf10:.4f}")
    print("  (RR<0: base-ccy puts bid — EURUSD-style risk-off skew. "
          "Pricing here is flat-vol GK; smile construction in project 9.)")


def main() -> None:
    t0 = time.time()
    print("FX OPTIONS PRICING & GREEKS ENGINE — pipeline report")
    print("Conventions: BASE/QUOTE quotes; r_d = quote ccy, r_f = base ccy.")
    print("USDJPY notes: pips are 0.01; premia conventionally in USD "
          "(base ccy) -> premium-adjusted deltas are standard; Tokyo cut.")
    forwards_section()
    pricing_section()
    greeks_section()
    delta_conventions_section()
    implied_vol_section()
    convergence_section()
    hedging_section()
    vol_quotes_section()
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
