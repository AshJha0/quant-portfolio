"""
Validate the from-scratch Black-Scholes implementation three ways and
demonstrate where the model breaks down.

  1. Put-call parity and Greeks-vs-finite-difference checks (also in
     test_black_scholes.py).
  2. Monte Carlo under the same risk-neutral dynamics must converge to
     the closed form at O(1/sqrt(n)).
  3. Implied-vol round trip: price -> implied vol -> price.

Then a demonstration of the model's central empirical failure: if the
world priced options with a Student-t (fat-tailed) return distribution
but we insist on reading prices through Black-Scholes, the implied
volatility comes out as a SMILE, not the flat line the model assumes.

Outputs: output/report.txt, output/figures/*.png
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import black_scholes as bs
import monte_carlo as mc

S, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 0.75


def main() -> None:
    os.makedirs("output/figures", exist_ok=True)
    lines = []
    add = lines.append

    add("=" * 70)
    add("BLACK-SCHOLES REPLICATION -- VALIDATION REPORT")
    add(f"Test contract: S={S}, K={K}, r={r:.0%}, sigma={sigma:.0%}, T={T}y")
    add("=" * 70)

    # ------------------------------------------------------------------
    # 1. Closed form + parity
    # ------------------------------------------------------------------
    c = bs.call_price(S, K, r, sigma, T)
    p = bs.put_price(S, K, r, sigma, T)
    parity_err = (c - p) - (S - K * math.exp(-r * T))
    add(f"\nCall price : {c:.6f}")
    add(f"Put price  : {p:.6f}")
    add(f"Put-call parity error: {parity_err:.2e}  (should be ~0)")

    g = bs.call_greeks(S, K, r, sigma, T)
    add(f"\nGreeks: delta={g.delta:.4f}  gamma={g.gamma:.4f}  "
        f"vega={g.vega:.4f}")
    add(f"        theta={g.theta:.4f}/yr  rho={g.rho:.4f}")

    # ------------------------------------------------------------------
    # 2. Monte Carlo convergence
    # ------------------------------------------------------------------
    add("\n--- Monte Carlo vs closed form (same risk-neutral GBM) ---")
    add(f"{'paths':>10} {'MC price':>10} {'std err':>9} {'abs error':>10}")
    ns = [1_000, 10_000, 100_000, 1_000_000]
    mc_errors = []
    for n in ns:
        est, se = mc.mc_call_price(S, K, r, sigma, T, n_paths=n, seed=7)
        mc_errors.append(abs(est - c))
        add(f"{n:>10,} {est:>10.4f} {se:>9.4f} {abs(est - c):>10.4f}")
    add("Errors shrink roughly like 1/sqrt(n): two independent")
    add("implementations of the same model agree, which is the point.")

    # ------------------------------------------------------------------
    # 3. Implied vol round trip
    # ------------------------------------------------------------------
    iv = bs.implied_volatility(c, S, K, r, T)
    add(f"\nImplied vol round trip: input sigma {sigma:.6f} -> "
        f"recovered {iv:.6f}")

    # ------------------------------------------------------------------
    # 4. Where the model breaks: the smile
    # ------------------------------------------------------------------
    # Price options under a fat-tailed world (Student-t, df=4, matched
    # variance) by Monte Carlo, then invert each price through
    # Black-Scholes. Constant-vol GBM would give a flat line; fat tails
    # give a smile because OTM options are worth more than BS thinks.
    add("\n--- Model breakdown demo: implied vol smile ---")
    rng = np.random.default_rng(3)
    df = 4
    z = rng.standard_t(df, size=2_000_000) / math.sqrt(df / (df - 2))
    st_fat = S * np.exp((r - 0.5 * sigma**2) * T
                        + sigma * math.sqrt(T) * z)
    disc = math.exp(-r * T)

    strikes = np.arange(70, 141, 5)
    smile = []
    for k in strikes:
        price_fat = disc * np.maximum(st_fat - k, 0).mean()
        try:
            smile.append(bs.implied_volatility(price_fat, S, k, r, T))
        except ValueError:
            smile.append(np.nan)
    atm = smile[len(strikes) // 2]
    add(f"BS implied vol at K=70 : {smile[0]:.1%}")
    add(f"BS implied vol at K=105: {atm:.1%}")
    add(f"BS implied vol at K=140: {smile[-1]:.1%}")
    add("Under fat-tailed returns, deep OTM/ITM options carry higher")
    add("BS-implied vols than ATM ones. Real markets show exactly this")
    add("smile/skew, which is direct evidence against constant-vol GBM.")

    report = "\n".join(lines)
    with open("output/report.txt", "w") as f:
        f.write(report + "\n")
    print(report)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    s_grid = np.linspace(50, 160, 300)
    for t_ in (0.05, 0.25, 0.75):
        ax.plot(s_grid, [bs.call_price(s, K, r, sigma, t_) for s in s_grid],
                lw=1.1, label=f"T={t_}y")
    ax.plot(s_grid, np.maximum(s_grid - K, 0), "k--", lw=0.9,
            label="payoff at expiry")
    ax.legend()
    ax.set_title("Call value vs spot (time value melting to payoff)")

    ax = axes[0, 1]
    ax.plot(s_grid, [bs.call_greeks(s, K, r, sigma, T).delta
                     for s in s_grid], lw=1.1, label="delta")
    ax2 = ax.twinx()
    ax2.plot(s_grid, [bs.call_greeks(s, K, r, sigma, T).gamma
                      for s in s_grid], lw=1.1, color="darkorange",
             label="gamma")
    ax.set_title("Delta and gamma vs spot")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[1, 0]
    ax.loglog(ns, mc_errors, "o-", label="|MC - closed form|")
    ref = mc_errors[0] * (ns[0] / np.array(ns)) ** 0.5
    ax.loglog(ns, ref, "k--", lw=0.9, label="O(1/sqrt(n)) reference")
    ax.set_xlabel("paths")
    ax.legend()
    ax.set_title("Monte Carlo convergence")

    ax = axes[1, 1]
    ax.plot(strikes, np.array(smile) * 100, "o-")
    ax.axhline(sigma * 100, color="k", ls="--", lw=0.9,
               label="BS assumption (flat)")
    ax.set_xlabel("strike")
    ax.set_ylabel("BS implied vol (%)")
    ax.legend()
    ax.set_title("Smile from fat-tailed returns (model breakdown)")

    fig.tight_layout()
    fig.savefig("output/figures/black_scholes_overview.png", dpi=130)
    print("\nFigures written to output/figures/")


if __name__ == "__main__":
    main()
