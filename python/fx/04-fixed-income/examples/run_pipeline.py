"""End-to-end FX-linked fixed income pipeline.

Bootstraps USD & EUR curves from synthetic quotes, builds the CIP and
basis-adjusted FX forward curves, demonstrates the 5y basis mispricing,
values a sample book, produces the full risk report (FX delta, per-currency
DV01/KRD ladders, basis DV01), runs joint scenarios including the 2008-style
basis blowout, shows forward carry, and runs the CIP arbitrage detector.

Run from the project root:  python examples/run_pipeline.py   (< 5 s)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd

from fx_rates import (
    CIPQuotes,
    FXForward,
    Scenario,
    apply_scenario,
    carry_table,
    cip_forward,
    detect_cip_arbitrage,
    forward_points_table,
    historical_scenarios,
    key_rate_dv01,
    market_forward,
    no_arb_bounds,
    book_risk_report,
    reprice_deposits,
    reprice_swaps,
    scenario_table,
    solve_par_basis,
    solve_par_rate_quote,
)
from fx_rates.data import build_market_state, generate_market_quotes, sample_book

pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
T0 = time.time()
SEED = 42


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ------------------------------------------------------------------ #
hr("1. BOOTSTRAP USD & EUR CURVES (regime: normal, seed 42)")
quotes = generate_market_quotes("normal", seed=SEED)
market = build_market_state(quotes)
print(f"EURUSD spot: {market.spot:.4f}")
for name, curve, deps, swps in [
    ("USD", market.domestic_curve, quotes.domestic_deposits, quotes.domestic_swaps),
    ("EUR", market.foreign_curve, quotes.foreign_deposits, quotes.foreign_swaps),
]:
    zs = {t: curve.zero_rate(t) for t in [0.25, 1.0, 2.0, 5.0, 10.0]}
    print(f"{name} zeros (cc): " + "  ".join(f"{t}y={z * 100:.3f}%" for t, z in zs.items()))
    print(
        f"   reprice deposits: {reprice_deposits(curve, deps):.2e}"
        f"   reprice swaps: {reprice_swaps(curve, swps):.2e}   (contract: < 1e-10)"
    )

# ------------------------------------------------------------------ #
hr("2. FX FORWARD CURVE VIA CIP + FORWARD POINTS TABLE (pips)")
tbl = forward_points_table(market, [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
print(tbl.to_string(index=False))

# ------------------------------------------------------------------ #
hr("3. CROSS-CURRENCY BASIS: IGNORING IT MISPRICES THE 5Y FORWARD")
f_cip = cip_forward(market.spot, market.domestic_curve, market.foreign_curve, 5.0)
f_mkt = market_forward(market, 5.0)
print(f"5y EURUSD CIP forward (no basis) : {f_cip:.4f}")
print(f"5y EURUSD market forward (basis) : {f_mkt:.4f}")
print(f"Difference: {(f_mkt - f_cip) * 1e4:,.1f} pips "
      f"(5y basis {dict(quotes.basis_spreads)[5.0] * 1e4:.0f} bp)")
fwd_100m = FXForward(100e6, f_mkt, 5.0, label="100m 5y fwd @ market")
pv_true = fwd_100m.value(market)
pv_cip = fwd_100m.value(market.replace(basis_spreads=()))
print(f"100m EUR 5y forward struck at market forward:")
print(f"   PV with basis curve   : ${pv_true:,.2f}   (par by construction)")
print(f"   PV ignoring the basis : ${pv_cip:,.2f}   <-- fictitious 'edge'")

# ------------------------------------------------------------------ #
hr("4. VALUE THE BOOK (outrights + FX swap + 5y xccy swap)")
book = sample_book(market, seed=1)
for p in book:
    print(f"   {p.label:35s} PV = ${p.value(market):>15,.2f}")
xccy = book[-1]
print(f"\nXCCY par USD rate  : {solve_par_rate_quote(xccy, market) * 100:.4f}%  "
      f"(traded {xccy.rate_quote * 100:.4f}%)")
print(f"XCCY par basis     : {solve_par_basis(xccy, market) * 1e4:+.2f} bp flat shift to current basis curve")

# ------------------------------------------------------------------ #
hr("5. RISK REPORT (USD): FX DELTA, DV01 PER CURRENCY, BASIS DV01")
report = book_risk_report(book, market)
print(report.to_string())

hr("5b. KEY-RATE DV01 LADDERS FOR THE BOOK (USD per +1bp per pillar)")
class _Book:
    label = "BOOK"
    def value(self, m):
        return sum(p.value(m) for p in book)
agg = _Book()
krd = pd.DataFrame({
    "KRD_USD": key_rate_dv01(agg, market, "USD"),
    "KRD_EUR": key_rate_dv01(agg, market, "EUR"),
}).fillna(0.0)
print(krd[(krd.abs() > 1e-6).any(axis=1)].to_string())

# ------------------------------------------------------------------ #
hr("6. SCENARIO TABLE (joint spot / curves / basis shocks)")
scens = historical_scenarios() + [
    Scenario("Spot -10% only", spot_pct=-10.0),
    Scenario("USD +50bp only", domestic_bp=50.0),
]
sc_tbl = scenario_table(book, market, scens)
print(sc_tbl.to_string())
print(f"\nBase book PV: ${sc_tbl.attrs['base_pv']:,.2f}")

# ------------------------------------------------------------------ #
hr("7. CARRY / FORWARD-POINTS ROLL (long 10m EUR 1y forward)")
fwd = FXForward(10e6, market_forward(market, 1.0), 1.0, label="10m 1y fwd")
print(carry_table(fwd, market, [1 / 12, 0.25, 0.5]).to_string())
print("Long the low-yielder at a forward premium: negative carry, as expected.")

# ------------------------------------------------------------------ #
hr("8. CIP ARBITRAGE DETECTOR (6m EURUSD, bid/ask)")
r_d, r_f, tau = 0.0405, 0.0250, 0.5
mid_f = market.spot * (1 + r_d * tau) / (1 + r_f * tau)
clean = CIPQuotes(
    spot_bid=market.spot - 1e-4, spot_ask=market.spot + 1e-4,
    fwd_bid=mid_f - 1e-4, fwd_ask=mid_f + 1e-4,
    dom_rate_bid=r_d - 2.5e-4, dom_rate_ask=r_d + 2.5e-4,
    for_rate_bid=r_f - 2.5e-4, for_rate_ask=r_f + 2.5e-4, tau=tau,
)
res = detect_cip_arbitrage(clean)
lo, hi = no_arb_bounds(clean)
print(f"Consistent quotes: no-arb band [{lo:.4f}, {hi:.4f}] -> "
      f"arbitrage: {res.is_arbitrage}")
violated = CIPQuotes(
    spot_bid=clean.spot_bid, spot_ask=clean.spot_ask,
    fwd_bid=hi + 30e-4, fwd_ask=hi + 31e-4,   # forward 30 pips above the band
    dom_rate_bid=clean.dom_rate_bid, dom_rate_ask=clean.dom_rate_ask,
    for_rate_bid=clean.for_rate_bid, for_rate_ask=clean.for_rate_ask, tau=tau,
)
res_v = detect_cip_arbitrage(violated)
print(f"Planted violation (fwd 30 pips rich): arbitrage: {res_v.is_arbitrage}, "
      f"direction: {res_v.direction}")
print(f"   riskless P&L: {res_v.pnl * 1e4:,.2f} bp of notional "
      f"(${res_v.pnl * 100e6:,.0f} on $100m) at maturity")

print(f"\nPipeline completed in {time.time() - T0:.2f}s")
