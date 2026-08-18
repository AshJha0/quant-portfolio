"""End-to-end FX vol surface pipeline: EURUSD + USDJPY demos.

Broker quotes -> five-point smiles -> delta->strike solving (pa vs
unadjusted) -> SVI + vanna-volga fits -> surface + calendar check ->
Heston calibration (incl. ground-truth recovery) -> digital pricing
three ways -> Greeks incl. vanna/volga -> Monte Carlo cross-check.

Run from the project root:  python examples/run_pipeline.py   (< 150 s)
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from fx_surface import (
    HestonParams,
    calibrate_heston,
    gk_delta,
    gk_digital,
    gk_forward,
    gk_greeks,
    heston_digital,
    heston_greeks_fd,
    mc_price,
    price_cos,
    smile_digital,
    solve_pillar_strikes,
    strike_from_delta,
    vols_from_quotes,
)
from fx_surface.data import (
    calibration_slices,
    eurusd_market,
    market_from_heston,
    usdjpy_market,
)
from fx_surface.surface import build_slice, build_surface

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

PILLARS = ("10p", "25p", "atm", "25c", "10c")
T0 = time.time()


def hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- 1. quotes
hdr("1. Broker quotes -> five-point smiles (vol %, per expiry)")
markets = {"EURUSD": eurusd_market(), "USDJPY": usdjpy_market()}
for pair, mkt in markets.items():
    rows = []
    for ms in mkt.slices:
        v = vols_from_quotes(ms.quotes)
        rows.append(
            {"tenor": ms.label, "conv": ms.convention,
             "ATM": ms.quotes.atm * 100, "RR25": ms.quotes.rr25 * 100,
             "BF25": ms.quotes.bf25 * 100,
             **{p.upper(): v[p] * 100 for p in PILLARS}}
        )
    print(f"\n{pair}  (S = {mkt.S})")
    print(pd.DataFrame(rows).set_index("tenor").round(3))

# ------------------------------------------------- 2. delta -> strike table
hdr("2. Strike solving: USDJPY 1y, premium-adjusted vs unadjusted")
jpy = markets["USDJPY"]
ms = [s for s in jpy.slices if s.label == "1y"][0]
v = vols_from_quotes(ms.quotes)
k_pa = solve_pillar_strikes(v, jpy.S, ms.T, ms.r_d, ms.r_f, "spot_pa")
k_un = solve_pillar_strikes(v, jpy.S, ms.T, ms.r_d, ms.r_f, "spot")
tbl = pd.DataFrame(
    {
        "vol %": {p: v[p] * 100 for p in PILLARS},
        "K (pa, correct)": k_pa,
        "K (unadjusted)": k_un,
        "diff (JPY pips)": {p: (k_un[p] - k_pa[p]) * 100 for p in PILLARS},
    }
)
print(f"\nUSDJPY 1y: S={jpy.S}, F={gk_forward(jpy.S, ms.T, ms.r_d, ms.r_f):.3f}, "
      f"r_d(JPY)={ms.r_d:.3%}, r_f(USD)={ms.r_f:.3%}")
print(tbl.round(3))
print("\nSame quotes, different delta convention => strikes shift by up to "
      f"{max(abs(k_un[p] - k_pa[p]) for p in PILLARS):.2f} JPY.")

# wrong-convention vol error (desk-realistic bug demo, quoted in VALIDATION.md)
surf_jpy = build_surface(jpy, "svi")
wrong_slices = [
    build_slice(m.label, m.T, jpy.S, m.r_d, m.r_f, m.quotes,
                m.convention.replace("_pa", ""), "svi")
    for m in jpy.slices
]
from fx_surface.surface import FXVolSurface

surf_wrong = FXVolSurface(wrong_slices)
errs = {}
for sl in surf_jpy.slices:
    e = [abs(surf_wrong.vol(K, sl.T) - surf_jpy.vol(K, sl.T)) * 100
         for K in sl.strikes.values()]
    errs[sl.label] = max(e)
print("\nIf USDJPY pa quotes are mistakenly solved with unadjusted deltas, the")
print("marked surface is wrong at the pillar strikes by (vol points):")
print(pd.Series(errs).round(3).to_string())

# ----------------------------------------------------- 3. SVI vs vanna-volga
hdr("3. SVI vs vanna-volga smile fits (EURUSD 3m)")
eur = markets["EURUSD"]
ms3 = eur.slices[2]
sl_svi = build_slice(ms3.label, ms3.T, eur.S, ms3.r_d, ms3.r_f, ms3.quotes,
                     ms3.convention, "svi")
sl_vv = build_slice(ms3.label, ms3.T, eur.S, ms3.r_d, ms3.r_f, ms3.quotes,
                    ms3.convention, "vv")
rows = []
for name, delta, cp in [("35dP", 0.35, -1), ("15dP", 0.15, -1),
                        ("15dC", 0.15, +1), ("5dC", 0.05, +1)]:
    sig0 = sl_svi.vols["atm"]
    K = strike_from_delta(delta, cp, sig0, eur.S, ms3.T, ms3.r_d, ms3.r_f, ms3.convention)
    rows.append({"point": name, "K": K,
                 "SVI vol %": float(sl_svi.smile.vol(K)) * 100,
                 "VV vol %": float(sl_vv.smile.vol(K)) * 100})
df = pd.DataFrame(rows).set_index("point")
df["diff bp"] = (df["SVI vol %"] - df["VV vol %"]) * 100
print(df.round(3))
print("\nBody (35d-15d): the two constructions agree to a few bp.  In the far")
print("wings (5d and beyond) they diverge: VV extrapolates the quadratic")
print("vega/vanna/volga cost, SVI enforces linear total variance.")
ok, gmin = sl_svi.smile.is_butterfly_arbitrage_free()
print(f"Durrleman butterfly check (SVI 3m): min g = {gmin:.4f} -> "
      f"{'arbitrage-free' if ok else 'ARBITRAGE'}")

# --------------------------------------------------------------- 4. surface
hdr("4. Surfaces + calendar arbitrage check")
surf_eur = build_surface(eur, "svi")
for pair, surf in (("EURUSD", surf_eur), ("USDJPY", surf_jpy)):
    report = surf.calendar_arbitrage_report()
    atm_ts = {sl.label: surf.vol_atm(sl.T) * 100 for sl in surf.slices}
    print(f"{pair}: ATM term structure (%): "
          + "  ".join(f"{k}={v:.2f}" for k, v in atm_ts.items()))
    print(f"{pair}: calendar-arbitrage violations at fixed delta: {len(report)}")
q = [(1.12, 0.4), (1.05, 1.5)]
print("Query API: " + ", ".join(
    f"vol(K={K}, T={T:.2f}y) = {surf_eur.vol(K, T):.4%}" for K, T in q))

# ----------------------------------------------------------- 5. calibration
hdr("5. Heston calibration")
true_p = HestonParams(v0=0.0064, kappa=1.8, theta=0.008, xi=0.45, rho=-0.35)
gt = market_from_heston(true_p)
res_gt = calibrate_heston(gt.S, calibration_slices(gt))
rec = pd.DataFrame(
    {"true": [true_p.v0, true_p.kappa, true_p.theta, true_p.xi, true_p.rho],
     "recovered": [res_gt.params.v0, res_gt.params.kappa, res_gt.params.theta,
                   res_gt.params.xi, res_gt.params.rho]},
    index=["v0", "kappa", "theta", "xi", "rho"],
)
rec["rel err %"] = (rec.recovered / rec.true - 1) * 100
print("\nGround-truth recovery (quotes generated from known Heston params):")
print(rec.round(5))
print(f"RMSE: {res_gt.rmse_vol_pts:.4f} vol pts over 30 quotes")

results = {}
for pair, mkt in markets.items():
    res = calibrate_heston(mkt.S, calibration_slices(mkt))
    results[pair] = res
    print(f"\n{pair}: {res.summary()}")
print("\nFX pattern: EURUSD (mild, near-symmetric smile) -> small |rho|; "
      f"USDJPY (strong JPY-call skew) -> large negative rho "
      f"({results['EURUSD'].params.rho:+.2f} vs {results['USDJPY'].params.rho:+.2f}).")
print("kappa/xi ridge: vanillas pin xi^2/kappa, not each separately "
      f"(GT: {true_p.xi ** 2 / true_p.kappa:.4f} -> "
      f"{res_gt.params.xi ** 2 / res_gt.params.kappa:.4f}).")

# ------------------------------------------------------------- 6. digitals
hdr("6. Digital (one-touch approx): flat GK vs vanna-volga vs Heston")
h_eur = results["EURUSD"].params
T_d, K_d = 0.5, 1.15  # topside digital, ~15d region
ms6 = eur.slices[3]  # 6m slice
sl6_vv = build_slice(ms6.label, ms6.T, eur.S, ms6.r_d, ms6.r_f, ms6.quotes,
                     ms6.convention, "vv")
atm6 = sl6_vv.vols["atm"]
flat = gk_digital(eur.S, K_d, ms6.T, ms6.r_d, ms6.r_f, atm6, 1)
vv_dig = smile_digital(sl6_vv.smile, K_d, eur.S, ms6.T, ms6.r_d, ms6.r_f, 1)
hes_dig = heston_digital(eur.S, K_d, ms6.T, ms6.r_d, ms6.r_f, h_eur, 1)
print(pd.Series(
    {"flat GK (ATM vol)": flat, "vanna-volga (smile)": vv_dig,
     "Heston (calibrated)": hes_dig}).round(5).to_string())
print("\nThe three differ because a digital = -dC/dK picks up the smile SLOPE:")
print("flat GK ignores it; VV adds the static skew correction -vega*dsigma/dK;")
print("Heston embeds the model's own (dynamic) smile.  Touch products on an FX")
print("desk are marked VV-adjusted and reserved against the model spread.")

# --------------------------------------------------------------- 7. greeks
hdr("7. Greeks: Heston FD vs BS-world (EURUSD 6m, 25d call pillar)")
K_g = strike_from_delta(0.25, +1, sl6_vv.vols["25c"], eur.S, ms6.T,
                        ms6.r_d, ms6.r_f, ms6.convention)
hg = heston_greeks_fd(eur.S, K_g, ms6.T, ms6.r_d, ms6.r_f, h_eur, 1)
bg = gk_greeks(eur.S, K_g, ms6.T, ms6.r_d, ms6.r_f, sl6_vv.vols["25c"], 1)
cmp_tbl = pd.DataFrame({"Heston FD": hg, "GK analytic (25c vol)": bg})
print(f"K = {K_g:.4f} (25-delta call)")
print(cmp_tbl.round(5))
print("\nBoth rhos are first-class: rho_d > 0 > rho_f for a call.  Vanna/volga")
print("are the FX risk buckets hedged with RR / fly trades (docs/DESK_GUIDE.md).")
print("OTM signs agree across models (vanna > 0, volga > 0 topside); exactly")
print("AT the money both buckets pass through ~0 and their sign is model-")
print("dependent - desks bucket risk at the 25d/10d pillars, not at ATM.")

# ------------------------------------------------------------------- 8. MC
hdr("8. Monte Carlo cross-check (EURUSD calibrated Heston, 1y ATM-fwd call)")
T_mc = 1.0
rd1, rf1 = surf_eur.rates(T_mc)
K_mc = gk_forward(eur.S, T_mc, rd1, rf1)
ref = float(price_cos(eur.S, K_mc, T_mc, rd1, rf1, h_eur, 1))
for scheme, steps in (("euler_ft", 250), ("qe", 24)):
    p, se = mc_price(eur.S, K_mc, T_mc, rd1, rf1, h_eur, 1,
                     n_paths=200_000, n_steps=steps, scheme=scheme, seed=7)
    print(f"{scheme:9s} ({steps:3d} steps): {p:.6f} +- {se:.6f}  "
          f"(COS {ref:.6f}, |z| = {abs(p - ref) / se:.2f})")

print(f"\nTotal pipeline time: {time.time() - T0:.1f} s")
