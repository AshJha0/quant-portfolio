"""End-to-end equity vol surface pipeline.

Chain -> implied vols -> SVI slice fits (+ butterfly checks) -> total-variance
surface (+ calendar check) -> Heston calibration to a KNOWN-parameter surface
(recovery table) -> digital & variance-swap pricing under calibrated Heston vs
flat-vol BS -> Greeks table -> MC vs Fourier cross-check.

Run:  python examples/run_pipeline.py     (offline, seeded, < 150 s)
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

import eq_surface as es
from eq_surface.data import DEFAULT_TRUE_HESTON, generate_chain

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:10.6f}")

t_start = time.time()
SEED = 42


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
hr("1. Synthetic option chain (ground truth: KNOWN Heston parameters)")
chain = generate_chain(mode="heston", seed=SEED)
S, r, q = chain.spot, chain.rate, chain.div_yield
print(f"spot={S}  r={r}  q={q}  quotes={len(chain.df)}  "
      f"expiries={sorted(chain.df.expiry.round(4).unique())}")
print(f"true Heston: {chain.true_heston}")

# --------------------------------------------------------------------------- #
hr("2. Implied volatilities (robust Brent + Newton polish)")
expiries: list[float] = []
strikes_by_T: list[np.ndarray] = []
ivs_by_T: list[np.ndarray] = []
for T in sorted(chain.df.expiry.unique()):
    sl = chain.slice(T)
    iv = es.implied_vol_vector(sl.call_mid.values, S, sl.strike.values, float(T), r, q)
    ok = np.isfinite(iv)
    expiries.append(float(T))
    strikes_by_T.append(sl.strike.values[ok])
    ivs_by_T.append(iv[ok])
    i_atm = int(np.abs(sl.log_moneyness.values).argmin())
    print(f"  T={T:7.4f}y  inverted {ok.sum():2d}/{len(iv):2d} quotes   "
          f"ATM iv = {iv[i_atm]*100:6.2f}%")

# --------------------------------------------------------------------------- #
hr("3. SVI slice fits: params, RMSE, Durrleman butterfly check")
svi_fits: dict[float, es.SVIFitResult] = {}
rows = []
for T, Ks, ivs in zip(expiries, strikes_by_T, ivs_by_T):
    F = S * np.exp((r - q) * T)
    k = np.log(Ks / F)
    fit = es.fit_svi(k, ivs**2 * T, T, n_restarts=6, seed=SEED)
    svi_fits[T] = fit
    quad = es.fit_quadratic_delta(k, ivs, T)
    p = fit.params
    rows.append({
        "T": T, "a": p.a, "b": p.b, "rho": p.rho, "m": p.m, "sigma": p.sigma,
        "rmse_vp": fit.rmse_vol * 100, "min_g": fit.min_g,
        "butterfly_free": fit.arb_free, "quad_delta_rmse_vp": quad.rmse_vol * 100,
    })
svi_table = pd.DataFrame(rows)
print(svi_table.to_string(index=False))
print("(quad_delta_rmse_vp: naive quadratic-in-delta baseline, for comparison)")

# --------------------------------------------------------------------------- #
hr("4. Total-variance surface + calendar-arbitrage check")
surf = es.VolSurface(np.array(expiries), [svi_fits[T].params for T in expiries],
                     spot=S, rate=r, div_yield=q)
cal = surf.calendar
print(f"calendar-arbitrage free on pillar grid: {cal.is_free} "
      f"(worst total-variance decrease: {cal.worst_violation:.3e})")
print("sample queries vol(K, T):")
for K, T in [(80.0, 0.4), (100.0, 0.4), (120.0, 0.4), (100.0, 1.5), (100.0, 3.0)]:
    print(f"  vol(K={K:6.1f}, T={T:4.2f}) = {surf.vol(K, T)*100:6.2f}%   "
          f"{'(extrapolated in T)' if not (expiries[0] <= T <= expiries[-1]) else ''}")

# --------------------------------------------------------------------------- #
hr("5. Heston calibration -> recover the KNOWN parameters")
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", es.FellerWarning)
    calres = es.calibrate_heston(S, r, q, np.array(expiries), strikes_by_T, ivs_by_T,
                                 n_starts=3, seed=SEED)
print(f"(calibration took {time.time()-t0:.1f}s, {calres.n_starts} starts, "
      f"winner = start {calres.best_start})")
true = chain.true_heston
recov = pd.DataFrame({
    "true": true.as_array(),
    "calibrated": calres.params.as_array(),
    "abs_error": np.abs(true.as_array() - calres.params.as_array()),
}, index=["v0", "kappa", "theta", "rho", "xi"])
print(recov.to_string())
print(f"\noverall RMSE: {calres.rmse_vol_points:.4f} vol points | "
      f"Jacobian condition number: {calres.condition_number:.3e}")
print("RMSE by expiry (vol points):",
      {f"{T:.3f}": round(v, 4) for T, v in sorted(calres.rmse_by_expiry.items())})
print(f"Feller ratio: {calres.feller_ratio:.3f} "
      f"({'violated -- typical for equity fits' if calres.feller_ratio < 1 else 'satisfied'})")

# --------------------------------------------------------------------------- #
hr("6. Exotic-ish payoffs: calibrated Heston vs flat-vol Black-Scholes")
p_cal = calres.params
T_ex = 0.5
F_ex = S * np.exp((r - q) * T_ex)
sigma_flat = surf.vol(F_ex, T_ex)  # ATM-forward vol as the "flat" mark
print(f"flat BS vol used: {sigma_flat*100:.2f}% (ATM-forward, T={T_ex})")

# 6a. Cash-or-nothing digital call via -dC/dK (captures the skew slope).
print("\ncash-or-nothing digital call, T=0.5, payout 1:")
h = 1e-3
print(f"  {'K':>6} {'Heston':>10} {'BS flat':>10} {'diff':>10}")
for K in [90.0, 100.0, 110.0]:
    dig_h = -(float(es.heston_call_gl(S, K + h, T_ex, r, q, p_cal))
              - float(es.heston_call_gl(S, K - h, T_ex, r, q, p_cal))) / (2 * h)
    dig_b = -(es.bs_price(S, K + h, T_ex, r, q, sigma_flat) -
              es.bs_price(S, K - h, T_ex, r, q, sigma_flat)) / (2 * h)
    print(f"  {K:6.1f} {dig_h:10.6f} {dig_b:10.6f} {dig_h-dig_b:+10.6f}")

# 6b. Variance swap via static replication strip vs Heston analytic E[avg var].
print("\nvariance swap (T=1y): static strip on the surface vs Heston analytic")
T_vs = 1.0
F_vs = S * np.exp((r - q) * T_vs)
strip_K = np.linspace(0.5 * F_vs, 2.0 * F_vs, 400)
prices = np.array([
    es.bs_price(S, K, T_vs, r, q, surf.vol(K, T_vs), "put" if K < F_vs else "call")
    for K in strip_K
])
weights = 2.0 / strip_K**2 * np.exp(r * T_vs)
fair_var_strip = float(np.trapezoid(weights * prices, strip_K)) / T_vs
kap, th, v0 = p_cal.kappa, p_cal.theta, p_cal.v0
fair_var_heston = th + (v0 - th) * (1 - np.exp(-kap * T_vs)) / (kap * T_vs)
print(f"  strip (log-contract replication on surface): {np.sqrt(fair_var_strip)*100:6.2f}% vol")
print(f"  Heston analytic E[(1/T) int v dt]:           {np.sqrt(fair_var_heston)*100:6.2f}% vol")
print(f"  BS flat vol (no smile contribution):         {sigma_flat*100:6.2f}% vol")
print("  (strip > flat vol: the smile's wings add convexity value; Heston agrees")
print("   with the strip because calibration matched the whole smile)")

# --------------------------------------------------------------------------- #
hr("7. Greeks: Heston FD (Richardson) vs BS-equivalent, smile-adjusted delta")
rows = []
for K in [90.0, 100.0, 110.0]:
    g = es.heston_greeks(S, K, T_ex, r, q, p_cal, richardson=True)
    bse = es.bs_equivalent_greeks(S, K, T_ex, r, q, p_cal)
    kk = np.log(K / F_ex)
    fit = svi_fits[0.5]
    dsig_dk = float(es.svi_dw_dk(kk, fit.params)) / (2.0 * bse["implied_vol"] * T_ex)
    sm = es.smile_adjusted_delta(S, K, T_ex, r, q, bse["implied_vol"], dsig_dk)
    rows.append({
        "K": K, "price": g.price, "delta_H": g.delta, "delta_BS": bse["delta"],
        "gamma_H": g.gamma, "gamma_BS": bse["gamma"], "vega_v0": g.vega_v0,
        "rho_rate": g.rho_rate, "delta_sticky_K": sm["delta_sticky_strike"],
        "delta_sticky_mny": sm["delta_sticky_moneyness"],
    })
print(pd.DataFrame(rows).to_string(index=False))
print("(delta_sticky_mny > delta_sticky_K: negative skew raises call delta when")
print(" the smile rides the forward -- see docs/METHODOLOGY.md)")

# --------------------------------------------------------------------------- #
hr("8. Monte Carlo vs Fourier cross-check (calibrated parameters)")
rows = []
for K, T in [(90.0, 0.5), (100.0, 0.5), (110.0, 0.5), (100.0, 1.0)]:
    fourier = float(es.heston_call_gl(S, K, T, r, q, p_cal))
    mc_qe = es.heston_mc_price(S, K, T, r, q, p_cal, n_paths=200_000, n_steps=32,
                               scheme="qe", seed=SEED)
    mc_eu = es.heston_mc_price(S, K, T, r, q, p_cal, n_paths=200_000, n_steps=256,
                               scheme="euler_ft", seed=SEED)
    rows.append({
        "K": K, "T": T, "fourier": fourier,
        "qe_32": mc_qe.price, "qe_dev_se": (mc_qe.price - fourier) / mc_qe.stderr,
        "euler_256": mc_eu.price, "eu_dev_se": (mc_eu.price - fourier) / mc_eu.stderr,
    })
print(pd.DataFrame(rows).to_string(index=False))
print("(dev_se = (MC - Fourier)/SE; |dev| < 3 is agreement within Monte Carlo noise)")

print(f"\npipeline total: {time.time()-t_start:.1f}s")
