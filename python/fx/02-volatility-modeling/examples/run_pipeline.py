"""End-to-end FX volatility pipeline: data -> models -> validation -> decision.

Runs in well under 120 s and reproduces every number quoted in README.md and
docs/. Two synthetic pairs are studied side by side:

* "EURUSD-like" (G10): symmetric GARCH with Student-t innovations -- fat
  tails, no leverage. The claim to verify: GARCH-t is sufficient; asymmetric
  models add nothing.
* "USDMXN-like" (EM): asymmetric GJR dynamics + fat tails + one-sided
  depreciation jumps. The claim: asymmetry (GJR/EGARCH) now matters.

Sections: historical estimators & seasonality | full-sample model ranking |
parameter recovery on 20k-obs simulations | GARCH-X event dummies |
cross-vol triangle | 500-day OOS race with Diebold-Mariano | vol risk premium.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fx_vol as fv
from fx_vol.data import synthetic as syn

t_start = time.time()
PPY = 252
N_DAYS = 3000  # ~12 years of daily data per pair


def ann_vol(var_daily: float) -> float:
    return float(np.sqrt(var_daily * PPY))


def hr(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# ---------------------------------------------------------------- data ----
hr("1. DATA: two synthetic pairs (seeded, reproducible)")

EUR_TRUE = dict(omega=1.2e-6, alpha=0.045, beta=0.915, nu=6.0)
eurusd = syn.simulate_garch(
    N_DAYS, EUR_TRUE["omega"], EUR_TRUE["alpha"], EUR_TRUE["beta"],
    dist="t", nu=EUR_TRUE["nu"], seed=2024,
)
usdmxn = syn.simulate_em_series(N_DAYS, seed=2025)

for name, r in [("EURUSD-like", eurusd), ("USDMXN-like", usdmxn)]:
    kurt = np.mean(r ** 4) / np.mean(r ** 2) ** 2
    skew = np.mean(r ** 3) / np.std(r) ** 3
    print(
        f"{name}: n={r.size}, ann.vol {100 * fv.close_to_close_vol(r, PPY):.2f}% "
        f"(260-day conv: {100 * fv.close_to_close_vol(r, 260):.2f}%), "
        f"skew {skew:+.2f}, kurtosis {kurt:.1f}"
    )

# --------------------------------------- historical vol & seasonality ----
hr("2. HISTORICAL ESTIMATORS & WEEKLY SEASONALITY")

seasonal = syn.simulate_seasonal_returns(2600, seed=7)
factors = fv.day_of_week_vol_factors(seasonal)
print("Day-of-week vol factors (unit r.m.s., estimated from seasonal sim):")
for day, f in factors.items():
    print(f"  {day:<10s} {f:.3f}")
print("(Injected pattern: Mon 0.85, Tue 0.95, Wed 1.15, Thu 1.00, Fri 1.10)")

arch_p = fv.arch_lm(eurusd, lags=10)["pvalue"]
lb_p = fv.ljung_box(eurusd ** 2, lags=10)["pvalue"]
print(f"\nEURUSD-like pre-tests: ARCH-LM p={arch_p:.2e}, Ljung-Box(r^2) p={lb_p:.2e}")
print("=> strong conditional heteroskedasticity; GARCH-family modelling justified")

ew = fv.ewma_variance(eurusd)
print(f"EWMA(0.94) terminal ann.vol: {100 * ann_vol(ew[-1]):.2f}% "
      f"(flat forecast at all horizons -- no mean reversion)")

# -------------------------------------------- full-sample model races ----
hr("3. FULL-SAMPLE FITS & IN-SAMPLE RANKING (AIC)")

MODELS = {
    "GARCH": lambda r: fv.fit_garch(r, dist="gaussian"),
    "GARCH-t": lambda r: fv.fit_garch(r, dist="t"),
    "GJR": lambda r: fv.fit_gjr(r, dist="gaussian"),
    "GJR-t": lambda r: fv.fit_gjr(r, dist="t"),
    "EGARCH": lambda r: fv.fit_egarch(r, dist="gaussian"),
    "EGARCH-t": lambda r: fv.fit_egarch(r, dist="t"),
}

fits: dict[str, dict[str, fv.FitResult]] = {}
for pair, r in [("EURUSD-like", eurusd), ("USDMXN-like", usdmxn)]:
    fits[pair] = {name: fit for name, fit in ((n, f(r)) for n, f in MODELS.items())}
    print(f"\n{pair}:")
    rows = []
    for name, fit in fits[pair].items():
        k = len(fit.params)
        aic = 2 * k - 2 * fit.loglik
        rows.append((aic, name, fit))
    rows.sort()
    best_aic = rows[0][0]
    for aic, name, fit in rows:
        asym = fit.params.get("gamma", float("nan"))
        nu = fit.params.get("nu", float("nan"))
        print(
            f"  {name:<9s} loglik={fit.loglik:10.2f}  dAIC={aic - best_aic:8.2f}  "
            f"pers={fit.persistence:.4f}  gamma={asym:+.4f}  nu={nu:.1f}"
        )
    sb = fv.sign_bias_test(r, fits[pair]["GARCH"].sigma2)
    print(f"  Engle-Ng sign-bias joint p (on symmetric GARCH fit): {sb['joint_f_p']:.4f}")

# Quote direction matters: USDMXN asymmetry loads on POSITIVE pair returns
# (EM depreciation), which the gamma >= 0 GJR cannot see in this quote
# direction (gamma pins at 0). EGARCH's sign-free leverage catches it
# directly; alternatively, invert the pair (MXNUSD) and refit GJR.
gjr_inv = fv.fit_gjr(fv.invert_returns(usdmxn), dist="t")
em_eg = fits["USDMXN-like"]["EGARCH-t"]
print(f"\nQuote-direction check (EM asymmetry):")
print(f"  EGARCH-t on USDMXN:      gamma = {em_eg.params['gamma']:+.4f} "
      f"(se {em_eg.std_errors['gamma']:.4f})  -- positive: depreciation raises vol")
print(f"  GJR-t on USDMXN:         gamma = {fits['USDMXN-like']['GJR-t'].params['gamma']:+.4f} "
      f"(pinned at 0: asymmetry is on the other side in this quote direction)")
print(f"  GJR-t on inverted MXNUSD: gamma = {gjr_inv.params['gamma']:+.4f} "
      f"(se {gjr_inv.std_errors['gamma']:.4f})  -- recovered after inversion")
g10_eg = fits["EURUSD-like"]["EGARCH-t"]
print(f"  EGARCH-t on EURUSD:      gamma = {g10_eg.params['gamma']:+.4f} "
      f"(se {g10_eg.std_errors['gamma']:.4f})  -- G10: no material asymmetry")
print("=> asymmetric terms are dead weight for the G10-style pair, first-order for EM")

# ------------------------------------------------- parameter recovery ----
hr("4. PARAMETER RECOVERY ON 20k-OBS SIMULATIONS (for docs/VALIDATION.md)")

rec_r = syn.simulate_garch(20_000, 1e-6, 0.05, 0.92, seed=11)
rec = fv.fit_garch(rec_r)
print("GARCH(1,1) gaussian, true (omega, alpha, beta) = (1.0e-06, 0.050, 0.920):")
for k in ("omega", "alpha", "beta"):
    print(f"  {k:<6s} = {rec.params[k]:.4g}  (se {rec.std_errors[k]:.2g})")

rec_t = fv.fit_garch(syn.simulate_garch(20_000, 1.5e-6, 0.06, 0.90, dist="t", nu=6.0, seed=12), dist="t")
print("GARCH-t, true (omega, alpha, beta, nu) = (1.5e-06, 0.060, 0.900, 6.0):")
for k in ("omega", "alpha", "beta", "nu"):
    print(f"  {k:<6s} = {rec_t.params[k]:.4g}  (se {rec_t.std_errors[k]:.2g})")

rec_gjr = fv.fit_gjr(syn.simulate_gjr(20_000, 1e-6, 0.03, 0.10, 0.88, seed=13))
print("GJR, true (omega, alpha, gamma, beta) = (1.0e-06, 0.030, 0.100, 0.880):")
for k in ("omega", "alpha", "gamma", "beta"):
    print(f"  {k:<6s} = {rec_gjr.params[k]:.4g}  (se {rec_gjr.std_errors[k]:.2g})")

rec_eg = fv.fit_egarch(syn.simulate_egarch(15_000, -0.50, 0.15, -0.06, 0.95, seed=14))
print("EGARCH, true (omega, alpha, gamma, beta) = (-0.500, 0.150, -0.060, 0.950):")
for k in ("omega", "alpha", "gamma", "beta"):
    print(f"  {k:<6s} = {rec_eg.params[k]:.4g}  (se {rec_eg.std_errors[k]:.2g})")

# --------------------------------------------- GARCH-X event dummies ----
hr("5. GARCH-X: CENTRAL-BANK EVENT DUMMIES IN THE VARIANCE EQUATION")

rx, x = syn.simulate_garch_x(
    20_000, omega=1e-6, alpha=0.05, beta=0.90, gamma_x=5e-5, event_prob=0.05, seed=15
)
fx_fit = fv.fit_garch(rx, x=x)
gx, gx_se = fx_fit.params["gamma_x"], fx_fit.std_errors["gamma_x"]
print(f"true gamma_x = 5.0e-05; estimated {gx:.3g} (se {gx_se:.2g}, t = {gx / gx_se:.1f})")
base_var = fx_fit.params["omega"] / (1 - fx_fit.persistence)
print(f"event-day variance uplift at the long-run level: "
      f"{100 * (np.sqrt((base_var + gx) / base_var) - 1):.0f}% of daily vol")
f_event = fv.forecast_variance(fx_fit, 5, x_future=np.array([[0], [0], [1.0], [0], [0]]))
f_quiet = fv.forecast_variance(fx_fit, 5, x_future=np.zeros((5, 1)))
print("5-day forecast, ann.vol, with an FOMC-style event on day 3 vs none:")
print("  event: ", " ".join(f"{100 * ann_vol(v):5.2f}%" for v in f_event))
print("  quiet: ", " ".join(f"{100 * ann_vol(v):5.2f}%" for v in f_quiet))

# ------------------------------------------------ cross-vol triangle ----
hr("6. VOL TRIANGLE: EURJPY FROM EURUSD & USDJPY")

tri = syn.simulate_correlated_pairs(20_000, vol1=0.0055, vol2=0.0065, rho=-0.25, seed=16)
r_cross = fv.triangulate_returns(tri["returns1"], "EURUSD", tri["returns2"], "USDJPY", "EURJPY")
s1 = tri["returns1"].std(ddof=1)
s2 = tri["returns2"].std(ddof=1)
rho_hat = np.corrcoef(tri["returns1"], tri["returns2"])[0, 1]
tri_vol = fv.cross_volatility(s1, s2, rho_hat, "EURUSD", "USDJPY", "EURJPY")
print(f"leg vols (ann.): EURUSD {100 * s1 * np.sqrt(PPY):.2f}%, USDJPY {100 * s2 * np.sqrt(PPY):.2f}%, "
      f"corr {rho_hat:+.3f}")
print(f"EURJPY vol -- triangle: {100 * tri_vol * np.sqrt(PPY):.2f}%  "
      f"direct: {100 * r_cross.std(ddof=1) * np.sqrt(PPY):.2f}%  (identity, exact)")
inv_vol = fv.close_to_close_vol(fv.invert_returns(r_cross))
print(f"JPYEUR (inverted pair) ann.vol: {100 * inv_vol:.2f}%  -- inversion-invariant")

# ------------------------------------------------------- OOS race -------
hr("7. 500-DAY OUT-OF-SAMPLE RACE (rolling 1-step, refit every 125 days)")

RACE_MODELS = ["ewma", "garch", "garch_t", "gjr", "egarch"]
BENCH = "garch_t"
for pair, r in [("EURUSD-like", eurusd), ("USDMXN-like", usdmxn)]:
    print(f"\n{pair} (QLIKE, lower is better; DM vs {BENCH}, negative favours model):")
    losses = {}
    for m in RACE_MODELS:
        out = fv.rolling_one_step(r, model=m, window=1000, refit_every=125, n_oos=500)
        losses[m] = fv.qlike_loss(out["forecast"], out["realized"])
    for m in RACE_MODELS:
        if m == BENCH:
            print(f"  {m:<8s} QLIKE {losses[m].mean():9.4f}   (benchmark)")
        else:
            dm = fv.diebold_mariano(losses[m], losses[BENCH])
            print(f"  {m:<8s} QLIKE {losses[m].mean():9.4f}   DM {dm['stat']:+.2f} (p={dm['pvalue']:.3f})")

# --------------------------------------------------- vol risk premium ---
hr("8. VOL RISK PREMIUM (synthetic implied vs realized)")

rv_fwd = fv.realized_vol_forward(eurusd, window=21)
rng = np.random.default_rng(17)
valid = np.isfinite(rv_fwd)
implied = np.where(valid, rv_fwd, 0.08) + 0.012 + 0.004 * rng.standard_normal(rv_fwd.size)
implied = np.clip(implied, 0.01, None)
prem = fv.vol_risk_premium(implied, rv_fwd)
s = fv.premium_summary(prem)
print(f"premium (implied - subsequent 21d realized): mean {100 * s['mean']:.2f} vol pts, "
      f"median {100 * s['median']:.2f}, %positive {100 * s['frac_positive']:.0f}%, "
      f"min {100 * s['min']:.2f}")
pnl = fv.variance_swap_pnl(implied[valid], rv_fwd[valid], vega_notional=1.0)
print(f"short-var P&L per unit vega: mean {pnl.mean():+.4f}, worst day {pnl.min():+.4f} "
      f"(convexity works against the seller in spikes)")

print(f"\nTotal pipeline runtime: {time.time() - t_start:.1f} s")
