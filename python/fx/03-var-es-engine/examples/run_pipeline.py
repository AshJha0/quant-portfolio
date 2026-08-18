"""End-to-end FX VaR/ES pipeline on the canned demo book.

Sections
--------
1. Book & triangulation demo (EURJPY == EURUSD + USDJPY legs)
2. VaR/ES, all methods, 95%/99%, 1d/10d
3. Method disagreement on an EM fat-tailed book (normal MC underestimates)
4. 500-day rolling backtests with Basel traffic light
   (parametric-normal fails on GARCH data, FHS passes)
5. Stress table incl. peg break; sensitivity ladder
6. Reverse stress: worst direction at the 99% radius

Runs offline on seeded synthetic data in well under two minutes.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

import fx_var as fv
from fx_var.data.synthetic import (
    demo_book,
    demo_em_book,
    demo_market,
    simulate_fx_returns,
    simulate_history,
)

T0 = time.time()
pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:.4f}")


def hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def money(x: float) -> str:
    return f"{x:>14,.0f}"


# ---------------------------------------------------------------- 1. book
market = demo_market()
book = demo_book()

hdr("1. DEMO BOOK (base = USD) & TRIANGULATION")
for p in book.positions:
    print(f"   {type(p).__name__:8s} {getattr(p, 'pair', getattr(p, 'ccy', '')):7s} "
          f"notional {getattr(p, 'notional', getattr(p, 'amount', 0)):>13,.0f}")
print(f"   Book value (USD): {money(book.value_usd(market))}")
print(f"   Risk factors:     {book.factors(market)}")

# triangulation: EURJPY == EURUSD + USDJPY decomposition
n = 10e6
cross = fv.Book([fv.Spot("EURJPY", n)])
legs = fv.Book([fv.Spot("EURUSD", n), fv.Spot("USDJPY", n * market.spot("EUR"))])
sh = {"FX:EUR": -0.02, "FX:JPY": 0.015}
p_cross, p_legs = cross.pnl(market, sh), legs.pnl(market, sh)
print(f"\n   Triangulation demo: 10m EURJPY under EUR -2% / JPY +1.5% shocks")
print(f"     direct cross P&L      {money(p_cross)}")
print(f"     EURUSD+USDJPY legs    {money(p_legs)}   (diff {p_cross - p_legs:+.2e})")

# ------------------------------------------------------- 2. VaR/ES matrix
hdr("2. VaR / ES - ALL METHODS  (USD, positive = loss)")
hist = simulate_history(book, market, 1000, seed=42, garch=True)
cov = fv.sample_cov(hist)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", fv.PegBlindnessWarning)
    rows = []
    for alpha in (0.95, 0.99):
        for h in (1, 10):
            r_hs = fv.historical_var(book, market, hist, alpha, h, warn_pegs=False)
            r_age = fv.historical_var(book, market, hist, alpha, h, method="age",
                                      warn_pegs=False)
            r_fhs = fv.historical_var(book, market, hist, alpha, h, method="fhs",
                                      warn_pegs=False)
            r_pn = fv.parametric_var(book, market, hist, alpha, h, warn_pegs=False)
            r_pt = fv.parametric_var(book, market, hist, alpha, h, dist="t", df=5,
                                     warn_pegs=False)
            r_mc = fv.monte_carlo_var(book, market, cov, alpha, h,
                                      n_scenarios=100_000, seed=1)
            rows.append({
                "alpha": alpha, "horizon_d": h,
                "HS": r_hs.var, "HS_age": r_age.var, "FHS": r_fhs.var,
                "Param_N": r_pn.var, "Param_t5": r_pt.var, "MC_N": r_mc.var,
                "ES_HS": r_hs.es, "ES_Param_N": r_pn.es,
            })
    tbl = pd.DataFrame(rows).set_index(["alpha", "horizon_d"])
    print(tbl.to_string(float_format=lambda v: f"{v:,.0f}"))

# check the engine flags the pegged HKD position
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    flagged = fv.historical_var(book, market, hist, 0.99).flagged_peg_factors
print(f"\n   PegBlindnessWarning emitted for: {list(flagged)}"
      f"  ->  peg-break stress add-on required (section 5)")

mc99 = fv.monte_carlo_var(book, market, cov, 0.99, n_scenarios=100_000, seed=1)
print(f"   MC 99%/1d VaR standard error: {money(mc99.se_var)}  "
      f"({mc99.se_var / mc99.var:.2%} of VaR)")

# ------------------------------------------- 3. EM fat tails disagreement
hdr("3. METHOD DISAGREEMENT - EM FAT TAILS (long-EM book, 99%/1d)")
em_book = demo_em_book()
em_cov = simulate_fx_returns(["MXN", "BRL", "TRY", "ZAR"], 1500, seed=21).cov()
jumps = fv.JumpSpec(prob=0.02, mean={"FX:TRY": -0.15, "FX:BRL": -0.08},
                    std={"FX:TRY": 0.05, "FX:BRL": 0.03})
mc_n = fv.monte_carlo_var(em_book, market, em_cov, 0.99, n_scenarios=100_000,
                          dist="normal", seed=7)
mc_t = fv.monte_carlo_var(em_book, market, em_cov, 0.99, n_scenarios=100_000,
                          dist="t", df=5, seed=7)
mc_j = fv.monte_carlo_var(em_book, market, em_cov, 0.99, n_scenarios=100_000,
                          dist="jump", jumps=jumps, seed=7)
print(f"   normal MC        VaR {money(mc_n.var)}   ES {money(mc_n.es)}")
print(f"   Student-t(5) MC  VaR {money(mc_t.var)}   ES {money(mc_t.es)}"
      f"   (+{mc_t.var / mc_n.var - 1:.0%} vs normal)")
print(f"   jump-mixture MC  VaR {money(mc_j.var)}   ES {money(mc_j.es)}"
      f"   (+{mc_j.var / mc_n.var - 1:.0%} vs normal)")
print("   -> at equal covariance, normal MC underestimates the EM 99% tail.")

# ------------------------------------------------------- 4. backtesting
hdr("4. 500-DAY ROLLING BACKTESTS (99% VaR) + BASEL TRAFFIC LIGHT")
bt_book = fv.Book([fv.Spot("EURUSD", 10e6), fv.Spot("USDJPY", 8e6)])
bt_hist = simulate_history(bt_book, market, 750, seed=29, garch=True,
                           regime_switching=True)


def fn_param(bk, mkt, window):
    return fv.parametric_var(bk, mkt, window, 0.99, min_obs=50, warn_pegs=False).var


def fn_hs(bk, mkt, window):
    return fv.historical_var(bk, mkt, window, 0.99, min_obs=50, warn_pegs=False).var


def fn_fhs(bk, mkt, window):
    return fv.historical_var(bk, mkt, window, 0.99, method="fhs", min_obs=50,
                             warn_pegs=False).var


print("   method            exc/500  Kupiec p  Indep p   CC p     zone (multiplier)")
results = {}
for name, fn in (("parametric-N", fn_param), ("plain HS", fn_hs), ("FHS", fn_fhs)):
    bt = fv.rolling_backtest(bt_book, market, bt_hist, fn, window=250)
    res = fv.evaluate_var_backtest(bt["pnl"], bt["var"], 0.99)
    tl250 = fv.basel_traffic_light(int(res.exceedances[-250:].sum()), 250)
    results[name] = (bt, res, tl250)
    print(f"   {name:16s}  {res.n_exceptions:>4d}     {res.kupiec_p:7.4f}  "
          f"{res.independence_p:7.4f}  {res.cc_p:7.4f}  {tl250.zone} ({tl250.multiplier:.2f})")
print("   -> unconditional parametric-normal fails conditional coverage on")
print("      vol-clustered data; FHS passes.  (GARCH + regime-switch sim)")

# ES backtest on the FHS run
bt_f, _, _ = results["FHS"]
es_fc = bt_f["var"] * (fv.normal_es(1.0, 0.99) / fv.normal_var(1.0, 0.99))
z, p = fv.es_backtest_acerbi_szekely(bt_f["pnl"], bt_f["var"], es_fc, 0.99, seed=0)
print(f"   Acerbi-Szekely ES backtest (FHS, 99%): Z = {z:+.3f}, p = {p:.3f}")

# ------------------------------------------------------- 5. stress tests
hdr("5. STRESS TESTS (full revaluation, USD P&L)")
lib = fv.historical_scenarios()
scens = dict(lib)
scens["usd_up_10"] = fv.usd_broad_move(sorted(c for c in book.currencies() if c != "USD"), +0.10)
scens["usd_dn_10"] = fv.usd_broad_move(sorted(c for c in book.currencies() if c != "USD"), -0.10)
scens["hkd_peg_break"] = fv.peg_break_scenario("HKD", jump=-0.30, vol_spike=0.10,
                                               vol_pairs=["USDHKD"])
stress = fv.run_stress(book, market, scens)
print(stress[["pnl"]].to_string(float_format=lambda v: f"{v:,.0f}"))
hs99 = fv.historical_var(book, market, hist, 0.99, warn_pegs=False)
peg_loss = -stress.loc["hkd_peg_break", "pnl"]
print(f"\n   HS 99% VaR:            {money(hs99.var)}")
print(f"   HKD peg-break loss:    {money(peg_loss)}  "
      f"({peg_loss / hs99.var:.1f}x the HS VaR - the risk HS cannot see)")

lad = fv.sensitivity_ladder(book, market, "FX:EUR")
print("\n   EUR sensitivity ladder (USD P&L):")
print("   " + "  ".join(f"{s:+.0%}:{p / 1e6:+.2f}m" for s, p in
                        zip(lad["shock"].map(lambda x: np.expm1(x)), lad["pnl"])))

# ------------------------------------------------------- 6. reverse stress
hdr("6. REVERSE STRESS - worst joint move at the 99% Mahalanobis radius")
w = book.linear_exposures(market)
shocks, loss = fv.reverse_stress_linear(w, cov, radius=2.326)
top = shocks.abs().sort_values(ascending=False).head(5).index
print(f"   Linearised worst-case loss: {money(loss)}")
print("   Worst direction (top factors):")
for f in top:
    print(f"     {f:12s} {shocks[f]:+9.4%}")
full_reval = book.pnl(market, shocks.to_dict())
print(f"   Full-revaluation P&L in that direction: {money(full_reval)}")

print(f"\nPipeline completed in {time.time() - T0:.1f}s")
