# 09 — Equity Volatility Surface & Stochastic Volatility (Heston)

Implied vol → SVI smiles → no-arbitrage vol surface → Heston model →
calibration → option pricing → Greeks → model validation. Pure
numpy/scipy/pandas, fully offline, deterministic seeds.

```
option chain ──► implied vols ──► SVI smiles ──► total-variance surface
   (robust        (Brent+Newton,   (5-param fit,    (linear in w(k,T),
    synthetic)     nan-not-garbage) Durrleman g≥0)    calendar check)
                                             │
                                             ▼
                        Heston calibration (vega-weighted LSQ, multi-start,
                          identifiability diagnostics: cond(J), κ/ξ ridge)
                                             │
                     ┌───────────────────────┼───────────────────────┐
                     ▼                       ▼                       ▼
             Fourier pricing          Monte Carlo             Greeks (FD +
          (little-trap CF; P1/P2   (full-trunc Euler,        Richardson; BS-
           + damped integral +      Andersen QE)             equivalent; sticky-
           fast Gauss–Legendre)                              strike vs -delta)
```

## Quickstart

```bash
cd python/equity/09-vol-surface
pip install -e .
pytest -q                        # 118 tests, ~35 s, offline
python examples/run_pipeline.py  # end-to-end, ~60 s
```

```python
import numpy as np, eq_surface as es
from eq_surface.data import generate_chain

chain = generate_chain(mode="heston", seed=42)      # known ground truth inside
sl = chain.slice(0.5)
ivs = es.implied_vol_vector(sl.call_mid.values, chain.spot, sl.strike.values,
                            0.5, chain.rate, chain.div_yield)
k = np.log(sl.strike.values / sl.forward.values)
fit = es.fit_svi(k, ivs**2 * 0.5, T=0.5, seed=0)    # SVI + butterfly check
print(fit.params, fit.arb_free)
```

## Headline numbers (seed 42; full tables in the docs)

* **Cross-method pricing agreement**: Heston P1/P2 vs Carr–Madan damped vs
  Gauss–Legendre fast path agree to **< 1.5e-7** across strikes/expiries,
  including ρ=±1, ξ=0 and a Feller-violating ξ=1 stress set (tests assert 1e-6).
* **BS degenerate limit** (ξ→0, v0=θ): Fourier price matches Black–Scholes
  to ~1e-10.
* **Calibration recovery** of known Heston truth (v0, κ, θ, ρ, ξ) =
  (0.035, 1.8, 0.045, −0.65, 0.45): **exact to 6 decimals**, RMSE 0.0000 vol
  points, cond(J) = 1.2e3 (the κ/ξ ridge, reported not hidden).
* **MC bias** (ATM 1y, 400k paths, Feller-violating set): Euler at 8
  steps/yr biased **+135 SE**; QE at the same 8 steps within **1.1 SE**.
* **SVI vs naive baseline**: slice RMSE 0.0006–0.04 vol points vs 0.50–0.75
  for quadratic-in-delta.
* **Smile economics**: 1y variance-swap strip off the surface 19.94 % vs
  Heston analytic 20.09 % vs flat ATM 18.30 %; ATM digital differs from
  flat-BS by +0.078 (the skew slope); sticky-moneyness vs sticky-strike ATM
  delta gap ≈ 0.08.
* **Documented failure mode**: calibrated to a non-Heston (SVI) surface with
  realistic 1/√T skew, residuals are 2.79 vp at 1 week decaying to 0.12 vp at
  2 y — short-dated equity skew needs jumps (see docs/VALIDATION.md §5.1).

## Layout

```
src/eq_surface/
  black_scholes.py   BS pricer/vega + robust implied vol (Brent + Newton, nan-not-garbage)
  smile.py           raw SVI fits, analytic derivatives, Durrleman butterfly check,
                     quadratic-in-delta baseline
  surface.py         total-variance surface, calendar check/enforcement, vol(K,T)
  heston.py          little-trap CF, P1/P2 + damped + Gauss–Legendre pricers, Feller check
  heston_mc.py       full-truncation Euler + Andersen QE, seeded, bias-documented
  calibration.py     vega-weighted multi-start calibration + identifiability diagnostics
  greeks.py          FD Greeks (Richardson), BS-equivalent Greeks, sticky-strike vs
                     sticky-moneyness delta
  data/synthetic.py  seeded chain generator (known-Heston or SVI ground truth, bid/ask)
tests/               118 tests: analytic identities, cross-method, recovery, bias, edges
examples/run_pipeline.py
docs/                METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```

## Documentation contract

1. **Why this model?** — SVI vs polynomial/kernel; Heston vs local vol vs
   SABR, with trade-off tables → `docs/METHODOLOGY.md`
2. **Assumptions** — 10-item register, each with "what breaks" →
   `docs/METHODOLOGY.md`
3. **Validation** — analytic benchmarks, 3-way cross-method, MC bias tables,
   exact recovery → `docs/VALIDATION.md`
4. **Failure modes** — short-dated skew underfit (residuals by expiry), κ/ξ
   ridge instability (3-day study), wing extrapolation, calendar-vs-events →
   `docs/VALIDATION.md`
5. **Desk usage** — marking workflow, calibration gates, Greeks/scenario
   feeds, reserve policy → `docs/DESK_GUIDE.md`
6. **Scenarios & edge cases** — earnings gap, Volmageddon 2018, crash skew,
   dividends; every edge case unit-tested → `docs/DESK_GUIDE.md` +
   `tests/test_edge_cases.py`
