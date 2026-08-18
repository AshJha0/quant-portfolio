# 09 — FX Volatility Surface & Stochastic Volatility

FX-native volatility surface construction and Heston stochastic
volatility, built the way the OTC FX market actually works — **delta-
space quotes** (ATM DNS / risk reversals / butterflies), four delta
conventions incl. premium-adjusted, vanna–volga *and* SVI smiles, a
sticky-delta surface, and Heston (Garman–Kohlhagen) on top with
two-method Fourier pricing, QE/Euler Monte Carlo, calibration and
vega/vanna/volga risk buckets. This is explicitly **not** an equity
surface with renamed variables.

## Pipeline

```
broker quotes {ATM, RR25, BF25, RR10, BF10} x {1w,1m,3m,6m,1y,2y}
      │  exact linear map (smile BF)              smile_from_quotes.py
      ▼
five smile vols (10P, 25P, ATM, 25C, 10C)
      │  delta -> strike solve (spot/forward, pa; DNS ATM both variants)
      ▼
pillar strikes ──► smile fits: SVI (log-moneyness) + vanna-volga smile.py
      │                 │ Durrleman butterfly check
      ▼                 ▼
FX vol surface: total-variance interp AT FIXED DELTA,      surface.py
calendar-arb check, vol(K,T) and vol(delta,T)
      │
      ▼
Heston under GK (little-trap CF): Gil-Pelaez + COS         heston.py
      │  vega-weighted calibration (5 pillars x 6 expiries)
      ▼                                                calibration.py
pricing / digitals / FD Greeks (both rhos, vanna, volga)   greeks.py
      │
      ▼
validation: MC (full-truncation Euler + QE) within 3 SE  heston_mc.py
```

## Quickstart

```bash
cd python/fx/09-vol-surface
pip install -e .
pytest -q                      # 195 tests, ~11 s, offline & seeded
python examples/run_pipeline.py   # full EURUSD + USDJPY demo, ~6 s
```

```python
from fx_surface import calibrate_heston, heston_greeks_fd
from fx_surface.data import usdjpy_market, calibration_slices
from fx_surface.surface import build_surface

mkt  = usdjpy_market()                    # pa-quoted, strong JPY-call skew
surf = build_surface(mkt, smile_model="svi")
surf.vol(145.0, 0.75)                     # vol by strike/expiry
surf.vol_delta(0.25, 0.75, cp=-1)         # vol at the 25-delta put
res  = calibrate_heston(mkt.S, calibration_slices(mkt))
print(res.summary())   # rho=-0.47, rmse=0.18 vol pts
```

## Headline results (all reproduced by tests / pipeline)

* **Conventions**: Δ→K→Δ round trips to 1e-8 under all four
  conventions; ATM DNS `F·e^{±σ²T/2}` exact; pa-call two-strike
  ambiguity resolved to the market branch (both candidates tested).
* **Convention risk, measured**: solving USDJPY's pa quotes with
  unadjusted deltas shifts the 1y ATM strike by 1.58 JPY and corrupts
  the surface by up to **0.32 vol pts** — silently (VALIDATION F2).
* **Smiles**: VV replication weights solve the 3×3 vega/vanna/volga
  system to 1e-12 and reproduce pillars exactly; SVI interpolates
  EURUSD pillars to <1e-7 (USDJPY's extreme skew hits the ρ bound —
  ≤0.05 vol pts, documented); VV vs SVI: ±5 bp in the body, ~10 bp at
  5Δ (the honest wing error bar).
* **Heston**: little-trap CF (φ(0)=1, φ(−i)=F to 1e-13); Gil-Pelaez
  vs COS < 1e-6; parity 1e-8; ξ→0 → GK to 1e-6; Euler(250 steps) and
  QE(24 steps) within 3 SE of Fourier, incl. Feller-violated regime.
* **Calibration**: exact ground-truth recovery (RMSE 0.000 vol pts);
  EURUSD → ρ = −0.18, RMSE 0.23; USDJPY → ρ = −0.47, RMSE 0.18 (the
  cross-pair rho pattern); κ–ξ ridge quantified (ξ²/κ recovered to
  0.1125 vs 0.1125).
* **Digitals three ways** (EURUSD 6m 1.15): flat GK 0.2054 / VV
  0.1895 / Heston 0.1899 — the skew correction and the VV-vs-model
  reserve, in one table.

## Layout

```
src/fx_surface/          garman_kohlhagen, smile_from_quotes, smile,
                         surface, heston, heston_mc, calibration,
                         greeks, data/synthetic (seeded presets:
                         EURUSD, USDJPY-pa, 35%-vol EM, Heston ground truth)
tests/                   195 offline seeded tests
examples/run_pipeline.py end-to-end demo (<150 s budget; runs in ~6 s)
docs/                    METHODOLOGY.md, VALIDATION.md, DESK_GUIDE.md
```

Docs answer the portfolio contract: model choice vs alternatives and
assumptions register (METHODOLOGY), validation evidence and failure
modes F1–F7 (VALIDATION), desk workflow, risk buckets, scenarios and
governance (DESK_GUIDE).
