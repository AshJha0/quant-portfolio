# FX Volatility Modeling & Forecasting (`fx_vol`)

Historical vol → EWMA → GARCH(1,1)/GARCH-X → EGARCH → GJR-GARCH →
forecasting → evaluation, specialized for FX. All models are implemented
**from scratch** (numpy/scipy MLE with constraint transforms and Hessian
standard errors); the `arch` package appears only in the test suite as a
cross-validation oracle.

```
prices (BASE/QUOTE) ──► returns.py ──► historical.py ─► ewma.py ─► garch.py ──► forecasting.py ─► evaluation.py
   │  inversion / triangulation           │ close-close,           │ GARCH-X     │ egarch.py        │ analytic +      │ QLIKE/MSE, MZ,
   │  (vol-invariance, cross vol)         │ Parkinson/GK,          │ event       │ gjr.py           │ simulated,      │ DM (Newey-West),
   └──────────────────────────────────────┘ day-of-week factors    └ dummies     └ (asymmetry)      └ rolling OOS     └ LB / ARCH-LM / sign-bias
                                                                                                        │
data/synthetic.py (seeded generators: GARCH/GJR/EGARCH, correlated legs,                                ▼
event dummies, EM jumps, pegs, depegs)  · data/live.py (ECB, import-guarded)                       vol_premium.py
```

## Quickstart

```bash
cd python/fx/02-volatility-modeling
pip install -e ".[dev]"            # or: PYTHONPATH=src with numpy/scipy/pandas/statsmodels
python -m pytest tests -q          # 184 tests, offline, ~35 s
python examples/run_pipeline.py    # full pipeline, ~30 s, reproduces every number below
```

```python
import fx_vol as fv
from fx_vol.data import synthetic as syn

r = syn.simulate_garch(3000, 1.2e-6, 0.045, 0.915, dist="t", nu=6, seed=2024)
fit = fv.fit_garch(r, dist="t")               # from-scratch MLE, Hessian SEs
print(fit.summary())
var_path = fv.forecast_variance(fit, horizon=21)   # analytic multi-step
ann_vol = (252 * var_path.mean()) ** 0.5
```

## Headline results (all reproduced by `examples/run_pipeline.py`)

- **Parameter recovery** (20k obs): GARCH α 0.0487 (SE 0.0039) vs true
  0.050; GARCH-t ν 6.43 (0.27) vs 6.0; GJR γ 0.0958 (0.0072) vs 0.100;
  EGARCH γ −0.049 (0.0054) vs −0.060; GARCH-X event dummy γ_x 4.89e−5
  (t = 23.9) vs 5.0e−5.
- **arch cross-check**: α/β agree to ~1e−6, log-likelihood to ~1e−7 (percent
  scaling convention reconciled explicitly).
- **G10 vs EM ranking differs** (the FX point): on the EURUSD-like pair
  GARCH-t wins on AIC and *nothing* beats it out-of-sample; on the
  USDMXN-like pair EGARCH-t wins in-sample (γ = +0.046, ν = 3.7) and
  Gaussian models lose the 500-day OOS QLIKE race significantly
  (DM p = 0.017). Quote direction matters: GJR-t finds γ = 0 on USDMXN but
  γ = +0.123 on inverted MXNUSD.
- **Vol triangle**: EURJPY 11.57% from legs 8.66%/10.29% with ρ = −0.26 —
  exact identity vs the directly computed cross.
- **Event pricing**: a scheduled FOMC-style day lifts that day's forecast
  from 8.8% to 14.2% annualized.
- **Edge cases tested**: pegged pairs at 2 bp daily vol fit cleanly; a
  CHF-2015-style −15% depeg converges, spikes > 20× and decays; constant
  series, NaNs and short samples raise informative `ValueError`s.

## Layout

```
src/fx_vol/            returns · historical · ewma · garch · egarch · gjr ·
                       forecasting · evaluation · vol_premium · data/{synthetic,live}
tests/                 184 offline seeded tests (incl. arch cross-validation)
examples/run_pipeline.py
docs/                  METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```

## Conventions

Pairs quoted BASE/QUOTE (EURUSD = USD per EUR); log returns; daily decimal
units; annualization 252 by default with the FX 260-day (52×5) convention
available everywhere via `periods_per_year` (constant factor
sqrt(260/252) ≈ 1.016 — see METHODOLOGY §3.4); NaN policy: reject, never
impute; every stochastic component takes an explicit seed/Generator.
