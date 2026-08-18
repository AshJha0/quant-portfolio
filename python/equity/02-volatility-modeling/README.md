# Equity Volatility Modeling & Forecasting (`eq_vol`)

Daily equity volatility pipeline, implemented **from scratch** (numpy/scipy
only; the `arch` package is used exclusively as an independent test
benchmark):

```
Historical vol ──► EWMA ──► GARCH(1,1) ──► EGARCH(1,1) ──► GJR-GARCH(1,1)
(close-to-close,  (RiskMetrics   (MLE, variance   (leverage via     (leverage via
 Parkinson, GK,    lambda=0.94)   targeting, SEs)  log-variance)     indicator)
 Rogers-Satchell)
        │
        └────► Multi-step forecasting ────► Evaluation
               (analytic recursion /        (QLIKE & MSE, Mincer-Zarnowitz,
                seeded MC, term structure,   Diebold-Mariano + Newey-West,
                rolling OOS harness)         Ljung-Box, ARCH-LM, sign bias)
```

## Quickstart

```bash
cd python/equity/02-volatility-modeling
pip install -e .[test]          # or just run in place — tests need no install
python -m pytest tests -q      # 149 tests, ~10 s, offline, seeded
python examples/run_pipeline.py # full pipeline, ~10 s
```

```python
import numpy as np
from eq_vol import fit_gjr, term_structure, rolling_one_step_forecasts
from eq_vol.data.synthetic import simulate_gjr

r = simulate_gjr(3000, seed=7).returns          # daily log-returns (decimal)
res = fit_gjr(r)                                # MLE with std errors
print(res.summary())                            # parameter table
print(term_structure(res, horizon=21).tail(1))  # 1-month ann. vol forecast
oos = rolling_one_step_forecasts(r, "gjr", min_train=2000, refit_every=25)
```

## Highlight numbers (all reproducible: `examples/run_pipeline.py`, seed fixed)

**Out-of-sample forecast race** — true model GJR with leverage (persistence
0.97), 500 test days, 1-step forecasts, QLIKE vs squared-return proxy,
DM test vs GARCH benchmark:

| model | QLIKE | DM vs GARCH | p-value |
|---|---|---|---|
| **GJR-GARCH** | **−7.6027** | **−2.91** | **0.004** |
| EGARCH | −7.5785 | −0.99 | 0.32 |
| GARCH | −7.5638 | — | — |
| EWMA(0.94) | −7.4996 | +3.83 | 0.0001 |
| Rolling 21d historical | −7.4963 | +3.25 | 0.001 |

Exactly the theoretical ordering on asymmetric data — asymmetric models >
symmetric GARCH > EWMA > rolling window — and the fitted GJR closes ~94% of
the QLIKE gap between GARCH and the oracle (true conditional variance,
QLIKE −7.6067).

**Cross-validation vs `arch`:** fitted GARCH parameters agree to ~2e-6;
evaluating our likelihood at arch's parameters reproduces arch's
log-likelihood to **1.8e-12** (after the exact `n·ln 100` percent-scaling
Jacobian). GJR and EGARCH cross-checks in the same test file.

**Parameter recovery (20k simulated obs):** GARCH (0.05, 0.90) recovered as
(0.0551 ± 0.0047, 0.8813 ± 0.0116); GJR leverage gamma 0.10 recovered as
0.1051 ± 0.0083; EGARCH gamma −0.08 as −0.088 ± 0.008; Student-t nu 8.0 as
7.79. Full tables in [docs/VALIDATION.md](docs/VALIDATION.md).

**News impact asymmetry (fitted GJR):** a −2σ shock moves next-day
annualised vol to 18.2% vs 15.2% for a +2σ shock.

## Layout

```
src/eq_vol/
  historical.py    close-to-close + Parkinson/Garman-Klass/Rogers-Satchell,
                   window sensitivity (efficiency discussion in docstrings)
  ewma.py          RiskMetrics EWMA: recursive + lfilter-vectorised, half-life,
                   flat forecast (documented why)
  garch.py         GARCH(1,1) MLE (Gaussian/Student-t), transforms, variance
                   targeting, Hessian SEs, persistence/half-life/uncond vol
  egarch.py        EGARCH(1,1): log-variance recursion, leverage, news impact
                   curve, why no positivity constraints
  gjr.py           GJR-GARCH(1,1): indicator term, stationarity alpha+gamma/2+beta<1
  forecasting.py   analytic GARCH/GJR recursion, seeded-MC EGARCH forecasts,
                   term structure, rolling OOS harness (expanding/rolling, refit control)
  evaluation.py    QLIKE/MSE (Patton 2011), Mincer-Zarnowitz, Diebold-Mariano
                   (Newey-West HAC), Ljung-Box, ARCH-LM, Engle-Ng sign bias
  simulate.py      re-exports of data/synthetic.py (seeded GBM/GARCH/GJR/EGARCH
                   + crisis regime-jump generators)
  data/live.py     optional yfinance loader (import-guarded; never used in tests)
docs/              METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
tests/             149 tests: recovery, arch cross-check, forecast identities,
                   DM size Monte Carlo, edge cases & failure surfacing
```

## Documentation contract

1. **Why these models** vs stochastic vol / HAR-realized-vol / implied —
   [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §2.
2. **Assumptions register** (8 numbered, each with "what breaks") —
   METHODOLOGY.md §3.
3. **Validation**: recovery tables, arch reconciliation to machine precision,
   forecast identities, DM test size — [docs/VALIDATION.md](docs/VALIDATION.md).
4. **Failure modes**: structural breaks (COVID-style case study), IGARCH
   boundary, fat tails, noisy proxies, short samples — VALIDATION.md §5.
5. **Desk usage**: option marking, VaR, vol targeting, refit cadence, model
   governance — [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md).
6. **Edge cases**: constant/NaN/short series, outliers, convergence failures,
   crisis jumps — documented (VALIDATION.md §6) *and* unit-tested.
