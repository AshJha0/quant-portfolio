# FX Volatility Modeling & Forecasting (`fx_vol`)

Historical vol, EWMA, GARCH(1,1)/GARCH-X, EGARCH and GJR-GARCH, forecasting
and evaluation, specialised for FX. All models are implemented from scratch
(numpy/scipy MLE with constraint transforms, classic and
Bollerslev-Wooldridge robust standard errors). The `arch` package appears
only in the test suite as a cross-check.

```
prices (BASE/QUOTE) -> returns.py -> historical.py -> ewma.py -> garch.py -> forecasting.py -> evaluation.py
  inversion /          close-close,     GARCH-X        egarch.py   analytic +       QLIKE/MSE, MZ,
  triangulation        Parkinson/GK,    event dummies  gjr.py      simulated,       HAC DM + HLN,
  (cross vol)          day-of-week                                 rolling OOS      LB / ARCH-LM / sign bias
data/synthetic.py (seeded generators)  data/live.py (ECB, optional `requests`)  vol_premium.py
```

## Quickstart

```bash
cd python/fx/02-volatility-modeling
pip install -e ".[test]"           # test extra: pytest, arch, hypothesis
python -m pytest tests -q          # offline, seeded
python examples/run_pipeline.py    # reproduces every number below (synthetic data)
```

Measured on the development machine (2026-11-01, Python 3.12, numpy 2.5):

| command | result | wall time |
|---|---|---|
| `python -m pytest tests -q` | 350 passed | 165 s |
| `python examples/run_pipeline.py` | exit 0 | 66 s |

Without `arch` installed, `tests/test_arch_crosscheck.py` is skipped and the
rest of the suite runs. The `[dev]` extra is kept as an alias of `[test]`
for one release.

```python
import fx_vol as fv
from fx_vol.data import synthetic as syn

r = syn.simulate_garch(3000, 1.2e-6, 0.045, 0.915, dist="t", nu=6, seed=2024)
fit = fv.fit_garch(r, dist="t")         # MLE; raises ConvergenceError on failure
print(fit.summary())                     # classic and robust SEs
var_path = fv.forecast_variance(fit, horizon=21)
ann_vol = (252 * var_path.mean()) ** 0.5
```

## Results on synthetic data (all from `examples/run_pipeline.py`)

| result | value |
|---|---|
| GARCH alpha recovery (20k obs) | 0.0487 vs true 0.050, robust SE 0.0042 |
| GARCH-t nu | 6.43 vs 6.0, robust SE 0.28 |
| GJR gamma | 0.0958 vs 0.100, robust SE 0.0073 |
| EGARCH gamma | -0.049 vs -0.060, robust SE 0.0054 |
| GARCH-X event dummy gamma_x | 4.89e-5 vs 5.0e-5, t = 23.9 |
| USDMXN-like: GJR-t gamma, quoted / inverted | -0.123 / +0.123, same log-likelihood |
| USDMXN-like OOS QLIKE, Gaussian GARCH vs GARCH-t | HAC DM +2.42, p = 0.016 |
| EURJPY vol from legs 8.66% / 10.29%, rho -0.26 | 11.57%, equal to the direct cross |
| Scheduled FOMC-style day, forecast ann. vol | 8.8% to 14.2% |

On the EURUSD-like pair GARCH-t wins on AIC and nothing beats it out of
sample. On the USDMXN-like pair EGARCH-t wins in-sample (gamma = +0.046,
nu = 3.7), and the Gaussian GARCH pins persistence at 1 (flagged
`at_boundary`). The arch cross-check and the edge cases (pegs, a -15% depeg,
constant series, NaNs, short samples) are in
[docs/VALIDATION.md](docs/VALIDATION.md).

## Layout

```
src/fx_vol/            returns, historical, ewma, garch, egarch, gjr,
                       forecasting, evaluation, vol_premium, data/{synthetic,live}
tests/                 offline seeded tests (arch cross-check optional)
examples/run_pipeline.py
docs/                  METHODOLOGY.md, VALIDATION.md, DESK_GUIDE.md
```

## Conventions

Pairs quoted BASE/QUOTE (EURUSD = USD per EUR); log returns; daily decimal
units; annualisation 252 by default with the FX 260-day convention available
via `periods_per_year` (factor sqrt(260/252) = 1.016, METHODOLOGY section
3.4); NaN policy: reject, never impute; every stochastic component takes an
explicit seed or Generator.

Limitations: the squared daily return is the only evaluation proxy. GJR and
EGARCH multi-step forecasts assume symmetric innovations. The Gaussian
EGARCH unconditional variance is exact; under Student-t it does not exist
and is reported as `inf`.
