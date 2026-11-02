# Equity Volatility Modeling & Forecasting (`eq_vol`)

Daily equity volatility pipeline implemented from scratch with numpy and
scipy. The `arch` package is used only as an independent test benchmark.

```
Historical vol --> EWMA --> GARCH(1,1) --> EGARCH(1,1) --> GJR-GARCH(1,1)
(close-to-close,  (RiskMetrics  (MLE, variance  (leverage via    (leverage via
 Parkinson, GK,    lambda=0.94)  targeting,      log-variance)    indicator)
 RS, Yang-Zhang)                 classic+robust SEs)
        |
        +---> Multi-step forecasting ---> Evaluation
              (analytic recursion,        (QLIKE & MSE, Mincer-Zarnowitz,
               seeded MC, term structure,  HAC Diebold-Mariano with HLN,
               rolling OOS harness)        Ljung-Box, ARCH-LM, sign bias)
```

## Quickstart

```bash
cd python/equity/02-volatility-modeling
pip install -e .[test]            # test extra: pytest, arch, hypothesis
python -m pytest tests -q         # offline, seeded
python examples/run_pipeline.py   # full pipeline on synthetic data
python examples/validation_tables.py   # tables in docs/VALIDATION.md
```

Measured on the development machine (2026-11-01, Python 3.12, numpy 2.5):

| command | result | wall time |
|---|---|---|
| `python -m pytest tests -q` | 217 passed | 39 s |
| `python examples/run_pipeline.py` | exit 0 | 85 s |

Without `arch` installed, `tests/test_arch_crosscheck.py` is skipped and the
rest of the suite runs.

```python
from eq_vol import fit_gjr, term_structure, rolling_one_step_forecasts
from eq_vol.data.synthetic import simulate_gjr

r = simulate_gjr(3000, seed=7).returns          # daily log-returns (decimal)
res = fit_gjr(r)                                # MLE
print(res.summary())                            # classic and robust SEs
print(term_structure(res, horizon=21).tail(1))  # 1-month ann. vol forecast
oos = rolling_one_step_forecasts(r, "gjr", min_train=2000, refit_every=25)
```

## Results on synthetic data

All from `python examples/run_pipeline.py` (seeded synthetic data).

**Out-of-sample forecast race.** True model GJR with leverage (persistence
0.97), 500 test days, 1-step forecasts, QLIKE vs squared-return proxy, DM
test vs GARCH with Newey-West lag 5 and the HLN correction:

| model | QLIKE | DM vs GARCH | p-value |
|---|---|---|---|
| GJR-GARCH | -7.6027 | -2.75 | 0.006 |
| EGARCH | -7.5785 | -0.86 | 0.39 |
| GARCH | -7.5638 | n/a | n/a |
| EWMA(0.94) | -7.4996 | +2.91 | 0.004 |
| Rolling 21d historical | -7.4963 | +2.69 | 0.007 |

The QLIKE ranking is asymmetric models > GARCH > EWMA > rolling window. The
fitted GJR closes 90.7% of the QLIKE gap between GARCH and the oracle (true
conditional variance, QLIKE -7.6067).

**Cross-check vs `arch`** (`python examples/validation_tables.py`): our
GARCH and EGARCH likelihoods at arch's fitted parameters reproduce arch's to
3.6e-12. Classic and robust standard errors match arch's to about 1e-4
relative on Student-t data.

**Parameter recovery** (synthetic, 20k obs): GARCH alpha 0.05 and beta 0.90
recovered as 0.0551 and 0.8813 (robust SEs 0.0046, 0.012). Full tables in
[docs/VALIDATION.md](docs/VALIDATION.md).

**News impact (fitted GJR):** a -2 sigma shock moves next-day annualised
vol to 18.2% vs 15.2% for +2 sigma.

## Layout

```
src/eq_vol/
  historical.py    close-to-close, Parkinson, Garman-Klass, Rogers-Satchell,
                   Yang-Zhang, window sensitivity
  ewma.py          RiskMetrics EWMA: recursive and lfilter-vectorised,
                   half-life, flat forecast
  garch.py         GARCH(1,1) MLE (Gaussian/Student-t), transforms, variance
                   targeting, classic and sandwich SEs, boundary flag
  egarch.py        EGARCH(1,1): log-variance recursion, leverage, news
                   impact, exact unconditional mean (Gaussian)
  gjr.py           GJR-GARCH(1,1): indicator term, alpha >= 0, alpha+gamma >= 0
  forecasting.py   analytic GARCH/GJR forecasts, seeded-MC EGARCH forecasts,
                   term structure, rolling OOS harness (Gaussian or t)
  evaluation.py    QLIKE/MSE (Patton 2011), Mincer-Zarnowitz, HAC
                   Diebold-Mariano with HLN, Ljung-Box, ARCH-LM, sign bias
  simulate.py      re-exports of data/synthetic.py
  data/live.py     optional yfinance loader (never used in tests)
docs/              METHODOLOGY.md, VALIDATION.md, DESK_GUIDE.md
examples/          run_pipeline.py, validation_tables.py (print only, no files)
tests/             recovery, arch cross-check, sandwich SEs, HAC DM size,
                   forecast identities, edge cases
```

## Documentation

1. Why these models vs stochastic vol, HAR-RV and implied vol:
   [docs/METHODOLOGY.md](docs/METHODOLOGY.md) section 2.
2. Assumptions register with what breaks: METHODOLOGY.md section 3.
3. Validation: recovery tables, arch reconciliation, forecast identities,
   DM size: [docs/VALIDATION.md](docs/VALIDATION.md).
4. Failure modes: structural breaks, IGARCH boundary, fat tails, noisy
   proxies, short samples: VALIDATION.md section 5.
5. Desk usage: [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md).

Limitations: the squared daily return is the only forecast-evaluation
proxy, because the package has no intraday data. HAR-RV, the Model
Confidence Set and the Giacomini-White test are not implemented. The
rolling harness re-runs the recursion over the full history every day,
which is O(n^2).

## November 2026 additions

- `har_rv.py`: realised variance from intraday returns and the HAR-RV model
  of Corsi (2009), fitted by OLS with Newey-West standard errors, with
  iterated multi-day forecasts. Tests: `tests/test_har_rv.py` (5 tests).
