# FX and Cross-Border Credit Risk: Sovereign PD, Settlement and Counterparty Risk

Three linked risk blocks, specialised for sovereign and FX-counterparty credit
(not a renamed corporate scorecard):

1. **Sovereign PD scorecard**: seeded synthetic country-year panel with known
   ground truth (nonlinear reserve-cover threshold, Guidotti ratio, FX-regime
   dummy, contagion years), panel-aware cleaning with leakage guards, WOE/IV
   binning with missing bin and monotone merge, **from-scratch IRLS logistic**
   (sklearn cross-checked to 1e-6), PDO scorecard scaling and AAA…C rating
   bands with PD midpoints.
2. **FX settlement (Herstatt) risk**: time-zone payment-window matrix
   (JPY/EUR/GBP/USD RTGS hours), full-principal gross exposure vs CLS/PvP,
   bilateral payment netting; Herstatt 1974 encoded as the docs anchor.
3. **Counterparty pre-settlement risk**: GBM EE/PFE profiles for FX forwards
   (grows ~ sqrt(t) to maturity: the correct shape for an outright forward),
   netting-set comparison, and an expected loss on exposure from a
   real-world PD with a flat hazard (not a CVA: no CDS-implied PD, no
   collateral, no DVA); EL, Basel standardized RW (0% for AAA/AA sovereigns) and Vasicek
   99.9% economic capital with elevated sovereign asset correlation.

## Pipeline

```
data/synthetic.py          cleaning.py              woe.py            model.py
sovereign panel ──► clean + leakage guard ──► WOE/IV + monotone ──► IRLS logistic
(60 x 29, 5.6% base)  time / country split       merge, missing bin    + scorecard
                                                                        │
        capital.py            exposure.py            validation.py      ▼
EL / Basel RW / Vasicek ◄── EE/PFE/EL-on-exposure ◄── PD term structure ◄── ratings AAA…C
        ▲                                                        AUC/KS/HL/PSI
settlement.py: time-zone windows, gross vs CLS  (independent block 2)
```

## Quickstart

```bash
cd python/fx/06-credit-risk
pip install -e .          # or rely on the repo conftest.py path shim
pytest -q                 # 218 tests, offline, about 30 s
python examples/run_pipeline.py   # full run, about 5 s
```

## Headline results (seed 42, reproduced by `examples/run_pipeline.py`)

| Quantity | Value |
|---|---|
| Panel | 60 countries x 29 years, base default rate **5.63%** |
| Leakage screen | `imf_program_next_year` IV = **5.04**, `devaluation_next_year_pct` IV = **4.65**, both auto-flagged (>1.0) and dropped |
| Reserve-cover binning | bad rate **11.3%** below 3.3 months cover vs **3.3%** above 6.2 months (planted threshold recovered) |
| Expected-sign check | `fiscal_gdp` has the wrong sign (coef +0.66, z 0.71 in good/bad WOE space) and is dropped; 8 features remain |
| IRLS vs sklearn | max coefficient difference **3.5e-07** (9 iterations) |
| Discrimination | AUC **0.817** train / **0.690** out-of-time (2015-23); KS 0.47 / 0.32 |
| OOT AUC bootstrap 95% CI | **[0.599, 0.782]**, width 0.18 (35 OOT events) |
| Calibration | HL p = 0.47 in-time (df 8); p = 2.0e-04 out-of-time (df 10), driven by the planted **2020 contagion year: predicted 6.3% vs observed 16.7%** (other OOT years: 5.36% vs 5.21%). By rating, only BB is amber (binomial p 0.013) |
| Settlement book (6 trades) | all-gross **USD 180.1m** at risk; with CLS **142.0m**; with bilateral payment netting **66.2m** (operating-hours windows, a lower bound) |
| Herstatt window | operating hours: pay JPY / receive USD **23.5 h**, pay USD / receive JPY **0 h**. BCBS 2013 definition (24 h cancellation lead, 12 h reconciliation): 59.5 h and 30.5 h |
| PFE (EUR 10m 1y forward) | 99% PFE: 1.68m (3m) → 3.52m (1y); monotone increasing, concave (~sqrt t) |
| Netting | offsetting +/-10m forwards: peak PFE99 **3.54m gross → 0 netted** |
| Expected loss on exposure (BB cpty, real-world PD 2%, LGD 55%) | **USD 4,431** = 4.1 bp of USD notional; not a CVA |
| Capital (BB, per 100 EAD) | EL 0.90, standardized 8.00, Vasicek sovereign-rho **14.08** vs corporate-rho 7.66 |

## Layout

Per portfolio conventions: `src/fx_credit/` (library), `tests/` (218 offline
seeded tests), `examples/run_pipeline.py`, `docs/`
([METHODOLOGY](docs/METHODOLOGY.md): model choice and assumptions register,
[VALIDATION](docs/VALIDATION.md): evidence and failure modes,
[DESK_GUIDE](docs/DESK_GUIDE.md): who uses it and real-life scenarios).

FX conventions: pairs quoted BASE/QUOTE (EURUSD = USD per EUR), domestic rate
= quote-currency rate; all stochastic components take explicit seeds; no
network access anywhere in library or tests (`data/live.py` is import-guarded).

## Round 3 models

`xva` (risk-neutral bilateral CVA/DVA, CSA with threshold, MTA and MPoR, Hull-White wrong-way risk) and `saccr` (SA-CCR EAD for FX netting sets, Basel CRE52). See docs/METHODOLOGY.md and docs/VALIDATION.md.
