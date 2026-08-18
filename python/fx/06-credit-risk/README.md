# FX / Cross-Border Credit Risk — Sovereign PD, Settlement & Counterparty Risk

Three linked risk blocks, specialised for sovereign and FX-counterparty credit
(not a renamed corporate scorecard):

1. **Sovereign PD scorecard** — seeded synthetic country-year panel with known
   ground truth (nonlinear reserve-cover threshold, Guidotti ratio, FX-regime
   dummy, contagion years), panel-aware cleaning with leakage guards, WOE/IV
   binning with missing bin and monotone merge, **from-scratch IRLS logistic**
   (sklearn cross-checked to 1e-6), PDO scorecard scaling and AAA…C rating
   bands with PD midpoints.
2. **FX settlement (Herstatt) risk** — time-zone payment-window matrix
   (JPY/EUR/GBP/USD RTGS hours), full-principal gross exposure vs CLS/PvP,
   bilateral payment netting; Herstatt 1974 encoded as the docs anchor.
3. **Counterparty pre-settlement risk** — GBM EE/PFE profiles for FX forwards
   (grows ~ sqrt(t) to maturity — the correct shape for an outright forward),
   netting-set comparison, CVA off the block-1 PD term structure (flat
   hazard); EL, Basel standardized RW (0% for AAA/AA sovereigns) and Vasicek
   99.9% economic capital with elevated sovereign asset correlation.

## Pipeline

```
data/synthetic.py          cleaning.py              woe.py            model.py
sovereign panel ──► clean + leakage guard ──► WOE/IV + monotone ──► IRLS logistic
(60 x 29, 5.6% base)  time / country split       merge, missing bin    + scorecard
                                                                        │
        capital.py            exposure.py            validation.py      ▼
EL / Basel RW / Vasicek ◄── EE/PFE/CVA ◄── PD term structure ◄── ratings AAA…C
        ▲                                                        AUC/KS/HL/PSI
settlement.py: time-zone windows, gross vs CLS  (independent block 2)
```

## Quickstart

```bash
cd python/fx/06-credit-risk
pip install -e .          # or rely on the repo conftest.py path shim
pytest -q                 # 179 tests, offline, ~6 s
python examples/run_pipeline.py   # full run, ~2 s
```

## Headline results (seed 42, reproduced by `examples/run_pipeline.py`)

| Quantity | Value |
|---|---|
| Panel | 60 countries x 29 years, base default rate **5.63%** |
| Leakage screen | `imf_program_next_year` IV = **5.04**, `devaluation_next_year_pct` IV = **4.68** — both auto-flagged (>1.0) and dropped |
| Reserve-cover binning | bad rate **11.3%** below 3.3 months cover vs **3.3%** above 6.2 months (planted threshold recovered) |
| IRLS vs sklearn | max coefficient difference **2.5e-07** (9 iterations) |
| Discrimination | AUC **0.820** train / **0.691** out-of-time (2015-23); KS 0.48 / 0.32 |
| OOT AUC bootstrap 95% CI | **[0.598, 0.782]** — width 0.18, the low-default reality |
| Calibration | HL p = 0.47 in-time; p < 1e-4 out-of-time, driven by the planted **2020 contagion year: predicted 6.4% vs observed 16.7%** (other OOT years: 5.35% vs 5.21%) |
| Settlement book (6 trades) | all-gross **USD 180.1m** at risk; with CLS **142.0m**; with bilateral payment netting **66.2m** |
| Herstatt window | pay JPY / receive USD = **23.5 h** of full-principal risk; pay USD / receive JPY = **0 h** |
| PFE (EUR 10m 1y forward) | 99% PFE: 1.68m (3m) → 3.52m (1y); monotone increasing, concave (~sqrt t) |
| Netting | offsetting +/-10m forwards: peak PFE99 **3.54m gross → 0 netted** |
| CVA (BB cpty, PD 2%, LGD 55%) | **USD 4,431** = 4.1 bp of USD notional |
| Capital (BB, per 100 EAD) | EL 0.90, standardized 8.00, Vasicek sovereign-rho **14.08** vs corporate-rho 7.66 |

## Layout

Per portfolio conventions: `src/fx_credit/` (library), `tests/` (179 offline
seeded tests), `examples/run_pipeline.py`, `docs/`
([METHODOLOGY](docs/METHODOLOGY.md) — model choice and assumptions register,
[VALIDATION](docs/VALIDATION.md) — evidence and failure modes,
[DESK_GUIDE](docs/DESK_GUIDE.md) — who uses it and real-life scenarios).

FX conventions: pairs quoted BASE/QUOTE (EURUSD = USD per EUR), domestic rate
= quote-currency rate; all stochastic components take explicit seeds; no
network access anywhere in library or tests (`data/live.py` is import-guarded).
