# 06: Corporate Credit Risk and PD Modelling (Scorecard)

A from-scratch credit risk stack: seeded synthetic corporate loan
book with known ground truth → cleaning (leakage guard, winsorization,
train/OOT split) → supervised WOE/IV binning (monotone merging, missing bin,
leakage flag) → logistic regression via Newton-Raphson/IRLS (Fisher-info
standard errors, sklearn cross-checked to 1e-13) → points/PDO scorecard →
validation (AUC/Gini/KS, Hosmer-Lemeshow, per-grade binomial and Jeffreys
tests, PSI, rank ordering, bootstrap CIs)
→ expected loss → Basel III IRB corporate capital formula (hand-checked
against the published 92.32% risk weight at PD 1%/LGD 45%/M 2.5) → Vasicek
one-factor loss distribution: analytic, exact finite-portfolio quadrature,
Gordy-Lutkebohmert granularity adjustment and Monte Carlo with confidence
intervals, economic capital at 99.9%.

```
loan book ─► clean ─► WOE/IV ─► IRLS logit ─► PD ─► score ─► validation
                                                    │
                                EL = PD·LGD·EAD ◄───┴──► Basel K/RWA ─► Vasicek EC
```

## Quickstart

```bash
cd python/equity/06-credit-risk
pip install -e ".[dev]"
pytest -q                      # 157 tests, about 60 s, offline, seeded
python examples/run_pipeline.py  # full pipeline, about 70 s
```

sklearn is used ONLY for cross-checks and benchmarks; the scorecard logistic
regression, AUC, KS, PSI, HL, Basel and Vasicek maths are implemented from
scratch in `src/eq_credit/`.

## Headline results (synthetic: train 30k loans at 2.95% default rate, OOT 12k with drift)

All numbers are from `python examples/run_pipeline.py` on synthetic data.
The default labels are drawn from a logistic model whose component shapes
(caps, a U-shape, linear terms) are the shapes WOE binning approximates, so
the scorecard is close to the true-model ceiling by construction. The OOT
stress is a pure intercept shift. Defaults are independent, so the book
carries no information about the Vasicek ρ, which is assumed (0.15).

| Metric | Train | OOT |
|---|---|---|
| AUC (true-model ceiling 0.781) | 0.785 [0.770, 0.800] | 0.770 |
| Gini / KS | 0.570 / 0.428 | 0.540 / 0.403 |
| Hosmer-Lemeshow | χ² 17.9, df 8 (in-sample), p 0.022 | χ² 43.2, df 10 (out-of-sample), p 4.7e-06 |
| Score PSI | | 0.076 (stable) |

- OOT back-test by PD grade (binomial test, independent defaults): grades
  4 (2-5%) and 5 (5-10%) are red (p 1.2e-06 and 4.2e-04); the other four
  grades are green.
- Noise features: IV 0.006/0.003, screened out. Planted leaky field
  `writeoff_flag`: IV 6.4, `SuspiciousIVWarning` and a hard `LeakageError`.
- IRLS matches sklearn (no penalty) to 3.7e-14; SEs match analytic Fisher
  information.
- Scorecard: 600 points at 50:1 odds, PDO 20, doubling property exact.
- Portfolio (EAD 41.2bn): EL 605m (1.47%), Basel III RWA 51.6bn (average
  RW 125%), economic capital 8.19% of EAD against Basel K 10.02%.
- Basel hand-check: K(1%, 45%, 2.5) = 0.073853, RW 92.32% (published value).
- Vasicek 99.9% loss rate, homogeneous book (PD 2.96%, LGD 50.1%, ρ 0.15):

  | N loans | exact (quadrature) | Vasicek + granularity adjustment | MC, 30k sims [95% CI] |
  |---|---|---|---|
  | infinite | 11.369% | 11.369% | |
  | 100 | 12.522% | 12.414% | 12.021% [11.520%, 13.023%] |
  | 1,000 | 11.470% | 11.474% | 11.771% [11.019%, 12.522%] |
  | 10,000 | 11.380% | 11.380% | 11.751% [11.275%, 12.557%] |

  The granularity add-on is 0.01% at 10,000 loans. The Monte Carlo
  intervals are wider than the add-on for N ≥ 1,000.

## Layout

```
src/eq_credit/
├── data/synthetic.py   # seeded loan book, known true PD model (U-shape, caps,
│                       # informative missingness, outliers, planted leaks, OOT drift)
├── data/live.py        # German-credit / Lending-Club CSV loader stub (offline)
├── cleaning.py         # leakage deny-list, winsorization, duplicates, train/OOT split
├── woe.py              # quantile pre-bins, chi2 monotone merging, WOE/IV, missing bin
├── model.py            # IRLS logistic + Fisher SEs + Wald tests, stepwise, points/PDO
├── validation.py       # ROC/AUC, KS, HL, per-grade binomial/Jeffreys, Brier, PSI, deciles, bootstrap
└── portfolio_risk.py   # EL, Basel III IRB (R, b, K, RWA), Vasicek, granularity adjustment, MC, EC
```

Docs: [METHODOLOGY.md](docs/METHODOLOGY.md) (why WOE+logit vs GBM/structural,
assumptions register, TTC-vs-PIT), [VALIDATION.md](docs/VALIDATION.md) (all
evidence with numbers, failure modes), [DESK_GUIDE.md](docs/DESK_GUIDE.md)
(cutoffs, pricing, IFRS 9, ICAAP, governance, COVID/2008 scenarios).

## Round 3 models

`lgd_ead` (workout LGD, CCF/EAD, beta LGD, downturn LGD), `calibration` (central tendency plus MoC A/B/C and the CRR floor), `migration` (cohort matrix, lifetime PD, IRW generator) `cds` (hazard bootstrap, simplified conventions) and `isda_cds` (CDS on ISDA standard-model conventions: dated schedule, step-in, ACT/360, exact accrual on default, a real discount curve, par and quoted-spread upfront). See docs/METHODOLOGY.md and docs/VALIDATION.md.
