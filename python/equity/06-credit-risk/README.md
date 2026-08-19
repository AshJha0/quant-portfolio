# 06 — Corporate Credit Risk / PD Modelling (Bank-Style Scorecard)

A complete, from-scratch credit risk stack: seeded synthetic corporate loan
book with known ground truth → cleaning (leakage guard, winsorization,
train/OOT split) → supervised WOE/IV binning (monotone merging, missing bin,
leakage flag) → logistic regression via Newton-Raphson/IRLS (Fisher-info
standard errors, sklearn cross-checked to 1e-13) → points/PDO scorecard →
validation (AUC/Gini/KS, Hosmer-Lemeshow, PSI, rank ordering, bootstrap CIs)
→ expected loss → **exact Basel IRB corporate capital formula** (hand-checked
against the published 92.32% risk weight at PD 1%/LGD 45%/M 2.5) → Vasicek
one-factor loss distribution, analytic + Monte Carlo, economic capital at
99.9%.

```
loan book ─► clean ─► WOE/IV ─► IRLS logit ─► PD ─► score ─► validation
                                                    │
                                EL = PD·LGD·EAD ◄───┴──► Basel K/RWA ─► Vasicek EC
```

## Quickstart

```bash
cd python/equity/06-credit-risk
pip install -e ".[dev]"
pytest -q                      # 136 tests, ~30 s, offline, seeded
python examples/run_pipeline.py  # full pipeline, ~40 s
```

sklearn is used ONLY for cross-checks and benchmarks; the scorecard logistic
regression, AUC, KS, PSI, HL, Basel and Vasicek maths are implemented from
scratch in `src/eq_credit/`.

## Headline results (train 30k loans @ 2.95% default rate, OOT 12k with drift)

| Metric | Train | OOT |
|---|---|---|
| AUC (true-model ceiling 0.781) | 0.784 [0.769, 0.800] | 0.767 |
| Gini / KS | 0.569 / 0.426 | 0.534 / 0.401 |
| Hosmer-Lemeshow p | 0.022 | 3.8e-07 (drifted calibration caught) |
| Score PSI | — | 0.074 (stable) |

- WOE captures the planted U-shaped `current_ratio` effect: AUC 0.603 vs
  0.515 for a linear logit on the raw ratio.
- Noise features: IV 0.006/0.003 → correctly screened out; planted leaky
  field `writeoff_flag`: IV 6.4 → `SuspiciousIVWarning` + hard `LeakageError`.
- IRLS = sklearn (no penalty) to 9.9e-14; SEs match analytic Fisher info.
- Scorecard: 600 points at 50:1 odds, PDO 20 — doubling property exact.
- Portfolio (EAD 41.2bn): EL 605m (1.47%), Basel RWA 51.7bn (avg RW 125%),
  economic capital 8.20% of EAD vs Basel K 10.03%.
- Basel hand-check: K(1%, 45%, 2.5) = 0.073853 → RW 92.32% (published value).
- Vasicek: MC 99.9% tail converges to the analytic quantile from above
  (12.02% @ 100 loans → 11.75% @ 10k vs 11.37% analytic).

## Layout

```
src/eq_credit/
├── data/synthetic.py   # seeded loan book, known true PD model (U-shape, caps,
│                       # informative missingness, outliers, planted leaks, OOT drift)
├── data/live.py        # German-credit / Lending-Club CSV loader stub (offline)
├── cleaning.py         # leakage deny-list, winsorization, duplicates, train/OOT split
├── woe.py              # quantile pre-bins, chi2 monotone merging, WOE/IV, missing bin
├── model.py            # IRLS logistic + Fisher SEs + Wald tests, stepwise, points/PDO
├── validation.py       # ROC/AUC, KS, HL, Brier, PSI, deciles, bootstrap
└── portfolio_risk.py   # EL, Basel IRB (R, b, K, RWA), Vasicek CDF/quantile, MC, EC
```

Docs: [METHODOLOGY.md](docs/METHODOLOGY.md) (why WOE+logit vs GBM/structural,
assumptions register, TTC-vs-PIT), [VALIDATION.md](docs/VALIDATION.md) (all
evidence with numbers, failure modes), [DESK_GUIDE.md](docs/DESK_GUIDE.md)
(cutoffs, pricing, IFRS 9, ICAAP, governance, COVID/2008 scenarios).
