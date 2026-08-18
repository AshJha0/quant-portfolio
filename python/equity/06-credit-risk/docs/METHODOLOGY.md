# Methodology — Corporate Credit Risk / PD Modelling (WOE Scorecard)

## 1. The pipeline

```
loan book ──► cleaning ──► WOE / IV binning ──► IRLS logistic ──► PD ──► score
   │            │               │                    │                     │
   │      leakage guard,   missing = own bin,   from-scratch          points/PDO
   │      winsorization,   monotone merging,    Newton-Raphson,       600 @ 50:1,
   │      duplicates,      IV leakage flag      Fisher-info SEs       PDO = 20
   │      train/OOT split
   │
   └──► validation (AUC/KS, Hosmer-Lemeshow, PSI) ──► EL = PD·LGD·EAD
                                        ──► Basel IRB K / RWA ──► Vasicek economic capital
```

The model is a **bank-style scorecard**: each raw feature is binned, each bin
replaced by its Weight of Evidence (WOE), and a logistic regression is fitted
on the WOE-transformed features.  PDs map to points via the classic
points/PDO scaling (600 points at 50:1 odds, 20 points to double the odds).

## 2. Why a WOE + logistic scorecard? (vs at least two alternatives)

**Alternative 1 — gradient-boosted trees / ML classifiers.**
GBMs usually win 1–3 AUC points on tabular credit data.  We still choose the
scorecard because:

- **Interpretability and adverse action.** Every point movement traces to a
  named bin of a named ratio.  Under ECOA/GDPR-style regimes, declined
  obligors receive concrete reasons; SHAP explanations of a 500-tree
  ensemble do not satisfy a credit officer or many regulators the same way.
- **Regulatory acceptance and SR 11-7 model risk.** IRB approval, annual
  independent validation, and audit all require a model whose full state fits
  on two pages (binning tables + coefficient table).  The Fed's SR 11-7
  demands effective challenge: a challenger GBM is run as a *benchmark* (our
  sklearn cross-checks play this role in miniature), but the production model
  stays simple.
- **Monotonicity by construction.** Business logic requires "more leverage
  never lowers the PD".  Our binning enforces monotone WOE per feature
  (except where the U-shape is real and documented, e.g. current ratio);
  GBMs need bolt-on monotone constraints and still interact features opaquely.
- **Stability under thin data.** With a 3% base rate, 30k loans contain ~900
  defaults.  A 9-coefficient logit has ~100 defaults per parameter; a GBM
  happily memorises noise at this size.
- The cost — a few AUC points — is measured here directly: the scorecard
  achieves AUC 0.784 vs the true-model ceiling of 0.781 on the training
  sample (the generator's noise floor dominates), i.e. on this book the
  scorecard leaves essentially nothing on the table.

**Alternative 2 — linear logit on raw (unbinned) ratios.**
Cheaper, but it misses real nonlinearity.  On our book, `current_ratio` has a
genuine U-shaped risk profile: a raw-linear logit on it scores AUC **0.515**
(nothing), while the WOE-binned version scores **0.603** (see
`examples/run_pipeline.py`, section 3).  WOE binning also handles missing
values natively (own bin), absorbs outliers (top/bottom bins), and puts every
feature on the same log-odds scale so coefficients are comparable (all ≈ −1
for a well-specified binning).

**Alternative 3 — Merton/structural distance-to-default models.**
Appropriate for listed corporates with traded equity (KMV-style EDF), but our
book is mid-market lending: no market cap, no asset volatility. Structural
models also calibrate point-in-time, cyclical PDs, which conflicts with a
stable origination cutoff (see TTC vs PIT below).

## 3. WOE / IV mathematics

For bin *i* of a feature, with `dist_good_i` the share of all non-defaults in
bin *i* and `dist_bad_i` the share of all defaults:

```
WOE_i = ln( dist_good_i / dist_bad_i )
IV    = Σ_i (dist_good_i − dist_bad_i) · WOE_i  ≥ 0
```

IV thresholds (Siddiqi): `< 0.02` useless, `0.02–0.1` weak, `0.1–0.3` medium,
`0.3–0.5` strong, **`> 0.5` suspicious** — automatically flagged as a
potential leak (`SuspiciousIVWarning`); the planted post-outcome field
`writeoff_flag` triggers it with IV ≈ 6.4.

Binning algorithm: ~20 quantile pre-bins → merge bins below 5% of mass →
merge the adjacent pair with the smallest 2×2 chi-square until the bad rate
is monotone.  Missing values form their own bin (informative missingness on
`behavioral_score` gets WOE −0.80: thin-file borrowers are riskier).
Zero-count cells receive a +0.5 smoothing (that bin only) so WOE is finite.

## 4. Estimation

Newton-Raphson / IRLS on the log-likelihood, implemented from scratch:
`β ← β + (XᵀWX)⁻¹ Xᵀ(y − p)` with `W = diag(p(1−p))`, optional ridge for
separation control.  Standard errors from the inverse Fisher information;
Wald z and p-values per coefficient.  The fit matches
`sklearn.LogisticRegression(penalty=None)` to ~1e-13 (independent
implementation cross-check, tested to 1e-6 in CI).

Scorecard scaling: `factor = PDO/ln 2 = 28.854`, `offset = 600 −
factor·ln 50 = 487.12`, `score = offset + factor·ln((1−PD)/PD)`.  The PDO
property (odds double every 20 points) holds exactly and is unit-tested.

## 5. Assumptions register (what breaks if violated)

1. **Stationarity of the feature→default relationship.**  The scorecard fitted
   on 2019–21 originations is applied to 2022+.  If the relationship shifts
   (regime change, payment holidays), calibration breaks first, discrimination
   second — exactly what the OOT sample demonstrates (HL p = 3.8e-07 after a
   +0.5 log-odds calibration shift).  Monitoring: PSI and vintage HL tests.
2. **One-year default horizon, cohort sampling.**  Each loan contributes one
   Bernoulli observation.  Violated by cure-and-redefault dynamics or multiple
   observation periods per loan (would need panel methods / GEE corrections to
   the SEs).
3. **Independence of defaults given features (estimation).**  IRLS SEs assume
   independent observations.  Sector/systematic correlation inflates true
   sampling error; portfolio-level correlation is handled *separately* in the
   Vasicek layer, not in the regression.
4. **LGD and EAD independent of PD.**  EL = PD·LGD·EAD multiplies point
   estimates.  In downturns PD and LGD rise together (**wrong-way risk**):
   collateral values fall exactly when defaults spike (2008 CRE is the
   canonical example).  The downturn-LGD haircut (+25% → EL rises from 605m
   to 754m on the demo book) is a crude, regulator-style patch, not a joint
   model.
5. **Monotone risk in each ratio, except documented exceptions.**  Enforced in
   binning.  If a true U-shape is forced monotone the signal is destroyed —
   so `current_ratio` is explicitly exempted (`non_monotone_features`).
6. **Missingness pattern stability.**  The missing bin's WOE assumes the
   *reason* for missingness is stable.  If e.g. a data pipeline change makes
   `behavioral_score` missing at random, the −0.80 WOE penalty misprices
   clean borrowers.
7. **Vasicek single-factor, Gaussian copula, flat ρ.**  Economic capital
   assumes one systematic factor and normal asset returns.  Sector
   concentration (multi-factor reality) and tail dependence are outside the
   model — see VALIDATION.md failure modes.
8. **Synthetic ground truth stands in for market data.**  All numbers are
   generated from a known PD model (per portfolio conventions, tests are
   offline/deterministic); `data/live.py` documents the mapping for real
   German-credit / Lending-Club style CSVs.

## 6. TTC vs PIT calibration

- **Point-in-time (PIT)** PDs track the current state of the cycle: they use
  current behavioural data and are recalibrated frequently.  Best for IFRS 9 /
  CECL provisioning, which *wants* cycle-sensitive expected losses.
- **Through-the-cycle (TTC)** PDs average over the cycle: stable ratings,
  stable capital, at the cost of under-predicting defaults in busts and
  over-predicting in booms.  Basel IRB capital is designed around
  (approximately) TTC PDs.

**This model is essentially PIT**: it is calibrated to the realised one-year
default rate of the training window and includes a behavioural score, which
is intrinsically cyclical.  Consequences: (i) using these PDs directly in the
IRB formula makes capital **procyclical** (capital requirements rise in a
downturn exactly when capital is scarce); (ii) for IRB use, banks typically
apply a cycle adjustment / long-run average PD per grade.  The OOT
demonstration quantifies the PIT weakness: the realised OOT rate (4.78%)
exceeds the mean predicted PD (3.80%) once the cycle turns.

## 7. Basel IRB and Vasicek layer

The IRB capital function K(PD, LGD, M) is the Vasicek 99.9% conditional PD
minus expected loss, times a maturity adjustment — formulas and constants in
`portfolio_risk.py`, hand-checked at PD = 1%, LGD = 45%, M = 2.5 against the
published corporate risk weight of **92.32%** (BCBS Explanatory Note, July
2005).  The Vasicek analytic loss CDF, its quantile, seeded Monte Carlo for
finite portfolios, and the granularity comparison are all reproduced and
tested; details and numbers in VALIDATION.md.
