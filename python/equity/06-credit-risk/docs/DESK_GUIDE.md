# Desk Guide — Who Uses This and How

## 1. Consumers of the numbers

| Consumer | What they take | Cadence |
|---|---|---|
| Origination / underwriting | score + cutoff, reason codes from the points table | per application |
| Pricing | PD (with LGD/EAD) → risk-based spread | per application |
| Finance / impairment | PIT PDs → IFRS 9 / CECL expected credit losses | monthly/quarterly |
| Treasury / capital management | Basel K, RWA per exposure | monthly |
| Risk appetite / ICAAP | Vasicek economic capital, stress results | quarterly/annual |
| Model risk management | full validation pack (this repo's VALIDATION.md is the template) | annual + triggers |

## 2. Origination cutoffs

The score is calibrated so 600 points = 50:1 good:bad odds (PD ≈ 1.96%) and
every 20 points doubles the odds.  A typical policy on the demo book:

- **Auto-approve ≥ 620** (odds 100:1, PD ≤ 1.0%)
- **Manual review 560–620** (odds 12.5:1–100:1, PD 1.0%–7.4%)
- **Decline < 560**

Cutoffs are set from the KS/decile table: the demo scorecard concentrates
11.2% default rate in the worst decile vs 0.10% in the best (112×), so a
cutoff excluding the bottom two deciles removes ~55% of expected defaults at
the cost of ~20% of volume.  Reason codes for declines read directly off the
points table (e.g. "leverage > 0.94: 58 pts vs 117 for < 0.18").

## 3. Risk-based pricing

Spread over funding ≈ (PD·LGD)/(1−PD) + capital cost·K + operating cost.  A
BB obligor (PD 1.5%, LGD 50%) on the demo numbers: EL 75bp + 12% cost of
equity × K(1.5%, 50%) ≈ 9.3% × 12% ≈ 112bp → ~190bp before costs.  The same
loan at CCC (PD 7%) prices ~480bp — the scorecard PD is the single largest
pricing input.

## 4. IFRS 9 / CECL provisioning link

- Stage 1: 12-month ECL = PD₁ᵧ·LGD·EAD — exactly this model's output
  (605m / 1.47% of EAD on the demo book; 754m with the downturn-LGD haircut).
- Stage 2 (significant increase in credit risk): triggered by score
  migration — e.g. score drop > 40 points (= odds worse by 4×) since
  origination; requires lifetime PD term structures built from this 1-year PD
  (survival extension not in scope here).
- Because the model is PIT, ECL responds to the cycle — that is *intended*
  under IFRS 9, unlike for capital (see procyclicality, VALIDATION.md §5.5).

## 5. ICAAP / stress testing

The Vasicek layer is the ICAAP engine: economic capital at 99.9% (8.20% of
EAD on the demo book) vs Basel K (10.03%).  Stress scenarios are run by
(i) shifting the calibration intercept (the OOT generator does exactly this —
+0.5 log-odds ≈ default rate 3.0% → 4.8%), (ii) applying the downturn-LGD
haircut, (iii) raising ρ (sector concentration stress).  The MC engine
accepts the actual heterogeneous book, so name-level concentration shows up
directly in the tail.

## 6. Model governance — annual validation cycle

Per SR 11-7 / EBA model-governance expectations:

1. **Independent validation** (not the developers) annually: re-run
   `examples/run_pipeline.py`-equivalent on the latest cohort; the
   VALIDATION.md tables are the pack skeleton.  Effective challenge includes
   a challenger model (GBM benchmark) and the sklearn cross-check.
2. **Ongoing monitoring, with hard triggers:**
   - **PSI > 0.25 on the score or any input → recalibrate/rebin** (the
     0.10–0.25 band → investigate). Implemented in `psi_report` / `psi_status`.
   - Realised vs predicted default rate outside the binomial 95% band, or
     HL p < 0.01 on the latest vintage → recalibrate intercept (at minimum).
   - AUC drop > 5 points from development → redevelop.
3. **Change control**: binning tables and coefficients are the model; any
   change is versioned and re-approved.  The leakage deny-list
   (`FORBIDDEN_POST_OUTCOME_FIELDS`) is part of the model definition.
4. **Use test**: pricing, origination and provisioning must consume the same
   PD (one model, many uses) or divergences must be documented.

## 7. Realistic scenarios

- **COVID payment holidays (2020) breaking behavioural scores.**  Payment
  holidays suppressed arrears, so behavioural scores stayed artificially
  clean while true risk rose; simultaneously `behavioral_score` went missing
  for moratorium accounts.  This model survives the second effect gracefully
  (missing = own bin, WOE −0.80 fitted on genuine thin-file risk) but NOT the
  first: the fitted missing-bin WOE assumes the historical missingness
  mechanism (assumption 6).  Correct desk response in 2020 was to freeze the
  behavioural component and fall back to financials — i.e. refit with the
  behavioural feature excluded, which this pipeline supports by construction.
- **2008-style vintage effects.**  Loans originated at the top of the cycle
  (loose underwriting) default more at every score level.  Detection:
  vintage-level HL tests, exactly the OOT demonstration here (realised 4.78%
  vs predicted 3.80%).  Response: intercept recalibration by vintage, not
  full redevelopment, unless rank-ordering (AUC) also degrades.
- **Sector concentration.**  The demo book is 12% construction — the riskiest
  sector (WOE −0.4 vs manufacturing).  The single-factor Vasicek treats
  sectors as perfectly correlated through one factor; a construction bust is
  understated.  ICAAP overlay: re-run MC with a stressed construction PD
  (×2) and report the delta as a concentration add-on.
- **Score PSI drifting into the monitor band.**  On the OOT sample the score
  PSI is 0.074 (stable) while *calibration* already failed — the documented
  lesson that PSI monitors the input mix, not correctness; both monitors are
  required.
