# Validation — Evidence, Cross-Checks, Failure Modes

All numbers below are produced by `examples/run_pipeline.py` (seed 42) and
locked in by the 207-test suite (`pytest -q`, offline, ~8 s). Section 6 maps
every documented edge case to its test.

## 1. Cross-model and analytic benchmarks

| Check | Result | Test |
|---|---|---|
| IRLS coefficients vs `sklearn.LogisticRegression` (C=∞, tol 1e-12) | max diff **2.5e-07** on the pipeline fit, < 1e-6 enforced on 40k-row recovery data | `test_model.py::test_irls_matches_sklearn_1e6` |
| IRLS recovers known ground-truth β (n=40k) | every coefficient within 4 SE and 0.08 absolute | `test_irls_recovers_true_coefficients` |
| Intercept-only MLE analytic identity (logit(ȳ), SE = 1/√(np(1−p))) | exact to 1e-8 | `test_intercept_only_analytic` |
| AUC from scratch vs `roc_auc_score` incl. heavy ties | ≤ 1e-12 (identical rank construction) | `test_auc_matches_sklearn_*` |
| WOE/IV vs hand arithmetic (2-bin table, smoothing 0) | exact: WOE = ±ln2, IV = 2ln2/3 | `test_woe_hand_computed_exact` |
| Hosmer-Lemeshow vs hand chi-square (3 groups) | exact to 1e-12 | `test_hosmer_lemeshow_hand_computed` |
| PSI vs hand arithmetic (2 bins) | exact to 1e-12 | `test_psi_hand_computed` |
| PDO scaling identity (doubling odds = +20 pts), score↔PD round trip | exact to 1e-10 / 1e-12 | `test_pdo_identity_exact`, `test_score_pd_round_trip` |
| CVA vs hand 2-period case (LGD .6, EE [10,8], ΔPD [.01,.02]) | exact: 0.156 | `test_cva_hand_computed_two_period` |
| GBM martingale E[S_T] = S₀e^{(r_d−r_f)T} | within 3 SE, 200k paths | `test_paths_forward_martingale` |
| Vasicek vs granular portfolio simulation (5000 names, one factor) | loss-rate quantile matches ASRF conditional PD within 0.01 | `test_vasicek_matches_granular_portfolio_simulation` |
| Settlement window vs hand clock arithmetic (JPY→USD = 23.5 h) | exact | `test_window_pay_jpy_receive_usd_hand_checked` |

## 2. Statistical backtest — scorecard, train (≤2014) vs out-of-time (2015-23)

```
metric            train    out-of-time
AUC               0.820        0.691
Gini              0.639        0.382
KS                0.482        0.315
HL chi2 (dof 8)   7.65         35.47
HL p-value        0.468        3e-05
PSI (scores)        —          0.013
```

Discrimination degrades but survives out-of-time (0.69 vs literature ~0.65-0.75
for sovereign early-warning models); score population is stable (PSI 0.013,
well under the 0.10 watch level). Calibration **fails** out-of-time — by
design, see §4.2. The recovered binning shows the planted nonlinearity: bad
rate 11.3% below 3.3 months of import cover vs 3.3% above 6.2 months, WOE
+0.83 → −0.46, monotone after merge.

## 3. Out-of-time is the honest split (panel leakage)

Within-country residual lag-1 autocorrelation is **+0.114** (warning utility
fires above 0.10). A random row split would place adjacent country-years on
both sides and overstate AUC. Enforced by construction:
`test_time_split_no_same_country_future_row_in_train` proves no same-country
future row can land in train; the country-holdout split (AUC 0.72 on 15
never-seen countries, `test_country_holdout_generalisation`) tests
cross-sectional generalisation separately.

## 4. Failure modes (each reproducible)

### 4.1 Low-default portfolio ⇒ wide confidence intervals
OOT window has 35 events in 540 rows. Bootstrap 95% CI on the OOT AUC:
**0.691 [0.598, 0.782] — width 0.184** (i.e. ±9 Gini points). Any claim that
one sovereign model "beats" another by 2 AUC points is noise. Tests:
`test_bootstrap_ci_wide_for_low_default`, `test_low_default_bootstrap_ci_is_wide`.

### 4.2 Contagion breaks calibration (planted 2020 numbers)
The generator plants a *global* contagion year in 2020, inside the OOT
window; the scorecard (which correctly excludes the contemporaneous contagion
flag) predicts mean PD **6.4%** for 2020 while the realised default rate is
**16.7%** — a 2.6x understatement, and the entire OOT HL failure (chi2 35.5):
excluding 2020, other OOT years are nearly calibrated (predicted 5.35% vs
observed 5.21%). Lesson: a fundamentals scorecard prices *idiosyncratic*
sovereign risk; systemic waves need the stress overlay, not a recalibrated
intercept. Test: `test_planted_contagion_year_breaks_calibration`.

### 4.3 Pegs mask risk until they break
The peg dummy carries positive WOE-space coefficient (+0.90), but a peg
suppresses *observed* volatility while devaluation risk accumulates
(Argentina 2001: cover ratios fine until convertibility died). In the
generator, pegged countries carry +0.45 latent log-odds; in real data the
scorecard sees calm macro inputs right up to the break — PD jumps
discontinuously. Control: peg-country limits carry a mandatory qualitative
overlay (DESK_GUIDE §3).

### 4.4 Wrong-way risk: ignoring it understates CVA ~6x (toy, quantified)
Buy USD 10m forward vs an EM currency from the EM sovereign itself. Suppose
PD₁y = 4%, LGD 55%, and: conditional on default the currency devalues 30%,
so exposure at default ≈ 10m × 30% = **3.0m**; unconditionally EE ≈ 0.5m.
- Independence CVA = 0.55 × 0.04 × 0.5m = **USD 11,000**
- Wrong-way CVA = 0.55 × 0.04 × 3.0m = **USD 66,000** — **6.0x larger.**
The `cva()` engine assumes independence (Assumption A7); for EM sovereign
counterparties trading their own currency, multiply EE by a devaluation-
conditional factor or price off a jump-at-default model. The 2022 RUB episode
is the settlement-side cousin (DESK_GUIDE §3).

### 4.5 Data staleness
Panel features are annual; a Guidotti ratio observed at year-end reaches the
model 6-18 months later in production (A2). Russia's short-term debt build-up
in 1997-98 was visible in *vintage* data only with a lag — an annual-frequency
scorecard cannot be an early-warning siren; it sets through-the-cycle limits
while market-based overlays (CDS, reserves drain at weekly frequency) handle
timing.

### 4.6 Numerical limits
Separation: a WOE bin that perfectly predicts default makes the MLE diverge —
detected at |β| > 30 with an actionable error, ridge restores a finite fit
(`test_separation_raises` / `test_separation_handled_with_ridge`). Collinear
WOE columns (a constant regime dummy before the low-cardinality binning fix)
raise rather than silently blow up. Vasicek K is non-monotone above PD ≈ 25%
(EL crowds out UL: K(C) = 25.1 < K(CCC) = 28.5 per 100 EAD) — correct
behaviour, documented so nobody "fixes" it.

## 5. Settlement & exposure evidence

- 6-trade book: all-gross **USD 180.1m**, with CLS **142.0m**, with bilateral
  payment netting **66.2m**; window matrix reproduces the Herstatt asymmetry
  (JPY→USD 23.5h vs USD→JPY 0h). Tests: `test_settlement.py` (hand-checked
  windows, PvP zeroing, same-currency netting, cross-counterparty isolation).
- Forward PFE (EUR 10m, 1y, 100k paths): 99% PFE 1.68m → 2.41m → 3.00m →
  3.52m at quarterly points — strictly increasing, decreasing increments
  (concavity), corr(PFE, √t) > 0.99, PFE99 > PFE95 everywhere. Netting:
  offsetting pair 3.54m → 0; identical same-direction trades: netting benefit
  exactly zero (equality bound). Tests: `test_exposure.py`.
- CVA (BB midpoint PD 2%, LGD 55%, flat hazard, r_d 3%): **USD 4,431** =
  4.1 bp of USD notional.

## 6. Edge-case ↔ test map (contract item 6)

| Edge case | Test |
|---|---|
| Country with no crisis history still scored/rated | `test_scenarios.py::test_country_with_no_crisis_history_scored` |
| Same-country future row leaking into train | `test_cleaning.py::test_time_split_no_same_country_future_row_in_train` |
| Post-crisis leaky fields present at fit time | `test_leak_guard_raises_on_leaky_fields`, `test_leaky_feature_iv_flagged_on_panel` |
| Entirely-missing feature / missing bin | `test_all_missing_raises`, `test_missing_bin_created_and_exact` |
| Constant feature, degenerate bins | `test_constant_feature_zero_iv` |
| Perfect separation (low-default hazard) | `test_separation_raises`, `test_separation_handled_with_ridge` |
| Zero-notional trade | `test_zero_notional_trade_zero_exposure`, `test_zero_notional_zero_profile` |
| Matured forward (T ≤ 0, t > T) | `test_matured_forward_zero_profile`, `test_cva_matured_forward_zero` |
| Counterparty with PD = 0 | `test_cva_zero_pd_counterparty_is_zero` |
| Single-currency book (no FX settlement risk) | `test_single_currency_book_no_settlement_risk` |
| PD = 0 / PD = 1 in Vasicek | `test_vasicek_degenerate_pds`, `test_capital_zero_at_boundary_pds` |
| Unknown currency / rating / missing FX rate | `test_window_unknown_currency_raises`, `test_standardized_rw_table`, `test_missing_usd_rate_raises` |
| Network access attempted | `test_live_loader_is_network_guarded` |
| Pegged pair, vol = 0 (deterministic exposure, PFE = EE) | `test_edge_cases_review.py::test_pegged_pair_zero_vol_exposure_pfe_equals_ee` |
| Distressed sovereign PD → 1 (hazard explodes, K → 0, CVA bounded) | `test_hazard_explodes_as_pd_to_one`, `test_capital_vanishes_at_both_pd_extremes_high_rho`, `test_cva_bounded_by_lgd_times_peak_ee` |
| NaN/Inf inputs (paths, EL, Vasicek, CVA, AUC/KS/PSI/HL, USD rates) | `test_edge_cases_review.py` NaN/Inf block — every public entry point raises `ValueError` |
| Tiny samples (2-row AUC, 2-row WOE), constant scores | `test_auc_minimal_two_row_sample`, `test_woe_single_row_per_class`, `test_auc_constant_score_is_half`, `test_ks_constant_score_is_zero` |
| Empty / all-CLS settlement book | `test_empty_trade_book_zero_exposure`, `test_netting_all_cls_book_zero_exposure` |
