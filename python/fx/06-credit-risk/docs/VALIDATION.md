# Validation: Evidence, Cross-Checks, Failure Modes

All numbers below are produced by `examples/run_pipeline.py` (seed 42) and
locked in by the 218-test suite (`pytest -q`, offline, about 30 s). All data are synthetic. Section 6 maps
every documented edge case to its test.

## 1. Cross-model and analytic benchmarks

| Check | Result | Test |
|---|---|---|
| IRLS coefficients vs `sklearn.LogisticRegression` (C=∞, tol 1e-12) | max diff **3.5e-07** on the pipeline fit, < 1e-6 enforced on 40k-row recovery data | `test_model.py::test_irls_matches_sklearn_1e6` |
| IRLS recovers known ground-truth β (n=40k) | every coefficient within 4 SE and 0.08 absolute | `test_irls_recovers_true_coefficients` |
| Intercept-only MLE analytic identity (logit(ȳ), SE = 1/√(np(1−p))) | exact to 1e-8 | `test_intercept_only_analytic` |
| AUC from scratch vs `roc_auc_score` incl. heavy ties | ≤ 1e-12 (identical rank construction) | `test_auc_matches_sklearn_*` |
| WOE/IV vs hand arithmetic (2-bin table, smoothing 0, good/bad WOE) | exact: WOE = ±ln2, IV = 2ln2/3 | `test_woe_hand_computed_exact` |
| Smoothed WOE shares sum to 1 for any bin count | 1e-15 | `test_smoothed_shares_sum_to_one_for_any_bin_count` |
| HL out-of-sample size at 5% (known PDs, df = 10) | rejection in [3%, 7%] of 1,000 replications | `test_hl_out_of_sample_size_is_nominal` |
| Hosmer-Lemeshow vs hand chi-square (3 groups) | exact to 1e-12 | `test_hosmer_lemeshow_hand_computed` |
| PSI vs hand arithmetic (2 bins) | exact to 1e-12 | `test_psi_hand_computed` |
| PDO scaling identity (doubling odds = +20 pts), score↔PD round trip | exact to 1e-10 / 1e-12 | `test_pdo_identity_exact`, `test_score_pd_round_trip` |
| Expected loss on exposure vs hand 2-period case (LGD .6, EE [10,8], ΔPD [.01,.02]) | exact: 0.156 | `test_cva_hand_computed_two_period` |
| GBM martingale E[S_T] = S₀e^{(r_d−r_f)T} | within 3 SE, 200k paths | `test_paths_forward_martingale` |
| Vasicek vs granular portfolio simulation (5000 names, one factor) | loss-rate quantile matches ASRF conditional PD within 0.01 | `test_vasicek_matches_granular_portfolio_simulation` |
| Settlement window vs hand clock arithmetic (JPY→USD = 23.5 h) | exact | `test_window_pay_jpy_receive_usd_hand_checked` |

## 2. Statistical backtest: scorecard, train (≤2014) vs out-of-time (2015-23)

```
metric            train    out-of-time
AUC               0.817        0.690
Gini              0.635        0.380
KS                0.471        0.315
HL chi2           7.67         33.75
HL dof            8 (in-sample) 10 (out-of-sample)
HL p-value        0.466        2.0e-04
PSI (scores)                   0.018
```

Hosmer-Lemeshow uses n_groups - 2 degrees of freedom for the fitted
training sample and n_groups for the out-of-time sample, which the model
never saw. Discrimination degrades out-of-time (0.69); the score population
is stable (PSI 0.018). Calibration fails out-of-time because of the planted
2020 year (§4.2). The recovered binning shows the planted nonlinearity: bad
rate 11.3% below 3.3 months of import cover vs 3.3% above 6.2 months, WOE
-0.81 to +0.47 (good/bad convention), monotone after merge.

`fiscal_gdp` failed the expected-sign check (coefficient +0.66, z 0.71,
where every good/bad-WOE slope should be negative) and is dropped before the
final fit.

OOT back-test by rating (`pd_backtest_by_grade`; binomial test assumes
independent defaults, which the panel violates, so it is strict):

| Rating | N | Defaults | Mean PD | Observed | Binomial p | Jeffreys p | Light |
|---|---:|---:|---:|---:|---:|---:|---|
| AA | 1 | 0 | 0.09% | 0.00% | 1.000 | 0.039 | green |
| A | 22 | 1 | 0.24% | 4.55% | 0.051 | 0.009 | green |
| BBB | 96 | 1 | 0.70% | 1.04% | 0.488 | 0.279 | green |
| BB | 172 | 8 | 1.80% | 4.65% | 0.013 | 0.008 | amber |
| B | 168 | 12 | 5.25% | 7.14% | 0.174 | 0.137 | green |
| CCC | 54 | 9 | 14.23% | 16.67% | 0.359 | 0.292 | green |
| C | 27 | 4 | 34.04% | 14.81% | 0.993 | 0.987 | green |

Observed default rates are not monotone across ratings (A 4.55% > BBB
1.04%); only the predicted band PDs are.

## 3. Out-of-time is the right split (panel leakage)

Within-country residual lag-1 autocorrelation is **+0.116** (warning utility
fires above 0.10). A random row split would place adjacent country-years on
both sides and overstate AUC. Enforced by construction:
`test_time_split_no_same_country_future_row_in_train` proves no same-country
future row can land in train; the country-holdout split (AUC 0.72 on 15
never-seen countries, `test_country_holdout_generalisation`) tests
cross-sectional generalisation separately.

## 4. Failure modes (each reproducible)

### 4.1 Low-default portfolio ⇒ wide confidence intervals
OOT window has 35 events in 540 rows. Bootstrap 95% CI on the OOT AUC:
**0.690 [0.599, 0.782], width 0.183** (about ±9 Gini points). Any claim that
one sovereign model "beats" another by 2 AUC points is noise. Tests:
`test_bootstrap_ci_wide_for_low_default`, `test_low_default_bootstrap_ci_is_wide`.

### 4.2 Contagion breaks calibration (planted 2020 numbers)
The generator plants a *global* contagion year in 2020, inside the OOT
window; the scorecard (which correctly excludes the contemporaneous contagion
flag) predicts mean PD **6.3%** for 2020 while the realised default rate is
**16.7%**, a 2.6x understatement that drives the OOT HL failure (chi2 33.8,
df 10). Excluding 2020, other OOT years are nearly calibrated (predicted
5.36% vs observed 5.21%). The flag is excluded from the scorecard and the
year is planted, so this result holds by construction. Lesson: a fundamentals scorecard prices *idiosyncratic*
sovereign risk; systemic waves need the stress overlay, not a recalibrated
intercept. Test: `test_planted_contagion_year_breaks_calibration`.

### 4.3 Pegs mask risk until they break
The peg dummy carries a WOE-space coefficient of -1.00 (good/bad WOE, so
the expected sign), but a peg
suppresses *observed* volatility while devaluation risk accumulates
(Argentina 2001: cover ratios fine until convertibility died). In the
generator, pegged countries carry +0.45 latent log-odds; in real data the
scorecard sees calm macro inputs right up to the break: PD jumps
discontinuously. Control: peg-country limits carry a mandatory qualitative
overlay (DESK_GUIDE §3).

### 4.4 Wrong-way risk: ignoring it understates the loss about 6x (hand example, not code)
Buy USD 10m forward vs an EM currency from the EM sovereign itself. Suppose
PD₁y = 4%, LGD 55%, and: conditional on default the currency devalues 30%,
so exposure at default ≈ 10m × 30% = **3.0m**; unconditionally EE ≈ 0.5m.
- Independence: 0.55 × 0.04 × 0.5m = **USD 11,000**
- Wrong-way: 0.55 × 0.04 × 3.0m = **USD 66,000**, 6.0x larger.
`expected_exposure_loss()` assumes independence (Assumption A7); for EM sovereign
counterparties trading their own currency, multiply EE by a devaluation-
conditional factor or price off a jump-at-default model. The 2022 RUB episode
is the settlement-side cousin (DESK_GUIDE §3).

### 4.5 Data staleness
Panel features are annual; a Guidotti ratio observed at year-end reaches the
model 6-18 months later in production (A2). Russia's short-term debt build-up
in 1997-98 was visible in *vintage* data only with a lag: an annual-frequency
scorecard cannot be an early-warning siren; it sets through-the-cycle limits
while market-based overlays (CDS, reserves drain at weekly frequency) handle
timing.

### 4.6 Numerical limits
Separation: a WOE bin that perfectly predicts default makes the MLE diverge.
It is detected at |slope| > 30 (the intercept is excluded) with an
actionable error, ridge restores a finite fit
(`test_separation_raises` / `test_separation_handled_with_ridge`). Collinear
WOE columns (a constant regime dummy before the low-cardinality binning fix)
raise rather than silently blow up. Vasicek K is non-monotone above PD ≈ 25%
(EL crowds out UL: K(C) = 25.1 < K(CCC) = 28.5 per 100 EAD). This is
correct behaviour of the formula. IRLS warns with `ConvergenceWarning` when it
stops at `max_iter`, and the pipeline refuses a non-converged fit.

## 5. Settlement & exposure evidence

- 6-trade book: all-gross **USD 180.1m**, with CLS **142.0m**, with bilateral
  payment netting **66.2m** under operating-hours windows, which are a
  lower bound. The BCBS (2013) window runs from the unilateral cancellation
  deadline to confirmed receipt; with a 24 h lead and 12 h reconciliation
  JPY→USD is 59.5 h and USD→JPY 30.5 h, so receive-before-pay trades also
  carry exposure (`test_bcbs_window_creates_exposure_on_receive_before_pay_trade`). Tests: `test_settlement.py` (hand-checked
  windows, PvP zeroing, same-currency netting, cross-counterparty isolation).
- Forward PFE (EUR 10m, 1y, 100k paths): 99% PFE 1.68m → 2.41m → 3.00m →
  3.52m at quarterly points: strictly increasing, decreasing increments
  (concavity), corr(PFE, √t) > 0.99, PFE99 > PFE95 everywhere. Netting:
  offsetting pair 3.54m → 0; identical same-direction trades: netting benefit
  exactly zero (equality bound). Tests: `test_exposure.py`.
- Expected loss on exposure (real-world PD 2%, a fixed synthetic input;
  LGD 55%, flat hazard, r_d 3%): **USD 4,431** = 4.1 bp of USD notional.
  This is not a CVA: there is no risk-neutral PD, collateral, margin period
  of risk, DVA, EEPE or multi-pair netting.

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
| NaN/Inf inputs (paths, EL, Vasicek, expected exposure loss, AUC/KS/PSI/HL, USD rates) | `test_edge_cases_review.py` NaN/Inf block: every public entry point raises `ValueError` |
| Tiny samples (2-row AUC, 2-row WOE), constant scores | `test_auc_minimal_two_row_sample`, `test_woe_single_row_per_class`, `test_auc_constant_score_is_half`, `test_ks_constant_score_is_zero` |
| Empty / all-CLS settlement book | `test_empty_trade_book_zero_exposure`, `test_netting_all_cls_book_zero_exposure` |

## November 2026 oracle tests (`tests/test_xva_saccr.py`, 9 tests)

- Constant-EPE CVA equals LGD EPE (1 - e^{-lambda T}) to 1e-13. CVA and DVA swap exactly when the two parties are swapped.
- The uncollateralised EE of a forward matches the Black price within 4 standard errors at 3, 6 and 12 months.
- The MPoR EPE matches its closed form within 4 standard errors. Adding a threshold raises the exposure. A very large MTA gives the uncollateralised exposure.
- WWR with b = 0 equals the independent CVA to 1e-9. Wrong-way b > 0 raises CVA and right-way b < 0 lowers it.
- SA-CCR: the add-on, multiplier, margined RC and maturity factor match hand values from the CRE52 formulas.
