# Validation: Evidence, Numbers, Failure Modes

All numbers below are produced by `python examples/run_pipeline.py`
(train: 30,000 loans, seed 42, 2.95% default rate; OOT: 12,000 loans,
seed 123, drift 0.5 + calibration shift 0.5 → 4.78% default rate) and by the
test suite (`pytest -q`, 157 tests, offline, seeded). All data are synthetic.

## 1. Analytic and cross-model benchmarks

| Check | Result | Test |
|---|---|---|
| WOE of a bin = ln(dist_good/dist_bad), hand-computed 3-bin table | exact to 1e-14 | `test_woe_hand_computed_exact` |
| IV hand-computed on the same table | exact to 1e-14 | `test_iv_hand_computed_exact` |
| IRLS coefficients vs sklearn `LogisticRegression(penalty=None)` | max diff 3.7e-14 (pipeline), < 1e-6 (CI gate) | `test_matches_sklearn_to_1e6` |
| Standard errors vs independent Fisher-information calculation | rel. 1e-8 | `test_standard_errors_match_analytic_fisher` |
| AUC from scratch vs `sklearn.roc_auc_score` (incl. heavy ties) | 1e-12 | `test_auc_matches_sklearn_to_1e12` |
| Gini = 2·AUC − 1 | 1e-14 | `test_gini_identity` |
| PDO property: odds double ⇔ +20 points, at 3 anchor points | 1e-10 | `test_pdo_property_holds_exactly` |
| PSI hand-computed on known proportions | 1e-14 | `test_psi_hand_computed_exact_on_known_proportions` |
| Basel K at PD=1%, LGD=45%, M=2.5, re-derived independently in the test with the exact regulatory constants (R(PD), b(PD), N⁻¹(0.999)) | K = 0.073853 → RW = **92.32%**, matching the published Basel II corporate curve (BCBS Explanatory Note, July 2005) | `test_basel_k_reproduces_independent_hand_calculation` |
| Vasicek quantile/CDF inverse round-trip | 1e-10 | `test_vasicek_quantile_cdf_round_trip` |
| Vasicek 99.9% quantile ≡ Basel conditional-PD term at ρ = R(PD) | 1e-12 | `test_vasicek_matches_basel_conditional_pd` |

## 2. Coefficient recovery (known ground truth)

On 200k simulated observations, IRLS recovers the true generative
coefficients within 0.03 absolute (`test_irls_recovers_true_coefficients`).
On the pipeline book, all seven WOE coefficients are near the theoretical −1
(WOE is already on the log-odds scale) with |z| between 6.5 and 15.3. These
Wald statistics are post-selection: the same sample chose the features.

```
term                    coef     se        z
intercept            -3.4913 0.0391 -89.37
woe_leverage         -0.8886 0.0582 -15.28
woe_behavioral_score -0.4139 0.0637  -6.49
woe_interest_coverage-0.8997 0.0816 -11.03
woe_current_ratio    -1.0218 0.0805 -12.69
woe_log_assets       -1.0417 0.0977 -10.66
woe_roa              -0.8834 0.0991  -8.91
woe_sector           -1.0095 0.1555  -6.49
```

The two planted noise features have IV 0.0058 / 0.0032 (< 0.02) and are
dropped by the IV screen before estimation; when force-included in simulation
their Wald p-values are correctly insignificant
(`test_insignificant_noise_coefficient`).

## 3. Discrimination, calibration, stability (train vs OOT, synthetic)

| Metric | Train | OOT | Comment |
|---|---|---|---|
| AUC | 0.7850 [95% boot CI 0.770, 0.800] | 0.7699 | true-PD ceiling 0.7813. The generator and the scorecard share a functional form, so this closeness is by construction. |
| Gini | 0.5700 | 0.5398 | |
| KS | 0.4278 | 0.4033 | |
| Hosmer-Lemeshow | χ² = 17.9, df 8, p = 0.022 | χ² = 43.2, df 10, p = 4.7e-06 | OOT calibration rejected |
| Brier | 0.0272 | 0.0431 | |
| Mean PD vs realised | 2.95% vs 2.95% | 3.81% vs 4.78% | PIT under-prediction after the shift |

Hosmer-Lemeshow uses g - 2 degrees of freedom for in-sample PDs and g for
PDs scored on data not used in fitting (`hosmer_lemeshow(...,
out_of_sample=...)`). Size checks by simulation: with known PDs and df = g
the 5% test rejects within [3%, 7%] of 1,000 replications
(`test_hl_out_of_sample_size_is_nominal`); with g - 2 it rejected about
11%. With PDs from a logistic fit on the same rows and df = g - 2 the
rejection rate lies in [1.5%, 9%] of 300 replications
(`test_hl_in_sample_size_with_fitted_model`).

OOT back-test by PD grade (`pd_backtest_by_grade`; one-sided binomial and
Jeffreys p-values for PD underestimation; light from the binomial p-value:
green ≥ 0.05, amber [0.001, 0.05), red < 0.001):

| Grade | N | Defaults | Mean PD | Observed | Binomial p | Jeffreys p | Light |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 (<0.5%) | 939 | 1 | 0.34% | 0.11% | 0.958 | 0.905 | green |
| 2 (0.5-1%) | 1,820 | 17 | 0.75% | 0.93% | 0.211 | 0.176 | green |
| 3 (1-2%) | 2,725 | 47 | 1.46% | 1.72% | 0.144 | 0.128 | green |
| 4 (2-5%) | 3,743 | 175 | 3.22% | 4.68% | 1.2e-06 | 9.6e-07 | red |
| 5 (5-10%) | 1,775 | 161 | 6.95% | 9.07% | 4.2e-04 | 3.6e-04 | red |
| 6 (>10%) | 998 | 173 | 15.75% | 17.33% | 0.093 | 0.087 | green |

The binomial test assumes independent defaults. The synthetic defaults are
independent, so it is exact here. With correlated defaults it rejects too
often.

Rank ordering: decile default rates rise from 0.07% (decile 1) to 11.4%
(decile 10) on the training sample.

Stability (PSI, train to OOT): all features < 0.10 individually
(behavioral_score 0.086 and leverage 0.065 are the largest), score PSI
0.076 (stable band). The PSI machinery is validated on planted shifts: a
0.8σ location shift gives PSI > 0.25, 0.4σ lands in the 0.10-0.25 band,
identical samples give exactly 0.

## 4. Portfolio risk numbers (synthetic demo book, EAD 41.2bn)

- EL = 605.3m (1.47% of EAD); with a 25% downturn-LGD haircut: 754.2m.
- Basel III IRB (actual PDs/LGDs, M = 2.5, 5 bp PD floor): RWA 51.6bn,
  average risk weight 125.2%, 8% capital charge 4.13bn (K = 10.02% of
  EAD, EAD-weighted).
- Vasicek, homogeneous reference (PD 2.96%, LGD 50.1%, ρ = 0.15 assumed):

  | N loans | exact (quadrature) | Vasicek + GA | MC, 30k sims [95% CI] |
  |---|---|---|---|
  | infinite | 11.369% | 11.369% | |
  | 100 | 12.522% | 12.414% | 12.021% [11.520%, 13.023%] |
  | 1,000 | 11.470% | 11.474% | 11.771% [11.019%, 12.522%] |
  | 10,000 | 11.380% | 11.380% | 11.751% [11.275%, 12.557%] |

  The exact column integrates the conditional binomial over the factor
  (`finite_portfolio_quantile`). The granularity adjustment (Gordy 2003;
  Gordy and Lutkebohmert 2013, `granularity_adjustment`) agrees with it to
  within one lattice step 1/N for N from 200 to 20,000
  (`test_ga_matches_exact_quadrature_within_one_lattice_step`). The Monte
  Carlo order-statistic CIs (`mc_quantile_ci`) cover the exact value, and
  their width exceeds the adjustment for N ≥ 1,000.
- MC on the actual heterogeneous book: 9.66% [95% CI 9.17%, 10.13%];
  granularity adjustment 0.007%. It is below the homogeneous reference
  because PD dispersion at a fixed mean thins the tail.
- Economic capital (99.9% MC - EL) = 8.19% of EAD against Basel K 10.02%.
  Basel is higher here because R(PD) in [0.12, 0.24] mostly exceeds the
  flat ρ = 0.15, and the maturity adjustment adds about 10-30%.
- MC convergence: mean simulated loss = PD·LGD within 3 standard errors;
  simulated CDF at the analytic 95% quantile ≈ 0.95
  (`test_mc_converges_to_analytic_within_se`).

## 5. Failure modes (each reproducible)

1. **Data drift / regime change.**  OOT AUC -1.5 points, KS -2.5 points,
   out-of-sample HL p-value 4.7e-06, realised default rate 4.78% vs 3.81%
   predicted, two grades red. The synthetic OOT stress is a pure intercept
   shift, which the HL test is built to catch.  A PIT scorecard *cannot* see a
   calibration-level shift coming from features alone: note PSI stayed
   < 0.10 while calibration failed, which is why monitoring must track
   realised-vs-predicted rates, not PSI alone.
2. **Target leakage.**  Planted demo: `writeoff_flag` (a post-outcome field
   with 3% label noise) has IV ≈ 6.4; the IV screen flags anything > 0.5
   (`SuspiciousIVWarning`, `test_leaky_feature_triggers_suspicious_iv_warning`)
   and the cleaning-layer deny-list refuses it outright
   (`LeakageError`, demonstrated in the pipeline).  Both controls fire
   independently: defence in depth.
3. **Low-default portfolios.**  With zero defaults the WOE, the logit MLE and
   AUC are all undefined; every entry point raises an informative
   `ValueError` (`test_zero_defaults_raises_everywhere`) rather than
   returning garbage.  At ~30 observations / 3 defaults the fit stays finite
   but the standard errors become very large, as the Wald table shows.
4. **Separation.**  Perfectly separable data makes the unpenalised MLE
   diverge; the fitter detects the runaway iterates, warns
   (`SeparationWarning`) and recommends ridge; with ridge > 0 it converges
   (`test_separation_detected_and_warned`, `test_separation_ridge_regularizes`).
5. **Procyclicality of PIT PDs.**  Feeding PIT PDs into the IRB formula makes
   K itself cyclical: the OOT realised default rate is 4.78% against 3.81%
   predicted, so a recalibrated PIT model would raise PDs and capital in the
   downturn.  Basel expects long-run-average PDs per grade; this model
   would need a TTC overlay before IRB use (METHODOLOGY.md §6).
6. **Correlation model risk in the single-factor Vasicek.**  The 99.9%
   quantile is extremely sensitive to ρ: at PD 2%, moving ρ from 0.05 to 0.30
   multiplies the 99.9% default-rate quantile several-fold
   (`test_vasicek_higher_rho_fatter_tail`).  A single flat ρ also ignores
   sector concentration: the model treats a 12%-construction book as fully
   diversified.  Multi-factor or sector-stress overlays are required for
   ICAAP use.
7. **Basel K non-monotonicity at extreme PDs.**  K peaks near PD ≈ 30% and
   *falls* toward PD → 1 (EL absorbs the quantile).  Handled: K is tested
   monotone on the practical range [0.03%, 20%] and finite/non-negative
   beyond (`test_k_monotone_increasing_over_relevant_pd_range`). The Basel III
   corporate PD floor (0.05%), the maturity floor and cap [1, 5], the 1.25
   financial-institution correlation multiplier and K = max(0, LGD - EL_best)
   for defaulted exposures are applied (`test_granularity_and_basel3.py`).

## 6. Edge cases (documented and unit-tested)

Zero defaults (raises), all defaults (raises), single feature (end-to-end
scorecard works), all-missing feature (raises), constant feature (IV = 0),
PD = 0/1 (clamped to finite scores, ordering preserved), empty portfolio
(raises), ρ ∉ (0,1) (raises), LGD = 0 (K = 0), PD beyond the Basel floor
(floored), NaN bins in PSI (own bin), unseen categories at transform time
(raise by default; `unseen="missing"` or `"neutral"` maps them
explicitly), duplicate loan IDs (detected and dropped), train/OOT boundary
(strict temporal integrity checked with a `ValueError`, not `assert`).

Hardening additions (`test_input_validation.py`, `test_edge_cases.py`):

- **NaN/Inf rejection everywhere.**  NaN passes naive range checks
  (`NaN < 0` is `False`), so every entry point now checks `np.isfinite`
  explicitly: `fit_logistic` (features), `expected_loss`/`basel_k`/
  `asset_correlation` (PD/LGD/EAD), `brier_score`/`hosmer_lemeshow`
  (predicted PDs, also gated to [0, 1]), `woe_iv_from_counts` (counts, also
  rejects negatives).
- **Zero total EAD** in the Monte Carlo engine (loss *rate* undefined) raises
  instead of returning NaN.
- **Degenerate scorecard scalings** (`pdo <= 0`, `base_odds <= 0`, non-finite
  `base_score`) are rejected at construction.
- **Property checks**: Vasicek CDF monotone non-decreasing and in [0, 1];
  Vasicek quantile monotone in q; simulated loss rates bounded in [0, 1] with
  mean ≈ PD·LGD within 3 SE; Basel K bounded by LGD at M = 2.5; economic
  capital quantile gated to (0, 1].

## Round 3 oracle tests (`tests/test_round3_credit.py`, 18 tests)

| Model | Oracle |
|---|---|
| Workout LGD | Hand value at a 10% rate; the zero-rate case is undiscounted |
| EAD / CCF | Round trip CCF -> EAD -> CCF; NaN when nothing was undrawn |
| Beta LGD | Recovers (2, 5) from 200k draws within 3% |
| Calibration | Mean PD equals the target to 1e-12; ranking unchanged; a single obligor is exact |
| MoC C | Equals t_{0.9,n-1} s / sqrt(n) to 1e-14 |
| Lifetime PD | Two-state chain: 1 - (1-p)^t to 1e-14 |
| Generator | exp(Q) -> log round trip to 1e-10; fixed generator is valid |
| CDS | Credit triangle s = lambda (1-R) as the premium grid is refined; flat-segment protection leg closed form; bootstrap reprices five tenors to 1e-12 |
| ISDA CDS (`test_isda_cds.py`) | Standard dates (step-in, cash settle, weekend roll, maturity + 1, 21 periods); protection and premium legs equal adaptive quadrature to 1e-10 rel; flat hazard gives lambda (1-R) 360/365 / (1 - r tau/2) to 5e-4 rel; bootstrap reprices five dated tenors to 1e-13; quoted spread = coupon gives zero clean upfront; upfront ~ (S - C) RPV01 to 3%. ISDA's published test grids not checked (licensed, not in the repo) |
