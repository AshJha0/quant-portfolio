# Validation — Evidence, Numbers, Failure Modes

All numbers below are produced by `python examples/run_pipeline.py`
(train: 30,000 loans, seed 42, 2.95% default rate; OOT: 12,000 loans,
seed 123, drift 0.5 + calibration shift 0.5 → 4.78% default rate) and by the
test suite (`pytest -q`, 123 tests, offline, seeded).

## 1. Analytic and cross-model benchmarks

| Check | Result | Test |
|---|---|---|
| WOE of a bin = ln(dist_good/dist_bad), hand-computed 3-bin table | exact to 1e-14 | `test_woe_hand_computed_exact` |
| IV hand-computed on the same table | exact to 1e-14 | `test_iv_hand_computed_exact` |
| IRLS coefficients vs sklearn `LogisticRegression(penalty=None)` | max diff 9.9e-14 (pipeline), < 1e-6 (CI gate) | `test_matches_sklearn_to_1e6` |
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
(WOE is already on the log-odds scale) with |z| between 6.5 and 15.2:

```
term                    coef     se        z
intercept            -3.4912 0.0390 -89.45
woe_leverage         -0.8881 0.0583 -15.24
woe_behavioral_score -0.4216 0.0636  -6.63
woe_interest_coverage-0.8955 0.0816 -10.98
woe_current_ratio    -1.0234 0.0805 -12.71
woe_log_assets       -1.0441 0.0986 -10.59
woe_roa              -0.8826 0.0994  -8.88
woe_sector           -1.0125 0.1555  -6.51
```

The two planted noise features have IV 0.0058 / 0.0032 (< 0.02) and are
dropped by the IV screen before estimation; when force-included in simulation
their Wald p-values are correctly insignificant
(`test_insignificant_noise_coefficient`).

## 3. Discrimination, calibration, stability (train vs OOT)

| Metric | Train | OOT | Comment |
|---|---|---|---|
| AUC | 0.7843 [95% boot CI 0.769, 0.800] | 0.7671 | true-PD ceiling 0.7813 — the scorecard is at the information frontier (train AUC exceeds the ceiling by in-sample noise) |
| Gini | 0.5685 | 0.5342 | |
| KS | 0.4258 | 0.4006 | |
| Hosmer-Lemeshow | χ² = 17.9 (p = 0.022) | χ² = 44.9 (p = 3.8e-07) | OOT calibration correctly **rejected** |
| Brier | 0.0272 | 0.0431 | |
| Mean PD vs realised | 2.95% vs 2.95% | 3.80% vs **4.78%** | PIT under-prediction after regime shift |

Rank ordering: decile default rates rise monotonically from 0.10% (decile 1)
to 11.2% (decile 10), a 112× spread (`test_decile_default_rates_monotone_for_good_model`
allows 20bp noise per step).

Hosmer-Lemeshow behaviour under the null was verified by replication: with
known true probabilities the statistic averages ≈ 10 over 60 replications,
consistent with χ²(10) (`test_hosmer_lemeshow_null_distribution`).

Stability (PSI, train → OOT): all features < 0.10 individually
(behavioral_score 0.086, leverage 0.065 are the largest — direction
consistent with the planted drift), score PSI 0.074 (stable band).  The PSI
machinery itself is validated on planted shifts: a 0.8σ location shift gives
PSI > 0.25 ("shifted"), 0.4σ lands in the 0.10–0.25 "monitor" band, identical
samples give exactly 0.

## 4. Portfolio risk numbers (demo book, EAD 41.2bn)

- EL = 605.5m (1.47% of EAD); with 25% downturn-LGD haircut: 754.3m.
- Basel IRB (actual PDs/LGDs, M = 2.5): RWA 51.7bn, average risk weight
  125.4%, 8% capital charge 4.13bn (K = 10.03% of EAD, EAD-weighted).
- Vasicek, homogeneous reference (PD 2.96%, LGD 50.1%, ρ = 0.15):
  - analytic infinitely-granular 99.9% loss rate: **11.37%**
  - MC 99.9% loss rate: 100 loans **12.02%**, 1,000 loans 11.77%, 10,000
    loans 11.75% — finite ≥ analytic, converging from above (granularity
    adjustment), also gated in `test_finite_portfolio_tail_at_least_infinitely_granular`
    and `test_mc_quantile_approaches_analytic_as_n_grows`.
  - MC on the actual heterogeneous book: 9.67% — *below* the homogeneous
    reference because PD dispersion at fixed mean thins the tail.
- Economic capital (99.9% MC − EL) = **8.20%** of EAD vs Basel K 10.03%:
  Basel is more conservative here because R(PD) ∈ [0.12, 0.24] mostly exceeds
  the flat ρ = 0.15, and the maturity adjustment adds ~10–30%.
- MC convergence: mean simulated loss = PD·LGD within 3 standard errors;
  simulated CDF at the analytic 95% quantile ≈ 0.95
  (`test_mc_converges_to_analytic_within_se`).

## 5. Failure modes (each reproducible)

1. **Data drift / regime change.**  Shown directly: OOT AUC −1.7 points, KS
   −2.5 points, HL p-value collapses from 0.022 to 3.8e-07, realised default
   rate 4.78% vs 3.80% predicted.  A PIT scorecard *cannot* see a
   calibration-level shift coming from features alone — note PSI stayed
   < 0.10 while calibration failed, which is why monitoring must track
   realised-vs-predicted rates, not PSI alone.
2. **Target leakage.**  Planted demo: `writeoff_flag` (a post-outcome field
   with 3% label noise) has IV ≈ 6.4; the IV screen flags anything > 0.5
   (`SuspiciousIVWarning`, `test_leaky_feature_triggers_suspicious_iv_warning`)
   and the cleaning-layer deny-list refuses it outright
   (`LeakageError`, demonstrated in the pipeline).  Both controls fire
   independently — defence in depth.
3. **Low-default portfolios.**  With zero defaults the WOE, the logit MLE and
   AUC are all undefined; every entry point raises an informative
   `ValueError` (`test_zero_defaults_raises_everywhere`) rather than
   returning garbage.  At ~30 observations / 3 defaults the fit stays finite
   but SEs balloon — reported honestly via the Wald table.
4. **Separation.**  Perfectly separable data makes the unpenalised MLE
   diverge; the fitter detects the runaway iterates, warns
   (`SeparationWarning`) and recommends ridge; with ridge > 0 it converges
   (`test_separation_detected_and_warned`, `test_separation_ridge_regularizes`).
5. **Procyclicality of PIT PDs.**  Feeding PIT PDs into the IRB formula makes
   K itself cyclical: repricing the demo book at the OOT-implied calibration
   raises every PD by ×1.26 on average — capital demand rises precisely in
   the downturn.  Basel expects long-run-average PDs per grade; this model
   would need a TTC overlay before IRB use (METHODOLOGY.md §6).
6. **Correlation model risk in the single-factor Vasicek.**  The 99.9%
   quantile is extremely sensitive to ρ: at PD 2%, moving ρ from 0.05 to 0.30
   multiplies the 99.9% default-rate quantile several-fold
   (`test_vasicek_higher_rho_fatter_tail`).  A single flat ρ also ignores
   sector concentration — the model treats a 12%-construction book as fully
   diversified.  Multi-factor or sector-stress overlays are required for
   ICAAP use.
7. **Basel K non-monotonicity at extreme PDs.**  K peaks near PD ≈ 30% and
   *falls* toward PD → 1 (EL absorbs the quantile).  Handled: K is tested
   monotone on the practical range [0.03%, 20%] and finite/non-negative
   beyond (`test_k_monotone_increasing_over_relevant_pd_range`); the PD floor
   (0.03%) is applied automatically.

## 6. Edge cases (documented and unit-tested)

Zero defaults (raises), all defaults (raises), single feature (end-to-end
scorecard works), all-missing feature (raises), constant feature (IV = 0),
PD = 0/1 (clamped to finite scores, ordering preserved), empty portfolio
(raises), ρ ∉ (0,1) (raises), LGD = 0 (K = 0), PD beyond the Basel floor
(floored), NaN bins in PSI (own bin), unseen categories at transform time
(mapped to missing-bin WOE), duplicate loan IDs (detected and dropped),
train/OOT boundary (strict temporal integrity asserted).
