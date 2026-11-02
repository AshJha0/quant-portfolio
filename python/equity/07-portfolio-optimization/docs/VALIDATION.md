# Validation: Equity Portfolio Optimization and Risk Allocation

All numbers below come from `python examples/run_pipeline.py` (seeded,
offline, synthetic data, about 90 s) and the pytest suite (`pytest -q`,
196 tests, about 16 s, offline). Tolerances follow CONVENTIONS.md: analytic identities to
1e-10–1e-8 (most hold to machine precision), solver-vs-closed-form to
documented tolerance.

## 1. Analytic identities

| Identity | Tolerance | Test |
|---|---|---|
| Min-variance weights = `Σ⁻¹1/(1'Σ⁻¹1)` | 1e-12 | `test_min_variance_matches_formula` |
| Tangency weights = `Σ⁻¹μₑ/(1'Σ⁻¹μₑ)` | 1e-12 | `test_tangency_matches_formula` |
| Reverse-optimization round trip: `tangency(δΣw_mkt) = w_mkt` | ≤ 3.5e-16 observed | `test_reverse_optimization_round_trip_identity` |
| Black-Litterman, no views ⇒ posterior = prior | exact (0) | `test_bl_no_views_posterior_equals_prior` |
| Black-Litterman, `Ω = 0` ⇒ `Pμ_BL = Q` | 1e-14 | `test_bl_infinitely_confident_view_holds_exactly` |
| BL posterior between prior and view (finite Ω) | strict inequality | `test_bl_posterior_between_prior_and_view` |
| James-Stein: convex combination, `φ ∈ [0,1]` | exact | `test_js_is_convex_combination` |
| EWMA recursion `S_T = λS_{T−1} + (1−λ)r_T r_T'` | 1e-18 | `test_ewma_recursion_identity` |
| LW shrunk = `δF + (1−δ)S`, `δ ∈ [0,1]`, variances preserved | 1e-18 | `test_lw_*` |
| Euler identity: `Σᵢ RCᵢ = w'Σw` | 1e-18 | `test_rc_euler_identity_exact` |
| ERC: all risk contributions equal | < 1e-8 relative spread (observed ~1e-15) | `test_erc_contributions_all_equal` |
| ERC = inverse-vol under constant correlation | 1e-9 | `test_erc_constant_correlation_equals_inverse_vol` |
| Vol targeting hits ex-ante target | 1e-14 | `test_vol_target_hits_target_exactly` |
| Two-fund theorem: frontier portfolio = affine combo of two others | 1e-10 | `test_two_fund_theorem` |
| SLSQP min-variance vs closed form | 1e-6 (observed ~6e-8) | `test_slsqp_min_variance_matches_closed_form` |
| Target-return portfolio is a variance minimum (feasible perturbations) | perturbation test, 30 directions | `test_target_return_variance_is_minimal_under_perturbation` |
| Effective N of equal weight = N; diversification ratio ≥ 1 | exact / 1e-12 | `test_effective_n_*`, `test_diversification_ratio_*` |
| Backtest cost ledger on hand-computed 2-rebalance scenario | 1e-15 | `test_hand_computed_two_rebalance_cost_ledger` |
| No lookahead: estimation window ends strictly before rebalance; spike on rebalance day uncapturable | exact | `test_no_lookahead_*` |

## 2. Maximum-Sharpe solver checks

| Check | Tolerance | Test |
|---|---|---|
| All means below rf, long-only: returns the min-variance portfolio, `fallback=True` | 1e-9 | `test_all_negative_means_returns_min_variance_not_max_vol` |
| Inactive box: QP equals closed-form tangency | 1e-7 in weights | `test_inactive_bounds_match_closed_form_tangency` |
| Long-only, N = 2..6, 60 random problems: Sharpe equals the exact optimum from support-set enumeration | 1e-7 relative | `test_long_only_matches_support_enumeration` (hypothesis) |
| Box cap 0.3 to 0.6: KKT residual of the ratio problem < 1e-6, and no random feasible point beats it | 1e-9 | `test_box_cap_kkt_and_dominates_random_feasible_points` (hypothesis) |
| Backtest strategies count fallbacks | exact | `test_tangency_strategy_logs_fallbacks` |
| Lo-adjusted Sharpe on iid data: 5-95% range of adjusted/raw within [0.85, 1.15], median within 0.03 of 1 | 200 simulations | `test_lo_sharpe_on_iid_data_is_close_to_iid_scaling` |
| Jobson-Korkie/Memmel test size at 5% under H0 | [0.03, 0.07], about 3 MC standard errors | `test_sharpe_difference_test_size_under_null` |

## 3. The estimation-error study (pipeline §2)

20 disjoint 252-day windows on each of 10 calm synthetic 21-year panels
(seeds 7-16). Each window's estimated portfolio is evaluated under the
true moments of its panel. Intervals are 95% t-intervals across the 10
panel means. Mean achievable (true tangency) Sharpe is 0.427.

| Strategy | mean true Sharpe | 95% CI | paired diff vs MinVar LW | 95% CI |
|---|---|---|---|---|
| Tangency, raw mean, unconstrained | 0.115 | 0.068 to 0.161 | -0.218 | -0.258 to -0.179 |
| Tangency, raw mean, long-only | 0.279 | 0.250 to 0.309 | -0.054 | -0.078 to -0.030 |
| Tangency, James-Stein mean, long-only | 0.328 | 0.286 to 0.370 | -0.005 | -0.012 to 0.001 |
| Min-variance, Ledoit-Wolf, long-only | 0.333 | 0.289 to 0.377 | | |
| ERC, Ledoit-Wolf | 0.322 | 0.304 to 0.340 | -0.011 | -0.038 to 0.016 |
| Equal weight | 0.314 | 0.301 to 0.327 | -0.019 | -0.051 to 0.013 |

- Raw-mean tangency is worse than min-variance, long-only or not.
- James-Stein tangency, ERC and equal weight cannot be told apart from
  min-variance on these synthetic panels.
- The min-variance fallback fired in 16 of 200 windows for raw tangency
  and 73 of 200 for James-Stein tangency. The average James-Stein
  intensity is 0.94, so a negative grand mean often makes every shrunk
  mean negative.
- The unconstrained raw tangency closed form is undefined
  (`1'Σ⁻¹μ ≤ 0`) in 69 of 200 windows. Where it exists its gross
  leverage averages 26.2x.
- The average Ledoit-Wolf intensity is 0.33.

## 4. Ledoit-Wolf conditioning

- One 252x8 synthetic window: cond(sample) 16.2 to cond(shrunk) 14.0.
- 12x10 panel (N close to T): conditioning improvement tested
  (`test_lw_improves_conditioning_when_n_close_to_t`).
- 5x8 panel (T < N, sample singular): LW returns a finite-condition PSD
  matrix and long-only min-variance solves on it
  (`test_short_window_singular_sample_but_lw_invertible`).
- The degenerate-shrinkage guard is relative, so the intensity does not
  depend on the units of the returns
  (`test_ledoit_wolf_degenerate_guard_is_scale_free`).

## 5. Out-of-sample walk-forward race (pipeline §6)

2400-day 8-asset synthetic panels with a 120-day crisis (market vol x3,
-120%/yr market drift during the crisis), 252-day window, monthly
rebalance, 10 bp costs. Main panel (seed 1), net of costs:

| Strategy | AnnRet | AnnVol | Sharpe | MaxDD | AnnTurnover | AvgEffN |
|---|---|---|---|---|---|---|
| EqualWeight | -0.041 | 0.201 | -0.110 | 0.490 | 0.57 | 8.00 |
| MinVar (LW) | -0.031 | 0.181 | -0.083 | 0.384 | 1.47 | 3.68 |
| Tangency raw | -0.101 | 0.241 | -0.320 | 0.682 | 7.28 | 1.92 |
| Tangency JS | -0.031 | 0.182 | -0.081 | 0.387 | 2.57 | 3.60 |
| ERC (LW) | -0.040 | 0.196 | -0.109 | 0.468 | 0.61 | 7.82 |
| Static tilt | -0.046 | 0.204 | -0.131 | 0.508 | 0.57 | 7.69 |

Jobson-Korkie/Memmel p-values against min-variance on this panel are all
above 0.2.

Across 20 independent synthetic panels (seeds 1-20):

| Strategy | mean net Sharpe | 95% CI | diff vs MinVar | 95% CI | panels above MinVar |
|---|---|---|---|---|---|
| MinVar (LW) | -0.083 | -0.252 to 0.087 | | | |
| Tangency JS | -0.105 | -0.260 to 0.051 | -0.022 | -0.069 to 0.025 | 7 of 20 |
| ERC (LW) | -0.107 | -0.260 to 0.046 | -0.024 | -0.093 to 0.045 | 10 of 20 |
| EqualWeight | -0.114 | -0.266 to 0.038 | -0.032 | -0.108 to 0.045 | 9 of 20 |
| Static tilt | -0.120 | -0.267 to 0.027 | -0.037 | -0.114 to 0.040 | 8 of 20 |
| Tangency raw | -0.200 | -0.330 to -0.070 | -0.117 | -0.239 to 0.004 | 6 of 20 |

No difference from min-variance is significant at 5%. Raw tangency has
the lowest mean, and its interval only just reaches zero. Mean annual
turnover is 6.76 for raw tangency, 2.68 for James-Stein tangency, 1.32
for min-variance and about 0.6 for ERC, equal weight and the static
tilt. James-Stein tangency sat in the min-variance fallback at 45.1% of
rebalances. Absolute Sharpes are negative because each synthetic panel
contains a crisis with a large negative market drift. "Static tilt" is a
fixed 15%/10% split over eight synthetic equity assets and contains no
bonds.

## 6. Failure modes

1. **Correlation breakdown in crises** (pipeline §7). Realised average
   pairwise correlation is 0.470 calm and 0.867 in the crisis (seed-1
   synthetic panel):

   | Strategy | Crisis AnnVol | Crisis MaxDD | Crisis TotRet |
   |---|---|---|---|
   | EqualWeight | 0.467 | 0.399 | -0.311 |
   | MinVar (LW) | 0.380 | 0.308 | -0.202 |
   | Tangency raw | 0.542 | 0.448 | -0.347 |
   | Tangency JS | 0.380 | 0.308 | -0.202 |
   | ERC (LW) | 0.452 | 0.386 | -0.298 |
   | Static tilt | 0.472 | 0.407 | -0.319 |

   This is one panel. James-Stein tangency equals min-variance here
   because it was in the fallback for the whole crisis.
2. **Mean estimation error.** §3 quantifies it. It shrinks only with
   calendar time (σ/√Y), not with sampling frequency.
3. **Turnover and cost sensitivity.** Cost drag is linear in turnover
   (tested exactly). Raw tangency trades about five times as much as
   min-variance.
4. **Concentration under constraints.** The long-only frontier ends at
   max(μ). Above it the long-only column of the pipeline §3 table is
   NaN, because no long-only portfolio reaches that return.
5. **Numerical limits.** Singular sample matrices make the closed forms
   raise a `ValueError` that points to `psd_repair` or `ledoit_wolf_cc`.
   Zero-vol assets make ERC and inverse-vol raise.
6. **Smoothed returns overstate Sharpe.** `sharpe_lo` reports the
   Bartlett-truncated Lo adjustment next to the iid Sharpe.

## 7. Test suite

`pytest -q`: 196 tests, all passing, about 16 s, offline and seeded
(measured 2026-11-01). Coverage: the identities in §1, the solver checks
in §2, property tests (frontier monotonicity, DR ≥ 1, RC Euler),
no-lookahead and exact cost accounting, and the edge cases: single
asset, two-asset closed forms, singular covariances, zero-vol assets,
all-negative means under long-only, T < N windows, NaN inputs,
dimension mismatches, and relative symmetry checks.

`test_infeasible_and_properties.py` checks that infeasible constraint
sets raise `ValueError` (target return outside the box-achievable range,
caps that cannot sum to the budget, a vol cap below the constrained
minimum-variance vol) and runs portfolio invariants over random SPD
inputs.

## November 2026 oracle tests (`tests/test_round3_portfolio.py`, 19 tests)

- HRP gives exactly the inverse-variance weights for a diagonal Sigma and matches the two-asset closed form. It stays long-only on a near-singular Sigma.
- The closed-form MV satisfies its first-order condition. Resampled weights converge to the classic weights as T grows. Robust MVO with kappa = 0 equals the classic solution, and with kappa > 0 it equals the classic solution at the effective risk aversion.
- The cost QP matches the closed-form no-trade band with and without impact. With zero cost it equals the classic MV. The turnover cap binds and the box bounds hold.
- The factor model recovers the true loadings and reproduces lstsq exactly. Its Euler sums match the totals to 1e-13.
- TE: the active weights are budget-neutral, hit the target TE and attain the IR bound. The SLSQP solution equals the closed form when the bounds are slack.
- Sharpe: with no lags, the HAC standard error equals Memmel's to within 2% on 200k normal draws. The gradient matches finite differences. HAC widens under AR(1) returns. The QS kernel matches its closed form and its k_2 = 18 pi^2 / 125 curvature at 0; the Andrews bandwidth matches the AR(1) plug-in; the prewhitened QS estimator recovers the AR(1) long-run variance 1 / (1 - rho)^2 and a VAR(1)'s (I - A)^-1 (I - A)^-T within 3-4% on 200k draws, and the sample covariance on iid data; with `kernel="qs_pw"` the standard error equals Memmel's within 3% on iid normals. The studentised bootstrap rejects at most 9 of 60 seeded null samples at 5% (AR(1) returns, T = 300) and rejects a true difference; `studentize=False` reproduces the old centred statistic exactly. A defect found while testing: the QS kernel's closed form loses about 5e-6 to cancellation at x = 1e-6; it now uses its Taylor series below |6 pi x / 5| = 0.05.
