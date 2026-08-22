# Validation — Equity VaR & Expected Shortfall Engine

Contract items 3 and 4: **how the engine was validated** (analytic
benchmarks, convergence, statistical backtests, cross-model consistency)
and **where it fails** (reproducible failure modes). All numbers below are
reproduced by `python -m pytest tests -q` (266 tests, ~16 s, offline,
seeded) and `python examples/run_pipeline.py` (~95 s).

## 1. Analytic benchmarks (exact identities)

| Check | Tolerance | Test |
|---|---|---|
| Historical VaR on known 100-point grid = hand-computed type-7 quantile (95.05 at α=0.05, 99.01 at α=0.01) | 1e-12 | `test_historical_var.py::test_exact_quantile_*` |
| Empirical ES on known grid exact incl. fractional tail: ES₀.₀₂₅ of −100..−1 = (100+99+0.5·98)/2.5 = **99.2** | 1e-12 | `test_expected_shortfall.py` |
| Normal ES identity `σ·φ(z_α)/α` vs numerical tail integration | **1e-10** | `test_normal_es_identity_to_1e10` |
| Student-t ES closed form vs numerical integration | 1e-9 | `test_t_es_identity_vs_numerical_integration` |
| Parametric VaR = `−z_α·√(wᵀΣw)` for known Σ | 1e-12 (rel) | `test_var_matches_closed_form_known_covariance` |
| Kupiec LR vs independent hand computation, 5 (T,x) pairs; LR(250, 0)=−2·250·ln 0.99 | 1e-10 | `test_backtesting.py::TestKupiec` |
| Reverse-stress closed form: loss = r·σ_p, shock satisfies Mahalanobis constraint xᵀΣ⁻¹x = r² | 1e-10 – 1e-12 | `test_stress_testing.py::TestReverseStress` |
| BS put-call parity; Greeks vs finite differences | 1e-10 / 1e-5 | `test_portfolio.py::TestBlackScholes` |
| BRW weights sum to 1 for n ∈ {10, 250, 1000} | 1e-12 | `test_weights_sum_to_one` |
| CF expansion reduces to normal quantile at S=K=0 | 1e-12 | `test_reduces_to_normal_when_moments_are_gaussian` |
| ES₉₇.₅ = 2.3378σ vs VaR₉₉ = 2.3263σ for normal P&L (FRTB continuity) | 1 % | `test_es975_close_to_var99_for_normal` |

## 2. Convergence and statistical validation

- **MC → closed form**: linear 2-asset portfolio, normal factors, 400k
  paths: MC 99 % and 95 % VaR within **3 bootstrap SEs** of the parametric
  closed form (`test_mc_converges_to_parametric_closed_form`). Simulated
  covariance matches target within 5 % at 300k paths; t-copula simulation
  matches its target covariance despite fatter tails.
- **Order-statistics CI**: on 100k N(0,1) draws, the distribution-free 95 %
  CI for 99 % VaR brackets the true value 2.3263
  (`test_order_statistic_ci_brackets_true_var`). Pipeline: MC-t 99 % VaR
  $51,025 with bootstrap SE $397 and CI [$50,241, $51,677].
- **Bootstrap SE scaling**: SE shrinks by > 2× from 2k to 50k scenarios
  (theory: 5×).
- **Basel calibration study**: 300 replications of 250 i.i.d. days with the
  *true* 99 % VaR: mean exception rate 0.7–1.3 % and **green zone in > 80 %
  of replications** (theoretical 89.2 % = Binom(250,0.01) CDF at 4;
  `test_calibration_true_var_mostly_green`).
- **Acerbi-Székely Z₂**: ≈ 0 (|Z₂| < 0.05 on 200k obs) when the model is
  correct; ≈ −3.4 (analytic) when true σ is 1.5× the model's — the test
  rejects (`TestAcerbiSzekely`).
- **Christoffersen power**: planted clustered exceptions (two 5-day runs,
  n₁₁=8) rejected at p < 1e-6; the same count evenly spread passes.

## 3. Cross-model consistency (pipeline, demo book, real numbers)

Demo book: $1.76m long equities, short 1 SPX future, long 2 SPX 4750 puts.
1000 days of t(6) factor history, α = 1 %:

| method | VaR 99 % 1d | ES 99 % 1d |
|---|---|---|
| historical | $54,256 | $67,530 |
| age-weighted (λ=0.98) | $49,402 | — |
| FHS (λ=0.94) | $38,968 | — |
| parametric normal | $46,864 | $53,690 |
| parametric t(6) | $51,691 | $66,327 |
| MC normal (100k) | $46,190 | $52,722 |
| MC t(6) (100k) | $51,025 | $65,060 |

Reading the disagreement: MC-normal ≈ parametric-normal (same model, +MC
noise: cross-validates both implementations); parametric-t ≈ MC-t ≈
historical (the history *is* t(6): fat-tail-aware methods agree); the
normal family sits ~10 % lower at 99 % — the missing kurtosis. FHS is
lowest because the simulated history ends in a below-average-vol spell:
FHS is a *conditional* measure and is supposed to disagree with the
unconditional ones whenever current vol ≠ average vol. At 95 % the spread
is only ~4 % (tail-shape differences grow with confidence level);
ES ≥ VaR holds everywhere (property-tested across 40 random books).

## 4. The headline backtest (failure mode as a feature)

500-day walk-forward, 99 % VaR, GARCH(1,1)-t(5) factor data
(α_g=0.13, β_g=0.85, seed 5), 250-day windows:

| method | exceptions (exp. 5) | Kupiec p | indep p | CC p | zone |
|---|---|---|---|---|---|
| parametric normal (unconditional) | **14** | **0.0009** | 0.40 | 0.003 | yellow |
| plain historical | **11** | **0.02** | 0.23 | 0.03 | yellow |
| **FHS** | 6 | 0.66 | 0.70 | 0.85 | **green** |

On the 250-day high-vol subwindow: parametric-normal 13/250 (**red**,
k=4.0), plain HS 11/250 (**red**), FHS 3/250 (**green**, k=3.0). On a
separate seed (10), plain HS *fails the Christoffersen independence test*
(p = 0.02 < 0.05) while FHS passes (p = 0.48) — clustered exceptions are
the fingerprint of an unconditional model in a clustered-vol world
(`test_hs_fails_clustering_test_while_fhs_passes_on_garch`).

## 5. Known failure modes (each reproducible)

1. **VaR is not a worst case.** It is a quantile: on the demo book the
   99 % VaR is $54k but the 2020-replay stress loss is $254k and the ES
   beyond VaR is $68k. Anyone reading VaR as "maximum loss" is off by 5×.
2. **VaR non-subadditivity.** Two independent default-style positions,
   each VaR₉₅ = −5 (gain), portfolio VaR₉₅ = 95: splitting the book
   "removes" the risk. ES stays subadditive on the same scenarios
   (`TestCoherence` — the classic counterexample, constructed exactly).
3. **√t scaling breaks under vol clustering — and the error has no fixed
   sign.** Pipeline: 10d 99 % VaR is $171.6k by √t vs $140.3k by
   overlapping windows on the same history. The direction depends on which
   VaR is being scaled:
   - *unconditional* (the pipeline case): √t **overstates**, because
     temporal aggregation thins the tail — measured excess kurtosis on the
     20k-day GARCH sample falls from **4.76 (1-day) to 1.94 (10-day)**, and
     √t lands ~8 % above the direct overlapping estimate
     (`test_sqrt_time_overstates_the_unconditional_10day_var_under_garch`,
     `test_temporal_aggregation_thins_the_tail_of_garch_returns`);
   - *conditional, from a calm state*: √t **understates**, because variance
     mean-reverts upward over the horizon
     (`test_sqrt_time_understates_from_a_calm_conditional_state`).

   √t is exact only for i.i.d. zero-drift returns (verified: on i.i.d.
   normal data the two agree within 15 %,
   `test_sqrt_time_matches_iid_normal_overlapping_roughly`, and within 5 %
   at 95 % on 60k draws, `test_sqrt_time_is_accurate_for_iid_normal_returns`).
   With a non-zero drift the mean scales like `h` and sigma like `√h`, so
   they must not be scaled together
   (`test_nonzero_mean_breaks_pure_sqrt_scaling`).
4. **Historical window myopia ("great moderation" problem).** Plain HS
   after 450 calm days + 50 wild days reports a VaR ~40 % below FHS
   (`test_fhs_scales_up_after_vol_regime_switch`); symmetrically it
   *overstates* risk after a crisis leaves the window. The 500-day GARCH
   backtest in §4 is this failure measured with p-values.
5. **MC model risk.** MC-normal converges beautifully — to the wrong
   number if the world is t: $46.2k vs $51.0k at 99 % (10 % understatement)
   with *zero* sampling-error warning. Convergence ≠ correctness; the SE
   quantifies only the sampling half of the error.
6. **Delta-gamma breakdown for large moves.** Approximation error on the
   demo book: −$1.2k (0.9 %) in a −15 % scenario but **+$93k (37 %)** in
   the −34 % COVID replay — the quadratic overstates long-gamma gains once
   the put goes deep ITM (payoff is bounded; z² is not). Error is
   monotone in shock size (`test_approximation_error_grows_with_shock_size`)
   and sign-correct: long gamma cushions, short gamma amplifies
   (`test_long_gamma_reduces_loss_vs_pure_delta`).
7. **Cornish-Fisher non-monotonicity.** S=3 or K=10 → the CF "99 %
   quantile" is not a quantile; the engine raises `ValueError` instead of
   returning it (`test_var_raises_outside_domain`). The monotonicity check
   is exact, not grid-sampled: `dz_cf/dz` is a quadratic in `z`, so its
   minimum on `[-3.5, 3.5]` has a closed form. A fixed-resolution grid (the
   previous implementation, 2001 points) can miss a thin non-monotone dip
   between two grid nodes — confirmed with a constructed counterexample,
   `(skew, excess_kurt) = (-0.0105, 8.0001)`: the grid check reported it as
   monotone while the true minimum derivative between two adjacent nodes is
   ≈ -1.0e-6 (non-monotone)
   (`test_domain_check_is_exact_not_grid_resolution_dependent`). The `fx_var`
   engine had the same class of gap (a coarser, value-diff grid rather than
   a derivative grid, with an even easier-to-hit counterexample) and
   received the same fix; all six language/asset-class implementations
   (Python, C++, Rust × equity, FX) now use the closed-form check.
8. **ES estimation uncertainty.** Bootstrap SE of ES₀.₀₁ > SE of ES₀.₁ on
   the same sample (`test_es_se_larger_for_smaller_alpha`): a 250-day
   ES₉₇.₅ averages ~6 points — quote it with error bars.
9. **Kupiec's chi2(1) reference is oversized exactly at the regulatory
   window.** `kupiec_pof`'s LR statistic is asymptotically chi2(1), but at
   `n_obs=250, alpha=0.01` (the Basel backtesting window: expected exception
   count 2.5) that asymptotic is a poor approximation for a rare binomial:
   the *exact* rejection probability of a nominally-5%-size test, computed
   by summing the exact `Binomial(250, 0.01)` mass over `{x : LR(x) >
   chi2.ppf(0.95, 1)}`, is **≈9.5%** — a correctly calibrated model is
   flagged "reject at 5%" roughly twice as often as the p-value implies.
   The effect is driven by the expected count, not `n_obs` alone (verified
   at several `(n_obs, alpha)` pairs): it falls to ≈7.1% at `n_obs=500`
   (expected count 5) and ≈5.5% at `n_obs=1000` (expected count 10), close
   to nominal. This is not a bug in the LR formula — it is the textbook
   asymptotic reference, unchanged — but it is an unstated caveat that
   changes how a p-value near 0.05 should be read at the regulatory window;
   documented and made exact (no simulation, no RNG) in
   `test_backtesting.py::TestKupiec::
   test_asymptotic_chi2_reference_is_oversized_at_the_regulatory_window`.
   The Christoffersen independence and CC tests share the same asymptotic
   chi2 machinery and are subject to the same caveat at low exception
   counts; only Kupiec's exact size was computed here because its
   rejection region depends on a single sufficient statistic (the
   exception count), making an exact calculation tractable — Christoffersen's
   depends on the full transition-count triple and would need Monte Carlo.

## 6. Edge cases (contract item 6 — documented *and* tested)

| Edge case | Behaviour | Test |
|---|---|---|
| Empty portfolio | P&L ≡ 0, exposures empty, value 0 | `TestEmptyPortfolio` |
| Single asset | parametric = MC within 2 % | `TestSingleAsset` |
| Zero-vol asset | contributes exactly 0 to σ_p; zero-P&L series → VaR = ES = 0 (incl. FHS's 0/0 guard) | `TestZeroVolAsset` |
| Perfectly correlated assets (singular Σ) | parametric: σ_p = |w₁σ₁+w₂σ₂| exact; MC: Cholesky jitter path, corr(sim) = 1.000 ± 0.001 | `TestSingularCovariance`, `TestSafeCholesky` |
| Options-only portfolio | VaR computes; short-option VaR > 0; vol floored at 0 in full reval | `TestOptionsOnlyPortfolio` |
| α edge values (0, 0.5, negative) | informative `ValueError` in every module | `TestAlphaEdges` |
| Insufficient history | `ValueError` naming the minimum (50 obs) and why | `TestInsufficientHistory` |
| T→0 / vol→0 options | intrinsic / forward-intrinsic limits, step-function delta | `test_zero_tau_is_intrinsic`, `bs_greeks` degenerate branch |
| x = 0 or x = T exceptions in Kupiec | 0·ln0 convention, LR finite | `test_zero_exceptions_known_value` |
| No exceptions in Christoffersen / AS Z₂ | LR = 0 / Z₂ = +1 (defined, not NaN) | `test_no_exceptions_*` |
| **NaN / ±Inf P&L** | `ValueError` ("NaN or infinite") from `historical_var`, `expected_shortfall`, `age_weighted_var`, `filtered_historical_var`, and now also `ewma_volatility` and `overlapping_horizon_pnl` — the two recursive/aggregating helpers that previously propagated the poison silently | `TestNonFiniteRejection` |
| Unknown `init` for `ewma_volatility` | `ValueError` instead of silently falling back to `x[0]²` | `test_ewma_volatility_rejects_unknown_init` |
| Kupiec with x > T or x < 0 | `ValueError` naming `n_exceptions` | `test_kupiec_rejects_invalid_counts` |
| ES quoted below VaR in the AS Z₂ backtest | `ValueError` ("ES must be >= VaR") — an inconsistent risk feed is rejected, not scored | `test_acerbi_szekely_rejects_es_below_var` |
| Exception on exactly `pnl = −VaR` | **not** an exception (strict `<`), pinned so the convention cannot drift | `test_exceptions_definition_is_strict_inequality` |
| Negative VaR passed to the exception counter | `ValueError` ("positive loss") | `test_exceptions_reject_negative_var` |
| Basel zone boundaries | exactly 4→green, 5→yellow, 9→yellow, 10→red; multiplier non-decreasing in exception count, 3.0 → 4.0 | `test_basel_zone_boundaries_are_exactly_at_5_and_10` |

### 6.1 Structural (property-based) invariants

Checked as inequalities/invariants rather than hard-coded values, so they
constrain the engine without pinning numbers the C++/Rust engines
cross-validate against:

| Invariant | Detail | Test |
|---|---|---|
| **ES ≥ VaR** for every family and every α ∈ {0.1 %, 1 %, 2.5 %, 5 %, 10 %} | strict for continuous P&L | `TestOrdering` |
| Risk measures **non-increasing in α** | historical, ES, age-weighted, FHS | `test_risk_measure_decreasing_in_alpha` |
| **ES subadditivity** ES(X+Y) ≤ ES(X)+ES(Y) | 25 random fat-tailed pairs | `test_es_subadditive_on_random_splits` |
| **VaR non-subadditivity** reproduced | constructed default-style counterexample where VaR is superadditive while ES is not | `test_var_can_violate_subadditivity` |
| **Positive homogeneity** VaR(cX) = c·VaR(X) | 1e-10 relative, c ∈ {0.5, 1, 3, 100} | `test_positive_homogeneity` |
| **Translation invariance** ρ(X+m) = ρ(X) − m | 1e-9 relative | `test_translation_invariance` |
| Parametric VaR **linear in exposures**; σ_p scales as √c in Σ; sign-flip invariant | 1e-12 relative | `TestParametricEquivariance` |
| Horizon scaling is exactly √h at zero mean, and **breaks** with drift | 1e-12 / directional | `test_horizon_scaling_is_sqrt_time_at_zero_mean`, `test_nonzero_mean_breaks_pure_sqrt_scaling` |
| **Variance-matched t is fatter than normal only in the deep tail** — t(5) charges *less* than the normal at 95 %, crossover near α ≈ 2.8 %. ES, which averages the whole tail, stays higher at every α. A real trap for anyone switching to t "to be conservative" | directional at α ∈ {0.1 %, 1 %, 2.5 %} vs {5 %, 10 %} | `test_variance_matched_t_is_thinner_than_normal_in_the_shoulder`, `test_t_vs_normal_var_crossover_is_between_2_5_and_5_percent` |
| Student-t → normal as df → ∞ | gap monotone, < 1 % of σ at df = 500 | `test_student_t_converges_to_normal_as_df_grows` |
| BRW weights sum to 1 and increase monotonically toward the present | 1e-12 | `TestWeightingSchemes` |
| Age-weighting → plain HS as λ → 1 | gap shrinks | `test_age_weighting_converges_to_plain_hs_as_lambda_goes_to_one` |
| FHS is **conditional**: same unconditional sample, turmoil at the end vs the start gives a higher VaR | directional | `test_fhs_tracks_the_current_vol_regime` |
| `ewma_volatility` is **causal**: σ[t] does not depend on x[t] | 1e-12 | `test_ewma_volatility_is_causal_and_positive` |
| Kupiec LR = 0 exactly at the expected rate, increasing away from it; p-value decreasing in exception count | 1e-9 | `TestBacktestProperties` |
| Christoffersen rejects planted 5-day exception runs (p < 0.01) and passes i.i.d. exceptions (p > 0.01) | — | `test_christoffersen_detects_clustering_and_passes_iid` |
| Bootstrap VaR SE **shrinks with sample size**; order-statistics CI brackets the point estimate and widens for deeper tails | — | `TestEstimationError` |
| MC VaR converges to the closed form as paths grow | within 2 % at 800k paths | `test_mc_var_converges_to_the_parametric_answer` |
| **Crisis regime**: a 5× vol burst lifts every family, and the conditional ones (FHS, BRW) react *more* than plain HS | directional | `test_crisis_regime_all_families_reprice_upward` |

## 7. What was *not* validated (honest scope)

- No comparison against live market data (the `data/live.py` loader exists
  but is out of the offline test scope by design).
- The GARCH generator is a data-generating device for backtests, not a
  calibrated model of any particular index.
- Stress shocks are approximations of published episode moves, not
  tick-level replays (METHODOLOGY.md A11).
