# Validation: Equity VaR and Expected Shortfall Engine

How the engine is validated and where it fails. The numbers come from
`python -m pytest -p no:cacheprovider -q` (offline, seeded; counts and
runtime in the README) and `python examples/run_pipeline.py`. Every pipeline
number below is on synthetic data.

## 1. Analytic benchmarks

| Check | Tolerance | Test |
|---|---|---|
| Historical VaR on -100..-1 is the S2 order statistic: 96 at alpha 0.05, 100 at 0.01, 248 on -1..-250 at 0.01 (k = 2.5, m = 3) | exact | `test_historical_var.py::TestPlainHistorical` |
| Empirical ES with the fractional atom: ES at 0.025 of -100..-1 = (100 + 99 + 0.5 x 98) / 2.5 = 99.2 | 1e-12 | `test_expected_shortfall.py` |
| Normal ES `sigma phi(z) / alpha` against numerical integration | 1e-10 | `test_normal_es_identity_to_1e10` |
| Student-t ES closed form against numerical integration | 1e-9 | `test_t_es_identity_vs_numerical_integration` |
| Parametric VaR `-z sqrt(w' Sigma w)` for a known Sigma | 1e-12 rel | `test_var_matches_closed_form_known_covariance` |
| Kupiec LR against a hand computation | 1e-10 | `test_backtesting.py::TestKupiec` |
| Basel zone, multiplier and binomial CDF at 9 (n, x, alpha) cases incl. 500 and 1000 days and alpha 0.025 | exact zone and k, CDF 1e-13 rel | `test_basel_spec_s4_any_window_any_alpha` |
| Reverse stress: loss = r sigma_p, `x' Sigma^{-1} x = r^2` | 1e-10 | `test_stress_testing.py::TestReverseStress` |
| BS put-call parity, Greeks against finite differences | 1e-10 / 1e-5 | `test_portfolio.py::TestBlackScholes` |
| Monte Carlo VaR of a single long stock against the exact lognormal quantile `-V expm1(sigma z)` | 3 bootstrap SE | `test_mc_converges_to_the_exact_lognormal_quantile` |
| Higham repair of `[[1,.9,.9],[.9,1,-.9],[.9,-.9,1]]` gives 0.5 off the diagonal | 1e-6 | `test_higham_repair` |

## 2. Statistical validation

- **Acerbi-Szekely p-value calibration.** 200 samples of 250 days drawn
  from the null: the 5 % rejection rate lies within 3 binomial SE of 5 %
  (`test_p_value_is_calibrated_under_the_null`). Power: realised P&L 1.5x
  the model sigma is rejected on 250 days; a Student-t(3) P&L with the same
  2.5 % quantile as the forecast (right VaR, understated ES) is rejected on
  10,000 days.
- **Order-statistic CI.** On 100,000 N(0, 1) draws the distribution-free
  95 % CI for the 99 % VaR brackets 2.3263
  (`test_order_statistic_ci_brackets_true_var`). The CI raises when the
  sample is too short for the requested coverage.
- **Bootstrap SE.** Chunked resampling gives exactly the same result as an
  unchunked reference with the same seed (`test_bootstrap_se_is_chunk_invariant`).
- **Basel calibration study.** 300 replications of 250 i.i.d. days with the
  true 99 % VaR: mean exception rate 0.7-1.3 % and green in more than 80 %
  of replications (theory 89.2 %).
- **Christoffersen power.** Planted clustered exceptions are rejected at
  p < 1e-6; the same count spread evenly passes.

## 3. Cross-model consistency

Demo book: $1.76m long equities, short 1 SPX future, long 2 SPX 4750 puts.
1000 days of synthetic t(6) factor history. Command:
`python examples/run_pipeline.py`.

| method | VaR 99 % 1d | ES 99 % 1d |
|---|---|---|
| historical | $54,681 | $67,530 |
| age-weighted (lam 0.98) | $49,402 | |
| FHS per factor (lam 0.94) | $41,337 | |
| parametric normal | $46,864 | $53,690 |
| parametric t(6) | $51,691 | $66,327 |
| MC normal (100k) | $45,238 | $51,535 |
| MC t(6) (100k) | $49,853 | $63,097 |

MC t(6) 99 % VaR has a bootstrap SE of $381 and a 95 % order-statistic CI of
[$49,142, $50,506]. MC normal sits about 3.5 % below parametric normal
because Monte Carlo moves are log returns and the book is long. The normal
family is about 10 % below the t family at 99 %. FHS is lowest because the
synthetic history ends in a below-average vol spell.

## 4. Walk-forward backtest

500 days, 99 % VaR, synthetic GARCH(1,1)-t(5) factor data
(alpha_g 0.13, beta_g 0.85, seed 5), 250-day windows:

| method | exceptions (expected 5) | Kupiec p | indep p | CC p | zone |
|---|---|---|---|---|---|
| parametric normal (unconditional) | 14 | 0.0009 | 0.40 | 0.00 | yellow |
| plain historical | 10 | 0.048 | 0.52 | 0.12 | yellow |
| FHS per factor | 4 | 0.64 | 0.80 | 0.87 | green |

On the first 250 forecast days: parametric normal 13 (red, k = 4.0), plain
HS 10 (red), FHS 2 (green, k = 3.0). This is one seed of a synthetic
process. The tests pin properties, not these counts: tripling the last 10
days of factor moves roughly triples the per-factor FHS VaR and lifts plain
HS by under half (`test_factor_fhs_reacts_to_a_vol_spike_and_hs_does_not`).

## 5. Known failure modes

1. **VaR is not a worst case.** On the synthetic demo book the 99 % VaR is
   $55k and the 2020-replay stress loss is $254k.
2. **VaR is not subadditive.** Two independent default-style positions each
   with 95 % VaR of -5 combine to 95. ES stays subadditive (`TestCoherence`).
3. **sqrt(t) scaling has no fixed sign of error under clustering.**
   Unconditional case: sqrt(t) gives $172.9k against $144.5k from
   overlapping windows on the synthetic history. Conditional case from a
   calm state: sqrt(t) understates, shown on a simulated GARCH path
   (`test_sqrt_time_understates_from_a_calm_conditional_state`).
4. **Window myopia.** Plain HS reacts slowly to a vol regime change
   (`test_fhs_scales_up_after_vol_regime_switch`,
   `test_factor_fhs_reacts_to_a_vol_spike_and_hs_does_not`).
5. **MC model risk.** MC normal converges to $45.2k when the history is
   t(6) and MC t gives $49.9k. The SE covers sampling error only.
6. **Delta-gamma breakdown.** Approximation error on the demo book is
   -$1.2k (0.9 %) in a -15 % scenario and +$93k (37 %) in the -34 % 2020
   replay (`test_approximation_error_grows_with_shock_size`).
7. **Cornish-Fisher outside its domain.** S = 3 or K = 10 raise
   `ValueError`. The check is the exact minimum of the quadratic derivative
   on `[-z_a, z_a]` (spec S6). `(S, K) = (0.122, -0.427)` has its minimum
   derivative at z = -4, outside the 99 % interval, so it is valid at
   alpha 0.01. `(-0.0105, 8.0001)` has its minimum, -2.5e-7, at
   z = 0.00175 and is invalid at every alpha; a 2001-point grid misses it
   (`test_domain_check_is_exact_not_grid_resolution_dependent`).
8. **ES estimation error.** The bootstrap SE of ES at 1 % exceeds that at
   10 % on the same sample (`test_es_se_larger_for_smaller_alpha`).
9. **Kupiec's chi2(1) reference is oversized at the regulatory window.**
   Summing the exact Binomial(250, 0.01) mass over the 5 % rejection region
   gives a size of 9.48 %, 7.09 % at 500 days and 5.51 % at 1000 days
   (`TestKupiecExactSize`). Christoffersen shares the asymptotic reference.
10. **Small samples.** With `n alpha < 1` the S2 rank is clamped to 1 and
    the VaR is the largest loss in the sample. The engine requires 50
    observations but does not refuse alpha = 0.001 on them.

## 6. Edge cases

| Edge case | Behaviour | Test |
|---|---|---|
| Empty book | `ValueError("book has no positions")` | `TestEmptyPortfolio` |
| Single asset | parametric and MC within 2 % at 95 % | `TestSingleAsset` |
| Zero-vol asset | contributes 0 to sigma_p; Cholesky jitter is at most 1e-6 x mean variance, with a warning | `TestZeroVolAsset`, `test_zero_variance_factor` |
| Singular Sigma (perfect correlation) | parametric exact; MC through the warned jitter path | `TestSingularCovariance` |
| Materially indefinite Sigma | `LinAlgError` naming the most negative eigenvalue; explicit `repair="clip"` or `"higham"` | `test_indefinite_matrix_raises_instead_of_silent_10x_jitter` |
| Heavy-tailed MC (t, df 2.2, 5x vol) | every path revalues, no exception | `test_heavy_t_paths_never_raise` |
| alpha 0, 0.5, 0.99, negative | `ValueError` "alpha is the tail probability" | `TestAlphaEdges`, fuzz test |
| Insufficient history | `ValueError` naming the 50-observation minimum | `TestInsufficientHistory` |
| tau or vol at 0 | intrinsic or forward-intrinsic value | `test_zero_tau_is_intrinsic` |
| Kupiec x = 0 or x = T | `0 ln 0 = 0`, finite LR | `test_zero_exceptions_known_value` |
| No exceptions in AS Z2 | Z2 = +1 | `test_no_exceptions_gives_plus_one` |
| ES below VaR in AS Z2 | `ValueError` | `test_acerbi_szekely_rejects_es_below_var` |
| `pnl == -VaR` | not an exception (strict `<`) | `test_exceptions_definition_is_strict_inequality` |
| Reverse stress optimiser fails | `RuntimeWarning`, closed-form delta direction, `converged=False` | `TestReverseStressOptimiserChecks` |
| NaN, +inf, -inf, empty, negative where invalid | `ValueError` at every public entry point | `test_fuzz_public_api.py` |

`test_fuzz_public_api.py` holds a registry with one entry for every name in
`eq_var.__all__`. Hypothesis poisons each argument with NaN, +inf, -inf, an
empty array, a negative value or a bad alpha. A new public name without a
registry entry fails the test.

### 6.1 Property tests

| Invariant | Test |
|---|---|
| ES >= VaR for every family and alpha | `TestOrdering` |
| Risk measures non-increasing in alpha | `test_risk_measure_decreasing_in_alpha` |
| Positive homogeneity and translation invariance | `test_positive_homogeneity`, `test_translation_invariance` |
| Parametric VaR linear in exposures | `TestParametricEquivariance` |
| Variance-matched t(5) charges less than the normal at 95 % and more at 99 % | `test_t_vs_normal_var_crossover_is_between_2_5_and_5_percent` |
| `ewma_volatility` is causal and its seed uses the first 30 observations only | `test_ewma_volatility_is_causal_and_positive`, `test_seed_is_mean_square_of_first_30` |
| `ewma_covariance` seed is mean(r r') of the first 30 rows only (S8) | `test_ewma_covariance_seed_uses_first_window_only` |
| Per-factor FHS scenarios equal `r / sigma_t * sigma_{T+1}` per column | `test_factor_scenarios_are_rescaled_per_factor` |
| MC VaR error falls with path count | `test_mc_var_converges_to_the_parametric_answer` |

The ES subadditivity tests on paired empirical samples cannot fail for the
step-integral estimator. They document the identity and are not evidence.

## 7. Cross-language golden vectors

`tests/golden/generate_golden.py` writes `tests/golden/golden_vectors.json`
(79 values: 30 engine-specific keys from three sin/cos cases, and the 49
`conv_*` keys of spec S13 shared with the FX family). The C++ header and
the Rust module `tests/golden/mod.rs` are generated from the JSON.
`tests/test_golden_vectors.py` checks that the committed JSON matches a live
run within 64 ulps (the bound is argued in that file) and checks every
`conv_*` key against a plain-Python computation that shares no code with
`eq_var`. The JSON records the Python, numpy and scipy versions that
produced it. Consumers compare at 1e-9 relative plus 1e-12 absolute.

`conv_as_z2` is 1.0 because no loss in the S13 sample exceeds the S13 VaR
of 1500, so that key does not exercise the exception branch.

## 8. Not validated

- No comparison against live market data.
- The GARCH generator is a data-generating device, not a calibrated model.
- Stress shocks approximate published episode moves.

## Round 3 validation

Every new model has at least one oracle test. Files under `tests/`:

| Test file | Oracle | Evidence class |
|---|---|---|
| `test_risk_attribution.py` | hand two-asset example; Euler sum to 1e-13; marginal vs central finite difference; incremental vs marginal for a small trade; scenario ES and VaR contributions converge to the closed form on 400,000 draws | closed form, convergence |
| `test_frtb_ima.py` | hand-computed cascade; sqrt(12) telescoping; plus-factor table; capital by hand; desk thresholds at 12/13 and 30/31; Spearman and KS equal scipy to 1e-12; PLA zones | closed form, published table, library cross-check |
| `test_stressed_var.py` | planted 250-day crisis found; window VaR equals the 3rd-worst loss at alpha n = 3; capital by hand | closed form, planted signal |
| `test_evt.py` | POT VaR equals the exact GPD quantile and POT ES the integrated quantile, for xi in (-0.2, 0, 0.25, 0.6); MLE recovers xi and beta on 200,000 draws; t_4 tail gives xi near 1/4; flat mean excess for the exponential | closed form, convergence |
| `test_copula_mc.py` | Kendall tau = (2/pi) arcsin(rho); empirical tail dependence vs the closed form; Gaussian limit VaR vs closed form | closed form, convergence |
| `test_revaluation.py` | Greeks vs finite differences; put-call parity; cross-check against the package's own Black-Scholes `bs_price`/`bs_greeks`; linear book exact; explain residual scales by about 8 when moves halve | closed form, library cross-check, convergence |
| `test_limits.py` | statuses, ordering and aggregation by hand | exact |

Compiled `option_reval` (C++ and Rust, both VaR engines): golden values from
`eq_var.revaluation` to 1e-11, put-call parity, input rejection.
