# Validation

How the engine is validated and where it fails. Every claim below names the
test that enforces it. `cargo test --release --locked -p eq-var-engine`, run from `rust/`, runs
14 integration files (127 tests) and 16 doctests, all offline and deterministic.
`cargo clippy --release --all-targets --locked -p eq-var-engine -- -D warnings` is clean.

## 1. Goldens from the Python reference

Both golden suites read `tests/golden/mod.rs`, generated from
`python/equity/03-var-es-engine/tests/golden/golden_vectors.json` by
`tools/gen_golden_rs.py`. `--check` fails if the module and the JSON
disagree. The same JSON feeds the C++ engine's header.

**Shared conventions (`tests/test_conv_golden.rs`).** Input sample
`P_i = round2(1000 sin(0.37 i + 0.1)(1 + 0.5 cos(0.11 i)))`, `i = 0..249`,
with `round2(x) = floor(100 x + 0.5)/100` in every language.

| keys | what | tolerance and reason |
|---|---|---|
| `conv_var_a*`, `conv_es_a*` | S2 VaR and ES at alpha 0.010, 0.025, 0.050 (`k` = 2.5, 6.25, 12.5) | 1e-12 rel: same values summed in the same order on both sides |
| `conv_basel_*` | zone, multiplier, `F(x)` for nine `(n, x, alpha)` cases | zone and multiplier exact; CDF 1e-13 abs (pmf recursion of at most 16 terms) |
| `conv_kupiec_lr_*` | LR at (250, 5), (500, 9), (250, 0) | 1e-12 abs (difference of log-likelihoods of size 25) |
| `conv_as_z2` | Z2 with VaR 1000, ES 1300, alpha 0.025 | 1e-12 abs |
| `conv_cf_valid_*` | five Cornish-Fisher validity flags | exact |
| `conv_higham_*`, `conv_clip_*` | repaired entries of the 3x3 indefinite matrix | 1e-8 abs (spec) and 1e-12 rel (spec) |
| `conv_ewma_sigma_last` | one-step-ahead EWMA vol of `P/1e5` | 1e-14 rel (250-step recursion with contraction 0.94) |

**Older cases (`tests/test_cross_language.rs`).** Closed-form sin/cos inputs:
case A (historical family on 100 points), case B (parametric on a 3-asset
book), case C (covariances, Kupiec, Christoffersen on a 60x3 panel).
Tolerance `1e-9` relative plus `1e-12` absolute.

In 0.2.0 two case A values changed with the order-statistic rule:
`A_HIST_VAR_1PCT` 122.41 to 141.76 and `A_HIST_VAR_5PCT` 104.55 to 108.23.
`A_FHS_VAR_5PCT_LAM094` (109.38) is unchanged because the deprecated
P&L-level FHS keeps its old arithmetic.

## 2. Regression tests for 0.2.0 (`tests/test_conventions.rs`)

Each failed on the old code (wrong number, panic, hang, or missing entry
point) before the change. Run against the old sources: historical VaR 96
vs 95.05, Basel 5/500 green vs yellow, `x > n` an error vs clamped, the
EWMA seed, `erfc(+inf)`.

| test | checks |
|---|---|
| `alpha_is_tail_probability_and_confidence_levels_are_rejected` | 0.5, 0.95, 0.99, 0, NaN, inf rejected with the convention message |
| `order_statistic_var_and_fractional_es_by_hand` | `L(m)`, fractional ES, the `1e-9` rank guard |
| `weighted_estimator_matches_unweighted_with_equal_weights` | weighted S2 with equal weights equals S2 |
| `kupiec_struct_matches_formula_and_rejects_bad_counts` | LR by hand, `x > n` and `alpha >= 0.5` rejected |
| `basel_zone_follows_the_window_not_the_raw_count` | 250-day table, 5/500 green, thresholds on `F` for other windows |
| `basel_cdf_matches_incomplete_beta` | pmf recursion vs incomplete beta, 1e-12 |
| `acerbi_szekely_z2_sign_and_value` | value and sign |
| `cornish_fisher_validity_follows_alpha` | (0.122, -0.427) valid at 0.01 and 0.001, invalid at 1e-5 |
| `indefinite_matrix_errors_with_eigenvalue_and_repairs_work` | error names the eigenvalue `-0.8`; Higham unit diagonal, PSD; clip keeps the diagonal; Higham is closer to A than clip |
| `eigen_reconstructs_the_matrix` | `V Lambda V' = A` to 1e-13 |
| `singular_matrix_reports_jitter_within_cap` | jitter reported and below `1e-6 mean(diag)` |
| `relative_symmetry_check` | asymmetry of `1e-13` on entries of `1e-4` is rejected |
| `ewma_seed_uses_first_30_observations_only` | later data does not move `sigma[0]` |
| `per_factor_fhs_matches_hand_recipe` | S9 against a hand loop |
| `monte_carlo_run_simulates_once_and_bootstrap_is_primary` | VaR and ES from the returned P&L, bootstrap and KDE SE within 3x of each other, `n_boot = 1` rejected |
| `empty_book_errors_everywhere` | "book has no positions" |
| `special_functions_never_panic_or_hang_on_non_finite_input` | `erfc(NaN)` used to recurse without end, `erfc(+inf)` panicked, `Rng::gamma(NaN)` never left its loop |

## 3. Fuzz walk (`tests/fuzz_public_api.rs`)

Proptest properties take valid arguments, poison one argument or one slice
element with NaN or `+/-inf` (or pass empty, length-1, negative input where
invalid) and assert `Err`. Functions that return a float (`erf`, `erfc`,
`normal_cdf`, `normal_pdf`, `cornish_fisher_z`, `Matrix::matvec`,
`Rng::gamma`, `Rng::chi_square`) must map NaN to NaN.
`registry_covers_every_public_fn` scans `src/` and fails if a `pub fn` has
no entry. Writing it found two holes, now closed: `overlapping_horizon_pnl`
and `portfolio_pnl` accepted non-finite input.

## 4. Analytic identities

| check | tolerance | test |
|---|---|---|
| `Phi^{-1}(0.975) = 1.959963984540054` | 1e-9 | `test_stats::normal_ppf_known_quantiles` |
| normal ES identity vs Simpson quadrature | 1e-10 rel | `test_expected_shortfall::normal_es_identity_vs_numerical_quadrature` |
| Cholesky `A = LL'` | 1e-12 | `test_matrix::cholesky_reconstructs_spd` |
| BRW weights sum to 1, ratio `1/lam` | 1e-12 | `test_historical::brw_weights_*` |
| `chi^2(1)` sf at 3.841458 is 0.05 | 1e-11 | `test_stats::incomplete_gamma_chi2_pvalues` |
| Student-t quantiles vs `scipy.stats.t.ppf` | 1e-8 | `test_stats::student_t_quantiles_vs_scipy` |

## 5. Monte Carlo

- Normal MC VaR and ES against the closed forms, within SE bars;
  Student-t MC above normal at 99 % and near the variance-matched t closed
  form; simulated covariance against the target (`tests/test_monte_carlo.rs`).
- Same seed gives exactly equal VaR and ES on the same build
  (`bitwise_seed_determinism`). No claim is made across platforms.
- The allocation-free `fast` API returns the same bits as the main API
  (VaR, ES, both SEs and every scenario P&L) and repeats bit for bit on a
  second run with the same seed (`tests/test_fast_equiv.rs`). It makes
  zero heap allocations per call (`tests/test_fast_alloc.rs`).
- On the 250 x 100 bench book, MC normal VaR is 444 614 against parametric 444 396 with
  bootstrap SE 2 110 and KDE SE 2 157 (full-rank 100-factor covariance).

## 6. Known limits

1. **Tail resolution.** 250 days put 2.5 observations in a 1 % tail. For
   `alpha < 1/n` the VaR is the worst observed loss (*test*
   `deep_tail_alpha_on_small_sample`). Below `MIN_OBS = 50` historical
   methods refuse.
2. **Square-root-of-time** understates stressed multi-day risk.
3. **KDE SE** uses a bandwidth tuned to the bulk and tends to understate
   the SE deep in the tail; the bootstrap SE is the primary number.
4. **Kupiec at 250 days, 1 %.** The chi2(1) approximation is oversized
   there (expected count 2.5); the exact binomial is the better read.
5. **Basel plus factors** for windows other than 250 days and alphas other
   than 1 % use the equivalent-count mapping, a repository convention.
6. **Linear P&L only.** No gamma or vega.
7. **NaN.** Validation uses `is_finite()` and the NaN-rejecting form
   `!(x > 0.0)`, never `x <= 0.0`, which is false for NaN. Earlier holes:
   a NaN covariance gave a zero sigma through `NaN.max(0.0)`, a NaN pivot
   factorised into an all-NaN `L`, and a NaN VaR feed scored zero
   exceptions (*tests* in `test_edge_cases.rs`).
8. **RNG streams differ across engines.** Cross-engine MC agreement is
   statistical.
