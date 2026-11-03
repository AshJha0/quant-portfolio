# Validation

How the engine is validated and where it fails. Every claim names the test
that enforces it. `cargo test --release --locked -p fx-var-engine`, run from `rust/`, runs
71 unit tests, 62 integration tests (7 files) and 8 doctests, offline and deterministic.
`cargo clippy --release --all-targets --locked -p fx-var-engine -- -D warnings` is clean.

## 1. Analytic identities

| Identity | Tolerance | Test |
| --- | --- | --- |
| Triangulation: EURJPY equals EURUSD plus USDJPY legs, per scenario | 1e-12 rel | `book::tests::triangulation_identity_eurjpy` |
| Forward at CIP strike has zero value | 1e-10 | `book::tests::forward_zero_value_at_inception_and_cip` |
| Forward equals two discounted deposit legs | 1e-10 | `book::tests::forward_matches_two_leg_deposit_decomposition` |
| Base-currency cash has zero P&L | 1e-9 notional | `book::tests::base_currency_position_has_zero_risk` |
| Empirical VaR/ES by hand, fractional atom, weighted case | exact | `edge_cases::hand_exact_empirical_var_quantiles`, `hand_exact_acerbi_tasche_es`, `weighted_empirical_var_es` |
| Normal 99 % VaR 2.3263, ES 2.6652 at `alpha = 0.01` | 1e-9 | `edge_cases::closed_form_normal_textbook_numbers`, `test_conventions` |
| Standardised t tends to normal as df grows | 1e-4 | `edge_cases::student_t_converges_to_normal_as_df_grows` |
| Kupiec, Christoffersen LR by hand | exact | `backtest::tests` |
| Reverse stress closed form and loss-target inversion | 1e-9 | `stress::tests::reverse_stress_closed_form_loss_and_direction` |

## 2. Goldens from the Python reference

`tests/golden/mod.rs` is generated from
`python/fx/03-var-es-engine/tests/golden/golden_vectors.json` by
`tools/gen_golden_rs.py` (`--check` verifies it).

- `tests/test_conv_golden.rs`: every shared `conv_*` key (S2 VaR/ES at three
  alphas on the shared sample, nine Basel cases, three Kupiec LRs, Z2, five
  Cornish-Fisher flags, Higham and clip repairs at the JSON's declared
  tolerances, EWMA vol). Tolerances are argued in ulps in the file header.
- `tests/test_golden_python.rs`: case A (book P&L and plain/BRW HS on a
  300-day history), case B (parametric closed forms), case C (Kupiec,
  Christoffersen, Basel CDFs). The calls now pass tail alphas (0.01, 0.025);
  the golden values did not change beyond the last digit, because the old
  confidence rule picked the same order statistic.

## 3. Regression tests for 0.2.0

`tests/test_conventions.rs`. The behavioural ones were run against the old
sources first and failed there: 0.99 accepted as an alpha, the Basel plus
factor for 15/1000, the EWMA seed, `inv_norm_cdf(1.5)` panicking, the
10-day jump mean.

| test | checks |
|---|---|
| `confidence_levels_are_rejected_not_reinterpreted` | 0.99, 0.975, 0.95, 0.5 rejected; defaults are 0.01 |
| `deprecated_confidence_shims_convert` | old `(x, n, confidence)` calls forward with `alpha = 1 - confidence` |
| `basel_plus_factor_uses_the_equivalent_250_day_count` | plus factor by equivalent count, `x > n` rejected |
| `cornish_fisher_counterexample_is_valid_at_99` | (0.122, -0.427) valid at 0.01 |
| `indefinite_covariance_errors_and_repairs` | eigenvalue named; Higham and clip; MC refuses the indefinite matrix |
| `ewma_seed_is_first_30_observations` | seed independent of later data |
| `monte_carlo_reports_bootstrap_se_and_jitter` | `se_var` is the bootstrap SE; `n_boot = 1` rejected |
| `empty_book_error_message` | "book has no positions" |
| `former_panics_are_errors` | `inv_norm_cdf`, incomplete gamma/beta, `chi2_sf`, t quantile, Student-t df NaN, `Rng::gamma` |

Unit test `monte_carlo::tests::jump_count_scales_with_horizon`: with
`p = 0.02` per day and `h = 10`, the mean jump P&L over 200 000 scenarios
is `lam h mu_J` within 5 standard errors of the mean.

## 4. Fuzz walk

`tests/fuzz_public_api.rs`: proptest properties poison one argument or one
element with NaN or `+/-inf` (or pass empty, length-1, negative input where
invalid) and assert `Err` for every public function. Functions that return a
float (`norm_pdf`, `norm_cdf`, `cornish_fisher_z`, `Matrix::matvec`,
`Rng::gamma`, `Rng::chi_square`) must map NaN to NaN.
`registry_covers_every_public_fn` scans `src/` and fails on a `pub fn`
without an entry. Writing it found holes, now closed: `Market::forward`
(NaN expiry), `CompiledBook::value_usd` and `pnl_scenarios` (non-finite
shocks), `Matrix::quad_form` and `solve_spd` (non-finite vectors),
`usd_broad_move` (empty list), `flag_peg_factors` (NaN threshold flagged
nothing).

## 5. Monte Carlo

- Normal MC VaR on a single EURUSD spot position matches the exact
  lognormal quantile `11m (1 - e^{z sigma})` within 4 bootstrap SEs (a
  two-sided miss probability of 6e-5 for a correct engine); the delta
  closed form sits about 1 % higher because of convexity
  (`monte_carlo::tests::monte_carlo_var_normal_agrees_with_parametric`).
- Student-t realised tails are fatter than normal at equal covariance.
- Seed determinism on the same build (`simulate_is_deterministic_for_fixed_seed`).
- Bootstrap and KDE SE agree within 3x at `alpha = 0.001` on 20 000 draws.

## Reference: the independent mpmath oracle

The reference values for this engine come from an oracle that shares no
code with any engine: `python/equity/03-var-es-engine/tests/golden/var_oracle.py`.
It uses mpmath at 50 digits and exact `Fraction` arithmetic for order
statistics and the binomial CDF, and imports nothing but the standard
library and mpmath (a test scans its imports). The Student-t quantile is
found by bisection on the CDF written as a regularized incomplete beta, not
by a library ppf. It writes 196 values to
`python/fx/03-var-es-engine/tests/golden/oracle_vectors.json`, and
`emit_oracle_consts.py` turns them into the C++ and Rust constants; a
Python test fails if the emitted files are stale.

What it covers: normal and variance-matched Student-t VaR and ES at alpha
0.001, 0.01, 0.025 and 0.05 and df 3, 4, 6, 10 and 30; Cornish-Fisher;
the case-B portfolio sigma and its 1-day and 10-day VaR; historical VaR on
a 250-value sample under the regulatory k-th-worst rule and the
interpolated (type 7) rule; Acerbi-Tasche ES; Kupiec LR and p-value (18
cases); Christoffersen independence and conditional coverage; and the
Basel binomial CDF (15 cases).

Published tables. The oracle reproduces all 11 cells of the BCBS (1996)
traffic-light cumulative probabilities to 0.01%, and 7 of the 9 cells of
the Kupiec (1995) non-rejection regions as tabulated by Jorion. The other
two follow from the LR test itself and are pinned by tests: at T = 255,
p = 1%, N = 0 gives LR = 5.126 > 3.841, so it is rejected and the table
prints no lower bound; at T = 255, p = 5%, N = 20 gives LR = 3.727 < 3.841,
so it is accepted where the table stops at 19. The BCBS 1996
traffic-light and Kupiec 1995 tables were checked against the printed
sources by the owner, 2026-11-03.

This engine (Rust, `tests/test_oracle.rs`, constants in `tests/oracle/mod.rs`) agrees with the oracle: closed forms to
1e-10 relative, historical VaR exactly, ES to 1e-12, and Monte Carlo VaR (case B, normal and t5, 3 seeds, 200,000 paths) within 3 standard errors.

## 6. Failure modes

- **F1 Peg blindness.** A pegged currency contributes near-zero historical
  scenarios. The peg screen flags `FX:*` factors with daily vol below 5e-4;
  the jump overlay and `peg_break_scenario` price the break
  (`edge_cases::peg_break_jump_produces_loss_historical_simulation_misses`).
  Detected and priced, not fixed: no historical method can see it.
- **F2 Singular covariances.** Pegged blocks take a reported jitter
  (`pegged_currencies_trigger_cholesky_jitter`); a materially indefinite
  matrix is an error naming the most negative eigenvalue.
- **F3 Cornish-Fisher outside its region** at the requested alpha: error;
  `check_domain = false` exists for diagnostics.
- **F4 Square-root-of-time** on carry books understates 10-day risk.
- **F5 KDE SE** understates the SE deep in the tail; the bootstrap SE is
  primary.
- **F6 Engines do not match each other draw for draw.** Rust uses xoshiro256++,
  C++ `mt19937_64`; MC agreement across engines is statistical. No
  cross-platform reproducibility is claimed either.
- **F7 Kupiec at 250 days, 1 %.** The chi2(1) test is oversized there
  (expected count 2.5); read p-values near 0.05 with care.
- **F8 NaN.** Validation uses `is_finite()` and `!(x > 0.0)`; `x <= 0.0` is
  false for NaN. Earlier holes: a NaN covariance gave sigma 0 through
  `NaN.max(0.0)`; a `+/-inf` VaR feed scored zero exceptions; `Market::new`
  did not check rates (`edge_cases::non_finite_*`).

## 7. Benchmarks

See the README for the per-kernel latency table and the command. On
2026-11-01 (Intel Core 5 120U under WSL2), p50: historical 1.2 ms,
parametric 0.63 ms, MC 100k normal 0.45 s and t(5) 0.35 s, each MC call including 200 bootstrap resamples of the 100k
scenarios. MC normal VaR 424 401 against parametric 424 010.

The parallel Monte Carlo and the workspace kernels changed no result.
`tests/test_latency_determinism.rs` pins VaR, ES and both standard
errors to the bits printed by the serial code for normal, Student-t and
jump draws, checks thread counts 1 to 64 give the same P&L vector, and
checks each workspace kernel against its allocating twin with
`to_bits`. `tests/test_latency_alloc.rs` checks that the hot kernels make
no heap allocation on success.
