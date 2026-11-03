# Rust oracle kernels

## What this file is

Owner decision D1 keeps a full Rust implementation of the pricing and risk
engines. Most Rust code is a port: it is checked against the Python and C++
engines, so a shared mistake would pass in all three languages. The kernels
listed here have a stronger job. Each one is checked against a **shared
golden produced by an independent oracle** (mpmath at 50 digits, exact
rationals or closed forms), not against another engine. If one of them
drifts, CI fails even if every engine drifts the same way.

Plan reference: `review/2026-11-03/NEXT_RELEASE_PLAN.md`, W20 target 4 and
the D1 option list ("BS price and greeks, IV, historical VaR, Cholesky, MC
VaR").

## How it is checked

`rust/tools/check_oracle_kernels.py` reads the table below and, for every
row, checks that:

1. the crate directory and the Rust test file exist;
2. the test file has a `#[test] fn <test>` with that exact name;
3. every Rust symbol in the "Rust entry points" column is called in the
   test file, so a row cannot quietly stop exercising its kernel;
4. the generated constants module exists and names the oracle source
   (the "Oracle source" column), and the oracle source file exists, so the
   expected numbers still come from the independent generator.

`--run` also runs each listed test with
`cargo test --release --locked -p <package of crate> --test <file> -- --exact <test>`
and fails unless exactly one test ran and passed. The static check runs in
the Python suite (`tests/test_oracle_kernels.py`); the `--run` form needs a
Rust toolchain and belongs in the Rust CI job.

Adding a row: add the test first, make sure its constants come from an
oracle that imports nothing from the engines, then add the row. Removing a
row needs a CHANGELOG entry saying why the kernel no longer needs an
independent check.

## Kernels

| id | Kernel | Crate | Test file | Test | Rust entry points | Constants module | Oracle source |
|---|---|---|---|---|---|---|---|
| bs-price-greeks | Black-Scholes price and Greeks, Black-76, CRR | equity-options-engine | tests/golden.rs | all_records_match_oracle_and_track_worst_deviation | bs_price, bs_greeks, black76_greeks, crr_price | src/golden.rs | python/equity/01-options-pricing/tests/golden/generate_golden.py |
| bs-implied-vol | Implied volatility (equity) | equity-options-engine | tests/golden.rs | all_records_match_oracle_and_track_worst_deviation | implied_vol | src/golden.rs | python/equity/01-options-pricing/tests/golden/generate_golden.py |
| gk-price-greeks | Garman-Kohlhagen price, Greeks, delta conventions, CRR | fx-options-engine | tests/golden.rs | all_records_match_oracle_and_track_worst_deviation | gk_price, analytic_greeks, delta, binomial_price | src/golden.rs | python/fx/01-options-pricing/tests/golden/generate_golden.py |
| gk-implied-vol | Implied volatility (FX) | fx-options-engine | tests/golden.rs | all_records_match_oracle_and_track_worst_deviation | implied_vol | src/golden.rs | python/fx/01-options-pricing/tests/golden/generate_golden.py |
| eq-hist-var | Historical VaR and ES, k-th worst (MAR99) | equity-var-engine | tests/test_oracle.rs | historical_kth_worst_exact | empirical_var_es | tests/oracle/mod.rs | python/equity/03-var-es-engine/tests/golden/var_oracle.py |
| fx-hist-var | Historical VaR and ES, k-th worst (MAR99) | fx-var-engine | tests/test_oracle.rs | historical_kth_worst_exact | empirical_var_es | tests/oracle/mod.rs | python/equity/03-var-es-engine/tests/golden/var_oracle.py |
| eq-param-var | Normal and Student-t VaR and ES | equity-var-engine | tests/test_oracle.rs | normal_and_student_t_var_es | parametric_var, normal_es | tests/oracle/mod.rs | python/equity/03-var-es-engine/tests/golden/var_oracle.py |
| fx-param-var | Normal and Student-t VaR and ES | fx-var-engine | tests/test_oracle.rs | normal_and_student_t_var_es | normal_var, normal_es, student_t_var, student_t_es | tests/oracle/mod.rs | python/equity/03-var-es-engine/tests/golden/var_oracle.py |
| eq-backtests | Kupiec, Christoffersen, Basel zones | equity-var-engine | tests/test_oracle.rs | kupiec_christoffersen_basel | kupiec_pof | tests/oracle/mod.rs | python/equity/03-var-es-engine/tests/golden/var_oracle.py |
| fx-backtests | Kupiec, Christoffersen, Basel zones | fx-var-engine | tests/test_oracle.rs | kupiec_christoffersen_basel | kupiec_pof | tests/oracle/mod.rs | python/equity/03-var-es-engine/tests/golden/var_oracle.py |
| eq-cholesky | Cholesky with PSD repair (Higham, clip) | equity-var-engine | tests/test_conv_golden.rs | conv_cholesky_repairs | cholesky_repair | tests/golden/mod.rs | python/equity/03-var-es-engine/tests/golden/generate_golden.py |
| eq-mc-var | Monte Carlo VaR of a Gaussian and t book (3 SE) | equity-var-engine | tests/test_oracle.rs | monte_carlo_gaussian_book_within_3_se | monte_carlo_run | tests/oracle/mod.rs | python/equity/03-var-es-engine/tests/golden/var_oracle.py |
| fx-mc-var | Monte Carlo VaR of a Gaussian and t book (3 SE) | fx-var-engine | tests/test_oracle.rs | monte_carlo_gaussian_book_within_3_se | simulate_factor_returns, empirical_var | tests/oracle/mod.rs | python/equity/03-var-es-engine/tests/golden/var_oracle.py |

## Caveats

- `eq-cholesky` checks the repaired matrix against
  `golden_vectors.json`, which `generate_golden.py` builds with NumPy
  (Higham's alternating projections, Higham 2002, and eigenvalue clipping).
  That is a reference implementation, not mpmath; it is independent of the
  Rust code but not of floating point. The tolerance is 1e-8 absolute.
- The Monte Carlo rows are statistical: they pass when the estimate is
  within 3 standard errors of the exact quantile, for three seeds.
- The two FX VaR rows use the equity oracle: the FX book in the test is the
  same case-B covariance, and `var_oracle.py` covers both
  (`tests/oracle/mod.rs` header).
