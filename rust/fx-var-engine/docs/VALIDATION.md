# Validation

How the Rust engine was validated: analytic identities, cross-language
golden constants against the Python reference, statistical/ordering
tests, benchmarks, and known failure modes. Every claim below is enforced
by the test suite (`cargo test --release`): **91 tests + 2 doctests — 93 in
total — all passing under `RUSTFLAGS="-D warnings"`** (zero warnings, zero clippy
lint suppressions beyond documented `#[allow(...)]` on the two functions
that legitimately need more than the default argument-count lint allows).

---

## 1. Analytic identities

| Identity | Tolerance | Test |
| --- | --- | --- |
| Triangulation: long N EURJPY == long N EURUSD + long N*X(EURUSD) USDJPY, per scenario | 1e-12 (rel) | `book::tests::triangulation_identity_eurjpy` |
| Forward at CIP strike has zero inception value | 1e-10 | `book::tests::forward_zero_value_at_inception_and_cip` |
| Forward == two discounted deposit legs (+N e^{-r_f T} BASE, -N K e^{-r_d T} QUOTE) under FX shocks | 1e-10 | `book::tests::forward_matches_two_leg_deposit_decomposition` |
| Forward rate-leg deltas = -+T*N*e^{-+rT}*S closed form | 1e-4 rel (FD) | `book::tests::forward_rate_leg_sensitivities` |
| Base-ccy cash has zero P&L under any FX shock | 1e-9*notional | `book::tests::base_currency_position_has_zero_risk` |
| Forward at T=0 has zero rate-factor sensitivity | 1e-3 | `edge_cases::forward_at_zero_expiry_collapses_to_spot_difference` |
| Empirical VaR/ES hand-exact on 10-point samples (incl. fractional Acerbi-Tasche atom, weighted BRW case) | exact | `edge_cases::hand_exact_empirical_var_quantiles`, `hand_exact_acerbi_tasche_es`, `weighted_empirical_var_es` |
| Normal ES == sigma*phi(z_a)/(1-a); classic values (2.3263, 2.6652 at 99%) | 1e-9 | `edge_cases::closed_form_normal_textbook_numbers` |
| Standardised t: fatter than normal at equal sigma; df->inf recovers normal | ordering / 1e-4 | `edge_cases::student_t_converges_to_normal_as_df_grows` |
| Inverse normal CDF: Acklam + one Halley step | < 1e-9 (measured < 1e-13) | `stats.rs` doc comments; exercised transitively by every VaR test |
| Cholesky reconstruction L*L' = cov; exact-zero-pivot rank-1 matrix engages jitter and reports it | 1e-12 | `monte_carlo::tests::pegged_currencies_trigger_cholesky_jitter`, `edge_cases::singular_pegged_covariance_runs_with_jitter_end_to_end` |
| `var_covar` == hand-computed w'Sigma w quantile; sqrt-time scaling of VaR and ES (h=10) | 1e-6 | `parametric::tests`, `historical::tests::horizon_scaling_is_sqrt_time` |
| Kupiec & Christoffersen LR == hand-computed log-likelihoods; LR_cc = LR_uc + LR_ind | exact | `backtest::tests` |
| Basel zones exact at 250d/99%: green 0-4, yellow 5-9 (3.40/3.50/3.65/3.75/3.85), red 10+ (4.0) — from the cumulative binomial, not a table | exact | `backtest::tests::basel_zones_at_boundaries` |
| Reverse stress closed form: loss = k*sigma_p, shock reproduces loss via -w'dx; loss-target inversion | 1e-9-1e-12 | `stress::tests::reverse_stress_closed_form_loss_and_direction` |

## 2. Cross-model and ordering tests

- **MC -> parametric**: 200k-scenario normal MC VaR agrees with the
  var-covar closed form on the same book, within a small multiple of the
  KDE-based asymptotic standard error (`monte_carlo::tests::monte_carlo_var_normal_agrees_with_parametric`,
  the SE itself is asserted positive).
- **Tail ordering at 99%**: with identical covariance and seed, Student-t
  and jump-mixture VaR strictly exceed normal MC — pure tail shape, no
  variance leakage (`monte_carlo::tests::student_t_dist_has_fatter_realised_tails`,
  `jump_overlay_adds_tail_mass`).
- **Seed determinism**: identical seed => bitwise-identical scenario
  matrices and VaR figures on this platform (`monte_carlo::tests::simulate_is_deterministic_for_fixed_seed`);
  a different seed changes the draw
  (`monte_carlo::tests::different_seeds_differ`).
- **FHS regime response**: the crate's `historical::HsMethod::Filtered`
  path is exercised end to end (`historical::tests::filtered_hs_runs`); the
  EWMA-covariance regime-shift test
  (`edge_cases::ewma_covariance_reacts_faster_than_sample_after_regime_shift`)
  demonstrates the same lambda-decay responsiveness that makes FHS survive
  volatility-clustering backtests where plain HS fails.
- **Reverse stress**: an independent projected-gradient search in
  whitened coordinates (seeded random start, finite-difference gradients,
  never assumes the analytic optimum) matches the closed form to **1e-6**
  in loss and 1e-4 in the shock vector
  (`stress::tests::reverse_stress_numerical_search_confirms_closed_form`).

## 3. Cross-language golden constants (Python `fx_var`)

Three fully deterministic cases were run in the Python reference package
and their outputs embedded as constants in `tests/test_golden_python.rs`
(provenance header in the file; `repr()` 17-digit precision; generated and
independently re-confirmed against a live `python3` interpreter on
2026-08-18):

- **Case A — book + historical VaR.** 3-position book (10m EURUSD spot,
  5m USDJPY ATM CIP forward, -3m EURJPY cross) on a 300-day sinusoidal
  history reproduced bit-for-bit in Rust. Checked: single scenario P&L
  (58177.374898...), plain HS VaR/ES at 99% and 97.5%, BRW age-weighted
  VaR/ES at lambda=0.995 — agreement <= 1e-6 absolute on ~6e4 magnitudes
  (residual is libm sin/cos/exp ulp noise plus special-function
  implementation differences: SciPy Cephes vs this crate's Acklam+Halley).
- **Case B — parametric closed forms.** Fixed exposure vector and 3x3
  covariance through `var_covar`: normal and t(5) 99% VaR/ES, 10-day
  normal — 1e-8 relative.
- **Case C — backtest statistics.** `kupiec_pof(8, 250, 0.99)` LR and
  chi-square p-value, Christoffersen LR/p on a fixed 250-day 9-exception
  pattern, Basel cumulative binomial probabilities at x = 4, 5, 10 — 1e-10
  to 1e-12.

These are the same three cases committed in the C++ engine's
`tests/test_golden_python.cpp` (same fixture, same Python generator), so
all three stacks (Python, C++, Rust) are pinned to one source of truth.

Regeneration: run the snippet below from
`python/fx/03-var-es-engine` with `PYTHONPATH=src`; it is the exact
generator used to independently re-confirm the constants in this crate:

```python
import numpy as np, pandas as pd
from fx_var.book import Book, Market, Spot, Forward
from fx_var.historical_var import historical_var
from fx_var.parametric_var import var_covar
from fx_var.backtesting import kupiec_pof, christoffersen_independence, basel_traffic_light

market = Market(spot_usd={"EUR": 1.10, "JPY": 0.0090, "GBP": 1.27},
                rates={"USD": 0.050, "EUR": 0.030, "JPY": 0.001})
book = Book([Spot("EURUSD", 10e6, None), Forward("USDJPY", 5e6, 0.5, None),
             Spot("EURJPY", -3e6, None)], base="USD")
factors = ["FX:EUR", "FX:JPY", "IR:JPY", "IR:USD"]
scales = {"FX:EUR": .006, "FX:JPY": .007, "IR:JPY": .0004, "IR:USD": .0005}
t = np.arange(300.)
rets = pd.DataFrame({f: scales[f]*(np.sin(.1*t+j)+.5*np.cos(.05*t*(j+1.)))
                     for j, f in enumerate(factors)})
print(book.pnl(market, rets.iloc[17]))
print(historical_var(book, market, rets, alpha=.99, warn_pegs=False))
print(historical_var(book, market, rets, alpha=.99, method="age", decay=.995, warn_pegs=False))
w = pd.Series({"FX:EUR": 11e6, "FX:JPY": -4.5e6, "IR:USD": -2.4e6})
cov = pd.DataFrame([[3.6e-5,1.1e-5,-2e-6],[1.1e-5,4.9e-5,-1e-6],
                    [-2e-6,-1e-6,2.5e-7]], index=w.index, columns=w.index)
print(var_covar(w, cov, .99), var_covar(w, cov, .99, dist="t", df=5.),
      var_covar(w, cov, .99, horizon_days=10.))
print(kupiec_pof(8, 250, .99), basel_traffic_light(5, 250, .99))
```

## 3b. Input-validation contract (edge cases, CONVENTIONS item 6)

| Edge case | Behaviour | Test |
| --- | --- | --- |
| NaN / `+inf` / `-inf` in a **market** spot or rate | `FxVarError::Invalid` at `Market::new` | `edge_cases::non_finite_market_and_book_inputs_are_rejected` |
| NaN / Inf in a **position** notional, cash amount, entry rate, strike or expiry | `FxVarError::Invalid` at `Book::new` | same |
| NaN / Inf anywhere in a **return history** | rejected by `validate_returns`, `sample_cov`, `ewma_cov`, `ewma_volatility`, `historical_var`, `parametric_var` | `edge_cases::non_finite_returns_covariances_and_scalars_are_rejected` |
| NaN / Inf in a **covariance** or **exposure** vector | rejected by `portfolio_sigma`, `Matrix::cholesky(_with_jitter)`, `reverse_stress_*` | same |
| NaN / Inf in `sigma`, `mean`, `df`, `alpha`, `horizon_days` of the closed forms | rejected by `normal_var/es`, `student_t_var/es`, `var_covar` | same |
| NaN / Inf in an empirical **P&L sample** or its **weights** | rejected by `empirical_var/es` | same |
| NaN / Inf in a **backtest** P&L or VaR-forecast series | rejected by `evaluate_var_backtest` | same |
| **Negative** VaR forecast in a backtest | rejected (positive-loss convention) | same |
| `simple_to_log` at or below −100 % | `FxVarError::Invalid` instead of `-inf` / `NaN` | `edge_cases::simple_to_log_rejects_its_domain_boundary` |
| Boundary confidence levels `alpha` in {0, 1} and outside `(0, 1)` | rejected at every entry point, never clamped | `edge_cases::boundary_alpha_and_tiny_samples` |
| 1- and 2-observation samples | `empirical_var/es` exact; `sample_cov` / `ewma_volatility` need 2 rows; the backtest needs 2 days | same |
| Single-currency book (USD cash in USD, EUR cash in EUR) | carries no FX factor and is exactly flat to any shock; the same EUR cash reported in USD is fully exposed | `edge_cases::single_currency_book_has_no_fx_risk` |
| Identity triangulation (`CCYUSD`, `CCYCCY`, reciprocals, `forward(T=0)`) | `CCYUSD` is exactly the USD leg; every cross is the exact ratio of two USD legs and its reciprocal to 1e-14; identical-leg pairs are **rejected**, not vacuously priced at 1 | `edge_cases::identity_triangulation_is_exact` |
| Large-notional book (P&L covariance entries ~1e22) | scale-relative symmetry and PSD gates accept it; VaR scales exactly with sigma | `edge_cases::large_notional_books_are_not_rejected_by_absolute_tolerances` |
| Rank-1 / pegged covariance block | repaired at the first jitter rung, jitter reported | `edge_cases::jitter_repairs_pegged_blocks_but_refuses_indefinite_covariances` |
| Materially indefinite covariance (correlation > 1) | `FxVarError::Numerical` naming the materiality cap; never silently repaired | same |

## 4. Failure modes (mirrored from the Python/C++ references)

- **F0 — NaN is the failure mode a naive guard misses.** A validation
  written as `if x <= 0.0 { return Err(...) }` silently **accepts** NaN,
  because every IEEE-754 comparison against NaN is false. Rust does not
  help: `f64::NAN < 0.0` is `false` exactly as in C and C++. Worse,
  `f64::max` propagates the *other* operand, so `NaN.max(0.0)` is `0.0`.
  Five concrete holes were closed in the hardening pass, each of which
  turned corrupt input into a *plausible-looking* number rather than an
  error:

  1. `portfolio_sigma` computed `w' Sigma w`, tested it with
     `var < -tol` (false for NaN) and returned `var.max(0.0).sqrt()` — so
     a single NaN covariance entry reported a portfolio sigma, VaR and ES
     of exactly **zero**. A broken feed looked like a flat book.
  2. `Matrix::cholesky` tested its pivot with `s <= 0.0` (false for NaN),
     so a NaN covariance factorised into an all-NaN `L` and every Monte
     Carlo scenario and VaR downstream was NaN with no diagnostic.
     `is_symmetric` had the same shape of hole (`|NaN - NaN| > atol` is
     false, i.e. a NaN matrix counted as symmetric).
  3. `evaluate_var_backtest` screened with `is_nan()` only, so `+/-inf`
     passed. The exception test is `-p > v`; with a non-finite value on
     either side it is false, so the day was scored as "no exception" —
     a model whose VaR feed had broken recorded **zero breaches** and
     passed Kupiec, Christoffersen and the Basel traffic light green.
  4. `validate_returns` and the empirical estimators screened with
     `is_nan()` only, letting an infinite return into the covariance.
  5. `Market::new` validated spots but not **rates**, and the position
     validators checked neither notionals nor (for NaN) entry rates,
     strikes and expiries — `fw.expiry < 0.0` is false for NaN.

  All validation is now written with `is_finite()` rather than an ordering
  comparison, and the contract is pinned by the two `edge_cases` tests
  listed in section 3b.
- **F0b — Absolute tolerances do not survive a change of units.** The
  Cholesky pre-check used an **absolute** symmetry tolerance of 1e-12.
  That is correct for a covariance of daily returns (entries ~1e-5) and
  badly wrong for a covariance expressed in currency units: a 50bn book's
  P&L covariance has entries ~1e22, where one ulp is ~1e6 — eighteen
  orders of magnitude above the gate, so a *perfectly symmetric* matrix
  whose two triangles were accumulated in different orders was rejected as
  "not symmetric". The gate is now scale-relative
  (`Matrix::is_symmetric_rel`, `rtol * max(1, max|a|)`); the absolute
  `Matrix::is_symmetric(atol)` remains available for callers who know
  their units. The PSD gate in `portfolio_sigma` was already
  scale-relative (`1e-10 * max|w|^2`) and is unchanged.
- **F0c — The jitter ladder repairs rounding noise, not indefiniteness.**
  `cholesky_with_jitter` escalated by 10x per attempt from `1e-12 *
  mean(diag)` with no upper bound other than `max_tries`, so a caller
  passing a generous `max_tries` could have a materially indefinite
  matrix "repaired" by a jitter comparable to the variances themselves —
  returning a factor that simulates a different book, with no diagnostic.
  The escalation now stops at `matrix::MAX_RELATIVE_JITTER` (1e-6 x mean
  diagonal) and returns `FxVarError::Numerical` naming the cap. Pegged /
  rank-deficient blocks are repaired at the first rung and are
  unaffected.

- **F1 — Peg blindness (the headline).** A pegged currency contributes
  ~zero historical scenarios; HS and var-covar report ~=0 VaR while the
  true risk is a rare revaluation jump (CHF 2015: no daily USDCHF move
  over 1.9% in the prior 250 days, then +15-30% in hours). *Engine
  behaviour, tested end to end*
  (`edge_cases::peg_break_jump_produces_loss_historical_simulation_misses`):
  the peg screen flags `FX:*` factors with daily vol < 5e-4 into
  `flagged_peg_factors`/`warnings`; the jump-mixture MC on the same book
  reports > 20x the HS VaR; `peg_break_scenario` supplies the stress
  add-on. The failure mode is *detected and priced, not fixed* — no
  historical method can be.
- **F2 — Singular covariances.** Pegged blocks make Sigma rank-deficient;
  plain Cholesky errors. Escalating jitter factorises and *reports*
  (`MonteCarloResult::cholesky_warning`), tested with an exactly-rank-1
  matrix (`monte_carlo::tests::pegged_currencies_trigger_cholesky_jitter`).
- **F3 — Cornish-Fisher out of domain.** For |S|, K outside the Maillard
  monotonicity region the expansion is not a quantile function (99%
  "VaR" can undercut 95%). The engine checks the domain on a dense grid
  and errors; the escape hatch is explicit (`check_domain: false`).
  Tested at S=5, K=50
  (`parametric::tests::cornish_fisher_var_errors_outside_domain`).
- **F4 — sqrt-time scaling on carry books.** Negatively skewed, serially
  correlated unwinds violate i.i.d. aggregation; 10-day figures inherit
  the 1-day tail shape. Documented limitation (assumption A5); the
  scaling itself is tested so at least it is the *documented* error.
- **F5 — KDE-based VaR standard error** degrades in extremely discrete
  P&L distributions (near-degenerate books): the bandwidth floor guards
  the division but the SE is then conservative. MC convergence tests use
  well-spread books.
- **F6 — Rust/C++ engines are not bit-identical to each other.** Both
  engines are internally deterministic (fixed seed => fixed stream within
  each crate/library) and both are pinned to the same Python golden
  constants on the deterministic (non-MC) paths, but the Rust crate uses
  xoshiro256++ where the C++ engine uses `mt19937_64` — MC scenario draws
  and hence MC VaR/ES will differ (by noise-level amounts) between the two
  engines for the same seed. This is by design (assumption A9 in
  `docs/METHODOLOGY.md`) and is why the golden tests cover only the
  deterministic historical/parametric/backtest paths.

## 5. Benchmarks

Single thread, release build (`opt-level = 3`), 250-position / 50-factor
deterministic book, 500-day history (`cargo run --release --bin bench`):

```
book: 250 positions, 50 factors
historical  VaR (500 scen): var=62719 es=62727      2.231 ms
parametric  VaR (normal)  : var=103002 es=118006      1.215 ms
monte carlo VaR 100k normal: var=103273 es=118005   1445.1 ms
monte carlo VaR 100k t(5)  : var=115154 es=151195   1465.9 ms
```

The MC-vs-parametric agreement on the bench book (~0.3%) is itself a daily
cross-model consistency check: the two numbers share only the covariance
estimator. Absolute wall time is the same order of magnitude as the C++
engine's (historical ~2ms, parametric ~1.2ms, 100k MC well under a couple
of seconds single core); the MC path is dominated by `Rng::normal`'s
inverse-CDF transform, identical in structure between the two engines.

## 6. Reproducing

```bash
rm -rf target
RUSTFLAGS="-D warnings" cargo test --release
cargo run --release --bin bench
```

No network, no data files: every test input is generated deterministically
in-source (sinusoidal histories, fixed seeds).
