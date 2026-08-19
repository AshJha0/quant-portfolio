# Equity VaR Engine (Rust)

A dependency-free Rust production twin of the Python research stack
[`python/equity/03-var-es-engine`](../../python/equity/03-var-es-engine) and
the sibling [`cpp/equity-var-engine`](../../cpp/equity-var-engine): the full
VaR / Expected-Shortfall / backtesting toolchain a market-risk desk runs in
its overnight batch, re-implemented in safe Rust (`std` only, zero crates)
for the memory-safety and determinism guarantees a production risk pipeline
wants, and **cross-validated against the Python reference to 1e-9 relative on
committed golden values**.

```
returns panel (T x n) ── sample / EWMA covariance ── Cholesky (jitter fallback)
   │                              │                        │
   ├── Historical VaR ─ plain (type-7) / BRW age-weighted / filtered (FHS)
   ├── Parametric VaR ─ normal / variance-matched Student-t / Cornish-Fisher
   ├── Monte Carlo VaR ─ MVN & multivariate-t, seeded xoshiro256++, VaR+ES+SE
   │
   ├── Expected Shortfall ─ empirical exact tail integral / closed forms
   ├── Backtesting ──────── Kupiec POF · Christoffersen IND/CC · chi2 p-values
   └── Basel traffic light ─ 0-4 / 5-9 / 10+ → multiplier 3.0 … 4.0
```

## Why a Rust engine when the Python and C++ stacks exist?

- **Memory safety with C++-grade throughput**: the `#![deny(missing_docs)]`,
  `Result`-everywhere API compiles to the same dense, allocation-light hot
  loops as the C++ twin (a Cholesky-then-triangular-product Monte Carlo
  path), but the borrow checker rules out the whole class of aliasing /
  lifetime bugs that a hand-rolled dense-matrix engine is prone to — with no
  runtime cost.
- **`Result`, not exceptions**: every fallible entry point returns
  [`EqVarError`](src/lib.rs) (`InvalidInput` / `Numerical`) instead of
  throwing; a validation failure is a value the caller must handle, not a
  stack unwind that a batch job can silently swallow.
- **No dependencies**: `std` only. The deterministic RNG (xoshiro256++), the
  special functions (inverse normal CDF, incomplete beta/gamma, Student-t
  quantile) and the dense linear algebra are implemented in-crate — `cargo
  build` needs nothing from crates.io, so the risk numbers of record never
  depend on an upstream crate's release cadence or transitive dependency
  tree.
- **Determinism as a control**: every stochastic path draws from a seeded
  [`rng::Rng`](src/rng.rs) (xoshiro256++ + Box–Muller + Marsaglia–Tsang) —
  never the platform thread-RNG — so a given seed reproduces VaR **bitwise**
  across runs on a given build. Regulators and model validation ask for
  exactly this.

## Validation contract — why NaN is the enemy

Every fallible entry point validates with `is_finite()`, **never** with an
ordering comparison. The reason is specific: a guard written as
`if x < 0.0 { return Err(...) }` silently *accepts* NaN, because every
IEEE-754 comparison against NaN is false — and Rust is no different from C
here (`f64::NAN < 0.0` is `false`). Worse, `f64::max` propagates the other
operand, so `NaN.max(0.0)` is `0.0`.

That is the worst possible failure mode for a risk system. A NaN risk
number breaches no limit and colours no traffic light (`NaN <= limit` is
false, but so is `NaN > limit`), so it aggregates into a book-level number
unnoticed; and a NaN that silently becomes a **zero** is worse still,
because a zero looks like a perfectly hedged book. Three real instances were
found and closed in the hardening pass:

| Where | Before | Now |
|---|---|---|
| `portfolio_sigma` | a NaN covariance entry passed the PSD test (`var < -tol` is false for NaN) and `NaN.max(0.0).sqrt()` returned **0.0** — zero sigma, zero VaR, zero ES | rejects non-finite exposures and covariances |
| `Matrix::cholesky` | pivot test `sum <= 0.0` is false for NaN, so a NaN covariance factorised into an all-NaN `L` and every downstream MC VaR was NaN | rejects non-finite entries; pivot test is `!(sum > 0.0)` |
| `exceptions_from_pnl` | `pnl_t < -var_t` is false when either side is NaN, so a broken VaR feed recorded **zero breaches** and passed Kupiec / Christoffersen / Basel in the green zone | rejects non-finite P&L and VaR series |

Related hardening: the Cholesky **jitter ladder** now stops at
`matrix::MAX_RELATIVE_JITTER` (1e-6 x mean variance) instead of escalating
to ten times the mean variance. A PSD-but-singular covariance (riskless leg,
perfectly correlated factors, rank-deficient factor block) is still repaired
at the first rung; a materially indefinite one now returns
`EqVarError::Numerical` naming the cap rather than being silently patched
into something that simulates a different book.

## Layout

```
equity-var-engine/
├── Cargo.toml              # zero deps, [[bin]] bench, release LTO profile
├── src/
│   ├── lib.rs               # EqVarError, TailModel, validate_alpha, prelude
│   ├── matrix.rs            # dense Matrix, Cholesky+jitter, covariance (sample/EWMA)
│   ├── stats.rs              # phi/Phi/Phi^-1, betainc, gammainc, t-dist, moments
│   ├── historical.rs         # plain / BRW / FHS VaR, sqrt-time scaling
│   ├── parametric.rs         # sigma_p, normal/t VaR, Cornish-Fisher + domain check
│   ├── expected_shortfall.rs # empirical exact tail integral, normal/t closed forms
│   ├── monte_carlo.rs        # seeded Rng, MVN/MVt simulation, VaR+ES+SE
│   ├── backtest.rs           # Kupiec, Christoffersen, Basel traffic light
│   ├── rng.rs                 # xoshiro256++, Box-Muller normals, Marsaglia-Tsang gamma
│   └── bin/bench.rs          # end-to-end timings on a 250 x 100 panel
├── tests/                   # 9 integration test files, 81 tests
│   └── test_cross_language.rs  # golden values generated by the Python reference
└── docs/                    # METHODOLOGY / VALIDATION / DESK_GUIDE
```

## Quickstart

```bash
cd rust/equity-var-engine
RUSTFLAGS="-D warnings" cargo test --release   # 81 tests + 10 doctests (91), offline
cargo run --release --bin bench                # benchmark table below
```

```rust
use eq_var_engine::prelude::*;

let cov = ewma_covariance(&returns_panel, 0.94)?;      // (T x n) -> (n x n)
let v_h = historical_var(&pnl, 0.01)?;                 // 99% 1d, type-7 quantile
let v_f = filtered_historical_var(&pnl, 0.01, 0.94)?;  // EWMA-filtered (FHS)
let v_p = parametric_var(&w, &cov, 0.01, TailModel::StudentT { df: 6.0 })?;
let mc_var = monte_carlo_var(&w, &cov, 0.01, 100_000, TailModel::StudentT { df: 6.0 }, 42)?;
let mc_es  = monte_carlo_es(&w, &cov, 0.01, 100_000, TailModel::StudentT { df: 6.0 }, 42)?;
// bitwise reproducible for a given seed on a given build
let tl = basel_traffic_light(5, 250)?; // yellow, k = 3.40
```

## Cross-language validation (the headline)

`tests/test_cross_language.rs` pins ~30 quantities to golden constants
generated by running the Python reference (`eq_var`, numpy / scipy) on
closed-form deterministic inputs that every language regenerates
independently — the same inputs and the same constants as
`cpp/equity-var-engine/tests/test_cross_language.cpp`, so **Python, C++ and
Rust all agree to the same tolerance**. Agreement required: **1e-9
relative**; observed residual is libm-vs-numpy ulp noise. Highlights:

| quantity (deterministic input) | Python reference | Rust engine |
|---|---|---|
| historical VaR 99 % (case A)   | 122.4129222375264 | equal to 1e-9 rel |
| empirical ES 99 % (case A)     | 141.7568107549531 | equal to 1e-9 rel |
| FHS VaR 95 %, lam=0.94 (A)     | 109.3777910164513 | equal to 1e-9 rel |
| parametric VaR 99 % normal (B) | 23176.07650751402 | equal to 1e-9 rel |
| parametric VaR 99 % t(6) (B)   | 25563.37478743866 | equal to 1e-9 rel |
| Cornish-Fisher VaR 99 % (B)    | 28230.62626871169 | equal to 1e-9 rel |
| EWMA cov [0,0], lam=0.94 (C)   | 6.085653794066718e-05 | equal to 1e-9 rel |
| Kupiec LR / p (250 obs, 7 exc) | 5.496990447793 / 0.019049230891 | equal to 1e-9 rel |
| Christoffersen CC LR / p (9 exc, planted cluster) | 11.41838698 / 3.315345e-03 | equal to 1e-9 rel |

Regenerate the constants any time with the provenance command in the test
header (`PYTHONPATH=src python3 /tmp/gen_golden.py` from the Python project).

Independently of the golden values, the MC engine is validated **against the
closed forms it should converge to**: 200k-path normal MC VaR within 3
order-statistic SE of the parametric answer, Student-t MC against the
variance-matched t closed form, and simulated covariance against the target
matrix (see `docs/VALIDATION.md`).

## Benchmark (250-day x 100-asset panel, alpha = 1 %, single thread, release + LTO)

Best-of-R wall times, deterministic inputs (`src/bin/bench.rs`):

| stage | best ms |
|---|---|
| sample covariance (250 x 100 → 100 x 100) | 0.78 |
| EWMA covariance (lam = 0.94) | 1.62 |
| historical VaR 99 % (250 scenarios) | 0.004 |
| BRW age-weighted VaR 99 % | 0.008 |
| filtered (FHS) VaR 99 % | 0.006 |
| empirical ES 97.5 % | 0.004 |
| parametric VaR 99 % (normal) | 0.010 |
| parametric VaR 99 % (Student-t df=6) | 0.020 |
| Cholesky 100 x 100 | 0.21 |
| **MC VaR 99 % normal, 100k paths** | **764** |
| MC VaR 99 % Student-t df=6, 100k paths | 776 |
| **full daily batch (everything above once)** | **~767** |

Sanity check inside the bench run: MC normal VaR 68 584 vs parametric 69 002
(within 1 SE at SE = 422); MC ES 78 573 vs closed form 79 053.

## Documentation

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — the maths, model choices vs
  alternatives, and the numbered assumptions register.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — analytic identities,
  convergence studies, cross-language goldens, failure modes with tests.
- [`docs/DESK_GUIDE.md`](docs/DESK_GUIDE.md) — batch SLAs, intraday use,
  regression gates, reconciliation against the Python research stack.
