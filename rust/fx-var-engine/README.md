# FX VaR & ES Engine (Rust)

Multi-currency FX Value-at-Risk and Expected Shortfall engine — the Rust
twin of the Python research package
[`python/fx/03-var-es-engine`](../../python/fx/03-var-es-engine/) (package
`fx_var`) and the C++ engine
[`cpp/fx-var-engine`](../../cpp/fx-var-engine/) (namespace `fxvar`),
mirroring their semantics (factor conventions, estimators, test
identities) and cross-validated against both with committed golden
constants.

Zero external dependencies: the linear algebra, special functions (inverse
normal CDF, Student-t quantile, incomplete gamma/beta), RNG transforms and
estimators are implemented in the crate and validated in `tests/`.

## What it does

| Module | Contents |
| --- | --- |
| `src/book.rs` | Multi-currency book vs a base ccy: cash, spot, outright forwards (spot + two deposit legs, CIP-consistent). Cross positions (EURJPY) decompose into USD legs — triangulation by construction. Struct-of-arrays `CompiledBook` hot path: one `exp()` per leg per scenario. |
| `src/matrix.rs` | Dense matrix, Cholesky with escalating jitter (singular pegged-block covariances are a legitimate input), products/quadratic forms. |
| `src/returns.rs` | Labelled factor histories, sample + RiskMetrics-EWMA covariance, EWMA volatility, NaN-refusing validation, near-zero-vol **peg screen**. |
| `src/stats.rs` | Normal pdf/cdf, inverse normal CDF (Acklam + Halley, error < 1e-13), Student-t pdf/cdf/quantile, regularised incomplete gamma/beta, chi-square SF, exact binomial CDF, sample moments. |
| `src/historical.rs` | Plain HS, BRW age-weighted, filtered (EWMA-devolatilised) HS; sqrt-time horizon scaling; peg-blindness warning list. |
| `src/parametric.rs` | Variance-covariance VaR/ES (normal + standardised Student-t on the same sigma), Cornish-Fisher with explicit monotonicity-domain check. |
| `src/monte_carlo.rs` | Multivariate normal, covariance-matched Student-t, and **jump-mixture** (Bernoulli common jump, configurable size per ccy — the peg-break add-on) with full book revaluation; seeded xoshiro256++ + crate-owned inverse-CDF transforms → bit-reproducible; VaR standard error via KDE. |
| `src/expected_shortfall.rs` | Weighted empirical VaR/ES (coherent Acerbi–Tasche tail splitting), closed-form normal and standardised-t VaR/ES. |
| `src/backtest.rs` | Kupiec POF, Christoffersen independence, conditional coverage (chi-square p-values via regularised incomplete gamma), exact Basel traffic light with 1996 multiplier add-ons. |
| `src/stress.rs` | Joint scenarios {per-ccy spot shocks, rate shifts}; canned GBP −8.1% Brexit, CHF +14.9% depeg, JPY carry unwind, broad USD move, peg-break add-on with contagion; reverse stress: closed form for a linear book, verified against an independent numerical search. |

## Quickstart

```bash
rm -rf target
cargo test --release      # 83 tests + 1 doctest, all passing
cargo run --release --bin bench
```

Toolchain: Rust 2021 edition, `cargo test`, no external crates.
`#![deny(missing_docs)]` — every public item is documented.

```rust
use fx_var_engine::prelude::*;

let market = Market::new(
    [("EUR", 1.10), ("JPY", 0.0090)],
    [("USD", 0.05), ("EUR", 0.03), ("JPY", 0.001)],
)?;
let book = Book::new(
    vec![
        Position::Spot(SpotPosition::new("EURJPY", 5.0e6, None)),          // cross: 2 USD legs
        Position::Forward(ForwardPosition::new("USDJPY", 10.0e6, 0.5, None)), // CIP fwd, rate legs
    ],
    "USD",
)?;

let hs = historical_var(&book, &market, &returns, &HistoricalOptions::default())?;
// hs.var, hs.es, hs.flagged_peg_factors, hs.warnings

let mut opts = MonteCarloOptions { dist: McDist::Jump, seed: 42, ..Default::default() };
opts.jumps.prob = 0.02;
opts.jumps.mean.insert("FX:JPY".to_string(), -0.10); // peg-break overlay
let mc = monte_carlo_var(&book, &market, &sample_cov(&returns)?, &opts)?;
```

## Benchmark

250-position / 50-factor book (200 spots across 45 currencies + 50
forwards; 45 FX + 5 IR factors), full revaluation, single thread, release
build (`cargo run --release --bin bench`):

```
book: 250 positions, 50 factors
historical  VaR (500 scen): var=62719 es=62727      2.231 ms
parametric  VaR (normal)  : var=103002 es=118006      1.215 ms
monte carlo VaR 100k normal: var=103273 es=118005   1445.1 ms
monte carlo VaR 100k t(5)  : var=115154 es=151195   1465.9 ms
```

Monte Carlo normal 99% VaR agrees with the parametric closed form on the
same book to ~0.3% (103.3k vs 103.0k) — the two paths share no code beyond
the covariance. Numbers are the same order of magnitude as the C++ twin
(historical ~2 ms, parametric ~1.2 ms, 100k MC well under 2s single core);
absolute wall time differs with allocator/codegen, not algorithm.

## Validation summary

- **Triangulation identity**: EURJPY P&L == EURUSD + USDJPY leg
  decomposition to 1e-12, scenario by scenario.
- **Forwards**: zero value at CIP inception; equal to the two discounted
  deposit legs to 1e-10; rate-leg sensitivities match `∓T·N·e^(−rT)·S`
  closed form.
- **Cross-language goldens**: three deterministic cases (historical VaR/ES,
  parametric closed forms, backtest statistics) reproduce the Python
  `fx_var` reference to 1e-6 absolute / 1e-8 relative
  (`tests/test_golden_python.rs`; independently re-run against a live
  Python interpreter, provenance in the file header).
- **MC**: normal MC within a few standard errors of the closed form; t and
  jump-mixture strictly fatter at 99%; seed determinism (bit-reproducible
  on this platform).
- **Backtests**: Kupiec/Christoffersen against hand-computed LRs; Basel
  zones exact at the regulatory boundaries (green 0–4 / yellow 5–9 / red
  10+ at 250d/99%).
- **Reverse stress**: closed form vs independent projected-gradient search
  to 1e-6.
- **Peg blindness**: engine flags near-zero-vol FX factors and the
  jump-mixture MC reports the loss HS misses (tested end to end).

Full detail: [`docs/VALIDATION.md`](docs/VALIDATION.md). Model choices and
assumptions: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). Desk usage:
[`docs/DESK_GUIDE.md`](docs/DESK_GUIDE.md).

## Layout

```
fx-var-engine/
├── README.md
├── Cargo.toml
├── src/
│   ├── lib.rs              # crate root, FxVarError, TailDist/McDist, prelude
│   ├── book.rs              # Market, Book, CompiledBook, triangulation
│   ├── returns.rs           # ReturnsMatrix, FactorCov, covariance estimators, peg screen
│   ├── matrix.rs             # dense matrix + robust Cholesky
│   ├── stats.rs               # special functions, sample moments
│   ├── historical.rs           # plain / age / filtered HS
│   ├── parametric.rs            # var-covar, Cornish-Fisher
│   ├── monte_carlo.rs             # normal / t / jump-mixture MC
│   ├── expected_shortfall.rs       # empirical + closed-form VaR/ES
│   ├── backtest.rs                  # Kupiec, Christoffersen, Basel
│   ├── stress.rs                     # canned scenarios, reverse stress
│   ├── rng.rs                         # xoshiro256++ deterministic RNG
│   └── bin/bench.rs                    # 250x50 book timings
├── tests/                                # integration + golden tests
│   ├── edge_cases.rs
│   └── test_golden_python.rs
└── docs/                                   # METHODOLOGY / VALIDATION / DESK_GUIDE
```
