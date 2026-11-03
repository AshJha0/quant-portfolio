# FX VaR & ES Engine (Rust)

A multi-currency FX VaR and ES engine: the Rust port of the Python
package [`python/fx/03-var-es-engine`](../../python/fx/03-var-es-engine/)
(`fx_var`), alongside [`cpp/fx-var-engine`](../../cpp/fx-var-engine/). Books
hold cash, spot and outright forwards and are revalued in full under factor
scenarios. European FX options are repriced by Garman-Kohlhagen in every
historical or Monte Carlo scenario (see
[Option VaR](#option-var-full-revaluation)).

Zero runtime dependencies (`proptest` is a test-only dependency).

## Conventions (changed in 0.2.0)

- **`alpha` is the tail probability** (`0.01` = 99 % VaR), as in every other
  engine in the repository. Until 0.2.0 this crate took a confidence level.
  Passing `0.99` now returns `Err` ("alpha is the tail probability (e.g. 0.01
  for 99 %)"); it is never silently reinterpreted. Defaults are `0.01`.
- Empirical VaR is the order statistic `L(m)`, `m = ceil(n alpha - 1e-9)`;
  ES is Acerbi-Tasche with the exact fractional atom; BRW weights use the
  weighted form of the same rule.
- Kupiec and Basel take a struct with named fields. The Basel zone comes
  from the `Binomial(n, alpha)` CDF for any window; the yellow plus factor
  is read at the equivalent 250-day count.
- EWMA vol is seeded with the mean of `r^2` over the first `min(30, n)` days.
- Monte Carlo jumps are a Poisson count over the horizon.
- Bad input (NaN, infinities, empty, too short, negative where invalid)
  returns `Err`. No library function panics on user input; a proptest walk
  over every public function checks this.

## What it does

| Module | Contents |
| --- | --- |
| `src/book.rs` | Book vs a base currency: cash, spot, CIP forwards. Crosses decompose into USD legs. `CompiledBook` hot path: one `exp()` per leg per scenario. |
| `src/returns.rs` | Factor histories, sample and EWMA covariance, EWMA vol (seed window 30), peg screen. |
| `src/tail.rs` | Shared S2 estimator and alpha validation. |
| `src/psd.rs` | Jacobi eigen, `cholesky_repair(None, Clip, Higham)` with reported jitter. |
| `src/historical.rs` | Plain, BRW, per-factor filtered HS; sqrt-time scaling; peg warnings. |
| `src/parametric.rs` | Variance-covariance VaR/ES (normal, standardised t), Cornish-Fisher with the alpha-aware validity check. |
| `src/monte_carlo.rs` | Normal, covariance-matched t, Poisson jump overlay; one simulation gives VaR, ES, bootstrap SE (primary) and KDE SE (diagnostic). |
| `src/expected_shortfall.rs` | Empirical VaR/ES (weighted or not), closed forms. |
| `src/backtest.rs` | Kupiec, Christoffersen, conditional coverage, Basel (any window), Acerbi-Szekely Z2. |
| `src/stress.rs` | Canned scenarios (Brexit GBP, CHF depeg, JPY 1998), broad USD move, peg break with contagion, reverse stress. |

## Quickstart

From `rust/` (the Cargo workspace root; one `Cargo.lock`, one `target/`):

```sh
cargo test --release --locked -p fx-var-engine
cargo clippy --release --all-targets --locked -p fx-var-engine -- -D warnings
cargo run --release --locked -p fx-var-engine --bin bench -- --out /tmp/bench.json
```

The tests are 71 unit, 62 integration and 8 doc tests. Without `--out`
the bench writes `bench/results.json` in this crate.

The per-path Philox stream with AS 241 normals (`path_rng`), the Acklam guess, the Cholesky kernel and the
fixed-chunk thread helper come from the workspace's [`qp-core`](../core)
crate (std only).

```rust
use fx_var_engine::prelude::*;

let market = Market::new(
    [("EUR", 1.10), ("JPY", 0.0090)],
    [("USD", 0.05), ("EUR", 0.03), ("JPY", 0.001)],
)?;
let book = Book::new(
    vec![
        Position::Spot(SpotPosition::new("EURJPY", 5.0e6, None)),
        Position::Forward(ForwardPosition::new("USDJPY", 10.0e6, 0.5, None)),
    ],
    "USD",
)?;
let hs = historical_var(&book, &market, &returns, &HistoricalOptions::default())?; // alpha 0.01

let mut opts = MonteCarloOptions { dist: McDist::Jump, seed: 42, ..Default::default() };
opts.jumps.prob = 0.02;                              // per-day jump probability
opts.jumps.mean.insert("FX:JPY".to_string(), -0.10); // size per jump
let mc = monte_carlo_var(&book, &market, &sample_cov(&returns)?, &opts)?;
// mc.var, mc.es, mc.se_var (bootstrap), mc.se_var_kde, mc.jitter

let k = kupiec_pof(&KupiecInput { n_obs: 250, n_exceptions: 8, alpha: 0.01 })?;
```

## API changes in 0.2.0 (one-release deprecations)

| Old | New |
|---|---|
| `alpha` = confidence (0.99) everywhere | `alpha` = tail probability (0.01); 0.99 is an error |
| `kupiec_pof(x, n, confidence)` | `kupiec_pof(&KupiecInput { n_obs, n_exceptions, alpha })`; old form `kupiec_pof_confidence` deprecated |
| `basel_traffic_light(x, n, confidence)`, add-on by raw count | `basel_traffic_light(&BaselInput { .. })`, new `plus_factor` field; old form `basel_traffic_light_confidence` deprecated |
| `cornish_fisher_domain_ok(s, k, 4.0, grid)` | `cornish_fisher_valid(s, k, alpha) -> Result<bool>`; old predicate deprecated |
| `Matrix::cholesky_with_jitter(tries)` | `Matrix::cholesky_repair(PsdRepair)`; old method deprecated |
| Bernoulli jump per scenario | Poisson count over the horizon; `prob` must be in `[0, 1)` |
| `MonteCarloResult.se_var` = KDE SE | `se_var` = bootstrap SE; `se_var_kde`, `jitter`, `MonteCarloOptions.n_boot` added |
| `inv_norm_cdf`, `reg_lower_gamma`, `reg_upper_gamma`, `reg_inc_beta`, `ln_gamma`, `student_t_pdf`, `student_t_cdf`, `sample_std`, `flag_peg_factors` returned bare values (some panicked) | return `Result` |
| EWMA vol seeded with full-sample variance | seed window `min(30, n)` |
| new | `acerbi_szekely_z2`, `simulate_factor_returns_jitter`, `psd`, `tail` |

## Benchmark

250 positions (200 spots across 45 currencies, 50 forwards), 50 factors
(45 FX, 5 IR), 500-day history (sinusoid plus seeded noise, full-rank
covariance), full revaluation, release build with `lto = "fat"` and
`codegen-units = 1`. `cargo run --release --bin bench` times each kernel
call by call after a warm-up, cycling through varied inputs, and writes
`bench/results.json` (p50, p90, p99, p99.9, max in ns). The closed-form
kernel is timed in batches of 64 calls and reported per call
(`calls_per_sample` in the JSON).

Numbers from this machine: Intel Core 5 120U (12 threads) under WSL2,
rustc 1.99.1, 2026-11-01. The run waited for the 1-minute load average to
fall below 1.5, but other crates were building in the same VM, so the
tails (p99, p99.9) include scheduler and VM jitter. "Before" is the same
bench on the pre-change code (opt-level 3, no LTO), run right after.

> **How these figures were timed.** Old harness: `std::time::Instant` over a fixed batch per sample (16 or 64 calls for fast kernels, 1 for slow ones). They are one laptop run (Intel Core 5 120U, WSL2) kept for the record; `bench/results.json` holds the same run. Benches now use one method in C++ and Rust: a batch of back-to-back calls sized to at least 2 us per sample, median with p10/p90 ([docs/BENCHMARKS.md](../../docs/BENCHMARKS.md)). Current figures, measured base against head on one CI runner, are in the `bench-results` artifact of the [CI bench job](https://github.com/AshJha0/quant-portfolio/actions/workflows/ci.yml).

| Kernel | Before p50 | After p50 | After p99 | After p99.9 |
|---|---:|---:|---:|---:|
| `CompiledBook::pnl`, one scenario | 1.83 us | 2.03 us | 4.1 us | 10.8 us |
| `portfolio_sigma`, 50 factors | 2.58 us | 3.09 us | 3.4 us | 17 us |
| `normal_var` (per call, batch of 64) | 281 ns | 307 ns | 468 ns | 928 ns |
| `empirical_var_es`, 100k | 240 us | 274 us | 440 us | 570 us |
| `empirical_var_es_into`, 100k | n/a | 228 us | 466 us | 0.99 ms |
| `pnl_aligned_into`, 500 scenarios | n/a | 1.17 ms | 3.19 ms | 4.13 ms |
| `historical_var`, 500 scenarios | 1.25 ms | 1.23 ms | 2.38 ms | 2.71 ms |
| `parametric_var`, normal | 676 us | 595 us | 695 us | 1.23 ms |
| `var_standard_error_bootstrap`, 100k x 200 | 90 ms | 54.5 ms | 74 ms | 74 ms |
| `monte_carlo_var`, 100k normal | 961 ms | 346 ms | 437 ms | 437 ms |
| `monte_carlo_var`, 100k t(5) | 943 ms | 312 ms | 378 ms | 378 ms |

Monte Carlo is 2.8x (normal) to 3.0x (t) faster at p50. The serial RNG
pass bounds the gain. The bootstrap splits its replicates into one
contiguous chunk per thread. A serial pass only advances the generator to
record each chunk's start state, then each thread redraws its own
indices, so the bits match the serial loop. It stays serial below 2^19
index draws (measured crossover, see `BOOT_PAR_MIN_DRAWS`). Small kernels
in this run sit within the run-to-run spread of the before numbers.
Each MC call is one simulation plus 200 bootstrap resamples and the KDE
standard error on the same 100k scenarios.

### Where the time goes and what changed

- Monte Carlo draws its one RNG stream serially, in the reference order,
  into the scenario matrix as raw uniforms. The inverse normal CDF, the
  Cholesky product, the Student-t scale or jump overlay and the
  revaluation then run on a fixed contiguous partition of rows inside
  one `std::thread::scope`. Every row depends only on itself, so the
  result is the same bits for any thread count
  (`tests/test_latency_determinism.rs`).
- The bootstrap draws each batch of resample indices serially, then runs
  the selections in parallel, one replicate per thread.
- The KDE standard error uses two selections instead of a full sort.
- The allocation-free kernels (`empirical_var_es_into`,
  `empirical_var_into`, `CompiledBook::pnl_row`,
  `CompiledBook::pnl_aligned_into`) take a caller workspace and return a
  small `Copy` `KernelError`. `Matrix::quad_form` and so
  `portfolio_sigma` no longer allocate. `tests/test_latency_alloc.rs`
  counts heap allocations with a global allocator.
- No loop in the crate auto-vectorises (`-Cremark=loop-vectorize`). The
  revaluation loop calls `exp()` and gathers factor columns. The dot
  products and sums are ordered reductions, and reordering them would
  change the bits. The inverse CDF calls `ln` and `erfc`. Speed comes
  from threads, not SIMD.

## Validation summary

- `tests/test_conv_golden.rs`: every shared `conv_*` key (S2 VaR/ES, nine
  Basel cases, Kupiec, Z2, Cornish-Fisher flags, Higham and clip repairs,
  EWMA vol), tolerances argued in ulps.
- `tests/test_golden_python.rs`: cases A, B, C against the Python `fx_var`
  reference, now with tail alphas.
- `tests/test_conventions.rs`: regression tests for 0.2.0; each failed on
  the old code first.
- `tests/fuzz_public_api.rs`: NaN, `+/-inf`, empty, length-1 and negative
  inputs for every public function, with a registry test.
- Triangulation, CIP forwards, reverse stress against a numerical search,
  peg-break MC (`tests/edge_cases.rs` and unit tests).

Full detail: [`docs/VALIDATION.md`](docs/VALIDATION.md). Model choices:
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). Desk usage:
[`docs/DESK_GUIDE.md`](docs/DESK_GUIDE.md).

## Option VaR (full revaluation)

`option_reval` prices European options by Garman-Kohlhagen (Black-Scholes-
Merton with the foreign rate as carry), with closed-form Greeks and
full-revaluation and delta-gamma P&L, mirroring `fx_var.revaluation`. The
VaR entry points are `historical::historical_option_var` and
`monte_carlo::monte_carlo_option_var`: the same arguments as `historical_var`
/ `monte_carlo_var` plus `options`, `option_factors` (leg `factor` indexes
this list, e.g. `"FX:EUR"`) and `dt` (years of ageing). The book may be
empty. Options are on CCYUSD with P&L in USD, so a book holding them must
report in USD; historical `horizon_days` must be 1, since square-root-of-time
scaling is wrong for a non-linear book. Options are held at today's implied
vol unless you call the `_vol` entry points (below).

Vol scenario modes. `full_revaluation_pnl_vol_mode(legs, x, dv, n_factors, dt, mode,
skews)` takes `VolMode::StickyStrike` (`sigma + dv`), `StickyDelta`
(`sigma - skew x + dv`, `skews` one per leg or empty) or `HistoricalAtm`
(`sigma e^{dv}`, `dv` a historical log change of ATM vol). `full_revaluation_pnl`
is sticky strike. All three are pinned to the Python reference at
1e-12 relative (`vol_modes_golden` in `tests/test_option_reval.rs`).

The VaR drivers take vol scenarios too. The plain idea: a vol shock is one
more named risk factor, so it sits on the same factor axis as the spot
returns. Formally, the vol factor of `"FX:CCY"` is `"VOL:CCY"` (vol points
for sticky strike and sticky delta, a log change of ATM vol for historical
ATM). The returns panel or covariance must carry `"VOL:CCY"` for every
option factor; the scenario axis is the option layout's factors (book
factors first, then option factors the book lacks) followed by those vol
factors, so Monte Carlo simulates spot and vol jointly with their cross
terms, and filtered HS rescales the vol columns by their own EWMA vol like
any other column. With sticky strike and all-zero vol columns the answer is
the old one bit for bit. Worked example: the EUR call plus short AUD put
test book, with EUR 1m cash and one day of ageing, has 99 % historical VaR
47,005 USD sticky strike, 47,689 sticky delta and 37,268 historical ATM,
against 36,908 with vol held fixed; all three match Python at 1e-11
relative.
Entry points: `historical::historical_option_var_vol`,
`monte_carlo::monte_carlo_option_var_vol`,
`option_reval::option_book_pnl_vol` and `option_combined_factors`; tests in
`tests/test_option_var_vol.rs`.

Parity with Python. The historical numbers are pinned to the Python
reference (`fx_var.revaluation.full_revaluation_pnl` followed by the S2 order
statistic) at 1e-11 relative, for an options-only book and for options plus
a linear book, with and without ageing (`dt = 1/252`). The goldens come from
[`cpp/equity-var-engine/tools/gen_option_var_golden.py`](../../cpp/equity-var-engine/tools/gen_option_var_golden.py),
which builds a closed-form scenario panel so no RNG is shared. Monte Carlo
cannot match numpy draw for draw, so its test uses a case with a known
answer: a long call on one normal factor has P&L monotone in the factor, so
the exact VaR is minus the P&L at sigma z_alpha and the exact ES is a
one-dimensional integral. The engine must land within 4 bootstrap standard
errors (ES within 2 %). A book with no option legs goes down the unchanged
linear path, and the tests check the two give identical bits.
