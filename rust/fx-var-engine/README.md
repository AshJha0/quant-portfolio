# FX VaR & ES Engine (Rust)

A multi-currency FX VaR and ES engine: the linear-book port of the Python
package [`python/fx/03-var-es-engine`](../../python/fx/03-var-es-engine/)
(`fx_var`), alongside [`cpp/fx-var-engine`](../../cpp/fx-var-engine/). Books
hold cash, spot and outright forwards and are revalued in full under factor
scenarios. FX options are revalued in the Python package only.

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

```bash
cargo test --release      # 71 unit + 55 integration + 7 doctests
cargo clippy --all-targets -- -D warnings
cargo run --release --bin bench
```

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
covariance), full revaluation, single thread, release build. Measured
2026-11-01 on the development laptop (Windows 11, MSVC toolchain):

```
historical  VaR (500 scen): var=436774 es=488533      1.660 ms
parametric  VaR (normal)  : var=424010 es=485774      0.898 ms
monte carlo VaR 100k normal: var=424401 es=485813 se=2067 (kde 2032)     2525.3 ms
monte carlo VaR 100k t(5)  : var=473404 es=620076 se=4082 (kde 3722)     6325.8 ms
```

Each MC line is one simulation plus 200 bootstrap resamples of 100k
scenarios. MC normal VaR is within one bootstrap SE of the parametric value
(the book is near-linear at this horizon).

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
