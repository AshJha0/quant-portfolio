# FX Options Pricing Engine (Rust)

Rust port of the Python reference project
[`python/fx/01-options-pricing`](../../python/fx/01-options-pricing/) and
sibling of the [C++ engine](../../cpp/fx-options-engine/), with no runtime
dependencies (proptest is test-only). It is checked against golden vectors
computed in mpmath and against the Python and C++ engines on seeded random
inputs.

Everything prices European (and, in the tree, American) vanilla FX options
under Garman–Kohlhagen: Black–Scholes with the continuous dividend yield
replaced by the foreign interest rate.

## Conventions

* Pairs are quoted **BASE/QUOTE** (EURUSD = USD per 1 EUR).
* `s` is spot in domestic (quote) currency per unit foreign (base) currency;
  all premiums are domestic currency per unit of foreign notional.
* `r_d` = domestic (quote-ccy) rate, `r_f` = foreign (base-ccy) rate, both
  continuously compounded, annualised, ACT/365F. Negative rates are legal.
* Theta is per year; vega per unit of vol.
* Invalid inputs return `Err(FxError::InvalidInput)` (the mirror of the
  Python `ValueError` contract), no panics on bad market data.

### Validation contract

Every public entry point rejects **NaN and +/-infinity in every float
argument**, spot, strike, tenor, both rates, vol, the observed premium
passed to `implied_vol`, the `pip_factor` of `forward_points`, the
`rel_bump` of `finite_difference_greeks`, and the delta/premium arguments
of the delta-conversion helpers. Validation is written with `is_finite()`,
never with an ordering comparison: `if x <= 0.0 { return Err(..) }`
silently **accepts** NaN, because every IEEE-754 comparison against NaN is
false, and Rust is no different from C here (`f64::NAN < 0.0` is `false`).
That matters operationally rather than aesthetically, a NaN price or a
NaN delta breaches no limit and colours no traffic light (`NaN <= limit`
is false, and so is `NaN > limit`), so it aggregates into a book number
unseen. The contract is pinned by
`tests/edge_cases.rs::every_entry_point_rejects_non_finite_inputs`, which
poisons each float slot of each public function in turn.

Two further guards worth knowing about:

* `finite_difference_greeks` requires `T > 0`, `sigma > 0` and a finite
  positive `rel_bump`, otherwise the theta bump `min(1e-6, T/4)` or the
  spot bump `S * rel_bump` collapses to exactly zero and the difference
  quotient becomes an unreported `0/0`.
* `mc_price` needs two independent samples (an antithetic pair is one),
  three with the control variate; fewer return `InvalidInput`.
* Finite inputs whose result overflows (a forward or discount factor
  beyond f64 range) return `InvalidInput`, never `Ok(inf)` or `Ok(NaN)`.

## Layout

```
fx-options-engine/
├── Cargo.toml                # edition 2021, zero runtime dependencies
├── src/
│   ├── lib.rs                 # OptionType, FxError, validation, N(x)/n(x)/N^-1(x), Brent
│   ├── garman_kohlhagen.rs    # d1/d2, gk_price/gk_call/gk_put, limits
│   ├── forwards.rs            # CIP forward, forward points, synthetic forward
│   ├── black76.rs             # Black-76 on the FX forward (== GK on CIP F)
│   ├── deltas.rs               # 4 delta conventions, strike-from-delta, ATM/DNS
│   ├── greeks.rs               # analytic Greeks (2 rhos, vanna, volga) + generic FD
│   ├── implied_vol.rs          # bounds checks + shared core (iv_core.rs), at most 10 evaluations
│   ├── binomial.rs             # CRR tree, foreign rate as yield, European + American
│   ├── monte_carlo.rs          # GBM under domestic measure, xoshiro256**, VR
│   ├── rng.rs                  # SplitMix64 seed expansion + xoshiro256** + inverse-CDF normal
│   ├── golden.rs                # GENERATED: 434 mpmath golden records
│   └── bin/bench.rs             # throughput benchmarks
├── tools/gen_golden_rs.py      # golden JSON -> src/golden.rs (--check)
├── tests/                       # integration test suite (93 cases) + doctests (19)
└── docs/                        # METHODOLOGY / VALIDATION / DESK_GUIDE
```

## Build, test, benchmark

```sh
cargo build --release
RUSTFLAGS="-D warnings" cargo test --release   # 93 integration tests + 19 doctests
cargo run --release --bin bench
```

Requires only a stable Rust toolchain (edition 2021). No external crates -
the deterministic RNG, normal-distribution kernel and root finders (Brent,
the Halley-based implied-vol core) are all implemented in this crate on `std` alone.

## Quick example

```rust
use fx_options_engine::{gk_price, analytic_greeks, strike_from_delta,
                        DeltaConvention, OptionType};

// EURUSD 6m call: S=1.10, K=1.12, r_d(USD)=4.25%, r_f(EUR)=2.90%, vol 9.25%
let px = gk_price(1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925,
                  OptionType::Call).unwrap();
let g = analytic_greeks(1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925,
                        OptionType::Call).unwrap();

// 25-delta strike under the premium-adjusted spot convention (Brent solve
// on the market-standard high-strike branch):
let k25 = strike_from_delta(0.25, 1.10, 0.5, 0.0425, 0.0290, 0.0925,
                            OptionType::Call, DeltaConvention::SpotPa)
    .unwrap();
```

## Why Rust, and why zero dependencies

A Rust port has the properties the C++ engine has (closed-form pricing at
millions of quotes per second, no interpreter, no GC pause) plus compile-time
memory safety. The normal-distribution kernel, root finders and RNG are
small numerical primitives the crate owns, so there is no runtime
dependency surface to review. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Validation headline

- **Golden vectors:** 434 records computed in mpmath at 50 digits (GK
  price and Greeks incl. both rhos, vanna, volga and all four delta
  conventions; implied vol; strike-from-delta in all four conventions;
  DNS ATM; CRR; forwards), each output within its own tolerance; worst
  deviation 0.186 of tolerance.
- **Differential test:** `src/bin/fx_jsonl.rs` lets the Python test drive
  this crate with 790 seeded random requests; largest deviation from
  Python 5.7e-14 on prices, 6.8e-13 on strikes, 4.1e-14 on implied vol.
- **Property tests** (proptest) and the identity, convergence and
  non-finite-rejection suites: see [docs/VALIDATION.md](docs/VALIDATION.md).

`cargo test --release` runs 93 integration tests and 19 doctests. On
Windows, Application Control may block test executables (os error 4551);
run them in WSL or Linux.

## Benchmark results

`cargo run --release --bin bench`, WSL2 on an Intel Core 5 120U, Rust
1.99.1, single run:

| Benchmark | Throughput |
|---|---|
| Garman-Kohlhagen vanilla price | 16.9 M prices/s (1M prices in 0.059 s) |
| Strike-from-delta, spot / forward (analytic `N^-1`) | 13.6 - 14.0 M solves/s |
| Strike-from-delta, premium-adjusted (Brent, branch-aware) | 1.1 M solves/s |
| Monte Carlo, antithetic + control variate | 19.6 M paths/s (1M paths in 0.051 s, SE 9.4e-6) |
| Implied vol (8 quotes) | 442 ns per solve |
| CRR American, 2000 steps | 0.99 ms |

Single runs on a laptop; expect variation of a factor of about 2.

## Docs

* [docs/METHODOLOGY.md](docs/METHODOLOGY.md), why a Rust engine, design
  choices, model assumptions (register lives with the Python reference).
* [docs/VALIDATION.md](docs/VALIDATION.md), cross-language golden
  validation, identity tests, convergence, failure modes, bench table.
* [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md), where this engine sits on an
  FX desk: pricing tiles, risk service, delta-convention pitfalls, golden
  regression as a release gate.
