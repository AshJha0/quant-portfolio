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
│   ├── rng.rs                  # xoshiro256** and SplitMix64 from qp-core + inverse-CDF normal
│   ├── fast.rs                 # low-latency API: Copy errors, workspaces, SoA batch, threaded MC
│   ├── golden.rs                # GENERATED: 434 mpmath golden records
│   └── bin/bench.rs             # per-call latency benchmark -> bench/results.json
├── tools/gen_golden_rs.py      # golden JSON -> src/golden.rs (--check)
├── tests/                       # integration test suite (100 cases) + doctests (20)
└── docs/                        # METHODOLOGY / VALIDATION / DESK_GUIDE
```

## Build, test, benchmark

From `rust/` (the Cargo workspace root; one `Cargo.lock`, one `target/`):

```sh
cargo test --release --locked -p fx-options-engine
cargo clippy --release --all-targets --locked -p fx-options-engine -- -D warnings
cargo run --release --locked -p fx-options-engine --bin bench -- --out /tmp/bench.json
```

Without `--out` the bench writes `bench/results.json` in this crate.

The toolchain is pinned in `rust/rust-toolchain.toml` (1.99.1, edition
2021). No third-party runtime crates. The RNG, the Cody erfc, Brent and the
implied-vol core live in the workspace's own [`qp-core`](../core) crate
(std only) and are re-exported, so public paths are unchanged.

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

`cargo test --release --locked -p fx-options-engine` (from `rust/`) runs
100 integration tests and 20 doctests. On
Windows, Application Control may block test executables (os error 4551);
run them in WSL or Linux.

## Low-latency API

`fx_options_engine::fast` holds the kernels for a hot path. Each returns
the same bits as its classic counterpart; `tests/fast_bits.rs` checks this
with `f64::to_bits` on thousands of varied inputs.

* `gk_price_k`, `greeks_k`, `implied_vol_k`: scalar kernels returning
  `Result<_, KernelError>`. `KernelError` is a one-byte `Copy` enum, so
  neither the success path nor the error path allocates.
* `gk_price_batch`, `greeks_batch`: struct-of-arrays batch pricers over
  slices (`GkBatch` in, `&mut [f64]` or `GreeksBatchOut` out). A
  branch-free scan validates the batch first; a `BatchError` names the
  first bad index.
* `binomial_price_ws`: CRR tree on a reusable `BinomialWorkspace`.
* `mc_price_ws`, `mc_price_par`: Monte Carlo on a reusable `McWorkspace`.
  `mc_price_par` splits the path fill over scoped threads with a fixed
  partition. Each chunk starts its own RNG copy at its offset in the one
  stream, and the reductions stay sequential, so the result equals
  `mc_price` for every thread count.

`tests/fast_zero_alloc.rs` installs a counting global allocator and
asserts zero allocations for each kernel after its workspace is sized.
The classic API is unchanged. Its error type carries a `String`, built
only on the error path.

## Latency

Per-call latency from `cargo run --release --bin bench` (writes
[bench/results.json](bench/results.json)). Numbers from this machine: Intel
Core 5 120U, 12 threads, under WSL2, Rust 1.99.1, release profile
`opt-level = 3`, `lto = "fat"`, `codegen-units = 1`. Other builds shared the
machine during the run (load average about 7), so the tails carry
scheduler jitter. Kernels under 1 us are timed in batches of 64 or 16
calls and reported per call; the batch rows time one 1,024-option batch
and report per option.

| Kernel | p50 ns | p99 ns | p99.9 ns |
|---|---:|---:|---:|
| `gk_price` | 64 | 72 | 332 |
| `fast::gk_price_k` | 61 | 146 | 711 |
| `fast::gk_price_batch`, per option | 59 | 181 | 782 |
| `analytic_greeks` | 48 | 54 | 221 |
| `fast::greeks_k` | 30 | 75 | 627 |
| `fast::greeks_batch`, per option | 67 | 177 | 265 |
| `implied_vol` | 602 | 1,111 | 2,263 |
| `fast::implied_vol_k` | 654 | 1,521 | 17,999 |
| `strike_from_delta`, spot_pa | 1,333 | 1,581 | 6,763 |
| `binomial_price`, American, 500 steps | 77,507 | 101,014 | 186,863 |
| `fast::binomial_price_ws`, same tree | 89,060 | 182,541 | 264,781 |
| `mc_price`, 100k paths, antithetic + CV | 5.79 M | 7.89 M | 7.99 M |
| `fast::mc_price_ws`, same | 7.27 M | 11.42 M | 12.97 M |
| `mc_price`, 1M paths | 73.1 M | 174.2 M | 174.2 M |
| `fast::mc_price_par`, 1M paths, 12 threads | 28.0 M | 35.4 M | 37.9 M |

Read the table this way. The closed-form kernels sit at 30 to 70 ns, and
`fast::` matches the classic call within the noise of this machine. The
gain of the `fast::` API is the contract: no heap use, a `Copy` error, and
caller-owned workspaces. The one large speed-up is the threaded Monte
Carlo, about 2.6 times on 1M paths. The batch loops do not vectorise:
`norm_cdf` (Cody erfc, branchy) and the `exp`/`ln` calls into libm block
it. The validation scan of the batch, the tree's backward induction and
the antithetic pair averages do vectorise (SSE2, width 2), checked with
`RUSTFLAGS=-Cremark=loop-vectorize` and the emitted assembly.

## Docs

* [docs/METHODOLOGY.md](docs/METHODOLOGY.md), why a Rust engine, design
  choices, model assumptions (register lives with the Python reference).
* [docs/VALIDATION.md](docs/VALIDATION.md), cross-language golden
  validation, identity tests, convergence, failure modes, bench table.
* [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md), where this engine sits on an
  FX desk: pricing tiles, risk service, delta-convention pitfalls, golden
  regression as a release gate.
