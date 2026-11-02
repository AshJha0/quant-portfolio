# eq-options-engine (Rust)

Rust port of the Python reference package
[`eq_options`](../../python/equity/01-options-pricing/) and sibling of the
[C++ engine](../../cpp/equity-options-engine/): the same models and
conventions, checked against golden vectors computed in mpmath and against
the Python and C++ engines on seeded random inputs. No runtime
dependencies (proptest is a test-only dependency).

## Models

| Module (`src/`) | Contents |
|---|---|
| `black_scholes.rs` | BSM price with continuous dividend yield; `d1`/`d2`; Cody erfc and erf, erfc-based normal CDF; `T=0` and overflow-safe `sigma=0` limits |
| `greeks.rs`        | Analytic delta, gamma, vega, theta, rho, vanna, volga; generic central-FD Greeks over any pricer closure |
| `implied_vol.rs`   | Out-of-the-money-side solver on normalised Black (`iv_core.rs`, same algorithm as Python and C++): at most 10 evaluations, about 0.5 µs per solve; rejects quotes at or outside the bounds as `ArbitrageBound` |
| `binomial.rs`      | CRR tree, European and American, O(n) memory, exercise values precomputed per level |
| `black76.rs`       | Options on forwards/futures: price and Greeks; equals BSM at `F = S e^{(r-q)T}` |
| `monte_carlo.rs`   | GBM exact scheme; explicit `u64` seed; antithetic pairs and a control variate fitted on pair averages; `McResult{price, std_error, ci_low, ci_high}` |
| `rng.rs`           | re-exports xoshiro256++ (SplitMix64 seeding, Box-Muller normals) from `qp-core` |

Conventions match the Python reference: continuously compounded annualised
`r`/`q` (ACT/365F), `T` in years; theta per year, vega per unit vol, rho
per unit rate. Errors are `Result<_, PricingError>`; non-finite inputs and
overflowing results are `InvalidInput`.

## Build and test

From `rust/` (the Cargo workspace root; one `Cargo.lock`, one `target/`):

```sh
cargo test --release --locked -p eq-options-engine
cargo clippy --release --all-targets --locked -p eq-options-engine -- -D warnings
cargo run --release --locked -p eq-options-engine --bin bench -- --out /tmp/bench.json
```

The tests are 81 integration tests and 32 doctests. Without `--out` the
bench writes `bench/results.json` in this crate.

From this directory, `python3 tools/gen_golden_rs.py --check` checks that
`src/golden.rs` matches the golden JSON.

The RNG, Cody erfc, normal CDF, Acklam guess and the implied-vol core live
in [`rust/core`](../core) (`qp-core`, std only) and are re-exported here;
public paths such as `eq_options_engine::rng::Xoshiro256PlusPlus` and
`eq_options_engine::black_scholes::erfc` are unchanged.

On Windows, Application Control may block the doctest executables
(os error 4551); run the doctests in WSL or Linux.

## Validation

- **Golden vectors:** 456 mpmath records (50 digits) from the Python
  project, each output with its own tolerance; worst deviation 0.175 of
  tolerance. See [docs/VALIDATION.md](docs/VALIDATION.md).
- **Differential test:** `src/bin/eq_jsonl.rs` lets the Python test
  `tests/test_differential.py` drive this crate with 640 seeded random
  requests; largest deviation from Python 1.1e-13 on prices, 7.1e-15 on
  implied vol.
- **Property tests** (proptest): parity, monotone in vol and strike,
  strike convexity, delta bounds, implied-vol round trip.

## Latency

Numbers from this machine: Intel Core 5 120U (12 threads), WSL2 on
Windows 11, Rust 1.99.1, release profile with `lto = "fat"` and
`codegen-units = 1`. Command: `cargo run --release --bin bench`, which
writes [bench/results.json](bench/results.json). "Before" is the same
bench source built against the previous version of the crate with
`RUSTFLAGS="--cfg before"`. Both runs went back to back on one quiet
period. Rows with 16 calls per sample report 16-call averages, because a
single call is close to the timer resolution under WSL2. Batch rows
report one 1,024-row batch divided by 1,024.

| Kernel | Calls per sample | Before: p50 / p99 / p99.9 | After: p50 / p99 / p99.9 |
|---|---|---|---|
| `bs_price` | 16 | 55 ns / 146 ns / 609 ns | 44 ns / 116 ns / 576 ns |
| `bs_greeks` | 16 | 46 ns / 136 ns / 814 ns | 46 ns / 71 ns / 193 ns |
| `black76_price` | 16 | 44 ns / 91 ns / 397 ns | 38 ns / 82 ns / 107 ns |
| `implied_vol` | 1 | 522 ns / 1.13 us / 2.40 us | 507 ns / 1.21 us / 1.75 us |
| `crr_price n=100 european` | 1 | 1.86 us / 2.23 us / 13 us | 1.46 us / 1.88 us / 8.63 us |
| `crr_price n=100 american` | 1 | 5.75 us / 16 us / 62 us | 5.14 us / 13 us / 55 us |
| `crr_price n=1000 european` | 1 | 86 us / 245 us / 592 us | 95 us / 206 us / 483 us |
| `crr_price n=1000 american` | 1 | 468 us / 1.24 ms / 1.61 ms | 396 us / 777 us / 1.42 ms |
| `mc_price 10k paths anti+cv` | 1 | 120 us / 316 us / 576 us | 111 us / 228 us / 339 us |
| `mc_price 1M paths anti+cv` | 1 | 24.89 ms / 33.02 ms / 33.02 ms | 16.16 ms / 24.82 ms / 24.82 ms |
| `try_bs_price` | 16 | n/a (new API) | 42 ns / 98 ns / 140 ns |
| `try_bs_greeks` | 16 | n/a (new API) | 44 ns / 113 ns / 556 ns |
| `try_implied_vol` | 1 | n/a (new API) | 486 ns / 1.02 us / 1.55 us |
| `bs_price_batch (per row)` | 1024 | n/a (new API) | 38 ns / 54 ns / 100 ns |
| `bs_greeks_batch (per row)` | 1024 | n/a (new API) | 48 ns / 102 ns / 196 ns |
| `try_crr_price n=100 european` | 1 | 1.86 us / 2.23 us / 13 us | 1.92 us / 4.29 us / 27 us |
| `try_crr_price n=100 american` | 1 | 5.75 us / 16 us / 62 us | 7.17 us / 22 us / 192 us |
| `try_crr_price n=1000 european` | 1 | 86 us / 245 us / 592 us | 152 us / 422 us / 2.22 ms |
| `try_crr_price n=1000 american` | 1 | 468 us / 1.24 ms / 1.61 ms | 775 us / 2.00 ms / 4.04 ms |
| `try_mc_price 10k paths anti+cv 1 thread` | 1 | 120 us / 316 us / 576 us | 196 us / 502 us / 828 us |
| `try_mc_price 1M paths anti+cv 1 thread` | 1 | 24.89 ms / 33.02 ms / 33.02 ms | 20.89 ms / 27.87 ms / 27.87 ms |
| `try_mc_price 1M paths anti+cv 8 threads` | 1 | 24.89 ms / 33.02 ms / 33.02 ms | 11.16 ms / 14.87 ms / 14.87 ms |

WSL2 adds scheduler jitter, so read p99.9 as an upper bound. The rows
run in order, and the machine load rose during the run: the `try_crr`
and `try_mc` rows run the same kernels as the classic rows above them
(the classic call now builds a workspace and calls them), so their
higher numbers are drift, not a regression. The 8-thread Monte Carlo
draws the raw stream serially and runs Box-Muller and payoffs on a
fixed partition; it returns the same bits as one thread.

## Low-latency API

`eq_options_engine::fast` holds the entry points for latency-sensitive
callers. Results equal the classic API bit for bit (tested with
`to_bits` in `tests/fast_equivalence.rs`).

- `try_bs_price`, `try_bs_greeks`, `try_implied_vol`,
  `try_black76_price`, `try_black76_greeks` return
  `Result<_, ErrorCode>`. `ErrorCode` is a one-byte `Copy` enum, so no
  path builds a `String`. It converts into `PricingError` with `.into()`.
- `try_crr_price` takes a `CrrWorkspace`; `try_mc_price` takes a
  `McWorkspace` and a thread count (0 is rejected as
  `ErrorCode::BadThreads`, as in the C++ engine). A sized workspace makes repeated
  calls heap-free (`tests/zero_alloc.rs` counts allocations).
- `bs_price_batch` and `bs_greeks_batch` price struct-of-arrays slices
  (`BsBatch`, `GreeksOut`) and report the first failing row.

The batch loops do not auto-vectorise: each row calls `exp`, `ln` and
Cody's `erfc`, which LLVM cannot widen (checked with
`RUSTFLAGS=-Cremark=loop-vectorize`). The Monte Carlo sums stay scalar
because reordering them would change the bits. A few elementwise loops
(buffer fills, the control-variate adjustment) vectorise at width 2.

## Documentation

- [docs/METHODOLOGY.md](docs/METHODOLOGY.md): why a Rust engine next to
  Python and C++; design choices; assumptions register.
- [docs/VALIDATION.md](docs/VALIDATION.md): golden and differential
  validation, identities, convergence, failure modes, benchmarks.
- [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md): where a Rust engine could sit
  in a desk workflow; golden-vector regression as a release gate.

## Ports of the newer models (November 2026)

Leisen-Reimer tree, Bachelier price/Greeks/implied normal vol, and charm,
speed, colour, zomma, veta: `src/lr_tree.rs`, `src/bachelier.rs`, `src/greeks_higher.rs` (unit tests in each module). They match the Python reference
golden values to 1e-12 relative (1e-13 for Bachelier). The other newer
models are Python only for now.
