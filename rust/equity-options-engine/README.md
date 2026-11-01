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
| `rng.rs`           | xoshiro256++ seeded by SplitMix64, Box-Muller normals |

Conventions match the Python reference: continuously compounded annualised
`r`/`q` (ACT/365F), `T` in years; theta per year, vega per unit vol, rho
per unit rate. Errors are `Result<_, PricingError>`; non-finite inputs and
overflowing results are `InvalidInput`.

## Build and test

```sh
cargo test --release                       # 74 integration tests + 28 doctests
cargo clippy --all-targets -- -D warnings
cargo run --release --bin bench
python3 tools/gen_golden_rs.py --check     # src/golden.rs matches the golden JSON
```

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

## Benchmark

`cargo run --release --bin bench`, WSL2 on an Intel Core 5 120U, Rust
1.99.1, single run:

| Kernel | Work | Time | Throughput |
|---|---|---|---|
| Black-Scholes analytic | 1,000,000 prices | 57.7 ms | 17.3M prices/sec |
| Full analytic Greek set | 100,000 evals | 4.7 ms | 21.3M evals/sec |
| CRR tree n=1000, European | 100 trees | 0.09 ms/tree | 10,700 trees/sec |
| CRR tree n=1000, American | 100 trees | 0.20 ms/tree | 5,050 trees/sec |
| MC 1M paths (antithetic + CV) | 1 price + CI | 24.4 ms | 40.9M paths/sec (SE 0.0028) |
| Implied vol (8 quotes) | 100,000 solves | 51.3 ms | 513 ns/solve |

Single runs on a laptop; expect variation of a factor of about 2.

## Documentation

- [docs/METHODOLOGY.md](docs/METHODOLOGY.md): why a Rust engine next to
  Python and C++; design choices; assumptions register.
- [docs/VALIDATION.md](docs/VALIDATION.md): golden and differential
  validation, identities, convergence, failure modes, benchmarks.
- [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md): where a Rust engine could sit
  in a desk workflow; golden-vector regression as a release gate.
