# Equity Options Pricing Engine (C++20)

C++ port of the Python reference package
[`eq_options`](../../python/equity/01-options-pricing/): the same models and
conventions, checked against golden vectors computed in mpmath and against
the Python and Rust engines on seeded random inputs. It runs about 12
million analytic prices per second on one core of the test machine and has
no dependencies beyond the standard library (GoogleTest for tests only).

## Models

| Header (`include/eqopt/`) | Contents |
|---|---|
| `black_scholes.hpp` | European call/put with continuous dividend yield `q`; `d1`/`d2`; erfc-based normal CDF; `T=0` and `sigma=0` limits (the latter overflow-safe) |
| `greeks.hpp`        | Analytic delta, gamma, vega, theta, rho, vanna, volga; templated central finite-difference Greeks for any pricer |
| `implied_vol.hpp`   | Out-of-the-money-side solver on normalised Black (shared core `iv_core.hpp`, same algorithm as Python and Rust): at most 10 evaluations, about 0.8 µs per solve |
| `binomial.hpp`      | CRR tree, European and American, dividend yield, O(n) memory, exercise values precomputed per level |
| `black76.hpp`       | Options on forwards/futures: price and Greeks; equals BSM at `F = S e^{(r-q)T}` |
| `monte_carlo.hpp`   | GBM exact scheme; `std::mt19937_64` explicit seed; antithetic pairs and a control variate fitted on pair averages; `{price, std_error, ci_low, ci_high}` |

Conventions match the Python reference: continuously compounded annualised
`r`/`q` (ACT/365F), `T` in years, `sigma` annualised; theta per **year**,
vega per **unit** of vol, rho per **unit** of rate. Invalid inputs (negative
`S`, `K`, `T`, `sigma`; NaN or +/-Inf anywhere, including `r` and `q`) and
results that overflow throw `std::invalid_argument`.

## Build, test, benchmark

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # add -DEQOPT_WERROR=ON for -Werror
cmake --build build -j
ctest --test-dir build --output-on-failure        # 66 tests
./build/eqopt_bench
```

C++20, CMake >= 3.20. GoogleTest is found with `find_package(GTest)` and
fetched (v1.15.2) when absent. Warnings: `-Wall -Wextra -Wpedantic` on
GCC/Clang, `/W4` on MSVC. Built and tested with GCC 15.2, Clang 21.1 and a
GCC ASan+UBSan build (66/66 each).

## Validation

- **Golden vectors:** `tools/gen_golden_header.py` turns the Python
  project's `tests/golden/golden_vectors.json` (456 records computed in
  mpmath at 50 digits: BSM price and Greeks incl. vanna/volga, Black-76,
  implied vol, CRR European/American, edge limits) into
  `tests/golden_vectors.hpp`. Every output must lie within its own
  tolerance; the worst deviation is 0.18 of tolerance. `--check` verifies
  the header matches the JSON.
- **Differential test:** `tools/jsonl_main.cpp` builds `eqopt_jsonl`, a
  JSON-lines front end that the Python test `tests/test_differential.py`
  drives with 640 seeded random requests against Python and Rust. Largest
  deviation from Python: 1.1e-13 on prices, 2.1e-13 on implied vol.
- Put-call parity, Black-76 = BSM, CRR convergence, American ordering,
  Monte Carlo within 3 SE, analytic vs FD Greeks, NaN-free domain scan:
  see [docs/VALIDATION.md](docs/VALIDATION.md).

## Benchmark

`./build/eqopt_bench`, GCC 15.2 `-O3` (CMake Release), WSL2 on an Intel
Core 5 120U, single run:

| Kernel | Work | Time | Throughput |
|---|---|---|---|
| Black-Scholes analytic | 1,000,000 prices | 85.5 ms | 11.7M prices/sec |
| Full analytic Greek set | 100,000 evals | 7.9 ms | 12.7M evals/sec |
| CRR tree, n=1000, European | 1 tree | 0.10 ms | 9,600 trees/sec |
| CRR tree, n=1000, American | 1 tree | 0.61 ms | 1,600 trees/sec |
| Monte Carlo, 1M paths (antithetic+CV), 1 thread | 1 price + CI | 49.2 ms | 20.3M paths/sec |
| Monte Carlo, 1M paths, 4 threads | 1 price + CI | 25.2 ms | 39.6M paths/sec |
| Implied vol | 100,000 solves | 81.4 ms | 814 ns/solve |

Single runs on a laptop; numbers vary by a factor of about 2 between runs.

## Layout

```
equity-options-engine/
├── CMakeLists.txt             # C++20, per-compiler warnings, EQOPT_WERROR, GTest
├── include/eqopt/             # public headers (iv_core.hpp: shared IV core)
├── src/                       # implementations
├── tools/gen_golden_header.py # golden JSON -> constexpr header (--check)
├── tools/jsonl_main.cpp       # eqopt_jsonl shim for the differential test
├── tests/                     # GoogleTest suite + generated golden_vectors.hpp
├── bench/bench_main.cpp       # chrono micro-benchmark
└── docs/                      # METHODOLOGY, VALIDATION, DESK_GUIDE
```

## Documentation

- [docs/METHODOLOGY.md](docs/METHODOLOGY.md): why a C++ engine alongside
  Python; design choices; assumptions register.
- [docs/VALIDATION.md](docs/VALIDATION.md): golden and differential
  validation, identities, convergence, failure behaviour, benchmarks.
- [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md): where a C++ engine sits in a
  desk workflow; golden-vector regression as a release gate.
