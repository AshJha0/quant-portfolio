# Equity Options Pricing Engine (C++23)

C++ port of the Python reference package
[`eq_options`](../../python/equity/01-options-pricing/): the same models and
conventions, checked against golden vectors computed in mpmath and against
the Python and Rust engines on seeded random inputs. Every kernel has a
`noexcept` entry point that returns `std::expected` and makes no heap
allocation; the original throwing functions are thin wrappers over it. No
dependencies beyond the standard library (GoogleTest for tests only).

## Models

| Header (`include/eqopt/`) | Contents |
|---|---|
| `black_scholes.hpp` | European call/put with continuous dividend yield `q`; `d1`/`d2`; erfc-based normal CDF; `T=0` and `sigma=0` limits (the latter overflow-safe) |
| `greeks.hpp`        | Analytic delta, gamma, vega, theta, rho, vanna, volga; templated central finite-difference Greeks for any pricer |
| `implied_vol.hpp`   | Out-of-the-money-side solver on normalised Black (shared core `iv_core.hpp`, same algorithm as Python and Rust): at most 10 evaluations, about 0.9 µs per solve |
| `binomial.hpp`      | CRR tree, European and American, dividend yield, O(n) memory, exercise values precomputed per level |
| `black76.hpp`       | Options on forwards/futures: price and Greeks; equals BSM at `F = S e^{(r-q)T}` |
| `monte_carlo.hpp`   | GBM exact scheme; `std::mt19937_64` explicit seed; antithetic pairs and a control variate fitted on pair averages; `{price, std_error, ci_low, ci_high}`; `McWorkspace` for allocation-free reuse |
| `batch.hpp`         | Structure-of-arrays batch prices and Greeks over `std::span`; bit-equal to the scalar kernels |
| `thread_pool.hpp`   | Fixed-size persistent worker pool for Monte Carlo chunks; `default_thread_pool()` |
| `error.hpp`         | `Error` (code, field, values), `Result<T> = std::expected<T, Error>`, `message()` |

Conventions match the Python reference: continuously compounded annualised
`r`/`q` (ACT/365F), `T` in years, `sigma` annualised; theta per **year**,
vega per **unit** of vol, rho per **unit** of rate. Invalid inputs (negative
`S`, `K`, `T`, `sigma`; NaN or +/-Inf anywhere, including `r` and `q`) and
results that overflow are errors: the `try_*` kernels return an `Error`, the
throwing functions raise `std::invalid_argument` with the same text.

## Low-latency API

```cpp
#include "eqopt/batch.hpp"
#include "eqopt/binomial.hpp"
#include "eqopt/monte_carlo.hpp"

// Analytic: noexcept, no allocation, error as a value.
eqopt::Result<double> p = eqopt::try_bs_price(S, K, T, r, sigma, q, eqopt::OptionType::Put);
if (!p) log(eqopt::message(p.error()));   // message() allocates; error path only

// Trees and Monte Carlo: size buffers and threads once, reuse per call.
std::vector<double> tree_ws(eqopt::crr_workspace_size(1000));
auto am = eqopt::try_crr_price(S, K, T, r, sigma, q, type,
                               eqopt::ExerciseStyle::American, 1000, tree_ws);
eqopt::ThreadPool pool(3);                // 3 workers + the calling thread
eqopt::McWorkspace mc_ws;
mc_ws.reserve(1'000'000, /*control_variate=*/true);
auto mc = eqopt::try_mc_price(S, K, T, r, sigma, q, type, 1'000'000, true, true,
                              /*seed=*/42, /*threads=*/4, mc_ws, &pool);

// Batch: one span per field, one output span per Greek.
auto ok = eqopt::bs_price_batch({S_span, K_span, T_span, r_span, sigma_span, q_span, type_span},
                                out_span);
```

For a fixed `(seed, threads)` the Monte Carlo result does not depend on the
pool, and it equals the throwing `mc_price` bit for bit. See
[docs/METHODOLOGY.md](docs/METHODOLOGY.md) section 2 for the design.

## Build, test, benchmark

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # add -DEQOPT_WERROR=ON for -Werror
cmake --build build -j
ctest --test-dir build --output-on-failure        # 89 tests
./build/eqopt_bench --out results.json            # per-call latency, JSON
```

C++23 (`std::expected`), CMake >= 3.25. GoogleTest is found with
`find_package(GTest)` and fetched (v1.15.2) when absent. Warnings:
`-Wall -Wextra -Wpedantic` on GCC/Clang, `/W4` on MSVC. Options:

| Option | Default | Effect |
|---|---|---|
| `EQOPT_WERROR` | OFF | warnings are errors (CI sets ON) |
| `EQOPT_TESTS` | ON | build the GoogleTest suite |
| `EQOPT_NATIVE` | OFF | `-O3 -march=native` on every target; local latency work only, never CI |
| `EQOPT_LTO` | OFF | link-time optimisation via `CMAKE_INTERPROCEDURAL_OPTIMIZATION` |
| `EQOPT_VEC_REPORT` | OFF | print the vectorisation report for `batch.cpp`, `binomial.cpp`, `monte_carlo.cpp` |

`CMAKE_CXX_FLAGS` is never modified, so sanitizers pass through it.
Measured on 2026-11-01 with `EQOPT_WERROR=ON`:

| Build | Tests |
|---|---|
| GCC 15.2 Release | 82/82 |
| Clang 21.1 Release | 82/82 |
| GCC 15.2 ASan+UBSan | 77/77 |
| Clang 21.1 ASan+UBSan | 77/77 |
| Clang 21.1 TSan | 77/77 |

The sanitizer builds skip `eqopt_alloc_tests` (5 tests): it replaces the
global `operator new`, which the sanitizer runtimes also define.

## Shared core, presets and C++26

The normal cdf and inverse, the implied-vol core, splitmix64 seeding and the thread pool come from `qp::core`, the
header-only library in `cpp/core` (see `cpp/core/README.md`). This engine
keeps its own headers and API; moved names are using-declarations. The
CMake file adds the core with `add_subdirectory(../core)` unless a parent
project already did.

`CMakePresets.json` defines the presets `gcc`, `clang`, `gcc16-cxx26`,
`asan`, `tsan` and `msvc`. From this directory:

```bash
cmake --preset clang
cmake --build --preset clang
ctest --preset clang
```

Build trees go to `build/<preset>`. The presets turn `EQOPT_WERROR` on. `msvc`
must run inside a Visual Studio developer prompt. `QP_CXX26=ON` (the
`gcc16-cxx26` preset) compiles with C++26 and uses contracts and
`std::simd` where the compiler reports them; results are the same as the
C++23 build. CMake 3.25 or newer is required.

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

## Latency

Per-call latency, `./build/eqopt_bench --out results.json` (raw records for
every run, both compilers, before and after, are in
[bench/results.json](bench/results.json)). Sizes and inputs are in
`bench/bench_main.cpp`; every call gets a different input from a 16,384-entry
table.

> **How these figures were timed.** Old harness: each call timed on its own with rdtscp, empty-call overhead subtracted. They are one laptop run (Intel Core 5 120U, WSL2) kept for the record; `bench/results.json` holds the same run. Benches now use one method in C++ and Rust: a batch of back-to-back calls sized to at least 2 us per sample, median with p10/p90 ([docs/BENCHMARKS.md](../../docs/BENCHMARKS.md)). Current figures, measured base against head on one CI runner, are in the `bench-results` artifact of the [CI bench job](https://github.com/AshJha0/quant-portfolio/actions/workflows/ci.yml).

| Kernel | Compiler | p50 before | p50 after | p99 before | p99 after | p99.9 after | max after |
|---|---|---|---|---|---|---|---|
| `bs_price` | GCC 15.2 | 129 ns | 135 ns | 379 ns | 385 ns | 686 ns | 462.6 µs |
| `bs_price` | Clang 21.1 | 115 ns | 109 ns | 297 ns | 306 ns | 503 ns | 90.3 µs |
| `bs_greeks` | GCC 15.2 | 129 ns | 113 ns | 358 ns | 315 ns | 583 ns | 155.5 µs |
| `bs_greeks` | Clang 21.1 | 104 ns | 117 ns | 286 ns | 304 ns | 500 ns | 128.8 µs |
| `black76_price` | GCC 15.2 | 123 ns | 107 ns | 323 ns | 297 ns | 627 ns | 208.0 µs |
| `black76_price` | Clang 21.1 | 117 ns | 108 ns | 300 ns | 290 ns | 518 ns | 451.7 µs |
| `implied_vol` | GCC 15.2 | 941 ns | 889 ns | 2.0 µs | 2.0 µs | 10.1 µs | 1.10 ms |
| `implied_vol` | Clang 21.1 | 882 ns | 915 ns | 1.9 µs | 1.8 µs | 19.0 µs | 790.5 µs |
| `crr_price eu n=1000` | GCC 15.2 | 124.5 µs | 124.2 µs | 471.6 µs | 471.6 µs | 1.18 ms | 2.65 ms |
| `crr_price eu n=1000` | Clang 21.1 | 116.5 µs | 125.8 µs | 204.7 µs | 263.0 µs | 701.9 µs | 1.01 ms |
| `crr_price am n=1000` | GCC 15.2 | 631.1 µs | 681.7 µs | 1.71 ms | 2.45 ms | 3.26 ms | 4.76 ms |
| `crr_price am n=1000` | Clang 21.1 | 422.9 µs | 435.0 µs | 1.09 ms | 1.10 ms | 1.48 ms | 1.78 ms |
| `mc_price 10k paths 1 thr` | GCC 15.2 | 412.2 µs | 399.2 µs | 1.22 ms | 1.11 ms | 1.74 ms | 2.31 ms |
| `mc_price 10k paths 1 thr` | Clang 21.1 | 382.2 µs | 438.3 µs | 1.20 ms | 1.32 ms | 1.62 ms | 2.04 ms |
| `mc_price 10k paths 4 thr` | GCC 15.2 | 583.8 µs | 548.9 µs | 1.36 ms | 880.1 µs | 1.21 ms | 1.52 ms |
| `mc_price 10k paths 4 thr` | Clang 21.1 | 543.2 µs | 562.2 µs | 1.38 ms | 1.10 ms | 1.41 ms | 1.56 ms |
| `mc_price 1M paths 4 thr` | GCC 15.2 | 33.62 ms | 32.31 ms | 42.91 ms | 37.13 ms | 37.13 ms | 39.06 ms |
| `mc_price 1M paths 4 thr` | Clang 21.1 | 30.63 ms | 32.97 ms | 35.73 ms | 41.49 ms | 41.49 ms | 42.40 ms |
| `try_bs_price` | GCC 15.2 | - | 133 ns | - | 361 ns | 730 ns | 782.5 µs |
| `try_bs_price` | Clang 21.1 | - | 128 ns | - | 353 ns | 726 ns | 420.5 µs |
| `try_bs_greeks` | GCC 15.2 | - | 117 ns | - | 318 ns | 666 ns | 849.8 µs |
| `try_bs_greeks` | Clang 21.1 | - | 116 ns | - | 331 ns | 776 ns | 633.1 µs |
| `try_implied_vol` | GCC 15.2 | - | 1.1 µs | - | 2.7 µs | 31.4 µs | 1.02 ms |
| `try_implied_vol` | Clang 21.1 | - | 939 ns | - | 1.8 µs | 11.4 µs | 861.5 µs |
| `try_crr_price am n=1000 (ws)` | GCC 15.2 | - | 681.0 µs | - | 1.98 ms | 3.36 ms | 5.77 ms |
| `try_crr_price am n=1000 (ws)` | Clang 21.1 | - | 446.9 µs | - | 1.20 ms | 1.88 ms | 2.22 ms |
| `try_mc_price 10k paths 4 thr (pool)` | GCC 15.2 | - | 429.5 µs | - | 1.01 ms | 1.26 ms | 1.29 ms |
| `try_mc_price 10k paths 4 thr (pool)` | Clang 21.1 | - | 422.2 µs | - | 1.11 ms | 1.31 ms | 1.59 ms |
| `try_mc_price 1M paths 4 thr (pool)` | GCC 15.2 | - | 23.95 ms | - | 29.75 ms | 29.75 ms | 30.50 ms |
| `try_mc_price 1M paths 4 thr (pool)` | Clang 21.1 | - | 26.44 ms | - | 35.61 ms | 35.61 ms | 37.40 ms |
| `bs_price_batch x1024` | GCC 15.2 | - | 95.4 µs | - | 204.8 µs | 451.2 µs | 1.11 ms |
| `bs_price_batch x1024` | Clang 21.1 | - | 116.7 µs | - | 418.5 µs | 1.06 ms | 2.84 ms |
| `bs_greeks_batch x1024` | GCC 15.2 | - | 95.4 µs | - | 209.8 µs | 852.2 µs | 1.23 ms |
| `bs_greeks_batch x1024` | Clang 21.1 | - | 114.0 µs | - | 373.4 µs | 957.6 µs | 2.30 ms |

Median of 3 runs per cell; each run times every call separately after a warm-up. Intel Core 5 120U (12 logical CPUs) under WSL2 on a shared laptop, process pinned to logical CPUs 2-9 with `taskset`; timer: rdtscp calibrated against steady_clock, with the median cost of an empty timed call (17-20 ns) subtracted; CMake Release (`-O3 -DNDEBUG`), no `-march=native`; 2026-11-01. "Before" is the previous commit of this engine built with the same benchmark source.

How to read it. WSL2 and a shared machine add scheduling jitter: on
sub-microsecond kernels, p50 differences below about 20% between "before"
and "after" are inside run-to-run noise, and `max` is dominated by
hypervisor preemption, not by the kernel. The analytic kernels were already allocation-free, so their latency is unchanged within noise; the change there is the noexcept error path. The gains are on the Monte Carlo path, where the pool and the reusable workspace remove thread creation and the per-call buffer allocation (`try_mc_price`, 1M paths, 4 chunks: 24-26 ms against 31-34 ms for the old call), and in the batch pricer, which costs about 95-115 ns per option, Greeks included.

## Layout

```
equity-options-engine/
├── CMakeLists.txt             # C++23, warnings, EQOPT_WERROR/NATIVE/LTO/VEC_REPORT, GTest
├── include/eqopt/             # public headers (iv_core.hpp: shared IV core)
├── src/                       # implementations
├── tools/gen_golden_header.py # golden JSON -> constexpr header (--check)
├── tools/jsonl_main.cpp       # eqopt_jsonl shim for the differential test
├── tests/                     # GoogleTest suite + generated golden_vectors.hpp
├── bench/                     # per-call latency harness (harness: qp/bench/latency.hpp in cpp/core), results.json
└── docs/                      # METHODOLOGY, VALIDATION, DESK_GUIDE
```

## Documentation

- [docs/METHODOLOGY.md](docs/METHODOLOGY.md): why a C++ engine alongside
  Python; design choices; assumptions register.
- [docs/VALIDATION.md](docs/VALIDATION.md): golden and differential
  validation, identities, convergence, failure behaviour, kernel API tests.
- [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md): where a C++ engine sits in a
  desk workflow; golden-vector regression as a release gate.

## Ports of the newer models (November 2026)

Leisen-Reimer tree, Bachelier price/Greeks/implied normal vol, and charm,
speed, colour, zomma, veta: `include/eqopt/lr_tree.hpp`, `include/eqopt/bachelier.hpp`, `include/eqopt/greeks_higher.hpp` (tests in `tests/test_phase5.cpp`). They match the Python reference
golden values to 1e-12 relative (1e-13 for Bachelier). The other newer
models are Python only for now.
