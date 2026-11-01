# FX Options Pricing Engine (C++23)

C++ port of the Python reference project
[`python/fx/01-options-pricing`](../../python/fx/01-options-pricing/). It
prices European (and, in the tree, American) vanilla FX options under
Garman-Kohlhagen: Black-Scholes with the dividend yield replaced by the
foreign interest rate. It is checked against golden vectors computed in
mpmath and against the Python and Rust engines on seeded random inputs.

## Conventions

* Pairs are quoted **BASE/QUOTE** (EURUSD = USD per 1 EUR).
* `S` is spot in domestic (quote) currency per unit foreign (base) currency;
  premiums are domestic currency per unit of foreign notional.
* `r_d` = domestic (quote-ccy) rate, `r_f` = foreign (base-ccy) rate, both
  continuously compounded, annualised, ACT/365F. Negative rates are legal.
* Theta is per year; vega per unit of vol.
* Invalid inputs and non-finite results throw `std::invalid_argument`;
  solver non-convergence throws `std::runtime_error`.
* Every pricing kernel also has a `noexcept` form, `try_*`, that returns
  `std::expected<T, fxopt::Error>` and never throws or allocates. The
  throwing functions are thin wrappers over it.

## Layout

```
fx-options-engine/
├── CMakeLists.txt              # C++23, warnings, FXOPT_WERROR/NATIVE/LTO/VEC_REPORT, GTest
├── include/fxopt/
│   ├── error.hpp               # Errc, Error, Result<T> = std::expected<T, Error>
│   ├── common.hpp              # OptionType, validation, N(x), n(x), N^{-1}(x)
│   ├── garman_kohlhagen.hpp    # d1/d2, gk_price/gk_call/gk_put, limits
│   ├── forwards.hpp            # CIP forward, forward points, synthetic forward
│   ├── black76.hpp             # Black-76 on the FX forward (== GK on CIP F)
│   ├── deltas.hpp              # 4 delta conventions, strike-from-delta, ATM/DNS
│   ├── greeks.hpp              # analytic Greeks (2 rhos, vanna, volga) + templated FD
│   ├── implied_vol.hpp         # out-of-the-money-side solver (iv_core.hpp)
│   ├── iv_core.hpp             # shared IV core, same algorithm as Python and Rust
│   ├── binomial.hpp            # CRR tree, foreign rate as yield, Eur + Amer
│   ├── monte_carlo.hpp         # GBM under domestic measure, mt19937_64, McWorkspace
│   ├── thread_pool.hpp         # persistent fixed-size pool for chunked MC
│   └── batch.hpp               # structure-of-arrays price and Greeks batch
├── src/                        # implementations (+ internal brent.hpp)
├── tools/gen_golden_header.py  # golden JSON -> tests/golden_vectors.hpp (--check)
├── tools/jsonl_main.cpp        # fxopt_jsonl shim for the differential test
├── tests/                      # GoogleTest suite + generated golden_vectors.hpp
├── bench/                      # per-call latency harness (latency.hpp) and kernels
└── docs/                       # METHODOLOGY / VALIDATION / DESK_GUIDE
```

## Build, test, benchmark

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # add -DFXOPT_WERROR=ON for -Werror
cmake --build build -j
ctest --test-dir build --output-on-failure        # 87 tests
./build/fxopt_bench --out /tmp/fxopt.json         # latency percentiles, JSON
```

Build options: `FXOPT_NATIVE` adds `-O3 -march=native` (off by default,
for local latency work only, never in CI), `FXOPT_LTO` turns on
`CMAKE_INTERPROCEDURAL_OPTIMIZATION`, `FXOPT_VEC_REPORT` prints the
compiler vectorisation report for the batch, tree and Monte Carlo files.
`CMAKE_CXX_FLAGS` is never modified. The 87 CTest cases are 86 GoogleTest
cases (including the separate `fxopt_alloc_tests` binary) plus the shim
smoke test; sanitizer builds skip `fxopt_alloc_tests` (82 cases), because
the sanitizer runtimes define `operator new` themselves.

C++23, CMake >= 3.20. GoogleTest is found with `find_package(GTest)` and
fetched (v1.15.2) when absent. Built and tested with GCC 15.2, Clang 21.1
and ASan+UBSan builds with both compilers, plus a Clang TSan build for
the thread pool.

## Quick example

```cpp
#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/deltas.hpp"
#include "fxopt/greeks.hpp"
#include "fxopt/monte_carlo.hpp"

using namespace fxopt;

// EURUSD 6m call: S=1.10, K=1.12, r_d(USD)=4.25%, r_f(EUR)=2.90%, vol 9.25%
double px = gk_price(1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925, OptionType::Call);
GreeksResult g = analytic_greeks(1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925,
                                 OptionType::Call);
// 25-delta strike under the premium-adjusted spot convention:
double k25 = strike_from_delta(0.25, 1.10, 0.5, 0.0425, 0.0290, 0.0925,
                               OptionType::Call, DeltaConvention::SpotPa);
// Monte Carlo takes a typed seed, so a bool cannot bind to it:
MCResult mc = mc_price(1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925, OptionType::Call,
                       McConfig{100'000, Seed{42}, true, true});
```

## Validation headline

- **Golden vectors:** 434 records computed in mpmath at 50 digits (GK
  price and Greeks incl. both rhos, vanna, volga and all four delta
  conventions; implied vol; strike-from-delta in all four conventions;
  DNS ATM; CRR; forwards), each output within its own tolerance; worst
  deviation 0.19 of tolerance.
- **Differential test:** `fxopt_jsonl` against Python and Rust on 790
  seeded random requests; largest deviation from Python 1.1e-13 on prices,
  6.8e-13 on strikes, 1.9e-13 on implied vol.
- Parity, GK == Black-76, foreign-domestic symmetry, delta relations,
  tree convergence, MC within 3 SE: see [docs/VALIDATION.md](docs/VALIDATION.md).

## Low-latency API

```cpp
#include "fxopt/batch.hpp"
#include "fxopt/binomial.hpp"
#include "fxopt/monte_carlo.hpp"

// Scalar, noexcept: a value or an Error (code, field, numbers; no heap).
const fxopt::Result<double> px =
    fxopt::try_gk_price(1.0850, 1.10, 0.5, 0.045, 0.030, 0.08, fxopt::OptionType::Call);
if (!px) log(fxopt::message(px.error()));     // formats only on the error path

// Tree and MC on caller-owned buffers, sized once at start-up.
std::vector<double> tree(fxopt::binomial_workspace_size(1000));
auto am = fxopt::try_binomial_price(1.0850, 1.10, 0.5, 0.045, 0.030, 0.08,
                                    fxopt::OptionType::Put, 1000,
                                    fxopt::Exercise::American, tree);
fxopt::ThreadPool pool(3);                    // threads created once
fxopt::McWorkspace ws;
ws.reserve(1'000'000, true);
auto mc = fxopt::try_mc_price(1.0850, 1.10, 0.5, 0.045, 0.030, 0.08,
                              fxopt::OptionType::Call, 1'000'000, fxopt::Seed{7},
                              true, true, /*threads=*/4, ws, &pool);

// Structure of arrays: every output equals the scalar kernel bit for bit.
fxopt::GkBatchInputs in{S, K, T, r_d, r_f, sigma, type};   // std::span each
auto ok = fxopt::gk_price_batch(in, prices);                // BatchError on failure
```

* `try_gk_price`, `try_d1`, `try_analytic_greeks`, `try_black76_price`,
  `try_implied_vol`, `try_binomial_price` and `try_mc_price` are `noexcept`
  and allocate nothing on success or failure. The checks, their order and
  the error messages are the same as in the throwing API.
* `try_mc_price` with `threads = 1` draws the legacy single stream
  (`mt19937_64(seed)`) and equals `mc_price` bit for bit. With
  `threads > 1` the draws are split into `threads` chunks, chunk `c`
  seeded with `splitmix64(seed + k c)`; the result is fixed for a given
  `(seed, threads)` whichever pool runs it, and is statistically
  equivalent to, not bit-equal with, the single stream.
* `gk_price_batch` and `gk_greeks_batch` validate the whole batch in one
  vectorised pass, then price element by element with the scalar inline
  kernel. The pricing loop is not vectorised: `erfc`, `exp` and `log` are
  scalar libm calls, and vector libm variants return different bits.
* `strike_from_delta`, `delta` and the forward helpers keep only the
  throwing form.

No numerical result changed: a program printing 181,543 legacy-API results
and exception texts in hex-float gives identical output against the
previous and the current library, with GCC 15.2 and Clang 21.1.

## Latency

Per-call latency, `./build/fxopt_bench --out results.json` (raw records for
every run, both compilers, before and after, are in
[bench/results.json](bench/results.json)). Sizes and inputs are in
`bench/bench_main.cpp`; every call gets a different input from a 16,384-entry
table.

| Kernel | Compiler | p50 before | p50 after | p99 before | p99 after | p99.9 after | max after |
|---|---|---|---|---|---|---|---|
| `gk_price` | GCC 15.2 | 127 ns | 113 ns | 256 ns | 329 ns | 617 ns | 139.2 µs |
| `gk_price` | Clang 21.1 | 133 ns | 115 ns | 437 ns | 278 ns | 567 ns | 456.3 µs |
| `analytic_greeks` | GCC 15.2 | 138 ns | 126 ns | 310 ns | 317 ns | 682 ns | 893.3 µs |
| `analytic_greeks` | Clang 21.1 | 142 ns | 134 ns | 220 ns | 323 ns | 806 ns | 659.8 µs |
| `black76_price` | GCC 15.2 | 117 ns | 111 ns | 241 ns | 218 ns | 446 ns | 132.8 µs |
| `black76_price` | Clang 21.1 | 126 ns | 125 ns | 180 ns | 377 ns | 700 ns | 900.2 µs |
| `implied_vol` | GCC 15.2 | 980 ns | 945 ns | 1.7 µs | 1.6 µs | 8.2 µs | 508.2 µs |
| `implied_vol` | Clang 21.1 | 961 ns | 929 ns | 2.7 µs | 1.5 µs | 10.1 µs | 919.6 µs |
| `strike_from_delta 25d spot` | GCC 15.2 | 166 ns | 155 ns | 232 ns | 173 ns | 282 ns | 71.1 µs |
| `strike_from_delta 25d spot` | Clang 21.1 | 183 ns | 171 ns | 196 ns | 250 ns | 435 ns | 596.9 µs |
| `binomial_price eu n=1000` | GCC 15.2 | 149.7 µs | 163.6 µs | 206.5 µs | 203.9 µs | 293.9 µs | 409.8 µs |
| `binomial_price eu n=1000` | Clang 21.1 | 157.5 µs | 153.5 µs | 217.1 µs | 769.7 µs | 1.04 ms | 1.09 ms |
| `binomial_price am n=1000` | GCC 15.2 | 256.8 µs | 269.9 µs | 873.9 µs | 328.6 µs | 612.2 µs | 754.0 µs |
| `binomial_price am n=1000` | Clang 21.1 | 398.2 µs | 376.7 µs | 864.0 µs | 1.18 ms | 2.83 ms | 3.43 ms |
| `mc_price 10k paths 1 thr` | GCC 15.2 | 686.6 µs | 766.1 µs | 1.19 ms | 911.8 µs | 1.21 ms | 2.09 ms |
| `mc_price 10k paths 1 thr` | Clang 21.1 | 946.3 µs | 725.5 µs | 2.36 ms | 1.95 ms | 3.15 ms | 3.45 ms |
| `mc_price 1M paths 1 thr` | GCC 15.2 | 78.56 ms | 82.96 ms | 99.78 ms | 109.48 ms | 109.48 ms | 111.04 ms |
| `mc_price 1M paths 1 thr` | Clang 21.1 | 124.77 ms | 87.79 ms | 197.52 ms | 96.59 ms | 96.59 ms | 110.07 ms |
| `try_gk_price` | GCC 15.2 | - | 133 ns | - | 254 ns | 433 ns | 70.6 µs |
| `try_gk_price` | Clang 21.1 | - | 123 ns | - | 301 ns | 537 ns | 70.9 µs |
| `try_analytic_greeks` | GCC 15.2 | - | 133 ns | - | 351 ns | 794 ns | 104.5 µs |
| `try_analytic_greeks` | Clang 21.1 | - | 127 ns | - | 312 ns | 565 ns | 125.7 µs |
| `try_implied_vol` | GCC 15.2 | - | 941 ns | - | 2.1 µs | 7.3 µs | 532.5 µs |
| `try_implied_vol` | Clang 21.1 | - | 1.0 µs | - | 3.4 µs | 7.8 µs | 246.8 µs |
| `try_binomial_price am n=1000 (ws)` | GCC 15.2 | - | 261.2 µs | - | 508.3 µs | 673.4 µs | 1.75 ms |
| `try_binomial_price am n=1000 (ws)` | Clang 21.1 | - | 420.2 µs | - | 755.8 µs | 894.3 µs | 948.9 µs |
| `try_mc_price 10k paths 1 thr (ws)` | GCC 15.2 | - | 699.2 µs | - | 842.2 µs | 2.76 ms | 2.88 ms |
| `try_mc_price 10k paths 1 thr (ws)` | Clang 21.1 | - | 764.4 µs | - | 1.08 ms | 1.38 ms | 1.66 ms |
| `try_mc_price 10k paths 4 thr (pool)` | GCC 15.2 | - | 562.9 µs | - | 855.8 µs | 3.47 ms | 4.17 ms |
| `try_mc_price 10k paths 4 thr (pool)` | Clang 21.1 | - | 513.5 µs | - | 809.5 µs | 1.13 ms | 2.71 ms |
| `try_mc_price 1M paths 4 thr (pool)` | GCC 15.2 | - | 39.60 ms | - | 46.21 ms | 46.21 ms | 47.08 ms |
| `try_mc_price 1M paths 4 thr (pool)` | Clang 21.1 | - | 30.23 ms | - | 35.35 ms | 35.35 ms | 36.27 ms |
| `gk_price_batch x1024` | GCC 15.2 | - | 103.1 µs | - | 251.5 µs | 399.0 µs | 5.72 ms |
| `gk_price_batch x1024` | Clang 21.1 | - | 110.8 µs | - | 163.7 µs | 223.9 µs | 1.05 ms |
| `gk_greeks_batch x1024` | GCC 15.2 | - | 111.5 µs | - | 290.3 µs | 466.4 µs | 1.76 ms |
| `gk_greeks_batch x1024` | Clang 21.1 | - | 115.0 µs | - | 184.8 µs | 340.9 µs | 1.20 ms |

Median of 3 runs per cell; each run times every call separately after a warm-up. Intel Core 5 120U (12 logical CPUs) under WSL2 on a shared laptop, process pinned to logical CPUs 2-9 with `taskset`; timer: rdtscp calibrated against steady_clock, with the median cost of an empty timed call (17-20 ns) subtracted; CMake Release (`-O3 -DNDEBUG`), no `-march=native`; 2026-11-01. "Before" is the previous commit of this engine built with the same benchmark source.

How to read it. WSL2 and a shared machine add scheduling jitter: on
sub-microsecond kernels, p50 differences below about 20% between "before"
and "after" are inside run-to-run noise, and `max` is dominated by
hypervisor preemption, not by the kernel. The analytic kernels were already allocation-free and are unchanged within noise. The old `mc_price` has one RNG stream and no threads argument, so the parallel rows exist only for `try_mc_price`; with 4 chunks on the pool, 1M paths take 30-40 ms against 80-90 ms single-threaded.

## Docs

* [docs/METHODOLOGY.md](docs/METHODOLOGY.md): why a C++ engine, design
  choices, model assumptions (register lives with the Python reference).
* [docs/VALIDATION.md](docs/VALIDATION.md): golden and differential
  validation, identity tests, convergence, failure modes.
* [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md): where this engine could sit in
  an FX workflow; delta-convention pitfalls; golden regression as a
  release gate.
