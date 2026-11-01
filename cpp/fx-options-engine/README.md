# FX Options Pricing Engine (C++20)

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

## Layout

```
fx-options-engine/
├── CMakeLists.txt              # C++20, per-compiler warnings, FXOPT_WERROR, GTest
├── include/fxopt/
│   ├── common.hpp              # OptionType, validation, N(x), n(x), N^{-1}(x)
│   ├── garman_kohlhagen.hpp    # d1/d2, gk_price/gk_call/gk_put, limits
│   ├── forwards.hpp            # CIP forward, forward points, synthetic forward
│   ├── black76.hpp             # Black-76 on the FX forward (== GK on CIP F)
│   ├── deltas.hpp              # 4 delta conventions, strike-from-delta, ATM/DNS
│   ├── greeks.hpp              # analytic Greeks (2 rhos, vanna, volga) + templated FD
│   ├── implied_vol.hpp         # out-of-the-money-side solver (iv_core.hpp)
│   ├── iv_core.hpp             # shared IV core, same algorithm as Python and Rust
│   ├── binomial.hpp            # CRR tree, foreign rate as yield, Eur + Amer
│   └── monte_carlo.hpp         # GBM under domestic measure, mt19937_64, Seed/McConfig
├── src/                        # implementations (+ internal brent.hpp)
├── tools/gen_golden_header.py  # golden JSON -> tests/golden_vectors.hpp (--check)
├── tools/jsonl_main.cpp        # fxopt_jsonl shim for the differential test
├── tests/                      # GoogleTest suite + generated golden_vectors.hpp
├── bench/bench_main.cpp        # throughput benchmarks
└── docs/                       # METHODOLOGY / VALIDATION / DESK_GUIDE
```

## Build, test, benchmark

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # add -DFXOPT_WERROR=ON for -Werror
cmake --build build -j
ctest --test-dir build --output-on-failure        # 70 tests
./build/fxopt_bench
```

C++20, CMake >= 3.20. GoogleTest is found with `find_package(GTest)` and
fetched (v1.15.2) when absent. Built and tested with GCC 15.2, Clang 21.1
and a GCC ASan+UBSan build.

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

## Benchmark results

`./build/fxopt_bench`, GCC 15.2 `-O3` (CMake Release), WSL2 on an Intel
Core 5 120U, single run:

| Benchmark | Throughput |
|---|---|
| Garman-Kohlhagen vanilla price | 14.1 M prices/s (1M prices in 0.071 s) |
| Strike-from-delta, spot / forward (analytic `N^{-1}`) | 8.4 - 9.4 M solves/s |
| Strike-from-delta, premium-adjusted (Brent, branch-aware) | 0.68 - 0.79 M solves/s |
| Monte Carlo, antithetic + control variate | 12.6 M paths/s (1M paths in 0.079 s, SE 9.4e-6) |
| Implied vol (8 quotes, scratch timing loop) | 0.8 µs per solve (was 2.1 µs) |
| CRR American, 2000 steps | 0.9 ms (was 23 ms) |

Single runs on a laptop; expect variation of a factor of about 2.

## Docs

* [docs/METHODOLOGY.md](docs/METHODOLOGY.md): why a C++ engine, design
  choices, model assumptions (register lives with the Python reference).
* [docs/VALIDATION.md](docs/VALIDATION.md): golden and differential
  validation, identity tests, convergence, failure modes, bench table.
* [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md): where this engine could sit in
  an FX workflow; delta-convention pitfalls; golden regression as a
  release gate.
