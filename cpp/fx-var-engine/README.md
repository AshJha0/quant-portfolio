# FX VaR & ES Engine (C++20)

High-performance Value-at-Risk and Expected Shortfall engine for a
multi-currency FX book — the C++ twin of the Python research package
[`python/fx/03-var-es-engine`](../../python/fx/03-var-es-engine/), mirroring
its semantics (factor conventions, estimators, test identities) and
cross-validated against it with committed golden constants.

No external dependencies beyond GoogleTest for the test suite: the linear
algebra, special functions (inverse normal CDF, Student-t quantile,
incomplete gamma/beta), RNG transforms and estimators are implemented in
the library and validated in `tests/`.

## What it does

| Module | Contents |
| --- | --- |
| `fxvar/book.hpp` | Multi-currency book vs a base ccy: cash, spot, outright forwards (spot + two deposit legs, CIP-consistent). Cross positions (EURJPY) decompose into USD legs — triangulation by construction. Struct-of-arrays `CompiledBook` hot path: one `exp()` per leg per scenario. |
| `fxvar/matrix.hpp` | Dense matrix, Cholesky with escalating jitter (singular pegged-block covariances are a legitimate input), products/quadratic forms. |
| `fxvar/returns.hpp` | Labelled factor histories, sample + RiskMetrics-EWMA covariance, EWMA volatility, NaN-refusing validation, near-zero-vol **peg screen**. |
| `fxvar/stats.hpp` | Normal pdf/cdf, inverse normal CDF (Acklam + Halley, error < 1e-13, tested < 1e-9), Student-t pdf/cdf/quantile, regularised incomplete gamma/beta, chi-square SF, exact binomial CDF, sample moments. |
| `fxvar/historical.hpp` | Plain HS, BRW age-weighted, filtered (EWMA-devolatilised) HS; sqrt-time horizon scaling; peg-blindness warning list. |
| `fxvar/parametric.hpp` | Variance-covariance VaR/ES (normal + standardised Student-t on the same sigma), Cornish-Fisher with explicit monotonicity-domain check. |
| `fxvar/monte_carlo.hpp` | Multivariate normal, covariance-matched Student-t, and **jump-mixture** (Bernoulli common jump, configurable size per ccy — the peg-break add-on) with full book revaluation; seeded `mt19937_64` + library-owned inverse-CDF transforms → bitwise-deterministic; VaR standard error via KDE. |
| `fxvar/expected_shortfall.hpp` | Weighted empirical VaR/ES (coherent Acerbi–Tasche tail splitting), closed-form normal and standardised-t VaR/ES. |
| `fxvar/backtest.hpp` | Kupiec POF, Christoffersen independence, conditional coverage (chi-square p-values via regularised incomplete gamma), exact Basel traffic light with 1996 multiplier add-ons. |
| `fxvar/stress.hpp` | Joint scenarios {per-ccy spot shocks, rate shifts, jumps}; canned GBP −8.1% Brexit, CHF +14.9% depeg, broad USD ±10%, peg-break add-on with contagion; reverse stress: closed form for a linear book, verified against an independent numerical search. |

## Quickstart

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure   # 72 tests, 282 assertions
./build/fxvar_bench
```

Toolchain: g++ 13 (C++20), CMake ≥ 3.20, GoogleTest ≥ 1.14 via
`find_package(GTest)`. Warnings-as-errors (`-Wall -Wextra -Werror -O2`).

```cpp
#include "fxvar/historical.hpp"
#include "fxvar/monte_carlo.hpp"

using namespace fxvar;

Market market({{"EUR", 1.10}, {"JPY", 0.0090}},
              {{"USD", 0.05}, {"EUR", 0.03}, {"JPY", 0.001}});
Book book({SpotPosition{"EURJPY", 5e6, {}},              // cross: 2 USD legs
           ForwardPosition{"USDJPY", 10e6, 0.5, {}}});   // CIP fwd, rate legs

HistoricalResult hs = historical_var(book, market, returns, {});
// hs.var, hs.es, hs.flagged_peg_factors, hs.warnings

MonteCarloOptions mc;
mc.dist = McDist::kJump;                       // peg-break overlay
mc.jumps = {0.02, {{"FX:JPY", -0.10}}, {}};    // 2%/day, JPY -10% (log)
mc.seed = 42;                                  // bitwise reproducible
MonteCarloResult r = monte_carlo_var(book, market, sample_cov(returns), mc);
```

## Benchmark

250-position / 50-factor book (200 spots across 45 currencies + 50
forwards; 45 FX + 5 IR factors), full revaluation, single thread,
g++ 13 `-O2`, Intel Xeon @ 2.80 GHz (`./build/fxvar_bench`):

| Method | Scenarios | Wall time |
| --- | --- | --- |
| Historical simulation (full reval) | 500 | **2.0 ms** |
| Parametric (FD exposures + var-covar) | — | **1.2 ms** |
| Monte Carlo, normal (full reval) | 100 000 | **0.78 s** |
| Monte Carlo, Student-t(5) (full reval) | 100 000 | **0.85 s** |

Monte Carlo normal 99% VaR agrees with the parametric closed form on the
same book to ~0.7% (102.7k vs 103.0k) — the two paths share no code beyond
the covariance. A full intraday risk sweep (HS + parametric + stress
library) on a 250-position book costs single-digit milliseconds — the point
of the C++ twin (see `docs/METHODOLOGY.md`).

## Validation summary

- **Triangulation identity**: EURJPY P&L == EURUSD + USDJPY leg
  decomposition to 1e-12, scenario by scenario.
- **Forwards**: zero value at CIP inception; equal to the two discounted
  deposit legs to 1e-10; rate-leg sensitivities match ∓T·N·e^(−rT)·S closed
  form.
- **Cross-language goldens**: three deterministic cases (historical VaR/ES,
  parametric closed forms, backtest statistics) reproduce the Python
  `fx_var` reference to 1e-6 absolute / 1e-8 relative
  (`tests/test_golden_python.cpp`, provenance in the file header).
- **MC**: normal MC within 3 SE of the closed form; t and jump-mixture
  strictly fatter at 99%; bitwise seed determinism.
- **Backtests**: Kupiec/Christoffersen against hand-computed LRs; Basel
  zones exact at the regulatory boundaries (green 0–4 / yellow 5–9 / red
  10+ at 250d/99%).
- **Reverse stress**: closed form vs independent projected-gradient search
  to 1e-6.
- **Peg blindness**: engine flags near-zero-vol FX factors and the
  jump-mixture MC reports the loss HS misses (tested end to end).

Full detail: [`docs/VALIDATION.md`](docs/VALIDATION.md). Model choices and
assumptions: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). Desk usage:
[`docs/DESK_GUIDE.md`](docs/DESK_GUIDE.md).

## Layout

```
fx-var-engine/
├── README.md
├── CMakeLists.txt
├── include/fxvar/         # public headers (one per pipeline stage)
├── src/                   # implementations
├── tests/                 # GoogleTest suite (72 tests / 282 assertions)
├── bench/bench_main.cpp   # 250x50 book timings
└── docs/                  # METHODOLOGY / VALIDATION / DESK_GUIDE
```
