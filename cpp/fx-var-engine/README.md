# FX VaR & ES Engine (C++20)

Value-at-Risk and Expected Shortfall for a multi-currency FX book. It is
the linear-book port of the Python engine
[`python/fx/03-var-es-engine`](../../python/fx/03-var-es-engine/): cash,
spot and outright forwards with full revaluation, and no options. Results
are checked against the Python engine with committed golden values.

No external dependencies beyond GoogleTest for the tests. Linear algebra,
special functions (inverse normal CDF, log-gamma, Student-t quantile,
incomplete gamma and beta), RNG transforms and estimators are in the
library.

## Conventions

`alpha` is the tail probability (0.01 = 99 % VaR) in every function and
option struct. Values outside (0, 0.5) throw `std::invalid_argument`, so a
confidence level such as 0.99 is rejected rather than reinterpreted. VaR and
ES are positive losses. The empirical VaR is the order statistic L(m) of
the losses, m = ceil(n alpha - 1e-9), and ES is the Acerbi-Tasche average
with the fractional atom. NaN, infinite, empty or out-of-domain input
throws. `docs/METHODOLOGY.md` has the full table.

## What it does

| Module | Contents |
| --- | --- |
| `fxvar/book.hpp` | Book in a base currency: cash, spot, outright forwards (spot plus two deposit legs, CIP-consistent). Crosses decompose into USD legs. `CompiledBook` revalues with one `exp()` per leg per scenario. |
| `fxvar/matrix.hpp` | Dense matrix; Cholesky with a capped jitter ladder (1e-12 to 1e-6 x mean(diag)) and optional eigenvalue-clip or Higham repair; cyclic Jacobi eigen solver. |
| `fxvar/returns.hpp` | Labelled factor histories, sample and EWMA covariance, EWMA volatility (30-observation seed), input validation, near-zero-vol peg screen. |
| `fxvar/stats.hpp` | Normal pdf/cdf/quantile, thread-safe Lanczos `log_gamma`, Student-t, incomplete gamma and beta, chi-square SF, binomial CDF by pmf recursion, moments. |
| `fxvar/historical.hpp` | Plain, BRW age-weighted and per-factor filtered HS; sqrt-time scaling; peg warnings. |
| `fxvar/parametric.hpp` | Variance-covariance VaR/ES (normal and standardised t on the same sigma); Cornish-Fisher with the alpha-aware validity check. |
| `fxvar/monte_carlo.hpp` | Multivariate normal, covariance-matched Student-t, and a jump mixture (Poisson number of common jumps over the horizon, size per currency) with full revaluation. Seeded `mt19937_64` with library-owned transforms. Bootstrap VaR SE (primary) and KDE SE (diagnostic). |
| `fxvar/expected_shortfall.hpp` | Empirical VaR/ES (uniform or weighted), normal and standardised-t closed forms. |
| `fxvar/backtest.hpp` | Kupiec POF (`KupiecInput`), Christoffersen independence and conditional coverage, Basel traffic light for any window and alpha, Acerbi-Szekely Z2. |
| `fxvar/stress.hpp` | Joint scenarios (spot shocks, rate shifts, jumps), a scenario library, peg-break add-on, reverse stress (closed form for a linear book, checked against a numerical search). |

## Quickstart

```bash
cmake -S . -B build                          # GTest from the system, else fetched (v1.15.2)
cmake --build build -j
ctest --test-dir build --output-on-failure   # 114 tests, about 3 s serial
./build/fxvar_bench
```

Options: `-DFXVAR_WERROR=ON` (warnings as errors on GCC/Clang),
`-DBUILD_TESTING=OFF` (library only), `-DFXVAR_BUILD_BENCH=OFF`.
`CMAKE_BUILD_TYPE` defaults to Release.

```cpp
#include "fxvar/historical.hpp"
#include "fxvar/monte_carlo.hpp"

using namespace fxvar;

Market market({{"EUR", 1.10}, {"JPY", 0.0090}},
              {{"USD", 0.05}, {"EUR", 0.03}, {"JPY", 0.001}});
Book book({SpotPosition{"EURJPY", 5e6, {}},              // cross: two USD legs
           ForwardPosition{"USDJPY", 10e6, 0.5, {}}});   // CIP forward

HistoricalOptions ho;                          // ho.alpha = 0.01 (99 %)
HistoricalResult hs = historical_var(book, market, returns, ho);

MonteCarloOptions mc;
mc.dist = McDist::kJump;                       // peg-break overlay
mc.jumps = {0.02, {{"FX:JPY", -0.10}}, {}};    // 2 % per day, JPY -10 % (log)
mc.seed = 42;
MonteCarloResult r = monte_carlo_var(book, market, sample_cov(returns), mc);
// r.var, r.es, r.se_var (bootstrap), r.se_var_kde (diagnostic)
```

## Benchmark

Synthetic 250-position / 50-factor book (200 spots across 45 currencies and
50 forwards; 45 FX and 5 IR factors). The 500-day history is drawn from a
full-rank AR(1) covariance (rho = 0.5). Full revaluation, one thread, GCC
15.2 Release (`-O3`), Intel Core 5 120U under WSL2 (`./build/fxvar_bench`):

| Method | Scenarios | Wall time |
| --- | --- | --- |
| Historical simulation | 500 | 2.9 ms |
| Parametric (FD exposures + var-covar) | n/a | 1.2 ms |
| Cholesky 50 x 50 (no jitter needed) | n/a | 0.015 ms |
| Monte Carlo, normal, with 200-replicate bootstrap SE | 100 000 | 1.20 s |
| Monte Carlo, Student-t(5), with bootstrap SE | 100 000 | 1.04 s |

Same run: MC normal 99 % VaR 272 386 (bootstrap SE 1 624, KDE SE 1 300)
against the parametric closed form 273 191. The two paths share only the
covariance.

## Validation summary

- **Triangulation**: EURJPY P&L equals the EURUSD + USDJPY legs to 1e-12.
- **Forwards**: zero value at CIP inception; equal to the two discounted
  deposit legs to 1e-10.
- **Cross-language goldens**: 75 values from the Python `fx_var` engine,
  including the shared `conv_*` keys (order-statistic VaR/ES, Basel on nine
  windows, Kupiec, Z2, Cornish-Fisher flags, Higham and clip repairs, EWMA),
  with per-key tolerances (`tests/test_golden_python.cpp`; header generated
  by `tools/gen_golden_header.py`, `--check` verifies it).
- **MC**: normal MC within 3 SE of the closed form; t and jump mixture
  fatter at 99 %; the jump count scales with the horizon as Poisson.
- **Backtests**: Kupiec and Christoffersen against hand-computed LRs; Basel
  zones and plus factors on any window.
- **Inputs**: a fuzz test calls every public function with NaN, +inf, -inf,
  empty and negative inputs and requires `std::invalid_argument`.

Details: [`docs/VALIDATION.md`](docs/VALIDATION.md). Model choices and
assumptions: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). Desk use:
[`docs/DESK_GUIDE.md`](docs/DESK_GUIDE.md).

## Limitations

- No options: the book has cash, spot and forwards only.
- MC repeatability holds on one platform; other libms can change the last
  bits.
- The bootstrap SE accounts for most of the MC time.

## Layout

```
fx-var-engine/
+-- CMakeLists.txt
+-- include/fxvar/         # public headers
+-- src/                   # implementations
+-- tests/                 # 11 GoogleTest executables, 114 tests
+-- bench/bench_main.cpp   # 250 x 50 book timings
+-- docs/                  # METHODOLOGY / VALIDATION / DESK_GUIDE
```
