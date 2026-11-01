# FX VaR & ES Engine (C++23)

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
| `fxvar/kernels.hpp` | noexcept kernel API: `try_*` functions returning `std::expected<T, fxvar::Error>`, no allocation, caller-provided buffers. See "Low-latency API" below. |
| `fxvar/stress.hpp` | Joint scenarios (spot shocks, rate shifts, jumps), a scenario library, peg-break add-on, reverse stress (closed form for a linear book, checked against a numerical search). |

## Quickstart

```bash
cmake -S . -B build                          # GTest from the system, else fetched (v1.15.2)
cmake --build build -j
ctest --test-dir build --output-on-failure   # 121 tests, about 3 s serial
./build/fxvar_bench --out /tmp/fxvar_latency.json
```

Options: `-DFXVAR_WERROR=ON` (warnings as errors on GCC/Clang),
`-DBUILD_TESTING=OFF` (library only), `-DFXVAR_BUILD_BENCH=OFF`,
`-DFXVAR_NATIVE=ON` (adds `-O3 -march=native`; local latency work only,
never CI), `-DFXVAR_LTO=ON` (link-time optimisation through
`CMAKE_INTERPROCEDURAL_OPTIMIZATION`), `-DFXVAR_VEC_REPORT=ON` (prints the
compiler's vectorisation report for the hot sources). `CMAKE_CXX_FLAGS` is
never modified. `CMAKE_BUILD_TYPE` defaults to Release.

## Low-latency API

`fxvar/kernels.hpp` and two `CompiledBook` members give a `noexcept`
entry point for every per-call and per-scenario kernel. Each returns
`std::expected<T, fxvar::Error>`. An `Error` holds a code, a pointer to a
static (or book-owned) string and two numbers, so the error path does not
allocate either. `fxvar::message(e)` gives the same text the throwing API
puts in its `std::invalid_argument`.

| Kernel | Caller storage | Same result as |
| --- | --- | --- |
| `try_empirical_var_es(pnl, alpha, ws)` | `ws`: `pnl.size()` doubles | `empirical_var_es` (uniform weights) |
| `try_var_standard_error(pnl, alpha, ws)` | `ws`: `pnl.size()` doubles | `var_standard_error` |
| `try_normal_var/es`, `try_t_var/es` | none | `normal_var/es`, `t_var/es` |
| `try_matvec(A, x, y)`, `try_quad_form(w, A)` | `y`: `A.rows()` doubles | `matvec`, `quad_form` |
| `try_portfolio_sigma`, `try_var_covar` | none | `portfolio_sigma`, `var_covar` |
| `try_cholesky_lower(A, L)` | `L`: n*n doubles | `cholesky(A).lower` when no jitter is needed |
| `try_simulate_factor_returns(L, k, n, dist, df, seed, out)` | `out`: n*k doubles | `simulate_factor_returns(...).data` for normal and Student-t |
| `CompiledBook::try_pnl(shocks)` | none | `CompiledBook::pnl(const double*)` |
| `CompiledBook::try_pnl_batch(scenarios, out)` | `out`: one double per scenario row | `CompiledBook::pnl(ReturnsMatrix)` |

"Same result" means the same bits: the throwing functions are thin
wrappers over these kernels where the two overlap, and
`tests/test_latency_api.cpp` checks the rest. A program that prints 87,761
results of the legacy API (with 1,525 exception texts) in hex-float gives
identical output against the library before and after this API was added,
with GCC 15.2 and with Clang 21.1. `tests/test_no_alloc.cpp` replaces the
global `operator new` and checks that none of the kernels above allocates
on its success path. The weighted VaR/ES path is not in the noexcept API:
it stable-sorts index pairs, and `std::stable_sort` allocates a temporary
buffer.

## Latency

<!-- LATENCY_TABLE -->

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

The benchmark book is synthetic: 250 positions (200 spots across 45
currencies and 50 forwards; 45 FX and 5 IR factors), with histories of
1,000 and 10,000 days drawn from a full-rank AR(1) covariance (rho = 0.5).
`./build/fxvar_bench` times every kernel call by call after a warm-up and
reports p50, p90, p99, p99.9 and max in nanoseconds; `--out FILE` writes
the same numbers as JSON (engine, kernel, compiler, flags, cpu, date,
percentiles), `--quick` runs a tenth of the iterations. Inputs change on
every call. Under WSL2 the hypervisor adds scheduling jitter that shows in
p99.9 and max.

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
- The scenario revaluation and the inverse normal CDF call scalar libm
  `exp` and `log` per element, so those loops do not vectorise.

## Layout

```
fx-var-engine/
+-- CMakeLists.txt
+-- include/fxvar/         # public headers
+-- src/                   # implementations
+-- tests/                 # 13 GoogleTest executables, 121 tests
+-- bench/                 # per-call latency harness (latency.hpp) and kernels
+-- docs/                  # METHODOLOGY / VALIDATION / DESK_GUIDE
```
