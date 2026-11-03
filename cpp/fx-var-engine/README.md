# FX VaR & ES Engine (C++23)

Value-at-Risk and Expected Shortfall for a multi-currency FX book. It is
the C++ port of the Python engine
[`python/fx/03-var-es-engine`](../../python/fx/03-var-es-engine/): cash,
spot and outright forwards with full revaluation, plus European FX options
repriced by Garman-Kohlhagen in every historical or Monte Carlo scenario
(see [Option VaR](#option-var-full-revaluation)). Results are checked
against the Python engine with committed golden values.

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
| `fxvar/stats.hpp` | Wrappers over `qp/var_stats.hpp` in cpp/core: normal pdf/cdf/quantile, thread-safe Lanczos `log_gamma`, Student-t, incomplete gamma and beta, chi-square SF, binomial CDF by pmf recursion, moments. |
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
ctest --test-dir build --output-on-failure   # 133 tests, about 3 s serial
./build/fxvar_bench --out /tmp/fxvar_latency.json
```

Options: `-DFXVAR_WERROR=ON` (warnings as errors on GCC/Clang),
`-DBUILD_TESTING=OFF` (library only), `-DFXVAR_BUILD_BENCH=OFF`,
`-DFXVAR_NATIVE=ON` (adds `-O3 -march=native`; local latency work only,
never CI), `-DFXVAR_LTO=ON` (link-time optimisation through
`CMAKE_INTERPROCEDURAL_OPTIMIZATION`), `-DFXVAR_VEC_REPORT=ON` (prints the
compiler's vectorisation report for the hot sources). `CMAKE_CXX_FLAGS` is
never modified. `CMAKE_BUILD_TYPE` defaults to Release.

## Shared core, presets and C++26

The normal cdf and inverse, the special functions, the Gamma sampler and the dense matrix with Cholesky, eigen and PSD repair come from `qp::core`, the
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

Build trees go to `build/<preset>`. The presets turn `FXVAR_WERROR` on. `msvc`
must run inside a Visual Studio developer prompt. `QP_CXX26=ON` (the
`gcc16-cxx26` preset) compiles with C++26 and uses contracts and
`std::simd` where the compiler reports them; results are the same as the
C++23 build. CMake 3.25 or newer is required.

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

Per-call latency, `./build/fxvar_bench --out results.json` (raw records for
every run, both compilers, before and after, are in
[bench/results.json](bench/results.json)). Sizes and inputs are in
`bench/bench_main.cpp`. Inputs are synthetic (250 positions on 45 currencies,
50 factors); each call gets a different history window, P&L vector or
exposure scale. The rotated P&L rows include a copy of the input inside the
timed call (about 1 µs per 1,000 values), the same for old and new kernels.

> **How these figures were timed.** Old harness: each call timed on its own with rdtscp, empty-call overhead subtracted. They are one laptop run (Intel Core 5 120U, WSL2) kept for the record; `bench/results.json` holds the same run. Benches now use one method in C++ and Rust: a batch of back-to-back calls sized to at least 2 us per sample, median with p10/p90 ([docs/BENCHMARKS.md](../../docs/BENCHMARKS.md)). Current figures, measured base against head on one CI runner, are in the `bench-results` artifact of the [CI bench job](https://github.com/AshJha0/quant-portfolio/actions/workflows/ci.yml).

| Kernel | Compiler | p50 before | p50 after | p99 before | p99 after | p99.9 after | max after |
|---|---|---|---|---|---|---|---|
| `historical_var 1k obs 250 pos` | GCC 15.2 | 2.04 ms | 2.14 ms | 3.39 ms | 3.89 ms | 4.39 ms | 4.57 ms |
| `historical_var 1k obs 250 pos` | Clang 21.1 | 2.26 ms | 2.25 ms | 2.77 ms | 5.82 ms | 5.85 ms | 6.73 ms |
| `empirical_var_es n=1000` | GCC 15.2 | 11.4 µs | 10.6 µs | 19.0 µs | 22.0 µs | 52.0 µs | 238.7 µs |
| `empirical_var_es n=1000` | Clang 21.1 | 9.5 µs | 9.8 µs | 14.6 µs | 27.6 µs | 102.4 µs | 765.5 µs |
| `empirical_var_es n=10000` | GCC 15.2 | 111.1 µs | 107.9 µs | 146.8 µs | 189.3 µs | 246.9 µs | 602.0 µs |
| `empirical_var_es n=10000` | Clang 21.1 | 91.9 µs | 95.8 µs | 140.4 µs | 239.4 µs | 417.6 µs | 850.2 µs |
| `var_standard_error n=10000` | GCC 15.2 | 640.4 µs | 317.6 µs | 871.3 µs | 700.2 µs | 871.9 µs | 1.11 ms |
| `var_standard_error n=10000` | Clang 21.1 | 748.8 µs | 324.1 µs | 1.19 ms | 511.3 µs | 943.8 µs | 1.58 ms |
| `parametric_var 1k obs 50 factors` | GCC 15.2 | 770.2 µs | 887.1 µs | 957.6 µs | 1.14 ms | 1.64 ms | 1.78 ms |
| `parametric_var 1k obs 50 factors` | Clang 21.1 | 966.9 µs | 941.9 µs | 1.58 ms | 1.36 ms | 1.43 ms | 1.74 ms |
| `var_covar 50 factors` | GCC 15.2 | 3.3 µs | 3.5 µs | 4.6 µs | 9.0 µs | 22.3 µs | 228.4 µs |
| `var_covar 50 factors` | Clang 21.1 | 4.2 µs | 3.6 µs | 7.8 µs | 7.9 µs | 15.7 µs | 250.3 µs |
| `cholesky 50x50` | GCC 15.2 | 8.4 µs | 8.9 µs | 15.4 µs | 16.6 µs | 27.1 µs | 48.4 µs |
| `cholesky 50x50` | Clang 21.1 | 12.4 µs | 10.5 µs | 25.1 µs | 32.7 µs | 59.1 µs | 169.8 µs |
| `matvec 50x50` | GCC 15.2 | 641 ns | 716 ns | 849 ns | 1.9 µs | 7.5 µs | 145.3 µs |
| `matvec 50x50` | Clang 21.1 | 812 ns | 728 ns | 1.4 µs | 1.2 µs | 1.8 µs | 65.5 µs |
| `monte_carlo_var 10k scen normal` | GCC 15.2 | 55.42 ms | 63.20 ms | 78.04 ms | 85.59 ms | 85.59 ms | 95.08 ms |
| `monte_carlo_var 10k scen normal` | Clang 21.1 | 63.73 ms | 60.59 ms | 91.55 ms | 79.39 ms | 79.39 ms | 83.53 ms |
| `compiled pnl 10k scen 250 pos` | GCC 15.2 | 16.04 ms | 16.88 ms | 26.20 ms | 29.00 ms | 29.00 ms | 37.08 ms |
| `compiled pnl 10k scen 250 pos` | Clang 21.1 | 19.00 ms | 18.78 ms | 34.38 ms | 22.18 ms | 22.18 ms | 29.51 ms |
| `run_stress library` | GCC 15.2 | 169.0 µs | 172.3 µs | 506.6 µs | 228.3 µs | 305.0 µs | 396.0 µs |
| `run_stress library` | Clang 21.1 | 265.3 µs | 251.2 µs | 819.6 µs | 286.8 µs | 335.6 µs | 1.01 ms |
| `evaluate_var_backtest 1000 days` | GCC 15.2 | 4.9 µs | 4.8 µs | 11.1 µs | 6.6 µs | 19.7 µs | 47.0 µs |
| `evaluate_var_backtest 1000 days` | Clang 21.1 | 5.7 µs | 5.1 µs | 7.0 µs | 6.8 µs | 22.1 µs | 36.7 µs |
| `try_empirical_var_es n=1000 (ws)` | GCC 15.2 | - | 9.6 µs | - | 17.1 µs | 35.2 µs | 87.1 µs |
| `try_empirical_var_es n=1000 (ws)` | Clang 21.1 | - | 8.6 µs | - | 15.4 µs | 32.2 µs | 441.4 µs |
| `try_empirical_var_es n=10000 (ws)` | GCC 15.2 | - | 110.4 µs | - | 283.5 µs | 367.8 µs | 410.4 µs |
| `try_empirical_var_es n=10000 (ws)` | Clang 21.1 | - | 79.0 µs | - | 134.2 µs | 155.0 µs | 193.9 µs |
| `try_var_standard_error n=10000` | GCC 15.2 | - | 306.7 µs | - | 405.1 µs | 430.7 µs | 460.8 µs |
| `try_var_standard_error n=10000` | Clang 21.1 | - | 272.7 µs | - | 398.7 µs | 558.8 µs | 694.5 µs |
| `try_var_covar 50 factors` | GCC 15.2 | - | 3.4 µs | - | 4.5 µs | 14.3 µs | 283.6 µs |
| `try_var_covar 50 factors` | Clang 21.1 | - | 3.3 µs | - | 4.6 µs | 14.9 µs | 239.0 µs |
| `try_cholesky_lower 50x50` | GCC 15.2 | - | 7.9 µs | - | 12.2 µs | 24.4 µs | 558.9 µs |
| `try_cholesky_lower 50x50` | Clang 21.1 | - | 6.5 µs | - | 11.1 µs | 20.8 µs | 58.7 µs |
| `try_matvec 50x50` | GCC 15.2 | - | 793 ns | - | 844 ns | 1.2 µs | 67.4 µs |
| `try_matvec 50x50` | Clang 21.1 | - | 675 ns | - | 747 ns | 1.1 µs | 70.4 µs |
| `try_simulate 10k scen 50 factors` | GCC 15.2 | - | 38.43 ms | - | 41.14 ms | 41.14 ms | 61.80 ms |
| `try_simulate 10k scen 50 factors` | Clang 21.1 | - | 46.20 ms | - | 65.68 ms | 65.68 ms | 71.43 ms |
| `try_pnl_batch 10k scen 250 pos` | GCC 15.2 | - | 16.34 ms | - | 33.79 ms | 33.79 ms | 35.01 ms |
| `try_pnl_batch 10k scen 250 pos` | Clang 21.1 | - | 17.61 ms | - | 19.32 ms | 19.32 ms | 22.41 ms |
| `try_pnl 1 scen 250 pos` | GCC 15.2 | - | 2.0 µs | - | 2.6 µs | 12.5 µs | 173.9 µs |
| `try_pnl 1 scen 250 pos` | Clang 21.1 | - | 1.7 µs | - | 2.0 µs | 11.1 µs | 208.1 µs |
| `MC VaR pipeline 10k (try_*, ws)` | GCC 15.2 | - | 56.74 ms | - | 79.12 ms | 79.12 ms | 99.76 ms |
| `MC VaR pipeline 10k (try_*, ws)` | Clang 21.1 | - | 59.57 ms | - | 78.08 ms | 78.08 ms | 80.91 ms |

Median of 3 runs per cell; each run times every call separately after a warm-up. Intel Core 5 120U (12 logical CPUs) under WSL2 on a shared laptop, process pinned to logical CPUs 2-9 with `taskset`; timer: rdtscp calibrated against steady_clock, with the median cost of an empty timed call (17-20 ns) subtracted; CMake Release (`-O3 -DNDEBUG`), no `-march=native`; 2026-11-01. "Before" is the previous commit of this engine built with the same benchmark source.

How to read it. WSL2 and a shared machine add scheduling jitter: on
sub-microsecond kernels, p50 differences below about 20% between "before"
and "after" are inside run-to-run noise, and `max` is dominated by
hypervisor preemption, not by the kernel. `var_standard_error` is about 2x faster (selection instead of a full sort). Book revaluation is bound by one scalar `exp` per leg and is unchanged within noise; `try_pnl` prices one scenario of a 250-position book in about 2 µs with no allocation.

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
+-- tests/                 # 15 GoogleTest executables, 133 tests
+-- bench/                 # per-call latency harness (harness: qp/bench/latency.hpp in cpp/core) and kernels
+-- docs/                  # METHODOLOGY / VALIDATION / DESK_GUIDE
```

## Option VaR (full revaluation)

The plain idea: a forward's scenario value is one exponential, an option's is
a pricing formula, so its VaR needs the formula run in every scenario.

`option_reval` has the pieces: Black-Scholes-Merton with a carry yield (the
foreign rate, which is Garman-Kohlhagen 1983), closed-form Greeks,
full-revaluation P&L per scenario and the delta-gamma approximation, all
mirroring `fx_var.revaluation`. The VaR entry points take options through
overloads:

```cpp
historical_var(book, market, options, option_factors, returns, opts, dt);
monte_carlo_var(book, market, options, option_factors, cov, opts, dt);
```

`options[k].factor` indexes `option_factors` (for example `"FX:EUR"`). The
engine builds one factor axis (the book's factors, then any option factor the
book lacks), revalues the linear book through `CompiledBook` and reprices
every option leg at `S e^x`, aged by `dt` years. All three historical methods
work (filtered HS rescales the spot returns before repricing). Monte Carlo
uses the same model, seed, horizon and repair as the linear overload.

Rules, each enforced with an exception: options are on CCYUSD (spot in USD
per unit of CCY, P&L in USD), so a book that holds them must report in USD;
historical `horizon_days` must be 1, because square-root-of-time scaling is
wrong for a non-linear book (pass h-day returns instead). Each option is held
at today's implied vol in every scenario unless you pass `VolScenarios`
(below).

Vol scenario modes. `full_revaluation_pnl(legs, x, dv, n_factors, dt, mode,
skews)` takes `VolMode::StickyStrike` (`sigma + dv`), `StickyDelta`
(`sigma - skew x + dv`, `skews` one per leg or empty) or `HistoricalAtm`
(`sigma e^{dv}`, `dv` a historical log change of ATM vol). The five-argument
overload is sticky strike. All three are pinned to the Python reference at
1e-12 relative (`VolModesGolden` in `test_option_reval.cpp`).

The VaR drivers take vol scenarios too. The plain idea: a vol shock is one
more named risk factor, so it sits on the same factor axis as the spot
returns. Formally, the vol factor of `"FX:CCY"` is `"VOL:CCY"` (vol points
for sticky strike and sticky delta, a log change of ATM vol for historical
ATM). The returns panel or covariance must carry `"VOL:CCY"` for every
option factor; the scenario axis is the option layout's factors (book
factors first, then option factors the book lacks) followed by those vol
factors, so Monte Carlo simulates spot and vol jointly with their cross
terms, and filtered HS rescales the vol columns by their own EWMA vol like
any other column. With sticky strike and all-zero vol columns the answer is
the old one bit for bit. Worked example: the EUR call plus short AUD put
test book, with EUR 1m cash and one day of ageing, has 99 % historical VaR
47,005 USD sticky strike, 47,689 sticky delta and 37,268 historical ATM,
against 36,908 with vol held fixed; all three match Python at 1e-11
relative.
Entry points: the `historical_var` and `monte_carlo_var` overloads that end
in `const VolScenarios&` (`mode`, `skews`), `option_book_pnl(layout,
scenarios, dt, vol)`, and `option_combined_factors(layout)` for the axis;
tests in `test_option_var.cpp` (`OptionVarVol.*`).

Parity with Python. The historical numbers are pinned to the Python
reference (`fx_var.revaluation.full_revaluation_pnl` followed by the S2 order
statistic) at 1e-11 relative, for an options-only book and for options plus
a linear book, with and without ageing (`dt = 1/252`). The goldens come from
[`cpp/equity-var-engine/tools/gen_option_var_golden.py`](../../cpp/equity-var-engine/tools/gen_option_var_golden.py),
which builds a closed-form scenario panel so no RNG is shared. Monte Carlo
cannot match numpy draw for draw, so its test uses a case with a known
answer: a long call on one normal factor has P&L monotone in the factor, so
the exact VaR is minus the P&L at sigma z_alpha and the exact ES is a
one-dimensional integral. The engine must land within 4 bootstrap standard
errors (ES within 2 %). A book with no option legs goes down the unchanged
linear path, and the tests check the two give identical bits.
