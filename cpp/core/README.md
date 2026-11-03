# qp::core

Header-only C++23 library shared by the four C++ engines
(`equity-options-engine`, `fx-options-engine`, `equity-var-engine`,
`fx-var-engine`). CMake target `qp::core`, namespace `qp`.

## What it holds

| Header | Contents | Used by |
|---|---|---|
| `qp/normal.hpp` | `norm_pdf`, `norm_cdf` (erfc form), `norm_cdf_rcp`, `acklam_ppf`, `ppf_halley_step`, `norm_ppf_unchecked<Steps>` | all four |
| `qp/iv_core.hpp` | implied volatility from normalised Black (`qp::iv_core`) | eqopt, fxopt |
| `qp/black_scholes.hpp` | Black-Scholes in cost-of-carry form (`b = r - q = r_d - r_f`): `carry_d1_d2`, `carry_value`, `carry_price`, `carry_greeks`, templated on the normal cdf (`NormCdfRcp` for eqopt, `NormCdfDiv` for fxopt) | eqopt, fxopt |
| `qp/black_scholes_vm.hpp` | the same kernel on the `qp/vmath.hpp` functions, force-inlined: the per-element body of the `_fast` batch pricers | eqopt, fxopt |
| `qp/rng.hpp` | `splitmix64`, `uniform_open53`, `gamma_marsaglia_tsang` | all four |
| `qp/special.hpp` | Lanczos `log_gamma_lanczos`, incomplete beta and gamma kernels, binomial cdf | eqvar, fxvar |
| `qp/matrix.hpp` | `qp::Matrix`, `cholesky_raw`, `matvec_rows`, and in `qp::linalg` the validated Cholesky with jitter ladder, Jacobi eigen, eigenvalue-clip and Higham repair | eqvar, fxvar |
| `qp/brent.hpp` | `brentq` | fxopt |
| `qp/thread_pool.hpp` | `ThreadPool`, `default_thread_pool()` | eqopt, fxopt |
| `qp/expected.hpp` | `qp::Result<T, E>`, `value_or_throw` | all four |
| `qp/config.hpp` | `QP_PRE` / `QP_POST` contract macros | core |
| `qp/simd.hpp` | optional vector path for `matvec_rows` | core |
| `qp/philox.hpp` | Philox4x32-10 counter-based generator, `PhiloxStream` (Random123 known answers) | new, opt-in |
| `qp/quantile.hpp` | `select_kth`, `quantile_type7`: O(n) selection, same bits as a sort | new, opt-in |
| `qp/gemm.hpp` | `gemm_blocked`, `syrk_upper_blocked`: cache-blocked, same bits as the naive loops | eqvar `sample_covariance` |
| `qp/vmath.hpp` | `qp::vm::exp_batch`, `erfc_batch`, `norm_cdf_batch`: branch-free batch kernels with documented tolerances | new, opt-in |
| `qp/bench/latency.hpp`, `qp/tools/jsonl.hpp` | bench harness and JSON-lines reader for the shims | benches, shims |

Each engine keeps its public headers, namespaces and API. Where a type or
function moved, the engine header has a using-declaration (`eqopt::ThreadPool`
is `qp::ThreadPool`) or a one-line wrapper. `eqvar::Matrix` and
`fxvar::Matrix` derive from `qp::Matrix` and add their own accessors, because
their `data()` members have different return types.

The core holds only the numerical cores. Input validation and error texts
stay in the engines, so every exception message is unchanged. Where the
engines differed in a constant (series tolerance, iteration cap, number of
Halley steps) the constant is a parameter and each engine passes its own.
`eqopt` has always computed the normal cdf as `erfc(-x * (1/sqrt 2))`, which
can differ from `erfc(-x / sqrt 2)` in the last bit, so the core keeps that
form as `norm_cdf_rcp`.

Moving the code changed no result. Programs that call the old public API of
each engine print the same hex-float values and exception texts when built
against the pre-core sources (76166fd) and against these, with GCC 15 and
Clang 21.

## Performance kernels (November 2026)

Two kinds, kept apart on purpose.

Same bits as before. `gemm_blocked` and `syrk_upper_blocked` tile the loops
for the cache but add each product into its output element in the original
k (or row) order, so every element is the same double as the textbook loop.
`select_kth` and `quantile_type7` use `std::nth_element`, which puts the
k-th value where a sort would. The core tests check both bit for bit.
`eqvar::sample_covariance` now runs on `syrk_upper_blocked`; its goldens
did not move.

Documented tolerance. `qp::vm` evaluates exp, erfc and the normal cdf over
a span with no data-dependent branch, so the compiler can vectorise the
loop. Tolerances, checked by `tests/test_perf.cpp`:

| Kernel | Tolerance | Range |
|---|---|---|
| `exp_batch` | 2 ulp (relative 4.5e-16) | [-708, 709] |
| `erfc_batch` | relative 2e-15 (measured 1.0e-15) | [-6, 26.5] |
| `norm_cdf_batch` | absolute 4e-16 and relative 2e-15 (measured 2.8e-16, 1.1e-15) | x >= -37 |

The scalar functions in `qp/normal.hpp` stay the default and every golden
uses them. A caller opts in to `qp::vm` and accepts the tolerance. Inputs
must be finite. `tools/gen_erfc_cheb.cpp` regenerates the erfc
coefficients.

Philox. `PhiloxStream(seed, stream)` gives path `stream` its own sequence,
so a Monte Carlo can hand paths to any thread and get the same numbers.
The engines' existing generators are unchanged, because switching them
would move every Monte Carlo golden. Moving an engine to Philox is a
release that regenerates those goldens.

Latency. `bench_core` (built with the tests) prints the batch size B and
p10, p50, p90, p99 and max per call for each new kernel next to the code it
replaces, timed with the method shared with the Rust benches
([docs/BENCHMARKS.md](../../docs/BENCHMARKS.md)). It records compiler,
flags and CPU model with each line, and writes
JSON with `--out`. The harness does not pin the thread to a core. Use
`taskset -c 2 ./bench_core` for a pinned run.

## Use from an engine

```cmake
if(NOT TARGET qp::core)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core ${CMAKE_BINARY_DIR}/qp_core)
endif()
target_link_libraries(<engine> PUBLIC qp::core)
```

The guard lets one CMake project include several engines.

## Build and test on its own

```bash
cmake --preset gcc          # or clang, gcc16-cxx26, asan, tsan, msvc
cmake --build --preset gcc
ctest --preset gcc
```

The presets are the same as the engines' (see below).

## Presets

`CMakePresets.json` here and in each engine directory defines configure,
build and test presets. Build trees go to `build/<preset>`. All use Ninja,
Release (sanitizers: RelWithDebInfo) and turn warnings into errors.

| Preset | Compiler | Notes |
|---|---|---|
| `gcc` | `$CXX`, else the default `c++` | C++23 |
| `clang` | `clang++-21` | C++23 |
| `gcc16-cxx26` | `$CXX` (GCC 16 in CI) | `QP_CXX26=ON` |
| `asan` | `$CXX` | `-fsanitize=address,undefined` |
| `tsan` | `$CXX` | `-fsanitize=thread` |
| `msvc` | `cl` | run inside a Visual Studio developer prompt |

## C++26 option

`QP_CXX26=ON` compiles with C++26. Two features are then used when the
compiler reports them through a feature-test macro:

- Contracts (`__cpp_contracts`, GCC 16 with `-fcontracts` and `libstdc++exp` for the default violation handler, added by the
  CMake file). `QP_PRE` and `QP_POST` state programmer preconditions only:
  pointers and sizes in `cholesky_raw` and `matvec_rows`, `0 < p < 1` for the
  unchecked inverse cdf, the tolerance of `brentq`, a non-null task in
  `ThreadPool::run`, and `0 <= N(x) <= 1` as a postcondition. Market data is
  never checked by a contract: whether a contract is checked depends on the
  build mode, so the kernels validate inputs themselves.
- `std::simd` (`__cpp_lib_simd`). `matvec_rows`, the scenario P&L kernel
  of the equity VaR engine, then runs W rows at a time. Each lane sums its
  row in the same order with the same roundings as the scalar loop, so the
  result is the same double. The core test
  `Matrix.MatvecRowsMatchesPlainLoopBitForBit` checks this.

Without the macros both features compile to nothing and the C++23 code runs.
GCC 15 and Clang 21 in C++26 mode take that path.

Compiler support, rechecked on 2026-11-02 against the cppreference C++26
support table (primary source: the vendor columns of that page): contracts
(P2900R14) are listed for GCC 16 only, and `std::simd` (P1928R15) for GCC 16
libstdc++ only, marked partial. Clang 23.1 and MSVC 14.51 list neither. A
feature leaves its macro once GCC, Clang and MSVC all ship it, so both stay
behind their macros for now. The C++23 floor is unchanged.

Limitations, stated plainly:

- The contract and `std::simd` branches have not been compiled locally:
  no GCC 16 was available on the build machine. CI's `cpp <engine> (gcc16-cxx26)` jobs
  are the first builds that take them.
- The vector path's bit identity was tested through GCC's
  `<experimental/simd>` (CMake option `QP_SIMD_EXPERIMENTAL=ON`), which runs
  the same code with a different vector type.
- With `-march=native` the compiler may fuse the scalar loop's multiply and
  add, and the scalar and vector paths can then differ in the last bit.
- `uniform_open53` maps the all-ones 64-bit word to exactly 1.0, because
  `(2^53 - 0.5) 2^-53` rounds up. That word has probability 2^-53 per draw.
  The engines had this before the core existed; changing it would change
  their random streams, so it is recorded here and left for a release that
  regenerates the Monte Carlo goldens.
