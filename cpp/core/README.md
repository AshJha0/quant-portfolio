# qp::core

Header-only C++23 library shared by the four C++ engines
(`equity-options-engine`, `fx-options-engine`, `equity-var-engine`,
`fx-var-engine`). CMake target `qp::core`, namespace `qp`.

## What it holds

| Header | Contents | Used by |
|---|---|---|
| `qp/normal.hpp` | `norm_pdf`, `norm_cdf` (erfc form), `norm_cdf_rcp`, `acklam_ppf`, `ppf_halley_step`, `norm_ppf_unchecked<Steps>` | all four |
| `qp/iv_core.hpp` | implied volatility from normalised Black (`qp::iv_core`) | eqopt, fxopt |
| `qp/rng.hpp` | `splitmix64`, `uniform_open53`, `gamma_marsaglia_tsang` | all four |
| `qp/special.hpp` | Lanczos `log_gamma_lanczos`, incomplete beta and gamma kernels, binomial cdf | eqvar, fxvar |
| `qp/matrix.hpp` | `qp::Matrix`, `cholesky_raw`, `matvec_rows`, and in `qp::linalg` the validated Cholesky with jitter ladder, Jacobi eigen, eigenvalue-clip and Higham repair | eqvar, fxvar |
| `qp/brent.hpp` | `brentq` | fxopt |
| `qp/thread_pool.hpp` | `ThreadPool`, `default_thread_pool()` | eqopt, fxopt |
| `qp/expected.hpp` | `qp::Result<T, E>`, `value_or_throw` | all four |
| `qp/config.hpp` | `QP_PRE` / `QP_POST` contract macros | core |
| `qp/simd.hpp` | optional vector path for `matvec_rows` | core |
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

- Contracts (`__cpp_contracts`, GCC 16 with `-fcontracts`, added by the
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

Limitations, stated plainly:

- The contract and `std::simd` branches have not been compiled locally:
  no GCC 16 was available on the build machine. CI's `gcc16-cxx26` job is the
  first compiler that takes them.
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
