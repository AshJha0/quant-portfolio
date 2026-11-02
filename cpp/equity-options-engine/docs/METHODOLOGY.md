# Methodology

This engine is the C++23 port of the Python reference package
[`eq_options`](../../../python/equity/01-options-pricing/). The *financial*
methodology, model selection (Black-Scholes-Merton vs CRR vs Monte Carlo,
and why each earns its place), the full assumptions register with
"what breaks if violated", and the edge-case policy, is documented once, in
the Python project's
[METHODOLOGY.md](../../../python/equity/01-options-pricing/docs/METHODOLOGY.md),
and applies here unchanged: this document covers what is specific to the
C++ engine.

## 1. Why a C++ engine alongside Python?

The Python package is the research reference: readable, NumPy/SciPy-backed,
easy to extend and audit. It is the wrong tool for three production
workloads a desk actually runs:

1. **Intraday risk on a large book.** Re-pricing 500k option positions with
   full Greeks on every meaningful underlier move needs millions of
   evaluations per second per core. The README latency table gives the
   measured per-call cost of every kernel.
2. **Quoting / auto-hedging latency.** A market-making loop has a per-event
   budget of microseconds. The `try_*` kernels price with no allocation, no
   exception and no thread creation on any path; implied vol takes 2-4
   evaluations of a normalised Black kernel (at most 10).
3. **Batch reval / scenario grids.** Overnight full-reval across thousands
   of scenarios is throughput-bound. `bs_price_batch` and `bs_greeks_batch`
   take structure-of-arrays spans; the validation pass vectorises, the
   pricing loop calls scalar libm `erfc`/`exp`/`log` and does not (section
   2, "Batch API").

The implementations are checked against each other in two ways. The golden
vectors come from mpmath (`python/equity/01-options-pricing/tests/golden/`),
not from either engine; `tools/gen_golden_header.py` compiles them into a
`constexpr` table and every output must lie within its own tolerance. A
differential test drives the Python, Rust and C++ engines with the same
seeded random inputs (README).

## 2. Design choices

### Two error interfaces: noexcept kernels and throwing wrappers
Invalid inputs (negative `S`, `K`, `T`, `sigma`; NaN or +/-Inf anywhere,
including the rates `r` and `q`) are errors in both interfaces.

- The kernel interface (`try_bs_price`, `try_d1_d2`, `try_bs_greeks`,
  `try_black76_price`, `try_black76_greeks`, `try_implied_vol`,
  `try_crr_price`, `try_mc_price`, `bs_price_batch`, `bs_greeks_batch`) is
  `noexcept` and returns `std::expected<T, eqopt::Error>`
  (`include/eqopt/error.hpp`). An `Error` is a code, a static field name
  and up to four numbers. Building one does not allocate, so the error path
  is as cheap as the success path. Validation branches are marked
  `[[unlikely]]`.
- The original interface (`bs_price`, `bs_greeks`, `implied_vol`, ...) is
  a thin wrapper: it calls the kernel and, on error, throws the same
  exception type with the same message as before (`message(Error)` builds
  the text). `std::invalid_argument` for bad inputs, `std::runtime_error`
  for implied-vol non-convergence.

A quoting loop calls the kernels and branches on the result. Research code
and tests keep the exceptions, which mirror the Python `ValueError`
contract. Domain-*limit* inputs that are financially meaningful
(`T == 0`, `sigma == 0`, `K == 0`, `S == 0`) are **not** errors: they
return the documented analytic limits, the same policy as Python.

### No allocation and no thread creation per call
`try_crr_price` takes the tree arrays as a caller span of
`crr_workspace_size(n) = 3n + 2` doubles. `try_mc_price` takes an
`McWorkspace` (path buffers, sized once with `reserve`) and a `ThreadPool*`.
`tests/test_no_alloc.cpp` replaces the global `operator new` with a
counting version and checks that every kernel success path makes zero heap
allocations. The throwing `crr_price` and `mc_price` still allocate their
buffers per call, as before; use the kernels on a latency path.

### Monte Carlo threading crossover
Splitting a small simulation across threads makes it slower. Handing work
to the pool costs a wake-up and a join, roughly 100 to 200 us on a WSL2
host, and a 10,000-path price takes only about 400 us on one thread. So
`try_mc_price` sends its chunks to the pool only when there are at least
`kMcParallelMinDraws` (16,384) independent draws (paths, or antithetic
pairs); below that the same chunks run one after another on the calling
thread. The chunking and the RNG streams depend only on `(seed, threads)`,
so the result is bit-identical either way
(`ThreadPoolMc.ParallelThresholdKeepsBitsOnBothSides`).

Measured before the threshold was added (Core 5 120U, 12 logical CPUs,
WSL2 Ubuntu, GCC 15.2 `-O3`, equity engine, antithetic and control
variate, `default_thread_pool()`, median of 201 calls, 15 for 1M paths):

| paths | 1 thread | 2 threads | 4 threads |
|---:|---:|---:|---:|
| 1,000 | 34 us | 150 us | 126 us |
| 10,000 | 424 us | 626 us | 531 us |
| 20,000 | 898 us | 1,068 us | 746 us |
| 50,000 | 2,099 us | 1,999 us | 1,462 us |
| 1,000,000 | 62.8 ms | 43.9 ms | 30.2 ms |

Worked example: 10,000 antithetic paths are 5,000 draws, under the
threshold, so `threads = 4` now runs four chunks inline at the one-thread
cost (about 0.42 ms rather than 0.53 ms) and returns the same bits as
before. 50,000 paths (25,000 draws) still use the pool. The threshold is
conservative: at 20,000 paths four threads were 17% faster on this host,
but two were slower, and the wake-up cost differs by host. Scaling stays
well below linear even at 1M paths (2.1x on 4 threads); the serial
control-variate and variance passes are part of that and are not split.

### Batch API
`bs_price_batch(BsBatchInputs, out)` and `bs_greeks_batch` take one
`std::span` per field. Each element is priced by the same inline kernel as
`try_bs_price` / `try_bs_greeks`, so batch and scalar results are equal bit
for bit (`tests/test_latency_api.cpp`). The batch runs a validation pass
first; on a bad element nothing is written and the error names the first
bad index. Vectorisation report (`-DEQOPT_VEC_REPORT=ON`): the validation
pass and the finite-result pass vectorise with GCC 15.2 and Clang 21.1; the
pricing loop does not, because `erfc`, `exp` and `log` are scalar libm
calls. glibc's vector variants are only enabled under `-ffast-math` and
return different bits, which this engine does not accept. `batch.cpp` is
compiled with `-fno-math-errno`, which lets the compiler inline `sqrt` and
changes no returned value.

### erfc-based normal CDF
`Phi(x) = erfc(-x/sqrt(2))/2` retains full relative precision in the left
tail (down to ~1e-300) where `0.5*(1 + erf(.))` cancels catastrophically
below x ~ -6. This is what keeps deep-wing prices positive, finite and
NaN-free in the domain-scan test rather than relying on clamping.

### Templated finite-difference Greeks
`fd_greeks` is a header-only template over any callable with the pricer
signature `(S, K, T, r, sigma, q, type) -> double`. The compiler inlines the
callable into the stencil loops with no virtual dispatch. `bs_price` itself
is defined out of line in `src/black_scholes.cpp` and the build does not
enable LTO, so the stencil calls it rather than inlining it.
Bump sizes are relative with an absolute floor; first derivatives default to
`h ~ 1e-5` (truncation ~ round-off for analytic pricers), second derivatives
to a larger `h ~ 2e-4` because round-off scales like `eps/h^2` (optimal
`h ~ eps^{1/4}`).

### Tree memory layout
The CRR backward induction uses **one array of n+1 values updated in
place** (in the caller's workspace), O(n) memory, no per-step allocation,
sequential access that stays in L1/L2 for any practical n. The
American/European choice is a template parameter, so the inner loop has no
per-node branch, and GCC and Clang both vectorise it. Node spots `S u^m` for
m = -n..n are formed once in log space (`exp(log S + m log u)`, O(n) exps)
and the American exercise values are read from that array, so backward
induction does no transcendental work and nothing accumulates by repeated
multiplication.

### Reproducible Monte Carlo
The generator is `std::mt19937_64` with an explicit seed, but normal
deviates do **not** use `std::normal_distribution` (its
algorithm is implementation-defined, so results would differ between
libstdc++/libc++/MSVC). Instead each 64-bit draw maps to a (0,1) uniform and
through an inverse normal CDF (Acklam's rational approximation plus one
Halley refinement against the erfc CDF, accurate to about 1e-15).
Consequences: one draw per deviate, no rejection loop, and identical
results for a given (seed, thread count) on a given compiler and standard
library. Results are not bit-identical across platforms: `exp`, `log` and
`erfc` come from the platform libm, which is not correctly rounded, and
compilers may contract to FMA. Multi-threading partitions paths into
`threads` deterministic chunks, each with a splitmix64-derived stream.

The chunks run on a persistent `ThreadPool` (`include/eqopt/thread_pool.hpp`):
threads are created once, idle workers spin briefly and then block, and a
job is a function pointer plus a context pointer (no `std::function`, no
allocation). The throwing `mc_price` uses `default_thread_pool()`;
`try_mc_price` uses the pool the caller passes. Which thread runs a chunk
does not change the result: for a fixed `(seed, threads)` the output is the
same with no pool or with pools of 1, 3 or 7 workers
(`ThreadPoolMc.ResultIndependentOfPoolSizeAndEqualsLegacyApi`). The
antithetic and control-variate options are template parameters of the
chunk loop, so the inner loop has no per-path branch, and the payoff floor
is `x > 0 ? x : 0`, which equals `std::fmax(x, 0.0)` for every `x` and
compiles to one `maxsd`.

**Generator choice.** A counter-based generator (Philox4x32-10) would make
the streams independent of the chunk count. It would also change every
Monte Carlo number in the repository, including the cross-language
differential test, which drives the Python and Rust engines with the same
`mt19937_64` streams. This round keeps `mt19937_64` plus the inverse CDF so
that the latency work changes no result; the chunk count stays part of the
reproducibility key `(seed, threads)`.

### Variance reduction
Antithetic pairs `(Z, -Z)` are averaged first; the pair averages are the
independent samples. The control variate `e^{-rT} S_T` (known mean
`S e^{-qT}`) is fitted on those pair averages, which is the optimal
coefficient for the pair-average estimator. Same estimator as the Python
and Rust engines.

## 3. Assumptions register (engine-specific)

Financial assumptions (GBM dynamics, constant vol/rates, continuous
dividend yield, frictionless hedging, ACT/365F) are inherited unchanged
from the Python reference, see its METHODOLOGY.md, items A1–A8, including
what breaks when each is violated. Additional engine-level assumptions:

1. **IEEE-754 binary64 semantics** (no `-ffast-math`). Violated, the
   golden tolerances and run-to-run reproducibility no longer hold. The
   build never enables fast-math.
2. **The golden header is regenerated whenever the golden JSON changes.**
   `tests/golden_vectors.hpp` is committed generated code; drift between
   JSON and header would silently validate against a stale reference. The
   generator is one command and the header carries a "do not edit" banner.
3. **`std::mt19937_64` sequence stability**, guaranteed by the C++ standard
   (the engine's parameters are normative), so seeds are portable.
4. **Single-writer memory model in threaded MC**: each chunk writes a
   disjoint slice; reductions are serial, in index order. The pool's only
   shared state is its job counter, hence determinism. The TSan build
   (`-fsanitize=thread`, Clang 21.1) runs the full suite clean.
5. **No numerical change from the latency work.** A program that prints
   64,203 legacy-API results and exception texts in hex-float gives the
   same output against the library before and after this change, with GCC
   15.2 and Clang 21.1.

## Ports of the newer models (November 2026)

Straight ports of the Python formulas, same constants and branches. The LR
tree recomputes node spots in the American loop (O(n^2) `exp`); a
precomputed spot table, as in the CRR kernel, is left for a later performance pass.
Bachelier's implied vol inverts the out-of-the-money side so in-the-money
quotes keep their digits.
