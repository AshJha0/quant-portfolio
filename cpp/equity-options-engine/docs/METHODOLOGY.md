# Methodology

This engine is the C++20 port of the Python reference package
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
   evaluations per second per core. This engine does about 12M analytic
   prices (about 13M full Greek sets) per second on one core of the test
   machine; the scalar Python reference does about 0.25M (4 µs per call).
2. **Quoting / auto-hedging latency.** A market-making loop has a per-event
   budget of microseconds. A single BSM price here costs about 85 ns with
   no interpreter and no allocation; implied vol takes 2-4 evaluations of a
   normalised Black kernel (at most 10), about 0.8 µs per solve.
3. **Batch reval / scenario grids.** Overnight full-reval across thousands
   of scenarios is throughput-bound; a compiled kernel with flat memory
   access is much faster than a Python loop and parallelises across
   scenarios. There is no batch or SIMD API yet; every entry point is
   scalar.

The implementations are checked against each other in two ways. The golden
vectors come from mpmath (`python/equity/01-options-pricing/tests/golden/`),
not from either engine; `tools/gen_golden_header.py` compiles them into a
`constexpr` table and every output must lie within its own tolerance. A
differential test drives the Python, Rust and C++ engines with the same
seeded random inputs (README).

## 2. Design choices

### Exceptions vs error codes
Invalid inputs (negative `S`, `K`, `T`, `sigma`; NaN or +/-Inf anywhere,
including the rates `r` and `q`) throw `std::invalid_argument`, mirroring
the Python `ValueError` contract.
Rationale: these are *programming/contract* errors at the API boundary, not
expected runtime states, an error-code or `std::expected` return would tax
every hot-path call site with branching and would let a corrupted input
silently propagate into a book-level number. Exceptions cost nothing on the
non-throwing path with zero-cost unwinding, and the throwing path is
by definition off the hot loop. Domain-*limit* inputs that are financially
meaningful (`T == 0`, `sigma == 0`, `K == 0`, `S == 0`) are **not** errors:
they return the documented analytic limits, the same policy as Python.

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
The CRR backward induction uses a **single `std::vector<double>` of n+1
values updated in place**, O(n) memory, no per-step allocation, sequential
access that stays in L1/L2 for any practical n. Node spots `S u^m` for
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
deterministic chunks, each with a splitmix64-derived stream.

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
4. **Single-writer memory model in threaded MC**: each worker writes a
   disjoint slice; reductions are serial. No atomics, no ordering
   sensitivity, hence determinism.
