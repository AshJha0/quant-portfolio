# Methodology

This engine is the C++20 twin of the Python reference package
[`eq_options`](../../../python/equity/01-options-pricing/). The *financial*
methodology — model selection (Black-Scholes-Merton vs CRR vs Monte Carlo,
and why each earns its place), the full assumptions register with
"what breaks if violated", and the edge-case policy — is documented once, in
the Python project's
[METHODOLOGY.md](../../../python/equity/01-options-pricing/docs/METHODOLOGY.md),
and is *identical* here by construction: this document covers what is
specific to the C++ engine.

## 1. Why a C++ engine alongside Python?

The Python package is the research reference: readable, NumPy/SciPy-backed,
easy to extend and audit. It is the wrong tool for three production
workloads a desk actually runs:

1. **Intraday risk on a large book.** Re-pricing 500k option positions with
   full Greeks on every meaningful underlier move needs millions of
   evaluations per second per core. This engine does ~16M analytic prices
   (~12M full Greek sets) per second per core — a full book reval in tens of
   milliseconds, versus seconds-to-minutes through a Python call stack.
2. **Quoting / auto-hedging latency.** A market-making loop has a per-event
   budget of microseconds. A single BSM price here costs ~60 ns with no
   interpreter, no GIL, no allocation; implied vol solves in a handful of
   iterations of the same kernel.
3. **Batch reval / scenario grids.** Overnight full-reval across thousands
   of scenarios is throughput-bound; a compiled kernel with flat memory
   access wins by two orders of magnitude and parallelises trivially.

The two implementations are kept honest against each other by
**golden-vector cross-validation**: the Python project emits
`golden_vectors.json`; `tools/gen_golden_header.py` compiles it into a
`constexpr` table; the C++ test suite must reproduce every number to 1e-9
(measured: ~5e-15). Same maths, two codebases, one set of numbers.

## 2. Design choices

### Exceptions vs error codes
Invalid inputs (negative `S`, `K`, `T`, `sigma`, NaN) throw
`std::invalid_argument`, mirroring the Python `ValueError` contract.
Rationale: these are *programming/contract* errors at the API boundary, not
expected runtime states — an error-code or `std::expected` return would tax
every hot-path call site with branching and would let a corrupted input
silently propagate into a book-level number. Exceptions cost nothing on the
non-throwing path with modern zero-cost unwinding, and the throwing path is
by definition off the hot loop. Domain-*limit* inputs that are financially
meaningful (`T == 0`, `sigma == 0`, `K == 0`, `S == 0`) are **not** errors:
they return the documented analytic limits, byte-matching the Python policy.

### erfc-based normal CDF
`Phi(x) = erfc(-x/sqrt(2))/2` retains full relative precision in the left
tail (down to ~1e-300) where `0.5*(1 + erf(.))` cancels catastrophically
below x ~ -6. This is what keeps deep-wing prices positive, finite and
NaN-free in the domain-scan test rather than relying on clamping.

### Templated finite-difference Greeks
`fd_greeks` is a header-only template over any callable with the pricer
signature `(S, K, T, r, sigma, q, type) -> double`. The compiler inlines the
pricer into the stencil loops — bump-and-revalue Greeks for the tree or any
future model come for free, at native speed, with zero virtual dispatch.
Bump sizes are relative with an absolute floor; first derivatives default to
`h ~ 1e-5` (truncation ~ round-off for analytic pricers), second derivatives
to a larger `h ~ 2e-4` because round-off scales like `eps/h^2` (optimal
`h ~ eps^{1/4}`).

### Tree memory layout
The CRR backward induction uses a **single `std::vector<double>` of n+1
values updated in place** — O(n) memory, no per-step allocation, sequential
access that stays in L1/L2 for any practical n. Node prices needed for the
American exercise comparison are recomputed in log space
(`S * exp((2j - i) * log u)`) rather than stored or accumulated
multiplicatively: this matches the Python reference bit-for-bit in structure
and avoids drift at large n from repeated multiplication.

### Reproducible Monte Carlo
The generator is `std::mt19937_64` with an explicit seed, but normal
deviates deliberately do **not** use `std::normal_distribution` (its
algorithm is implementation-defined, so results would differ between
libstdc++/libc++/MSVC). Instead each 64-bit draw maps to a (0,1) uniform and
through an inverse normal CDF (Acklam's rational approximation plus one
Halley refinement against the erfc CDF, accurate to ~1e-15). Consequences:
one draw per deviate, no rejection loop, and **bit-identical results for a
given (seed, thread-count) on any conforming platform** — the property the
regression suite asserts. Multi-threading partitions paths into
deterministic chunks, each with a splitmix64-derived stream.

### Variance reduction
Antithetic pairs `(Z, -Z)` with the standard error computed on *pair means*
(the unbiased estimator under pair correlation), and a control variate
`e^{-rT} S_T` with known mean `S e^{-qT}` and the sample-optimal
coefficient. Same estimators, same defaults as the Python reference.

## 3. Assumptions register (engine-specific)

Financial assumptions (GBM dynamics, constant vol/rates, continuous
dividend yield, frictionless hedging, ACT/365F) are inherited unchanged
from the Python reference — see its METHODOLOGY.md, items A1–A8, including
what breaks when each is violated. Additional engine-level assumptions:

1. **IEEE-754 binary64 semantics** (no `-ffast-math`). Violated =>
   golden-vector agreement and bit-reproducibility claims void. The build
   never enables fast-math; `-O2` preserves IEEE semantics on GCC.
2. **Golden vectors are regenerated whenever the Python reference changes.**
   `tests/golden_vectors.hpp` is committed generated code; drift between
   JSON and header would silently validate against a stale reference. The
   generator is one command and the header carries a "do not edit" banner.
3. **`std::mt19937_64` sequence stability**, guaranteed by the C++ standard
   (the engine's parameters are normative), so seeds are portable.
4. **Single-writer memory model in threaded MC**: each worker writes a
   disjoint slice; reductions are serial. No atomics, no ordering
   sensitivity, hence determinism.
