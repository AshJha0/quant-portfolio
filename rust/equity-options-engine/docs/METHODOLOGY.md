# Methodology — eq-options-engine (Rust)

The mathematics of every model in this crate is documented once, in the
Python reference project
(`python/equity/01-options-pricing/docs/METHODOLOGY.md`): Black-Scholes-
Merton with continuous dividend yield, CRR binomial trees, Black-76, and
exact-scheme GBM Monte Carlo, including the model-choice comparisons
(BSM vs local/stochastic vol, CRR vs trinomial/PDE, exact GBM vs Euler).
This document covers what is *specific to the Rust engine*: why Rust, the
design decisions, and the assumptions register as it applies here.

## 1. Why a Rust engine? (vs the two incumbents)

The portfolio deliberately implements the same pricing library three
times. Each implementation answers a different production question.

| Criterion | Python (`eq_options`) | C++ (`eqopt`) | **Rust (this crate)** |
|---|---|---|---|
| Role | Research reference, golden-vector source | Incumbent low-latency production shape | Memory-safe production engine for new services |
| Throughput (1 core, this container) | ~0.1M prices/sec | ~16M prices/sec | ~12M prices/sec |
| Memory safety | GC'd, safe | Manual; UB possible (dangling refs, data races) | Compiler-enforced, **without** a garbage collector |
| Latency profile | GC pauses + interpreter jitter | Deterministic | Deterministic — no GC, no runtime |
| Concurrency | GIL-bound | Possible but races are on the programmer | "Fearless": `Send`/`Sync` checked at compile time |
| Error handling | Exceptions (`ValueError`) | Exceptions (`std::invalid_argument`) | `Result<_, PricingError>` — errors in the signature |
| Dependency posture | numpy/scipy stack | std + GoogleTest | **zero** external crates |

The two arguments that matter on a desk:

1. **Memory safety without GC pauses.** A risk service reprices a book on
   market-data ticks; a garbage collector that stops the world for 10 ms
   at the wrong moment is a real operational cost (this is why
   JVM/CLR-based pricers are rare in the latency path). C++ avoids GC but
   buys undefined behaviour: use-after-free and out-of-bounds writes in
   pricing code corrupt *numbers*, not just crash processes — the worst
   failure mode in finance is a silently wrong Greek. Rust's borrow
   checker eliminates that class at compile time while producing the same
   flat, allocation-free machine code as C++ (see the benchmark table:
   both engines sit in the 12-16M prices/sec band on this container).
2. **Fearless concurrency for parallel revaluation.** Portfolio reval is
   embarrassingly parallel (positions x scenarios). In C++ a data race on
   a shared vol surface is a latent, non-deterministic bug; in Rust code
   that races does not compile (`&mut` aliasing rules, `Send`/`Sync`
   auto-traits). This crate keeps the core single-threaded and pure —
   every function takes its inputs by value — precisely so callers can
   shard work across threads trivially and safely.

## 2. Design decisions

### 2.1 `Result`-based errors vs C++ exceptions

The Python reference raises `ValueError`; the C++ engine throws
`std::invalid_argument`. This crate returns
`Result<T, PricingError>` with three variants: `InvalidInput`,
`NoConvergence`, `ArbitrageBound` (a manual enum with hand-written
`Display`/`Error` impls — the thiserror pattern without the dependency).
Rationale:

- The failure set of a pricer is *part of its contract* (a sub-intrinsic
  quote has no implied vol; that is domain information, not an
  exceptional event). Encoding it in the signature forces every caller —
  including a batch revaluation loop over a million quotes — to decide
  what a bad input does, instead of unwinding through the stack.
- No unwinding means the engine can be compiled with `panic = "abort"`
  and embedded behind an FFI boundary (C, Python via ctypes/PyO3) where
  unwinding across the boundary is undefined behaviour.
- Semantics are mirrored 1:1: every condition that raises in Python
  returns the corresponding `PricingError` here, tested case by case.
- **One deliberate divergence, in the safe direction**: the Python
  reference rejects NaN but lets `inf` propagate into `inf`/NaN outputs.
  Every public entry point here also rejects `+/-inf` in `S`, `K`, `T`,
  `sigma` and NaN/`inf` in the rates `r`/`q`, so a corrupt tick cannot
  silently become a NaN risk number. No admissible input changes value,
  so the golden vectors are unaffected.

### 2.2 Zero-dependency determinism

The crate depends on `std` only. This is a deliberate model-governance
stance, not asceticism:

- **Auditability**: a model validator can read every line that produces a
  number — including the normal CDF and the RNG — in ~2,500 lines of
  first-party code. No transitive crate graph to review or re-review on
  every `cargo update`.
- **Reproducibility**: results cannot drift because a dependency changed
  its algorithm. Same source + same seed => bit-identical Monte Carlo on
  every platform, forever.
- **Supply chain**: pricing engines are high-value targets; an empty
  dependency tree is the strongest possible position.

The two components a normal crate would import:

- **Normal CDF**: `Phi(x) = 0.5 * erfc(-x/sqrt(2))` with erfc implemented
  from W. J. Cody's rational Chebyshev approximation (SIAM 1969, the
  netlib `CALERF` algorithm used inside most libm implementations):
  three regimes with the `exp(-x^2)` factor split for tail accuracy.
  Verified in-tests against high-precision reference values to < 2e-15
  relative — an order of magnitude tighter than the 1e-12 requirement.
  The erfc route (rather than `0.5(1+erf)`) preserves *relative* accuracy
  deep in the tail, which is what keeps far-OTM prices and wing implied
  vols meaningful.
- **RNG** (`src/rng.rs`, choice documented there in full):
  **xoshiro256++** (Blackman & Vigna 2019 — 256-bit state, passes
  BigCrush, ~1 ns/u64) seeded through **SplitMix64** (the seeding the
  xoshiro authors specify; well-mixed non-zero state for every seed,
  including 0), and **Box-Muller (trigonometric form)** for normals —
  exact in distribution, two `u64` draws per pair, trivially auditable
  versus a Ziggurat table or a rational inverse-CDF whose tail error
  would need its own validation. Uniforms use the top 53 bits mapped to
  `(m + 0.5) * 2^-53`, so 0 and 1 are unattainable and `ln(u)` is safe.

### 2.3 Other choices

- **Pure functions, no state.** Mirroring the conventions: pricing is
  free functions over `f64`; the only stateful object is the RNG, which
  is explicit and seed-constructed.
- **O(n) tree memory.** The CRR backward induction reuses one `Vec<f64>`
  of `n+1` nodes, recomputing spot from log-space (`exp(logS + (2j-i)
  log u)`) rather than storing a 2-D lattice — same layout as the C++
  engine, cache-friendly at n=1000+.
- **Generic FD Greeks over closures.** `fd_greeks` takes any
  `Fn(S,K,T,r,sigma,q,OptionType) -> Result<f64, PricingError>`, so the
  same second-order stencils (with the documented `h ~ eps^0.25` bumps
  for second derivatives) validate the analytic Greeks, the tree, and
  any future pricer.
- **MC estimator semantics copied from Python exactly**: antithetic
  standard errors computed on pair averages; control-variate coefficient
  from the sample covariance (ddof=1); `n_paths` rounded up to even under
  antithetic. Numbers differ from NumPy only through the RNG stream.

## 3. Assumptions register

Inherited from the reference implementation (full versions with
derivations in the Python project's METHODOLOGY.md); consequences
restated here.

1. **GBM dynamics, constant `r`, `q`, `sigma`.** Breaks: smile/skew is
   not generated — one sigma per (K, T) input; term-structure effects
   need per-expiry inputs. Wrong hedges in strongly heteroskedastic
   regimes.
2. **Continuous dividend yield `q`.** Breaks: discrete cash dividends
   around ex-dates (mispriced early exercise for American calls).
3. **Frictionless markets** (no transaction costs, continuous hedging,
   unlimited shorting). Breaks: replication argument, so prices are
   mid-market references, not tradeable quotes.
4. **ACT/365F, continuously compounded, annualised inputs.** Breaks:
   silent unit errors if fed ACT/360 money-market rates or per-day vols —
   the doc comments state units on every argument for this reason.
5. **CRR risk-neutral probability inside (0,1).** Enforced: the tree
   *returns an error* when `dt` is too large for `|r - q|` vs `sigma`
   rather than extrapolating.
6. **MC error is purely statistical** (exact terminal-distribution
   sampling — no discretisation bias). Breaks: nothing to break for
   vanilla payoffs; path-dependent products would need a scheme with
   time-stepping and its own bias analysis.
7. **IEEE-754 f64 throughout.** Deep wings underflow: `Phi(x) = 0` below
   x ~ -37.5; prices below ~1e-300 are not representable. Edge tests pin
   the behaviour.
8. **Inputs are finite.** Non-finite inputs are treated as a caller/data
   error, never as a pricing regime: they return `InvalidInput` at the
   entry point. Breaks: a caller relying on `inf` as "infinitely far
   OTM" must clamp to a large finite number instead — deliberate, since
   the alternative is a NaN that survives aggregation into a book-level
   risk figure.
