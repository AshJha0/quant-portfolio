# Desk Guide, where a C++ pricing engine sits in the real workflow

The Python reference's
[DESK_GUIDE.md](../../../python/equity/01-options-pricing/docs/DESK_GUIDE.md)
covers the *financial* workflow: who consumes prices and Greeks, hedging
cadence, P&L attribution, limits and scenarios. This guide covers the
*systems* question: which jobs on a desk run through a compiled engine like
this one, and how such an engine is governed.

## 1. The three production seats for a C++ engine

### Intraday risk service
The desk's risk view ("what is my delta/gamma/vega *right now*") is a
long-running service that re-prices the whole position set on every
meaningful underlier/vol tick. Order of magnitude: 100k–1M option positions
x full Greek set, refreshed every few seconds. At the per-call cost in the
README latency table (about 0.1 µs for a full Greek set), a full book sweep
is tens of milliseconds on a handful of cores, fast enough that traders see
risk move *with* the market, not 30 seconds behind it. `bs_greeks_batch`
takes the book as structure-of-arrays spans and fills one output span per
Greek. The Python package in the
same role would be the bottleneck; instead it is the *reference* the
service is validated against.

### Quoting / auto-hedging engines
Market-making systems price two-sided quotes off a marked vol surface and
re-hedge on fills. The per-event budget is microseconds, allocation-free,
GC-free, GIL-free. The `try_*` kernels fit that shape: they are `noexcept`,
return `std::expected<double, Error>`, and make no heap allocation on any
path (pinned by `eqopt_alloc_tests`). A tree uses a workspace sized once at
start-up (`crr_workspace_size`); Monte Carlo uses an `McWorkspace` and a
`ThreadPool` created once, so no thread is started per price. (A real quoting stack adds surface
interpolation and position caching on top; the pricing kernel is the part
that must never be the slow link.)

### Batch reval and scenario grids
End-of-day official P&L, VaR/stress grids (hundreds of scenarios x full
book), and margin replication are throughput problems: billions of
pricings on a schedule. Compiled kernels parallelise trivially across
scenarios (the MC engine's deterministic seed-partitioned threading is the
same pattern), and for sign-off they agree with the research library
within stated tolerances, because both are pinned to the same mpmath golden
vectors and checked against each other by the differential test.

## 2. Split of responsibilities: Python researches, C++ serves

- Model changes, calibration experiments, new payoffs: prototyped and
  validated in the Python package (fast iteration, rich ecosystem).
- Once accepted, the change is ported here, and the *contract* between the
  two is renewed mechanically: the golden JSON (mpmath) is regenerated if
  the contract changes, `tools/gen_golden_header.py` regenerates the C++
  header, both suites must pass with every output inside its tolerance,
  and the differential test must pass on seeded random inputs.

This is the standard two-implementation control quants are asked about in
model governance: an independent implementation agreeing to near machine
precision is strong evidence against implementation error (it cannot
detect *model* error, both implementations share assumptions
A1–A8; that risk is handled by the model-validation material in the Python
docs).

## 3. Golden-vector regression as the release gate

Every release of this engine (compiler upgrade, flag change, refactor,
new feature) must pass, in order:

1. **Build gate:** `-DEQOPT_WERROR=ON` with GCC and Clang, zero warnings,
   plus an ASan+UBSan build.
2. **Golden gate:** all 456 mpmath records within their tolerances, and
   the differential test against Python and Rust.
   Catches: maths edits, accidental convention drift (per-day vs per-year
   theta is the classic), CDF/precision regressions, aggressive-optimisation
   breakage (e.g. someone enabling `-ffast-math`).
3. **Identity gate:** parity, Black-76==BSM, analytic-vs-FD Greeks -
   catches errors that golden vectors alone might miss (a bug symmetric in
   call/put, say).
4. **Statistical gate:** MC within 3 SE, variance reduction effective,
   identical results for a given (seed, threads) on the same toolchain,
   which makes "rerun yesterday's batch" a meaningful debugging tool.
5. **Convergence gate:** tree within 2e-3 of BSM at n=2000, error decaying.

A red gate blocks the release; there is no "small numeric diff, ship it
anyway" path, because a 1e-6 price drift across a million-position book is
a real P&L restatement.

## 4. Operational notes and controls

- **Determinism is a feature, not a nicety.** Official reval uses pinned
  seeds recorded with the run; any number can be reproduced on the same
  toolchain. The in-house inverse-CDF sampler removes the dependence on
  `std::normal_distribution`, whose algorithm varies by standard library;
  libm differences in `exp`/`log` still change the last bits across
  platforms.
- **Input hygiene at the boundary.** The engine rejects negative or
  non-finite (NaN/Inf) inputs, including non-finite rates, and
  sub-intrinsic implied-vol requests instead of returning a plausible
  number. The kernels return an `Error` (code, field, values); the
  throwing API turns it into the same exception text as before. Upstream
  data problems (stale quotes, crossed markets, bad dividend feeds) surface
  as errors in the service log, not as silent risk misstatements.
- **Latency builds.** `-DEQOPT_NATIVE=ON` adds `-O3 -march=native` and
  `-DEQOPT_LTO=ON` turns on link-time optimisation; both are off by default
  and never used in CI. `-march=native` can enable FMA contraction, which
  moves last bits, so the golden and bit-equality tests are the check to
  run on such a build.
- **Edge-case policy is part of the interface.** `T=0` intrinsic and
  `sigma=0` discounted-forward-intrinsic limits are documented and tested;
  expiry-day and pinned-strike behaviour therefore matches the reference
  library exactly, no reconciliation break at 3:59pm on expiry Friday.
- **Known limits.** Flat vol, continuous dividends, European MC/BSM:
  the same A1–A8 assumptions as the reference. The engine prices *a* model
  fast; choosing the right vol/dividend inputs per the desk's marking
  process is upstream of it (see the Python DESK_GUIDE for that workflow).

## Phase 5 ports (Round 3)

Use the compiled LR tree where an American mark must be reproduced outside
Python; Bachelier for normal-vol quotes on the serving path; higher-order
Greeks for the overnight drift report. Each must agree with the Python
golden values before release, as for the other kernels.
