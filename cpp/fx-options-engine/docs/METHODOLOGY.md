# Methodology

## Why a C++ FX engine at all?

The Python reference project (`python/fx/01-options-pricing`) is the
*model authority*: readable, scipy-backed, exhaustively documented. It is
not, however, what runs inside an FX e-trading stack, for three reasons:

1. **Quoting latency.** A market-making tile re-prices its full ladder
   (spots × strikes × tenors, both sides) on every spot tick and every curve
   or vol update. EURUSD ticks arrive in bursts of hundreds per second; a
   2,000-instrument tile refresh must fit in a fraction of a millisecond so
   the quote stream never lags the market it is derived from. The measured
   per-call latency of each kernel is in the README Latency table.
2. **RFQ response budgets.** Multi-dealer platforms effectively rank
   dealers by response time; an RFQ answer that takes tens of milliseconds
   loses flow to one that takes hundreds of microseconds. The pieces of an RFQ path this
   engine covers (strike from a quoted delta at a given vol, premium, hedge
   deltas) are a few engine calls, single-digit microseconds here. There is
   no smile and no smile-consistent strike-from-delta in this engine.
3. **Determinism and deployability.** No interpreter, no GIL, no dependency
   on a scipy build; a static library with reproducible (per toolchain) Monte Carlo
   (explicit `mt19937_64` seed + inverse-CDF normals, no
   implementation-defined `std::normal_distribution`) that risk can rerun
   and reconcile on the same toolchain.

The division of labour: **Python defines the semantics, C++
reproduces them fast.** Any behavioural question ("what is theta's sign
convention?", "which PA-delta branch?") is answered by the Python source,
and the mpmath golden vectors plus the differential test check that the
C++ engine gives the same answer within stated tolerances.

## Model choice (recap; full discussion in the Python METHODOLOGY)

* **Garman–Kohlhagen** for European vanillas: Black–Scholes with the
  foreign rate as a continuous yield. Chosen over local/stochastic vol
  because vanilla FX quoting happens *in GK implied-vol space*, the model
  is the market's coordinate system, not a claim about dynamics. Chosen
  over numerical-only engines because closed forms give exact Greeks and
  microsecond latency.
* **Black-76 on the CIP forward** is provided separately because forward-
  space market data (outrights + domestic discount factor) is how desks
  actually receive inputs; it is algebraically identical to GK and tested
  as such to 1e-12.
* **CRR binomial** for American exercise (foreign rate enters exactly like
  a dividend yield; early exercise of a call becomes optimal when
  `r_f > r_d`). Chosen over finite differences for transparency and easy
  convergence testing against GK.
* **Monte Carlo** (exact terminal-spot sampling under the domestic
  measure) exists as an independent cross-check of the analytic prices and
  as the template for payoffs with no closed form.

## Design choices in the C++ implementation

* **Pure free functions, no state.** Pricing is `double in -> double out`;
  the only stateful object is the MC RNG, which is constructed per call
  (per chunk) from an explicit seed. This makes the engine thread-safe for
  read-only market data.
* **Two error interfaces over one kernel.** Each pricing kernel is a
  `noexcept` function `try_*` returning `std::expected<T, Error>`. An
  `Error` holds a code, a static field name and up to three numbers, so
  neither the success nor the error path allocates; `message()` formats the
  text only when asked. The throwing functions wrap these and raise
  `std::invalid_argument` (or `std::runtime_error` for solver
  non-convergence), mirroring the Python `ValueError` contract (same
  conditions, similar messages), so the implementations reject the same
  inputs. Validation is *total* at the
  API boundary: every entry point rejects NaN/Inf in any argument -
  including the two rates and the observed premium, because a single
  non-finite quote propagating through a book-level aggregate is far more
  expensive to find than an exception in the service log.
* **Statistics are only reported when they exist.** The Monte Carlo
  standard error is estimated from *independent* samples (pair averages
  under antithetic sampling). With fewer than two of them the SE is
  reported as 0 rather than the `0/0` NaN the naive formula produces, and
  the header documents the minimum path count for a meaningful error bar.
  Likewise the finite-difference Greeks refuse a `T` or `sigma` too small
  for their own central down-bump instead of pricing at a negative input.
* **Own normal-distribution kernel.** `N(x)` via `std::erfc` (full double
  precision in both tails); `N^{-1}(p)` via Acklam's rational approximation
  polished with two Halley steps (~1e-15). No dependence on any math
  library beyond `<cmath>`.
* **Root finding.** Implied vol uses the shared core (`iv_core.hpp`):
  normalised Black solved on the out-of-the-money side by safeguarded
  Halley iteration, at most 10 evaluations, the same algorithm and
  constants as the Python and Rust engines. Premium-adjusted
  strike-from-delta locates the fold of `K·N(d2(K))` analytically-in-d2 and
  Brent-solves on the market-standard decreasing branch. The Brent routine is a Numerical-Recipes-style zbrent (not scipy's implementation) and throws when it does not converge.
* **Limits handled explicitly**: `T = 0` returns intrinsic;
  `sigma·sqrt(T) <= 1e-12` returns discounted forward intrinsic, the same
  constants and branches as the Python module, so the two engines agree even
  at the edges.
* **No allocation or thread creation on the hot path.** The tree runs in a
  caller workspace (`binomial_workspace_size(n) = 3n + 2` doubles), Monte
  Carlo in an `McWorkspace` reserved once, and chunked Monte Carlo on a
  persistent `ThreadPool` (threads created in its constructor, a function
  pointer plus context per job, no `std::function`). Inner loops are
  instantiated per exercise style and per antithetic/control-variate
  choice, so they carry no per-node or per-path branch. A test binary with
  a counting `operator new` pins zero allocations on every kernel success
  path.
* **Batch pricing.** `gk_price_batch` and `gk_greeks_batch` take
  structure-of-arrays spans and call the same inline kernels as the scalar
  `try_*` functions, so the outputs match bit for bit.
* **Optimisation never changes a number.** Divisions, the order of sums
  and every libm call are kept as they were; the only flag added to a
  source file is `-fno-math-errno` on `batch.cpp`, which changes no
  returned value. `FXOPT_NATIVE` (`-O3 -march=native`) is off by default
  because FMA contraction can move last bits.
* **`-Wall -Wextra -Wpedantic`** (`-Werror` with `FXOPT_WERROR=ON`), C++23,
  no external dependencies beyond GoogleTest for the test binaries.

## Monte Carlo threading crossover
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
WSL2 Ubuntu, GCC 15.2 `-O3`, equity engine; the FX engine uses the same chunking and threshold, antithetic and control
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

## Assumptions

The full numbered assumptions register (CIP without cross-currency basis,
lognormal spot, constant rates/vol per pricing call, ACT/365F, no default
risk, etc.), with "what breaks if violated" for each, lives with the model
authority:
[`python/fx/01-options-pricing/docs/METHODOLOGY.md`](../../../python/fx/01-options-pricing/docs/METHODOLOGY.md).
This engine inherits every one of those assumptions verbatim; nothing in the
C++ port adds or relaxes any of them.

## Vectorised batch pricing (`gk_price_batch_fast`)

**The plain idea.** Pricing a thousand options one at a time spends most of
its time in `std::log`, `std::exp` and `std::erfc`, which are library calls
the compiler cannot vectorise. `gk_price_batch_fast` replaces them with the
branch-free versions in `qp/vmath.hpp` so the compiler can price two (SSE2)
or four (AVX2) options per instruction. The price is not bit-identical to
`gk_price`; `gk_price_batch` stays the bit-exact batch and is unchanged.

**Formal statement.** Same validation, the same `BatchError` for the same
first bad index, and the same `NonFiniteResult` check as `gk_price_batch`.
For every element, `|fast - exact| <= 1e-13 * max(1, S, K)`
(`BatchApi.FastPricesWithinDocumentedBound`, 100 000 random quotes plus the
boundary grid). Elements with `T = 0`, `sigma sqrt(T) <= kMinVol`, or `S / K` outside
the normal double range, go through `gk_price_unchecked` and
match `gk_price` bit for bit. On x86-64 Linux with GCC or Clang the kernel is
built twice (`target_clones("avx2", "default")`) and the loader picks the
AVX2 copy where the CPU has it; AVX2 does not enable FMA, so both copies
return the same bits. On MSVC the fast functions take the exact path (see the second pass below).

Three changes were needed before the loop vectorised, each confirmed with
`-fopt-info-vec` / `-Rpass=loop-vectorize`: the erfc reflection
`x < 0 ? 2 - r : r` became a bit-mask blend (GCC turned it into a branch
under the default `-ftrapping-math`), the Clenshaw loop is fully unrolled,
and the `qp::vm` kernels are `always_inline` (Clang otherwise kept `erfc1` as
a call). `qp::vm::log_batch` was added for `log(S / K)` (fdlibm reduction,
tested within 2 ulp).

**Worked example (measured).** `eqopt_batch_bench` (tools/batch_bench.cpp):
1024 random quotes, median of 21 runs of 2000 batches, steady_clock,
i5-120U under WSL2 with other jobs running (load average about 7, so the
spread is wide). Release build, no `-march=native`:

| Compiler | `try_bs_price` loop | `gk_price_batch` | `gk_price_batch_fast` | Speed-up over `gk_price_batch` |
|---|---|---|---|---|
| GCC 15.2 | 161 ns (130-244) | 135 ns (122-179) | 33 ns (30-41) | about 4x |
| Clang 21.1 | 122-182 ns | 108-260 ns | 57-125 ns (47-318) | 1.2x to 2.4x, noisy |

With `EQOPT_NATIVE=ON` (AVX2 and FMA) GCC measured 16-38 ns per option.
Before the three fixes above the fast kernel was scalar and no faster than
`gk_price_batch` (about 159 ns against 137 ns). The table above is for the equity kernel; FX numbers are in the second pass below.

### Greeks batch, FX numbers and MSVC (second pass)

`bs_greeks_batch_fast` and `gk_greeks_batch_fast` use the same kernels.
Bound per field: `|fast - exact| <= 1e-12 * max(1, S, K, |exact|)`
(`BatchApi.FastGreeksWithinDocumentedBound`, 50 000 random quotes plus the
grid). The Greeks loop only vectorised once every kernel pointer was a
`__restrict__` parameter: with 7 inputs and up to 11 outputs GCC needs more
run-time alias checks than `--param vect-max-version-for-alias-checks=10`
allows.

Measured with `eqopt_batch_bench` and `fxopt_batch_bench` (same method as
above, a later and quieter run, median ns per option, GCC 15.2 Release):

| Kernel | exact batch | `_fast` | Speed-up |
|---|---|---|---|
| equity price | 94 | 23 | about 4x |
| equity Greeks | 91 | 29 | about 3x |
| FX price | 89 | 28 | about 3x |
| FX Greeks | 110 | 27 | about 4x |

Clang 21.1 in the same session: equity 185 to 73 (price) and 197 to 101
(Greeks), FX 81 to 33 and 91 to 55, so 1.6x to 2.5x.

MSVC (VS 2022, `/O2`) does not vectorise the kernel and the scalar vmath code
was 3-4x slower than the exact batch there (equity price 189 against 622 ns).
On MSVC the four `_fast` functions therefore call the exact batch and are
bit-exact; both engines build with `/W4 /WX` and pass ctest (92 and 90 tests).
