# Validation

Everything below is enforced by the GoogleTest suite
(`ctest --test-dir build --output-on-failure`, 66 tests). It passes with
GCC 15.2, Clang 21.1 and a GCC AddressSanitizer + UndefinedBehaviorSanitizer
build, all with `-DEQOPT_WERROR=ON`.

## 1. Golden vectors from an independent oracle

- `python/equity/01-options-pricing/tests/golden/generate_golden.py`
  computes **456 records** in mpmath at 50 significant digits: BSM price,
  delta, gamma, vega, theta, rho, vanna, volga (51 records, incl. T=1e-6,
  T=30, sigma=0.001, sigma=3, |d1| > 8); Black-76 (10); implied-vol round
  trips over ln(K/S) in [-3, 3], T in [1e-5, 50], sigma in [1e-3, 2] (353,
  incl. the K=90, T=0.02 OTM put at 1e-12); CRR European and American
  trees run in mpmath with the same recursion (40); and exact limits
  (sigma=0 with an overflowing forward; American put at S=0 with r<0).
- Each output carries an absolute tolerance from a stated rule:
  `64 eps (1 + d1^2)(|v| + max(S, K))` for closed forms,
  `4 max(eps max(S,K)/vega, 1e-12 sigma)` for implied vol,
  `4 n eps max(S, K)` for an n-step tree.
- `tools/gen_golden_header.py` emits `tests/golden_vectors.hpp`
  (shortest round-trip doubles); `--check` fails if the header and JSON
  differ. `test_golden.cpp` checks every record and records the worst
  deviation over tolerance: **0.18**.

## 2. Cross-language differential test

`eqopt_jsonl` (from `tools/jsonl_main.cpp`) answers JSON-lines requests.
`python/equity/01-options-pricing/tests/test_differential.py` sends 640
seeded random requests (closed form and Greeks, implied vol, CRR,
Black-76) to the Python, Rust and C++ engines and compares each answer
with Python. Largest deviations from Python (GCC 15 build):

| function | output | max abs | max rel |
|---|---|---:|---:|
| bs | price | 1.1e-13 | 4.3e-10 (tiny prices) |
| bs | theta | 3.4e-13 | 2.2e-13 |
| bs | volga | 4.5e-13 | 6.2e-16 |
| iv | sigma | 2.1e-13 | 2.2e-13 |
| crr | price | 0 | 0 |
| b76 | price | 8.5e-14 | 2.6e-10 (tiny prices) |

Equality is not expected: libm `erfc` and `exp` differ by ulps between
platforms, and the compilers may contract to FMA. The engines agree per
target and toolchain, not bit for bit across platforms.

## 3. Analytic identities

| Identity | Grid | Tolerance |
|---|---|---|
| Put-call parity `C - P = S e^{-qT} - K e^{-rT}` | 486 combinations | 1e-12 (rel to S) |
| Black-76 == BSM at `F = S e^{(r-q)T}` | 864 combinations x call/put | 1e-12 |
| `rho_B76 = -T * V` | both types | 1e-12 |
| Call/put gamma, vega, vanna, volga identical | spot check | exact |
| Analytic vs central-FD Greeks | 5 cases incl. negative rates | 1e-6 rel |

## 4. Convergence and statistical checks

- **CRR to BSM:** n=2000 within 2e-3; odd/even-averaged error decreases
  across n = 50, 200, 800, 2000.
- **American ordering:** American >= European; with `q = 0` the American
  call equals the European call to 1e-12.
- **Monte Carlo:** 200k paths with antithetic + control variate lands
  within 3 SE of the closed form. The control-variate coefficient is
  fitted on antithetic pair averages; over seven seeds the combined SE is
  no larger than the control variate alone and below 0.015 (the raw-draw
  fit gave 0.0247). Fewer than two independent samples (three with the
  control variate) throws.
- **Reproducibility:** same (seed, threads) gives an identical `MCResult`
  on a given toolchain.
- **Implied vol:** at most 10 evaluations over a grid of up to 180 points, and
  within `4 max(eps price / vega, 1e-12 sigma)` of the input vol on that
  grid. The K=90, T=0.02, sigma=10% OTM put recovers to 1e-12. The ITM
  call at the same strike carries less information (one ulp of the price
  moves the vol by about 5e-4), so the test checks that the returned vol
  reprices the quote to a few ulps.

## 5. Edge cases and failure behaviour

- `T = 0` gives intrinsic; `sigma = 0` gives `max(+-(S e^{-qT} - K e^{-rT}), 0)`,
  finite even when the forward overflows.
- Results that overflow (a discount factor `e^{-rT}` beyond double range)
  throw rather than return `inf` or NaN.
- American put at `S = 0`: `K` when `r >= 0`, `K e^{-rT}` when `r < 0`.
- NaN or +/-Inf in any input throws `std::invalid_argument`.
- Implied vol rejects prices at or outside the no-arbitrage bounds and
  `T = 0`; non-convergence (not observed) throws `std::runtime_error`.
- **NaN-free domain scan:** 864-point sweep; every price finite and inside
  its bounds.

Model failure modes (smile, discrete dividends, term structure, coarse
trees) are documented in the Python reference
[VALIDATION.md](../../../python/equity/01-options-pricing/docs/VALIDATION.md).

## 6. Kernel API, allocation and threading tests

| Test | What it pins |
|---|---|
| `NoexceptApi.*` (5) | `try_*` equals the throwing API bit for bit over a 1,280-point grid that includes every edge limit; errors carry the same message as the exception |
| `BatchApi.*` (3) | `bs_price_batch` and `bs_greeks_batch` equal the scalar API bit for bit; a bad element is reported by index and nothing is written |
| `ThreadPoolMc.*` (2) | `try_mc_price` equals `mc_price` for threads 1, 2, 4, 9 on no pool and on pools of 0, 1, 3, 7 workers, three runs each; workspace size is checked |
| `ThreadPool.RunsEveryTaskOnceUnderRepeatedUse` | 500 jobs of 64 tasks on 5 workers, every task runs once per job |
| `eqopt_alloc_tests` (5) | counting `operator new`: zero heap allocations on every kernel success path (analytic, implied vol, CRR with workspace, batch, MC with pool and workspace) and on the kernel error path |
| `static_assert`s | every kernel entry point is `noexcept`; `Error` is trivially copyable |

The allocation test is a separate binary because it replaces the global
`operator new`. Sanitizer runtimes define `operator new` themselves, so the
sanitizer builds skip it.

The latency work changed no result. Evidence: a program that calls only
the legacy API prints 64,203 results and exception texts in hex-float; the
output against the library before and after the change is identical with
GCC 15.2 and with Clang 21.1. The 66 pre-existing tests pass unchanged.

## 7. Performance

See the README latency table (`./build/eqopt_bench`).

## Phase 5 ports (Round 3)

Golden values printed from the Python reference (`repr`) for four LR cases
(European and American, negative rate), two Bachelier Greek sets and two
higher-Greek sets; LR error ratio per doubling in (3, 5); Bachelier
implied-vol round trip on both sides of the money; speed against a finite
difference of gamma. The models' own oracles are in the Python suite.
