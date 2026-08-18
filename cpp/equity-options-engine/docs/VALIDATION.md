# Validation

Everything below is enforced by the committed GoogleTest suite
(`ctest --test-dir build --output-on-failure`, 48 tests, ~130 assertion
sites, thousands of executed assertions across parameter grids). The build
is warnings-as-errors (`-Wall -Wextra -Werror`).

## 1. Headline: cross-language golden validation

The strongest check a twin engine can offer is *identical numbers* against
its reference implementation.

- The Python project (`eq_options`) generates and commits
  `tests/golden/golden_vectors.json`: **32 cases** spanning ATM/ITM/OTM
  (moneyness 0.5x–2.0x), expiries from ~1 week to 5 years, positive and
  negative rates, zero and non-zero dividend yields, calls and puts — each
  with price, delta, gamma, vega, theta, rho computed by the SciPy-backed
  reference.
- `tools/gen_golden_header.py` converts the JSON to
  `tests/golden_vectors.hpp` (a `constexpr` array; doubles emitted via
  shortest-roundtrip repr, so the header carries the Python values
  bit-exactly). Both files are committed; the header carries a "do not
  edit" banner.
- `test_golden.cpp` asserts every one of the 32 x 6 = 192 numbers to
  **1e-9 abs/rel**.

**Result: the measured worst-case deviation across all 192 numbers is
5.3e-15** — Python (SciPy `norm.cdf`) and C++ (`erfc`-based CDF) agree to
within a few ulps of double precision. Unit conventions were verified
against the Python `greeks` module before comparison: theta per *year*,
vega per *unit* of vol, rho per *unit* of rate.

This doubles as the release gate: any change to either codebase that moves
a price or Greek by more than 1e-9 fails CI on the other side.

## 2. Analytic identities

| Identity | Grid | Tolerance |
|---|---|---|
| Put-call parity `C - P = S e^{-qT} - K e^{-rT}` | 486 combinations (S, K, T, sigma, r, q) | 1e-12 (rel to S) |
| Black-76 == BSM at `F = S e^{(r-q)T}` | 864 combinations x call/put | 1e-12 |
| Black-76 forward parity `C - P = e^{-rT}(F - K)` | spot check | 1e-12 |
| `rho_B76 = -T * V` (pure discounting) | both types | 1e-12 |
| Call/put gamma, vega, vanna, volga identical | spot check | exact (`EXPECT_DOUBLE_EQ`) |
| `delta_C - delta_P = e^{-qT}`, `rho_C - rho_P = K T e^{-rT}` | spot check | 1e-14 / 1e-10 |
| Analytic vs central-FD Greeks (all seven) | 5 cases incl. negative rates | 1e-6 rel |

## 3. Convergence and statistical checks

- **CRR -> BSM (European):** n=2000 within 2e-3 of the closed form;
  odd/even-averaged error decreases monotonically across n = 50, 200, 800,
  2000 (the raw CRR error oscillates; averaging consecutive n kills the
  oscillating term and exposes the O(1/n) decay).
- **American ordering:** American >= European across a strike/vol/dividend
  grid; with `q = 0` the American call equals the European call on the same
  tree to 1e-12 (early exercise never optimal), and the early-exercise
  premium is exactly 0; the deep-ITM American put dominates intrinsic and
  carries a strictly positive premium when `r > q`.
- **Monte Carlo vs BSM:** 200k paths with antithetic + control variate lands
  within 3 standard errors of the closed form (call and put); the plain
  estimator is also unbiased. Variance reduction is *quantified*: the
  control variate alone must cut the standard error by more than half.
- **Reproducibility:** same (seed, threads) => bit-identical `MCResult`
  (field-by-field exact equality); different seeds differ but both stay
  consistent with BSM; the 4-thread run is deterministic and statistically
  consistent with the single-thread run.
- **Implied vol round trip:** sigma -> price -> sigma to **1e-8** across
  moneyness 0.5x–2.0x, T in {0.05, 0.5, 2.0}, vols {0.1, 0.3, 0.8} (vol
  floored so the strike stays within ~3 standard deviations of the forward
  — beyond that the time value underflows double precision and *no* solver
  can invert the price; that regime is covered by the rejection tests
  instead). Extreme vol (sigma = 4) recovered to 1e-7; short-dated wings
  (tiny vega) recovered via the bisection fallback.

## 4. Edge cases and failure behaviour

- `T = 0` -> intrinsic; `sigma = 0` -> discounted forward intrinsic;
  `K = 0` / `S = 0` limits — all byte-matching the Python policy, all
  `EXPECT_DOUBLE_EQ`-exact.
- Huge vol (sigma = 50): price finite and pinned just below the
  no-arbitrage upper bound.
- Deep wings: erfc-based CDF keeps a K = 100x OTM call finite and
  >= 0 (< 1e-100), never NaN.
- Negative `S`, `K`, `T`, `sigma` and NaN inputs throw
  `std::invalid_argument` (negative `r`, `q` are supported and tested).
- Implied vol rejects sub-intrinsic prices (at/below the sigma -> 0 bound)
  and prices at/above the sigma -> infinity bound, and refuses `T = 0`.
- Tree rejects `n_steps < 1` and a risk-neutral probability outside (0, 1).
- **NaN-free domain scan:** 864-point sweep over S, K in [1e-6, 1e6],
  T in [1e-6, 10], sigma in [1e-8, 3], r in [-0.05, 0.1]: every price
  finite, non-negative, and inside its no-arbitrage bounds.

Known failure modes are those of the models themselves (fat tails, discrete
dividends, non-flat term structure, discretisation at coarse n): they are
documented with reproducible magnitudes in the Python reference
[VALIDATION.md](../../../python/equity/01-options-pricing/docs/VALIDATION.md)
and apply verbatim to this engine since the two produce identical numbers.

## 5. Performance (micro-benchmark)

`./build/eqopt_bench`, g++ 13 `-O2`, 2-vCPU Linux container, single run:

| Kernel | Work | Time | Throughput |
|---|---|---|---|
| BSM analytic price | 1,000,000 varied contracts | 62.4 ms | 16.0M prices/sec (~62 ns each) |
| Full analytic Greek set | 100,000 evals | 8.1 ms | 12.3M evals/sec |
| CRR tree n=1000, European | 1 tree | 0.41 ms | 2,434 trees/sec |
| CRR tree n=1000, American | 1 tree | 5.25 ms | 191 trees/sec |
| MC 1M paths, antithetic+CV, 1 thread | price + SE + CI | 65.4 ms | 15.3M paths/sec |
| MC 1M paths, 4 threads | price + SE + CI | 46.7 ms | 21.4M paths/sec |

The American tree is ~13x the European cost at equal n because every node
recomputes its spot in log space for the exercise comparison (a deliberate
choice for large-n stability — see METHODOLOGY.md). The 4-thread MC gain is
capped by the 2-vCPU host; the path loop itself scales linearly.
