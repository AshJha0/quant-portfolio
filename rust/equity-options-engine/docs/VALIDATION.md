# Validation: eq-options-engine (Rust)

How this engine was validated, what tolerances are enforced in CI
(`RUSTFLAGS="-D warnings" cargo test --release`: 74 integration tests +
28 doctests), and where the models fail.

## 1. Golden vectors and the differential test

The golden file `python/equity/01-options-pricing/tests/golden/golden_vectors.json`
is computed in mpmath at 50 digits, not by any engine: 456 records
covering BSM price and all Greeks incl. vanna/volga, Black-76, implied vol
over ln(K/S) in [-3, 3] and T in [1e-5, 50], CRR European and American
trees evaluated in mpmath, and exact edge limits. Each output carries its
own absolute tolerance. `tools/gen_golden_rs.py` emits `src/golden.rs`
(`--check` verifies it) and `tests/golden.rs` checks every record and the
worst deviation over tolerance: 0.175.

`src/bin/eq_jsonl.rs` is a JSON-lines shim. The Python test
`tests/test_differential.py` sends 640 seeded random requests to Python,
Rust and C++; the largest deviation of Rust from Python is 1.1e-13 on
prices, 1.8e-12 on volga and 7.1e-15 on implied vol. A few tiny deep-wing
prices differ in relative terms because Cody's erfc underflows at
x = 26.543 while CPython's `math.erfc` continues.

Regenerate after the golden JSON changes:
`python3 tools/gen_golden_rs.py && cargo test --test golden`.

## 2. Analytic identities and convergence (enforced in `tests/`)

| Check | Tolerance | Test |
|---|---|---|
| Put-call parity `C - P = S e^{-qT} - K e^{-rT}` on a 486-point grid | 1e-12 | `black_scholes.rs` |
| erfc vs high-precision references; norm_cdf quantiles + deep-tail relative accuracy at `Phi(-8)` | 1e-12 (erfc), 1e-10 rel tail | `black_scholes.rs` |
| Black-76 == BSM at `F = S e^{(r-q)T}` (7 market configs x call/put) | 1e-12 | `black76.rs` |
| Black-76 parity `C - P = e^{-rT}(F - K)`; `rho = -T V`; Greeks vs FD | 1e-12 / 1e-5 | `black76.rs` |
| CRR -> BSM, 2000 steps | 2e-3 | `binomial.rs` |
| CRR error decays (odd/even-averaged, n=50 vs n=800: > 4x reduction) |, | `binomial.rs` |
| American >= European across grid; `q=0` American call == European | 1e-12 | `binomial.rs` |
| ITM American put premium positive; American >= intrinsic |, | `binomial.rs` |
| Analytic vs central-FD Greeks (delta/vega/theta/rho and price) | 1e-6 rel | `greeks.rs` |
| Analytic vs FD second-order (gamma/vanna/volga, `h ~ eps^0.25` bump) | 5e-6 rel | `greeks.rs` |
| Greek identities: `delta_C - delta_P = e^{-qT}`, `rho_C - rho_P = K T e^{-rT}`, shared gamma/vega/vanna/volga | 1e-12 | `greeks.rs` |
| Implied-vol round trip across moneyness 0.5x-2x, T in [0.04, 2], sigma in [0.05, 0.8], r/q incl. negative; no case skipped | 8 eps max(S,K)/vega + 2e-12 sigma | `implied_vol.rs` |
| Implied vol: at most 10 evaluations; K=90, T=0.02 OTM put to 1e-12; ITM call reprices its quote to a few ulps; root at the branch point in at most 4 evaluations | exact / 1e-12 | `numerical_safety.rs` |
| Property tests (proptest): parity, monotone in vol and strike, strike convexity, delta bounds, IV round trip | 300 cases each | `properties.rs` |
| MC price within 3 SE of analytic BS (4 configs, 200k paths) | 3 SE | `monte_carlo.rs` |
| Antithetic and control variate each reduce SE; antithetic + CV no worse than CV alone over 7 seeds and below 0.6x the raw-draw fit (0.0246) | n/a | `mc_variance_reduction.rs` |
| Same seed => bitwise-identical `McResult`; different seeds differ | exact bits | `monte_carlo.rs` |
| RNG: stream reproducibility, uniforms strictly in (0,1), normal mean/variance within 3-sigma bands at n=1e6 |, | `monte_carlo.rs` |
| Vol monotonicity and strike convexity (butterfly >= 0) | 1e-12 | `black_scholes.rs` |
| Extreme moneyness `S/K = 1e4` and `1e-4`: parity to relative 1e-12, worthless wing underflows to a clean non-negative zero | 1e-12 rel | `black_scholes.rs` |
| 30-year expiry: price finite and inside `[max(Se^{-qT}-Ke^{-rT},0), Se^{-qT}]`, parity holds | 1e-12 rel | `black_scholes.rs` |
| `T = 1e-6` (~30 s): price collapses to intrinsic and stays at/above the sigma->0 floor | 1e-3 / 1e-12 | `black_scholes.rs` |
| Non-finite rejection: `+/-inf` in `S`, `K`, `T`, `sigma`; NaN/inf in `r`, `q`; across BS, tree, MC, Black-76, IV | exact | all suites |
| Resource caps: `n_steps > MAX_STEPS` (1e7) and `n_paths > MAX_PATHS` (1e9) rejected as `InvalidInput` | exact | `binomial.rs`, `monte_carlo.rs` |
| Implied vol above the initial bracket top (`sigma = 15`, forces doubling to the 1e3 cap) | 1e-6 | `implied_vol.rs` |
| Implied vol at the *bottom* of the range: `sigma` in [1e-3, 5e-2] x `T` in [1/365, 1] round-trips within its condition bound; a quote exactly on the `sigma->0` floor is rejected as `ArbitrageBound` | vega-scaled | `implied_vol.rs` |
| Non-finite rejection at **both** Greek entry points: 11 NaN/inf placements x call/put x `bs_greeks`/`fd_greeks` all return `InvalidInput` | exact | `greeks.rs` |
| `fd_greeks` rejects `sigma < h_v2` (2e-4) and `s < h_s2` even when the pricer closure does *no* validation of its own | exact | `greeks.rs` |
| Extreme-moneyness Greeks (`S/K = 100x`, both directions): all 8 outputs finite, `gamma, vega >= 0`, `delta` inside `[-e^{-qT}, e^{-qT}]`, deep-ITM call delta -> `e^{-qT}` | 1e-12 | `greeks.rs` |
| CRR at 100x moneyness and `T` in {30y, 100y}: finite, `American >= European`, inside the static upper bounds, European within 5e-3 *relative* of BS | 5e-3 rel | `binomial.rs` |
| CRR with negative `r`, negative `q` and both: converges to BS; `n_steps = 1` matches the closed-form two-node value | 2e-3 / 1e-12 | `binomial.rs` |
| MC with the minimum admissible path counts (2, 3, 4 with antithetic): `std_error` finite and non-negative, CI ordered and self-containing; the single-pair case gives `SE = 0` exactly, never `0/0 = NaN` | exact | `monte_carlo.rs` |
| MC at 100x moneyness and `T = 30y`: non-negative, within `3 SE + 1e-9 x scale` of BS (scale-relative, since the control variate drives SE below f64 summation noise on a 1e4-sized price) | 3 SE + rel | `monte_carlo.rs` |

## 3. Edge cases (documented *and* tested)

- `T = 0` -> intrinsic (BS, tree, Black-76, MC with `std_error = 0`).
- `sigma = 0` -> discounted forward intrinsic; American tree takes the
  max discounted intrinsic along the deterministic path (e.g. sigma=0
  American put with high `r` exercises immediately at `K - S`).
- `S = 0` / `K = 0` degenerate limits (put -> discounted/immediate strike,
  call -> dividend-adjusted forward; American K=0 call worth `S` when
  `q > 0`).
- Negative `r` and `q` supported end-to-end (golden cases include them).
- Invalid inputs (negative `S`, `K`, `T`, `sigma`, NaN, **infinity**,
  `n_steps = 0`, `n_paths < 2`) return `Err(PricingError::InvalidInput)`
  with the offending parameter named; `Display` output is asserted.
  Non-finite **rates** (`r`, `q`) are rejected too, see the hardening
  note below.
- Extreme moneyness (`S/K` from 1e-4 to 1e4): the worthless wing
  underflows to a clean non-negative zero rather than a negative
  cancellation residue, and put-call parity still holds to 1e-12
  *relative*.
- Very long (`T = 30y`) and very short (`T = 1e-6y`, ~30 seconds)
  expiries: both stay inside the static no-arbitrage bounds. Note that a
  near-expiry European **put** with `r > 0` legitimately prices *below*
  its undiscounted intrinsic (`K - S`), the correct floor is the
  `sigma -> 0` discounted-forward intrinsic, which is what the test
  asserts.
- Implied vol: sub-intrinsic and above-upper-bound quotes rejected as
  `ArbitrageBound`; expired options rejected; sigma=3.0 premiums
  recovered via bracket expansion; tiny-vega wings exercise the bisection
  fallback; sigma down to 1e-3 at `T = 1/365` either round-trips to its
  identifiability bound or fails loudly, never silently.
- Minimum-size Monte Carlo: `n_paths = 2` with antithetic sampling is one
  independent sample, so no standard error exists and `mc_price` returns
  `InvalidInput`. With the control variate three samples are needed: two
  points fit the regression exactly and leave a zero residual.
- Finite-difference bump domain: `fd_greeks` bumps are floored at
  `rel_bump x 1`, so for `sigma < 2e-4` (or `S < 2e-4`) the down-leg of
  the central stencil would sit at a *negative* volatility or spot.
  `bs_price` would reject that, but an arbitrary caller-supplied pricer
  closure need not, so `fd_greeks` checks the bump domain itself, up
  front, and returns `InvalidInput`.

## 4. Known failure modes and numerical limits

1. **Deep-ITM implied vol carries little information.** The solver works
   on the out-of-the-money time value, but an ITM quote's time value is the
   difference of the quote and the intrinsic bound. With S=100, K=90,
   T=0.02, sigma=10%, one ulp of the call (1.8e-15) moves the vol by about
   5e-4. The solver returns a vol that reprices the quote; the OTM put at
   the same strike recovers the vol to 1e-12. Elsewhere accuracy is
   `max(eps price / vega, 1e-12 sigma)`.
2. **Tail underflow.** `Phi(x)` underflows to 0 below x ~ -37.5
   (erfc cutoff 26.543 * sqrt 2); far-wing prices below ~1e-300 are not
   representable in f64. Cody's split-exponential keeps *relative*
   accuracy down to the cutoff.
3. **CRR odd/even oscillation.** European tree error oscillates in n
   around O(1/n); single-n comparisons can flatter or damn the tree. The
   convergence test averages adjacent n. American prices additionally
   carry the exercise-boundary discretisation error.
4. **Tree probability domain.** For `n_steps` small and `|r - q| sqrt(dt)
   > sigma`, the risk-neutral `p` leaves (0,1); the engine returns an
   error rather than pricing with a pseudo-probability.
5. **MC constants are seed-dependent.** Variance-reduction factors are
   statistics; the tests gate conservative bounds over several seeds. MC
   numbers do not match NumPy's stream (different RNG); they match the
   analytic price statistically (3 SE) and repeat exactly per seed on the
   same target and toolchain.
6. **Non-finite inputs and results are errors.** Every public entry
   point rejects NaN and `+/-inf` in `S`, `K`, `T`, `sigma`, `r`, `q`, as
   the Python and C++ engines do. A finite input whose result overflows
   (a discount factor beyond f64 range) returns `InvalidInput` instead of
   `Ok(inf)` or `Ok(NaN)`; the sigma=0 price is computed as
   `S e^{-qT} - K e^{-rT}` so it stays finite when the forward overflows.
7. **Resource exhaustion is an input error.** `crr_price` is O(n^2) in
   time and O(n) in memory, `mc_price` O(n) in both. Requests above
   `binomial::MAX_STEPS` (1e7) and `monte_carlo::MAX_PATHS` (1e9) return
   `InvalidInput` rather than hanging for hours or aborting inside the
   allocator, a library must not take the process down on a bad
   argument.
8. **Finite-difference Greeks have a floor on `sigma`, `S` and `T`.**
   The central bumps are `1e-5 x max(|x|, 1)` (first order) and
   `2e-4 x max(|x|, 1)` (second order), floored at 1 so that they do not
   collapse to zero for small inputs. The price of that floor is that
   `fd_greeks` cannot be used below `sigma = 2e-4`, `S = 2e-4` or
   `T = 1e-5`; it returns `InvalidInput` there rather than differencing
   across the domain boundary. Analytic `bs_greeks` has no such
   restriction and should be used in that regime.
9. **Model risk proper** (GBM, constant vol, continuous dividends) is
   documented in METHODOLOGY.md, validation here is *internal*
   consistency plus cross-implementation agreement, not market fit.

## 5. Latency and allocation

`cargo run --release --bin bench` on an Intel Core 5 120U under WSL2,
Rust 1.99.1. Numbers from this machine; the full table, method and
caveats are in the README "Latency" section and in
`bench/results.json`. Selected p50 / p99 / p99.9 per call (after):

| Kernel | p50 | p99 | p99.9 |
|---|---|---|---|
| `bs_price` (16-call average) | 44 ns | 116 ns | 576 ns |
| `bs_greeks` (16-call average) | 46 ns | 71 ns | 193 ns |
| `bs_price_batch` per row | 38 ns | 54 ns | 100 ns |
| `implied_vol` | 507 ns | 1.21 us | 1.75 us |
| `crr_price` n=1000 American | 396 us | 777 us | 1.42 ms |
| `mc_price` 1M paths, anti + CV, 1 thread | 16.2 ms | 24.8 ms | 24.8 ms |
| `try_mc_price` 1M paths, 8 threads | 11.2 ms | 14.9 ms | 14.9 ms |

Tests added with the low-latency API:

| Test | What it checks |
|---|---|
| `tests/zero_alloc.rs` | Counting global allocator, one test in its own binary. Every fast kernel, the batch pricers, the tree with a sized workspace and single-thread Monte Carlo after warm-up make zero heap allocations. The classic `crr_price` is asserted to allocate, so the counter is live. |
| `tests/fast_equivalence.rs` | `try_*`, batch and workspace pricers equal the classic API by `to_bits` on 20,000 varied rows including edge and invalid inputs; Monte Carlo gives the same bits across runs and for 1 to 12 threads. |
