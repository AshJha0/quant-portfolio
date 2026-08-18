# Validation — eq-options-engine (Rust)

How this engine was validated, what tolerances are enforced in CI
(`RUSTFLAGS="-D warnings" cargo test` — 45 integration tests + 26
doctests, all green), and where the models fail.

## 1. Three-way cross-language golden validation (the headline)

The Python reference project commits
`tests/golden/golden_vectors.json`: 32 Black-Scholes-Merton cases —
price plus delta/gamma/vega/theta/rho — spanning ATM/ITM/OTM, ~1-week to
5-year expiries, negative rates, dividend yields, near-zero and 150% vols,
small and index-scale notionals. Three independent implementations consume
the *same file*:

| Stack | Consumer | Gate | Measured worst deviation |
|---|---|---|---|
| Python (`eq_options`) | generator (scipy `norm.cdf`) | — | reference |
| C++ (`eqopt`) | `tools/gen_golden_header.py` -> `tests/golden_vectors.hpp` | 1e-9 | 5.3e-15 |
| **Rust (this crate)** | `tools/gen_golden_rs.py` -> `src/golden.rs` | **1e-9** | **5.7e-14** |

**All three languages agree on all 32 x 6 = 192 numbers to ~1e-13 or
better** — i.e. to within a few ulps of double precision, despite three
unrelated normal-CDF implementations (scipy's Cephes `ndtr`, glibc's
`erfc`, and this crate's in-house Cody erfc). The generator emits doubles
via shortest-roundtrip `repr`, so the committed Rust constants are
bit-identical to the Python reference values. A separate test pins the
worst-case deviation below 1e-12, so accuracy regressions are caught long
before the 1e-9 gate.

Regenerate after the Python project changes:
`python3 tools/gen_golden_rs.py && cargo test --test golden`.

## 2. Analytic identities and convergence (enforced in `tests/`)

| Check | Tolerance | Test |
|---|---|---|
| Put-call parity `C - P = S e^{-qT} - K e^{-rT}` on a 486-point grid | 1e-12 | `black_scholes.rs` |
| erfc vs high-precision references; norm_cdf quantiles + deep-tail relative accuracy at `Phi(-8)` | 1e-12 (erfc), 1e-10 rel tail | `black_scholes.rs` |
| Black-76 == BSM at `F = S e^{(r-q)T}` (7 market configs x call/put) | 1e-12 | `black76.rs` |
| Black-76 parity `C - P = e^{-rT}(F - K)`; `rho = -T V`; Greeks vs FD | 1e-12 / 1e-5 | `black76.rs` |
| CRR -> BSM, 2000 steps | 2e-3 | `binomial.rs` |
| CRR error decays (odd/even-averaged, n=50 vs n=800: > 4x reduction) | — | `binomial.rs` |
| American >= European across grid; `q=0` American call == European | 1e-12 | `binomial.rs` |
| ITM American put premium positive; American >= intrinsic | — | `binomial.rs` |
| Analytic vs central-FD Greeks (delta/vega/theta/rho and price) | 1e-6 rel | `greeks.rs` |
| Analytic vs FD second-order (gamma/vanna/volga, `h ~ eps^0.25` bump) | 5e-6 rel | `greeks.rs` |
| Greek identities: `delta_C - delta_P = e^{-qT}`, `rho_C - rho_P = K T e^{-rT}`, shared gamma/vega/vanna/volga | 1e-12 | `greeks.rs` |
| Implied-vol round trip across moneyness 0.5x-2x, T in [0.04, 2], sigma in [0.05, 0.8], r/q incl. negative | 1e-8 | `implied_vol.rs` |
| MC price within 3 SE of analytic BS (4 configs, 200k paths) | 3 SE | `monte_carlo.rs` |
| Antithetic and control variate each reduce SE; combined < 0.75x plain (measured ~0.54x) | — | `monte_carlo.rs` |
| Same seed => bitwise-identical `McResult`; different seeds differ | exact bits | `monte_carlo.rs` |
| RNG: stream reproducibility, uniforms strictly in (0,1), normal mean/variance within 3-sigma bands at n=1e6 | — | `monte_carlo.rs` |
| Vol monotonicity and strike convexity (butterfly >= 0) | 1e-12 | `black_scholes.rs` |

## 3. Edge cases (documented *and* tested)

- `T = 0` -> intrinsic (BS, tree, Black-76, MC with `std_error = 0`).
- `sigma = 0` -> discounted forward intrinsic; American tree takes the
  max discounted intrinsic along the deterministic path (e.g. sigma=0
  American put with high `r` exercises immediately at `K - S`).
- `S = 0` / `K = 0` degenerate limits (put -> discounted/immediate strike,
  call -> dividend-adjusted forward; American K=0 call worth `S` when
  `q > 0`).
- Negative `r` and `q` supported end-to-end (golden cases include them).
- Invalid inputs (negative `S`, `K`, `T`, `sigma`, NaN, `n_steps = 0`,
  `n_paths < 2`) return `Err(PricingError::InvalidInput)` with the
  offending parameter named; `Display` output is asserted.
- Implied vol: sub-intrinsic and above-upper-bound quotes rejected as
  `ArbitrageBound`; expired options rejected; sigma=3.0 premiums
  recovered via bracket expansion; tiny-vega wings exercise the bisection
  fallback.

## 4. Known failure modes and numerical limits

1. **Deep-ITM implied vol is ill-posed.** When vega < ~1e-2, a price
   solved to the 1e-10 tolerance pins sigma no better than ~1e-8/vega;
   at vega ~ 1e-10 *any* vol in a multi-point band reprices identically.
   The round-trip test encodes exactly this identifiability bound; the
   desk-level answer is to quote wings from OTM options (see DESK_GUIDE).
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
5. **MC constants are seed-dependent.** The variance-reduction *factors*
   (~0.54x combined here) are statistics, not constants; the tests gate
   conservative bounds. MC numbers do not match NumPy's stream (different
   RNG); they match the analytic price statistically (3 SE), and match
   themselves bit-for-bit per seed.
6. **Model risk proper** (GBM, constant vol, continuous dividends) is
   documented in METHODOLOGY.md — validation here is *internal*
   consistency plus cross-implementation agreement, not market fit.

## 5. Benchmarks (this container: 2 vCPU, shared; single runs)

Rust (`cargo run --release --bin bench`, rustc 1.95, LTO):

| Kernel | Work | Time | Throughput |
|---|---|---|---|
| Black-Scholes analytic | 1M prices | 83.5 ms | 12.0M prices/sec |
| Full analytic Greek set | 100k evals | 7.7 ms | 13.0M evals/sec |
| CRR tree n=1000, European | 1 tree | 0.58 ms | 1,739 trees/sec |
| CRR tree n=1000, American | 1 tree | 5.51 ms | 182 trees/sec |
| MC 1M paths, antithetic + CV, 1 thread | 1 price + CI | 54.1 ms | 18.5M paths/sec |

Versus the C++ engine (`eqopt_bench`, g++ 13 -O2, same machine, from its
README): 16.0M BS prices/sec, 12.3M Greek evals/sec, 0.41 ms European /
5.25 ms American n=1000 trees, 15.3M MC paths/sec single-threaded. The
engines are within ~25-35% of each other in both directions (Rust faster
on MC, C++ faster on the erfc-bound BS kernel — glibc's assembly `erfc`
vs this crate's portable Cody), i.e. the *language* is not the bottleneck;
both are suitable for the latency path, and single-run numbers on a shared
container should be read as order-of-magnitude.
