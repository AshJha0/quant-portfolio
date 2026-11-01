# Validation

## Golden vectors from an independent oracle

`python/fx/01-options-pricing/tests/golden/generate_golden.py` computes 434
records in mpmath at 50 digits: GK price, spot/forward/PA deltas, gamma,
vega, theta, both rhos, vanna, volga (25, incl. T=1e-5, T=10, sigma=0.002,
sigma=0.6); implied vol (203, incl. the K=90, T=0.02 OTM put at 1e-12);
strike-from-delta in all four conventions, 10 and 25 delta calls and
10/25/40 delta puts, with premium-adjusted strikes solved by bisection in
mpmath above the fold (140); DNS ATM unadjusted and PA (14); CRR European
and American in mpmath arithmetic (40); CIP forwards (12); sigma = 0 with an
overflowing forward (1). Each output carries an absolute tolerance from a
stated rule (JSON `tolerance_rules`). `tools/gen_golden_header.py` emits the
header and `--check` verifies it; the worst deviation is 0.19 of tolerance.

## Cross-language differential test

`fxopt_jsonl` answers JSON-lines requests;
`python/fx/01-options-pricing/tests/test_differential.py` sends 790 seeded
random requests to the Python, Rust and C++ engines. Largest deviations
from Python (GCC 15 build): prices 5.7e-14 abs, Greeks 4.5e-13 abs, strikes
6.8e-13 abs (1.7e-15 rel), implied vol 1.9e-13, CRR, DNS and forwards 0.
Agreement is per toolchain within these tolerances, not bit for bit:
libm `exp`/`erfc` differ by ulps between platforms.

## Identity and property tests (87 CTest cases)

| Category | Check | Tolerance |
|---|---|---|
| Parity | `C - P = S e^{-r_f T} - K e^{-r_d T}` across a 4-point grid incl. negative rates | 1e-12 |
| Model equivalence | GK == Black-76 on the CIP forward, 162-point grid (3 spots × 3 moneyness × 3 tenors × 3 rate pairs × 2 types) | 1e-12 (scaled) |
| Foreign–domestic symmetry | `C_d(S,K,r_d,r_f) = S·K·P_f(1/S,1/K,r_f,r_d)` across an 81-point grid | 1e-10 (scaled) |
| Delta relations | `Δ_f = Δ_s e^{r_f T}` (plain and PA); `Δ_pa = Δ_s − V/S`; PA < unadjusted for calls | 1e-14 |
| Strike-from-delta | round trips at 10/25/40 delta, both wings, all 4 conventions; PA-call solver picks the decreasing (high-strike) branch; out-of-range targets throw | 1e-8 |
| ATM conventions | DNS straddle net delta = 0 in all 4 conventions; closed forms `F e^{±σ²T/2}` | 1e-12 / 1e-15 |
| Greeks | analytic vs central finite differences (delta, gamma, vega, theta, both rhos, vanna, volga), calls and puts, incl. negative-rate case | 1e-6 (1e-4 gamma stencil) |
| Rho signs | `rho_d > 0`, `rho_f < 0` for calls; reversed for puts | sign |
| Tree | CRR → GK convergence (error decays through 50/200/800 steps; 2,000 steps within 5e-6); American ≥ European; strictly positive early-exercise premium for ITM calls when `r_f > r_d` | see left |
| Monte Carlo | price within 3 SE of analytic; SE(antithetic+CV) <= SE(CV) over seven seeds; same seed ⇒ identical result on one toolchain; `mc_price(..., 100000, true, true, 42)` does not compile (typed `Seed`) | 3 SE / exact |
| Implied vol | round trips across the Python reference grid; at most 10 evaluations; bounds contract (0.0 at the intrinsic bound, throw at the upper bound) | 1e-10 / exact |
| Edge cases | negative rates (both legs), `r_d = r_f`, zero rates, `T = 0` intrinsic, deep ITM/OTM bounds, `sigma = 0` forward intrinsic, invalid inputs throw `std::invalid_argument` everywhere | various |
| Extreme moneyness | `S/K = 100x` and `K/S = 100x`: prices pinned on the arbitrage bounds, non-negative, finite, parity to 1e-12 at both wings | 1e-12 |
| Extreme tenor | `T = 1e-5` (≈5 minutes) collapses towards intrinsic and still equals Black-76 on the forward; `T = 30y` stays inside the bounds with an exact parity and a 1e-8 implied-vol round trip | 1e-15 / 1e-12 / 1e-8 |
| Vol near the upper bound | `sigma = 10` round-trips to 1e-6; at `sigma = 25` the premium equals the `sigma -> inf` bound in double precision and the solver throws | 1e-6 / throws |
| Non-finite rejection | NaN/Inf premium, delta, spot, strike, tenor, either rate or vol rejected by `implied_vol`, `premium_adjust_spot_delta`, `mc_price`, `binomial_price`, `black76_price`, `cip_forward`, `strike_from_delta` | throws |
| Degenerate MC | a single antithetic pair throws; two pairs without the control variate, or three with it, give a positive SE | throws |
| Robustness | Brent non-convergence throws; `forward_points` rejects an infinite pip factor; delta conversions reject non-finite deltas; sigma=0 price finite when the forward overflows | throws / exact |

## Known failure modes and numerical limits

* **Implied vol near the bounds.** The solver works on the
  out-of-the-money side, so a small out-of-the-money price still inverts
  fully (an OTM put worth 6e-15 recovers to 1e-12). An in-the-money quote
  whose time value is within `4 eps df max(F, K)` of the intrinsic bound
  returns 0.0, the sigma -> 0 limit. A price within `4 eps upper` of the
  sigma -> infinity bound throws `std::invalid_argument`: the premium no
  longer identifies a finite vol (`EdgeCases.VolNearSolverCapSaturatesAtTheArbitrageBound`).
* **PA call delta above the fold.** The premium-adjusted call delta has a
  maximum in strike; targets above it throw. The fold equation is solved in
  log form on a bracket derived from the Mills ratio, valid for every
  `sigma sqrt(T)`; a fold strike beyond double range throws.
* **Tree probability bounds.** Extreme `|r_d - r_f|` with tiny vol and
  coarse steps pushes the CRR up-probability outside [0,1]; the engine
  throws with instructions to increase steps.
* **d1/d2 undefined at `sigma sqrt(T) = 0`**: `d1()`/`d2()` throw; the
  pricers switch to the analytic limits instead.
* **Standard errors need independent samples.** With antithetic sampling
  the independent unit is the pair; fewer than two samples (three with the
  control variate, whose fit consumes one) throws.
* **Finite-difference Greeks need a bumpable domain.** `T <= 0`, or a
  `sigma` smaller than its own central bump, is rejected up front, the
  down-bumped volatility would be negative.

## Low-latency API checks

| Check | Test | Result |
|---|---|---|
| `try_*` equals the throwing API bit for bit on a grid of quotes, including edge cases | `NoexceptApi.*` (`tests/test_latency_api.cpp`) | exact |
| Error codes and messages match the exception text | `NoexceptApi.ErrorsCarryTheThrowingMessage` | exact |
| Batch price and all ten Greeks equal the scalar kernel bit for bit | `BatchApi.*` | exact |
| Single-stream `try_mc_price` equals `mc_price` (price, SE, method, path count) | `ThreadPoolMc.SingleStreamEqualsLegacyApi` | exact |
| Chunked MC identical across pools of 0, 1, 3 and 7 workers and repeated runs; within 5 combined SE of the single stream | `ThreadPoolMc.ChunkedResultIndependentOfPoolAndRuns` | exact / statistical |
| Every pool task runs once per job over 500 jobs | `ThreadPool.RunsEveryTaskOnceUnderRepeatedUse` | exact |
| Zero heap allocations on every kernel success path (counting `operator new`) | `fxopt_alloc_tests` | 0 allocations |
| Kernel entry points are `noexcept` | `static_assert`s in `test_latency_api.cpp` | compile time |

The thread pool also passes the full suite under Clang ThreadSanitizer.

No numerical result changed with the low-latency rewrite. A program that
calls only the legacy API (prices, Greeks, Black-76, implied vol, trees,
deltas, strike-from-delta, forwards, Monte Carlo across every
antithetic/control-variate setting and path counts from 1 to 20,000,
including invalid inputs) printed 181,543 results and exception texts in
hex-float; the output against the previous library and the current one is
identical with GCC 15.2 and with Clang 21.1.

## Latency

Per-call latency percentiles are in the README Latency table, measured
with `./build/fxopt_bench` (see `bench/latency.hpp` for the method).
