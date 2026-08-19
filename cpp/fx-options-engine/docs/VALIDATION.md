# Validation

## Headline: cross-language golden validation

The Python reference project emits 30 golden vectors
(`python/fx/01-options-pricing/tests/golden/golden_vectors.json`) spanning
EURUSD-like, USDJPY-like, EM (high rates/vol), negative-rate (EURCHF-era),
long-dated (5y) and one-week cases, ITM/ATM/OTM, calls and puts. Each case
carries price, spot delta, forward delta, gamma, vega, theta, rho_domestic
and rho_foreign at a source tolerance of 1e-10.

`tools/gen_golden_header.py` converts the JSON to
`tests/golden_vectors.hpp` (floats via `repr()`, i.e. shortest round-trip
representation — bit-exact transport). The committed header is regenerated
and diffed as part of maintenance; the suite asserts the case count.

**Result: all 30 cases × 8 outputs reproduce to better than 1e-9**
(the C++ gate), with typical agreement at 1e-13 or below — the residual is
double-rounding noise between scipy's `norm.cdf` and our `erfc`-based CDF,
not semantics. This proves the two implementations encode identical
conventions (BASE/QUOTE, two rates, theta per year, ACT/365F).

## Identity and property tests (86 CTest cases, ~170 assertion sites)

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
| Monte Carlo | price within 3 SE of analytic; SE(antithetic+CV) < SE(antithetic) < SE(plain); same seed ⇒ bit-identical result; different seed ⇒ different | 3 SE / exact |
| Implied vol | round trips across the Python reference grid (incl. negative rates, high vol 60%) | 1e-10 |
| Edge cases | negative rates (both legs), `r_d = r_f`, zero rates, `T = 0` intrinsic, deep ITM/OTM bounds, `sigma = 0` forward intrinsic, invalid inputs throw `std::invalid_argument` everywhere | various |
| Extreme moneyness | `S/K = 100x` and `K/S = 100x`: prices pinned on the arbitrage bounds, non-negative, finite, parity to 1e-12 at both wings | 1e-12 |
| Extreme tenor | `T = 1e-5` (≈5 minutes) collapses towards intrinsic and still equals Black-76 on the forward; `T = 30y` stays inside the bounds with an exact parity and a 1e-8 implied-vol round trip | 1e-15 / 1e-12 / 1e-8 |
| Vol at the solver cap | `sigma = 10` (top of the Newton bracket) round-trips to 1e-6; at `sigma = 25` the premium *equals* the `sigma -> inf` bound in double precision, and the solver returns a finite vol that reprices the premium to 1e-14 | 1e-6 / 1e-14 |
| Non-finite rejection | NaN/Inf premium, delta, spot, strike, tenor, either rate or vol rejected by `implied_vol`, `premium_adjust_spot_delta`, `mc_price`, `binomial_price`, `black76_price`, `cip_forward`, `strike_from_delta` | throws |
| Degenerate MC | a single antithetic pair (`n_paths = 2`) reports `std_error = 0` (unestimable) instead of NaN; two pairs give a strictly positive SE | exact |

## Known failure modes and numerical limits

* **Vol unrecoverable at zero time value.** When a premium sits within
  ~1e-16 of the discounted forward intrinsic, no double-precision vol
  reproduces it; `implied_vol` returns 0.0 by documented convention
  (identical to the Python reference). More broadly, deep-ITM/short-dated
  options have time value near machine epsilon — the price→vol map is a
  plateau there and *any* solver's answer is ill-conditioned; quote vol
  from the OTM wing instead (as desks do).
* **PA call delta above the fold.** The premium-adjusted call delta has a
  maximum in strike; targets above it have no solution and throw rather
  than silently returning the wrong branch.
* **Tree probability bounds.** Extreme `|r_d − r_f|` with tiny vol and
  coarse steps pushes the CRR up-probability outside [0,1]; the engine
  throws with instructions to increase steps.
* **d1/d2 undefined at `sigma·sqrt(T) = 0`** — `d1()`/`d2()` throw; the
  pricers switch to the analytic limits instead.
* **Vol is not identifiable above ~15.** For `sigma·sqrt(T)` large enough
  that `N(d2)` underflows, the GK premium equals the `sigma -> inf` bound
  `S e^{-r_f T}` to the last bit, so every vol above that point prices
  identically. `implied_vol` returns the smallest vol consistent with the
  observed premium (repricing it to ~1e-14) rather than throwing; treat
  any implied vol above ~10 as "at the bound", not as a measurement.
* **Standard errors need at least two independent samples.** With
  antithetic sampling the independent unit is the *pair*, so `n_paths = 2`
  yields one sample and no error estimate; the engine reports
  `std_error = 0` (and a degenerate CI) rather than NaN. Use >= 4
  antithetic paths whenever the error bar is consumed downstream.
* **Finite-difference Greeks need a bumpable domain.** `T <= 0`, or a
  `sigma` smaller than its own central bump, is rejected up front — the
  down-bumped volatility would be negative.

## Benchmark table

Single-threaded, g++ 13.3 `-O2`, x86-64 Linux container (indicative):

| Benchmark | Volume | Time | Throughput |
|---|---|---|---|
| GK vanilla prices (varying K, σ, type) | 1,000,000 | 0.077 s | 12.94 M prices/s |
| strike-from-delta, spot (analytic) | 50,000 | 0.007 s | 7.54 M solves/s |
| strike-from-delta, forward (analytic) | 50,000 | 0.006 s | 8.00 M solves/s |
| strike-from-delta, spot_pa (Brent) | 50,000 | 0.066 s | 761 k solves/s |
| strike-from-delta, forward_pa (Brent) | 50,000 | 0.075 s | 670 k solves/s |
| Monte Carlo, antithetic + control variate | 1,000,000 paths | 0.099 s | 10.12 M paths/s (SE ≈ 3.0e-5) |

Reproduce with `./build/fxopt_bench` (numbers vary with hardware; the
checksums printed guard against the compiler optimising the loops away).
