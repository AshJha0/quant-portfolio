# Validation

## Headline: cross-language golden validation

The Python reference project emits 30 golden vectors
(`python/fx/01-options-pricing/tests/golden/golden_vectors.json`) spanning
EURUSD-like, USDJPY-like, EM (high rates/vol), negative-rate (EURCHF-era),
long-dated (5y) and one-week cases, ITM/ATM/OTM, calls and puts. Each case
carries price, spot delta, forward delta, gamma, vega, theta, rho_domestic
and rho_foreign at a source tolerance of 1e-10.

`tools/gen_golden_rs.py` converts the JSON to `src/golden.rs` (floats via
Python's `repr()`, i.e. shortest round-trip representation — bit-exact
transport, no runtime file I/O in `cargo test`). The committed module is
regenerated and diffed as part of maintenance; `tests/golden.rs` asserts
the case count.

**Result: all 30 cases × 8 outputs reproduce to better than 1e-9**
(`tests/golden.rs`'s gate), with typical agreement at 1e-13 or below — the
residual is double-rounding noise between scipy's `norm.cdf` and this
crate's `erfc`-based CDF, not semantics. This proves the two
implementations encode identical conventions (BASE/QUOTE, two rates, theta
per year, ACT/365F).

## Identity and property tests (69 integration tests, 19 doctests)

| Category | Check | Tolerance |
|---|---|---|
| Parity | `C - P = S e^{-r_f T} - K e^{-r_d T}` across a grid incl. negative rates (`garman_kohlhagen::two_rate_put_call_parity_to_1e12`) | 1e-12 |
| Model equivalence | GK == Black-76 on the CIP forward across an S × K × T × rate-pair × type grid (`forwards_black76::gk_equals_black76_to_1e12_across_grid`) | 1e-12 (scaled) |
| Foreign–domestic symmetry | `C_d(S,K,r_d,r_f) = S*K*P_f(1/S,1/K,r_f,r_d)` and the flipped-put/PA-delta identities (`symmetry::*`) | 1e-10 – 1e-12 (scaled) |
| Delta relations | `delta_fwd = delta_spot * e^{r_f T}` (plain and PA); `delta_pa = delta_spot - V/S`; PA < unadjusted for calls (`deltas::*`) | 1e-14 – 1e-15 |
| Strike-from-delta | round trips across delta/strike grid, all 4 conventions; PA-call solver picks the decreasing (high-strike) branch; out-of-range targets error (`deltas::strike_from_delta_round_trips_all_four_conventions_to_1e8`) | 1e-8 |
| ATM conventions | DNS straddle net delta = 0 in all 4 conventions; closed forms `F e^{+-sigma^2 T/2}` (`deltas::dns_*`) | 1e-12 / 1e-14 |
| Greeks | analytic vs generic central finite differences (delta, gamma, vega, theta, both rhos, vanna, volga), calls and puts (`greeks::finite_differences_match_analytic_to_1e6`) | 1e-6 (1e-4/1e-5 for the 2nd-order gamma/volga stencils) |
| Rho signs | `rho_d > 0`, `rho_f < 0` for calls; reversed for puts (`greeks::rho_signs_two_rate_structure`) | sign |
| Tree | CRR to GK convergence as steps grow (`binomial::european_tree_converges_to_gk`); American >= European everywhere; strictly positive early-exercise premium for ITM options on the carry-favourable side | see left |
| Monte Carlo | price within 3 SE of analytic; `SE(antithetic+CV) < 0.5 * SE(plain)`; same seed => bit-identical result (`.to_bits()` equality); different seed => different (`monte_carlo::*`) | 3 SE / exact |
| Implied vol | round trips across the Python-reference-style grid incl. negative rates and a high-vol wing, to 1e-10; a dedicated short-dated-wing case to 2e-8 (see "Known failure modes" below); zero-time-value degenerate case (`implied_vol::*`) | 1e-9 – 1e-10 |
| Edge cases | negative rates (both legs), deep ITM/OTM bounds, tiny tenor + high vol stay finite, extreme-moneyness `N(d1)/N(d2)` don't degenerate, RNG stream pinned across releases (`edge_cases::*`) | various |

## Known failure modes and numerical limits

* **Vol unrecoverable at zero time value.** When a premium sits within
  `1e-16 * max(1, lower)` of the discounted forward intrinsic, no
  double-precision vol reproduces it; `implied_vol` returns `0.0` by
  documented convention (identical to the C++ engine's behaviour; the
  Python reference has no `implied_vol` module to compare against). The
  test (`implied_vol::zero_time_value_returns_zero_vol`) constructs the
  boundary price with the *same* internal arithmetic `implied_vol` uses
  (`F = S * df_f / df_d`), because the mathematically-equivalent
  `S * exp((r_d - r_f) * T)` is not bit-identical in floating point and
  the near-intrinsic detection is deliberately a last-ULP check.
* **Implied vol precision floor on extreme short-dated wings.** For a very
  short-dated (T ~ 0.02y), meaningfully ITM/OTM strike, the GK call/put
  formula `S e^{-r_f T} N(d1) - K e^{-r_d T} N(d2)` subtracts two O(1)
  terms to get an O(price) result, losing roughly a decimal digit to
  cancellation; combined with the wing's near-zero vega (`~1e-9` in the
  tested case), the objective `sigma -> price(sigma) - price` becomes a
  *staircase* in sigma with ULP-sized flats `~1e-8` to `~2e-8` wide near
  the root — confirmed by scanning the objective directly and by re-seeding
  Brent with arbitrarily tight brackets around the true vol (every variant
  lands in the same band, because many adjacent sigma values round to the
  identical price `f64`). No root finder operating on this exact price
  residual can resolve the root more tightly than that flat is wide; this
  is an IEEE-754 floor, not a solver deficiency, and it is *not* specific
  to this crate — an unmodified port of the same Newton-then-Brent
  algorithm into the C++ engine reproduces an error of the same order (in
  fact worse, ~2.5e-7, before the premature-Newton-exit fix described
  below). `implied_vol::round_trip_short_dated_wings` documents the floor
  in place with a 2e-8 tolerance instead of the 1e-10 achieved on
  well-conditioned strikes; three of its four cases still round-trip to
  <1e-12. More broadly, deep-ITM/short-dated options have time value near
  machine epsilon — the price-to-vol map is a plateau there and *any*
  solver's answer is ill-conditioned; quote vol from the OTM wing instead
  (as desks do).
* **The premature-Newton-exit bug this crate fixes.** An earlier version of
  `implied_vol` accepted a Newton iterate as soon as the *absolute* price
  residual dropped below `1e-14`, regardless of the local vega. For a wing
  option with vega `~1e-9`, a sub-ULP price residual still implies a sigma
  error of `residual / vega ~ 1e-5`-to-`1e-7` — orders of magnitude looser
  than the intended `1e-12` sigma tolerance. The fix removes the
  vega-blind absolute check, always falls through to a Brent polish seeded
  tightly around the last Newton iterate (expanding geometrically until it
  brackets a sign change), and only accepts the fast-path return when the
  objective is *already* flat (vega `< 1e-12`) *and* the residual is at
  the double-precision noise floor. This is what took the well-conditioned
  cases in `round_trip_grid_to_1e10` and `round_trip_with_negative_rates`
  from "passing by luck on this specific grid" to consistently `<1e-12`.
* **Variance-reduction interaction in Monte Carlo.** Fitting the
  control-variate coefficient on raw per-draw values and only pairing them
  antithetically *afterwards* under-uses the control variate: for the
  tested market (`S=1.10, K=1.12, T=0.5`), that ordering gave
  `SE(antithetic+CV) / SE(plain) ~ 0.76` — worse than control-variate alone
  at `~0.55`, i.e. adding antithetic pairing on top of a working control
  variate *increased* the combined standard error. The fix fits beta on
  the antithetic pair averages (the same statistical unit as the final
  estimator): the optimal beta for `Var(0.5(X(Z) - beta C(Z) + X(-Z) -
  beta C(-Z)))` is `Cov(pair-avg payoff, pair-avg control) / Var(pair-avg
  control)`, which is not the same number as the per-draw beta once
  antithetic pairing has already reshaped the payoff/control covariance
  structure. After the fix, `SE(antithetic+CV) / SE(plain) ~ 0.24` on the
  same market/seed — comfortably under the `0.5` gate in
  `monte_carlo::variance_reduction_shrinks_the_standard_error`.
* **PA call delta above the fold.** The premium-adjusted call delta has a
  maximum in strike; targets above it have no solution and `strike_from_delta`
  returns `Err` rather than silently returning the wrong branch.
* **`d1`/`d2` undefined at `sigma * sqrt(T) = 0`** — `d1()`/`d2()` return
  `Err`; the pricers switch to the analytic limits instead.

## Benchmark table

Single-threaded, release profile (`opt-level = 3`, `lto = true`,
`codegen-units = 1`), x86-64 Linux container (indicative — rerun
`cargo run --release --bin bench` on your hardware; the checksums printed
guard against the compiler optimising the loops away):

| Benchmark | Volume | Time | Throughput |
|---|---|---|---|
| GK vanilla prices (varying K, sigma, type) | 1,000,000 | 0.119 s | 8.41 M prices/s |
| strike-from-delta, spot (analytic) | 50,000 | 0.010 s | 5.04 M solves/s |
| strike-from-delta, forward (analytic) | 50,000 | 0.010 s | 5.01 M solves/s |
| strike-from-delta, spot_pa (Brent) | 50,000 | 0.099 s | 505 k solves/s |
| strike-from-delta, forward_pa (Brent) | 50,000 | 0.110 s | 455 k solves/s |
| Monte Carlo, antithetic + control variate | 1,000,000 paths | 0.143 s | 7.00 M paths/s (SE ~ 9.4e-6) |

Reproduce with `cargo run --release --bin bench` (numbers vary with
hardware).
