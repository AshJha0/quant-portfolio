# Validation

## Golden vectors and the differential test

The golden file `python/fx/01-options-pricing/tests/golden/golden_vectors.json`
is computed in mpmath at 50 digits, not by any engine: 434 records covering
GK price and Greeks (both rhos, vanna, volga, four delta conventions),
implied vol, strike-from-delta in all four conventions, DNS ATM strikes,
CRR trees and forwards, each output with its own absolute tolerance.
`tools/gen_golden_rs.py` emits `src/golden.rs` (`--check` verifies it) and
`tests/golden.rs` checks every record; the worst deviation is 0.186 of
tolerance.

`src/bin/fx_jsonl.rs` is a JSON-lines shim. The Python test
`tests/test_differential.py` sends 790 seeded random requests to Python,
Rust and C++; the largest deviation of Rust from Python is 5.7e-14 on
prices, 4.5e-13 on Greeks, 6.8e-13 on strikes and 4.1e-14 on implied vol.
Agreement holds per target and toolchain within these tolerances, not bit
for bit.

## Identity and property tests (100 integration tests, 20 doctests)

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
| Monte Carlo | price within 3 SE of analytic; `SE(antithetic+CV) < 0.5 * SE(plain)` and no worse than CV alone over 7 seeds; same seed => identical result on one target (`.to_bits()` equality); different seed => different (`monte_carlo::*`, `mc_variance_reduction.rs`) | 3 SE / exact |
| Implied vol | round trips across the Python-reference-style grid incl. negative rates and a high-vol wing; short-dated wings within the condition bound `8 eps max(S,K)/vega + 2e-12 sigma`; at most 10 evaluations; bounds contract; OTM put worth 6e-15 to 1e-12 (`implied_vol::*`, `numerical_safety.rs`) | 1e-10 / condition bound |
| Edge cases | negative rates (both legs), deep ITM/OTM bounds, tiny tenor + high vol stay finite, extreme-moneyness `N(d1)/N(d2)` don't degenerate, RNG stream pinned across releases (`edge_cases::*`) | various |
| **Non-finite rejection** | NaN, `+inf` and `-inf` in **every** float slot of **every** public entry point, `gk_price`, `d1`, `d2`, `analytic_greeks`, `finite_difference_greeks`, `delta`, `strike_from_delta`, `binomial_price`, `mc_price`, `black76_price`, `black76_from_spot`, `implied_vol` (incl. the premium), `cip_forward`, `forward_points` (incl. `pip_factor`), `synthetic_forward_from_options`, `atm_forward_strike`, `atm_dns_strike`, `premium_adjust_spot_delta`, `spot_to_forward_delta`, `forward_to_spot_delta` (`edge_cases::every_entry_point_rejects_non_finite_inputs`) | exact (must `Err`) |
| 100x moneyness | `S/K` from 1/100 to 100 x `T` in {1d, 1y, 30y, 100y}: prices finite, non-negative, inside `[0, S e^{-r_f T}]` / `[0, K e^{-r_d T}]`, parity to 1e-13 **relative to notional scale**; all nine Greeks finite and correctly signed | 1e-13 rel |
| Zero / negative rate regimes | `(r_d, r_f)` in {(0,0), (0,2%), (2%,0), (-0.75%,-0.5%), (-5%,+5%), (+5%,-5%)}: parity holds and implied vol round-trips | 1e-14 / 1e-9 |
| Boundary deltas | `target_delta` in {0, +/-1, just outside, +/-1e300} rejected for all four conventions; `1e-4` (just inside) still solves and round-trips; a spot delta whose forward equivalent reaches 1 is rejected | exact / 1e-8 |
| Delta bounds at extremes | at 100x moneyness every convention's delta stays inside its own cap (`e^{-r_f T}`, `1`, `(K/F) e^{-r_f T}`, `K/F`) and never goes NaN; deep-ITM spot delta pins to `e^{-r_f T}` | 1e-12 |
| Minimum path counts | fewer than two independent samples (three with the control variate) returns `InvalidInput`; the smallest valid counts give a positive, finite SE | exact |
| Property tests (proptest) | parity, monotone in vol and strike, strike convexity, delta bounds, IV round trip (`properties.rs`) | 300 cases each |
| PA fold for large total vol | the fold bracket is derived for every `sigma sqrt(T)`; v = 37.5 solves, v = 40 returns `InvalidInput` (strike beyond f64 range) instead of a Brent failure (`numerical_safety.rs`) | exact |
| MC at extremes | 100x moneyness and `T = 30y` within `3 SE + 1e-10 x scale` of analytic GK | 3 SE + rel |

## Known failure modes and numerical limits

* **Implied vol near the bounds.** The shared core solves on the
  out-of-the-money side, so a tiny out-of-the-money price inverts fully.
  An in-the-money quote's time value is the difference of the quote and
  the intrinsic bound; within `4 eps df_d max(F, K)` of that bound the
  solver returns `0.0` (the sigma -> 0 limit). Within `4 eps upper` of the
  sigma -> infinity bound it returns `InvalidInput`: no finite vol is
  identifiable (`implied_vol::long_dated_deep_itm_high_vol_flat_plateau_errs`).
* **Short-dated wings.** A short-dated ITM call (T = 0.02, vega about
  7.6e-9) carries little vol information: one ulp of the price moves the
  vol by about 3e-8. The OTM put at the same strike carries full
  information. `implied_vol::round_trip_short_dated_wings` uses the
  condition-number tolerance `8 eps max(S,K)/vega + 2e-12 sigma` for both.
  An earlier version of this document called the ITM error an IEEE-754
  floor of the solver; it is a property of the ITM quote, and the earlier
  solver also lost accuracy on the OTM side (2.7e-7) because it stopped on
  an absolute price residual.
* **Variance-reduction interaction in Monte Carlo.** The control-variate
  coefficient is fitted on the antithetic pair averages, the independent
  samples. Fitting it on raw draws and pairing afterwards gave
  `SE(antithetic+CV) / SE(plain) ~ 0.76` against `~0.55` for the control
  variate alone on this market; the pair-average fit gives `~0.24`. The
  Python, C++ and Rust engines now use the same estimator.
* **PA call delta above the fold.** The premium-adjusted call delta has a
  maximum in strike; targets above it have no solution and `strike_from_delta`
  returns `Err` rather than silently returning the wrong branch.
* **`d1`/`d2` undefined at `sigma * sqrt(T) = 0`**, `d1()`/`d2()` return
  `Err`; the pricers switch to the analytic limits instead.
* **NaN is the failure mode this engine guards hardest against.** A
  validation written as `if x <= 0.0 { return Err(...) }` silently
  *accepts* NaN, because every IEEE-754 comparison against NaN is false.
  Rust's type system does not help here, `f64::NAN < 0.0` is false
  exactly as in C and C++. Every public entry point therefore validates
  with `is_finite()` (via `require_finite`/`validate_inputs`) rather than
  with an ordering comparison, and the contract is pinned by
  `edge_cases::every_entry_point_rejects_non_finite_inputs`, which
  poisons each float slot of each function in turn. The motivation is
  operational, not aesthetic: a NaN price or a NaN delta breaches no
  limit and colours no traffic light (`NaN <= limit` is false, but so is
  `NaN > limit`), so it propagates into an aggregated book number
  unnoticed. Three concrete holes closed in this pass:
  `premium_adjust_spot_delta` accepted a NaN premium/delta and returned a
  NaN hedge ratio; `spot_to_forward_delta`/`forward_to_spot_delta`
  scaled a NaN delta through unchecked; and `forward_points` accepted
  `pip_factor = +inf` and returned `inf` points. Finite inputs whose result
  overflows (a forward or discount factor beyond f64 range) now return
  `InvalidInput` instead of `Ok(inf)` or `Ok(NaN)`.
* **Finite-difference Greeks need a live bump.** `finite_difference_greeks`
  now rejects `T <= 0`, `sigma <= 0` and any `rel_bump` that is not
  finite and strictly positive. Previously `T = 0` collapsed the theta
  bump `min(1e-6, T/4)` to exactly zero, making the difference quotient
  an unreported `0/0 = NaN`, and `rel_bump = 0` did the same for delta
  and gamma. Analytic `analytic_greeks` was, and remains, the right call
  in those regimes (it errors cleanly).
* **Control variates are unbiased, not non-negative.** At single-digit
  path counts the fitted beta is pure noise and the adjusted estimate
  `X - beta (C - E[C])` can come out slightly negative even though every
  raw payoff is non-negative. This is a property of the estimator, not a
  defect; the minimum-path-count test asserts non-negativity only for the
  plain estimator, and the desk rule is simply not to run a control
  variate at trivial path counts.

## Low-latency API checks

* `tests/fast_bits.rs`: every `fast::` kernel equals the classic call by
  `f64::to_bits` on 4,000 seeded random inputs (price, all ten Greeks,
  implied vol), on the `T = 0` and `sigma = 0` limits, and on 60 inputs
  times 5 step counts times both exercise styles for the tree. The batch
  pricers equal the scalar ones on 1,031 inputs. `mc_price_par` equals
  `mc_price` for all four variance-reduction settings, 5 path counts and
  1, 2, 3, 4, 7, 12 and 64 threads, and repeats across runs.
* `tests/fast_zero_alloc.rs`: a counting global allocator sees zero
  allocations in each kernel call once its workspace is sized. The same
  test sees the classic tree allocate, so the counter is live.

## Latency table

Per-call latency from `cargo run --release --bin bench` (writes
[bench/results.json](../bench/results.json)). Numbers from this machine: Intel
Core 5 120U, 12 threads, under WSL2, Rust 1.99.1, release profile
`opt-level = 3`, `lto = "fat"`, `codegen-units = 1`. Other builds shared the
machine during the run (load average about 7), so the tails carry
scheduler jitter. Kernels under 1 us are timed in batches of 64 or 16
calls and reported per call; the batch rows time one 1,024-option batch
and report per option.

| Kernel | p50 ns | p99 ns | p99.9 ns |
|---|---:|---:|---:|
| `gk_price` | 64 | 72 | 332 |
| `fast::gk_price_k` | 61 | 146 | 711 |
| `fast::gk_price_batch`, per option | 59 | 181 | 782 |
| `analytic_greeks` | 48 | 54 | 221 |
| `fast::greeks_k` | 30 | 75 | 627 |
| `fast::greeks_batch`, per option | 67 | 177 | 265 |
| `implied_vol` | 602 | 1,111 | 2,263 |
| `fast::implied_vol_k` | 654 | 1,521 | 17,999 |
| `strike_from_delta`, spot_pa | 1,333 | 1,581 | 6,763 |
| `binomial_price`, American, 500 steps | 77,507 | 101,014 | 186,863 |
| `fast::binomial_price_ws`, same tree | 89,060 | 182,541 | 264,781 |
| `mc_price`, 100k paths, antithetic + CV | 5.79 M | 7.89 M | 7.99 M |
| `fast::mc_price_ws`, same | 7.27 M | 11.42 M | 12.97 M |
| `mc_price`, 1M paths | 73.1 M | 174.2 M | 174.2 M |
| `fast::mc_price_par`, 1M paths, 12 threads | 28.0 M | 35.4 M | 37.9 M |

Read the table this way. The closed-form kernels sit at 30 to 70 ns, and
`fast::` matches the classic call within the noise of this machine. The
gain of the `fast::` API is the contract: no heap use, a `Copy` error, and
caller-owned workspaces. The one large speed-up is the threaded Monte
Carlo, about 2.6 times on 1M paths. The batch loops do not vectorise:
`norm_cdf` (Cody erfc, branchy) and the `exp`/`ln` calls into libm block
it. The validation scan of the batch, the tree's backward induction and
the antithetic pair averages do vectorise (SSE2, width 2), checked with
`RUSTFLAGS=-Cremark=loop-vectorize` and the emitted assembly.
