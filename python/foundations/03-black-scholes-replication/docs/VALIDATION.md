# Validation: Black-Scholes Replication

This document states how the implementation was validated (analytic
benchmarks, convergence studies, cross-model consistency) and where it
fails, with reproducible numbers. Every number comes from the committed
code: `python examples/run_pipeline.py` prints the report (add
`--out DIR` to save it, and `--plot` for the figure). `pytest -q` (304
tests, offline, seeded, about 3 s) enforces the numeric claims.

Reference contract unless stated otherwise: `S=100, K=105, r=3%,
sigma=25%, T=0.75y`.

---

## 1. Identities that must hold

| Check | Result | Test |
|---|---|---|
| Put-call parity `C - P = S - K e^{-rT}`, reference contract | error 0.00e+00 | `tests/test_pricing.py::test_put_call_parity` |
| Put-call parity, 270-point grid (S, K, r ∈ {−2%, 0, 5%}, T, sigma) | max abs error < 1e-9 | `tests/test_pricing.py::test_put_call_parity_grid` |
| `max(S − K e^{-rT}, 0) ≤ C ≤ S` and the put analogue | holds | `test_no_arbitrage_bounds`, `test_no_arbitrage_bounds_put` |
| Call and put increasing in sigma | sigma ∈ {10%, 20%, 30%, 40%} | `test_monotonic_in_vol`, `test_monotonic_in_vol_put` |
| Call non-increasing, put non-decreasing in K | K ∈ {80, 100, 120, 140} | `test_call_decreasing_in_strike`, `test_put_increasing_in_strike` |

Put-call parity follows from static replication of a forward, not from
any distributional assumption, so it catches implementation bugs
whether or not Black-Scholes is the right model.

## 2. Monte Carlo vs closed form

Closed-form call price on the reference contract: 7.467460.

From `python examples/run_pipeline.py` (seed=7, antithetic variates):

| paths | MC price | std error | abs error | SE·√n |
|---:|---:|---:|---:|---:|
| 1,000 | 6.8840 | 0.3294 | 0.5835 | 10.42 |
| 10,000 | 7.4104 | 0.1113 | 0.0571 | 11.13 |
| 100,000 | 7.4314 | 0.0354 | 0.0361 | 11.21 |
| 1,000,000 | 7.4517 | 0.0112 | 0.0158 | 11.25 |

`SE·√n` is about 11.0 across three orders of magnitude of sample size,
the O(n^-1/2) law. The fitted error-decay exponent is 0.523 against a
theoretical 0.5. Each row's error is within 3 standard errors
(`tests/test_monte_carlo.py::test_mc_agrees_with_closed_form_within_3_standard_errors`
for 10k/100k/1M; `test_mc_standard_error_shrinks_like_inverse_sqrt_n`
checks `SE·√n` constant to 15%).

The Monte Carlo pricer shares no code with the closed form
(`docs/METHODOLOGY.md` §1.2), so a shared bug would have to corrupt
both identically to agree at every sample size.

The pricer uses antithetic variates only. It has no control variate.

### 2.1 The antithetic standard error

Antithetic variates reduce variance at equal path count
(`tests/test_monte_carlo.py::test_mc_antithetic_reduces_variance`,
`tests/test_extremes.py::test_mc_antithetic_beats_plain_at_the_same_path_count`).

The `2m` antithetic payoffs come in negatively correlated pairs, so
`std(all 2m payoffs, ddof=1) / sqrt(2m)` misstates the error. On the ATM
reference contract at 20,000 paths, the empirical standard deviation of
the estimate across 400 seeds is 0.0750. The naive formula reports
0.0997, a 33% overstatement. The pair-average formula
`std(pair means, ddof=1) / sqrt(m)` reports 0.0743. An earlier version
used the naive formula, and every "within 3 standard errors" check
still passed against the looser bar.
`test_mc_antithetic_standard_error_matches_empirical_dispersion` runs
60 seeds and requires the mean reported error to match the empirical
dispersion within 20%.

Degenerate cases: with antithetic sampling, `n_paths=2` or 3 is one
independent unit, so the standard error is `NaN`. `n_paths=1` with
antithetic pairing raises `ValueError`.

## 3. Analytic Greeks vs finite differences

Central differences on the reference contract, `h = 1e-4` for delta and
vega and `h = 1e-2` for gamma (from the pipeline):

| Greek | analytic | finite-difference | \|diff\| |
|---|---:|---:|---:|
| delta | 0.49474373 | 0.49474373 | 9.30e-14 |
| vega | 34.54641612 | 34.54641607 | 5.16e-08 |
| gamma | 0.01842476 | 0.01842476 | 2.07e-10 |

Gamma uses a larger step because the second difference has round-off of
order `eps * C / h^2`: at `h = 1e-4` that is about 1e-6, at `h = 1e-2`
truncation and round-off are both near 1e-10. Rho and theta are checked
in `tests/test_greeks.py`. Put Greeks come from call Greeks by the
put-call-parity relations, and
`tests/test_greeks.py::test_put_greeks_parity_relations` checks all five
relations to < 1e-12.

## 4. Implied-vol round trips

Price to implied vol to price, `S=100, K=105, r=3%, T=0.75y` (from the
pipeline):

| input sigma | recovered | \|diff\| |
|---:|---:|---:|
| 0.08 | 0.08000000 | 1.80e-16 |
| 0.20 | 0.20000000 | 0.00e+00 |
| 0.55 | 0.55000000 | 4.44e-16 |
| 1.20 | 1.20000000 | 4.44e-16 |

`implied_volatility` accepts calls and puts (`option_type`). It first
maps the quote to the out-of-the-money side by put-call parity
(`P = C - S + K e^{-rT}`), then runs Newton-Raphson on vega with a
bisection fallback. An ITM price is mostly intrinsic value; the OTM
price holds the same time value without it, so the inversion is well
conditioned. Deep ITM at low vol (`K=60, sigma=10%`) now round-trips to
< 1e-6 (`tests/test_edge_cases.py::test_deep_itm_low_vol_round_trip_uses_the_otm_side`).
`K=80` priced at 5% vol recovers 0.05 to 1e-8; the call-side inversion
used before returned 0.0501
(`tests/test_tail_and_itm_iv.py::test_deep_itm_call_quote_inverts_to_its_true_vol`).
A hypothesis property test round-trips random calls and puts with OTM
price ≥ 1e-3 to 1e-6 in vol. Prices outside the no-arbitrage bounds
raise `ValueError`.

### 4.1 Where implied vol is not identified

Implied vol is only as well determined as the time value is
resolvable. The solver warns (`RuntimeWarning`) and returns `nan` when:

- the OTM-side price is at most `64 * eps * max(S, K)` (about 1.4e-12
  for `S=K=100`). A call quoted at its discounted intrinsic value, whose
  true implied vol is zero, falls here
  (`test_implied_vol_at_exactly_intrinsic_returns_nan_with_warning`). So
  does `K=50` priced at 5% vol: its time value is about 1e-48, far below
  what a double can hold next to a price of 51. The previous
  implementation returned 0.175 for it
  (`test_unresolvable_itm_quote_warns_and_returns_nan`).
- the price sits at the upper bound (`C = S`), where the implied vol is
  infinite (`test_implied_vol_at_exactly_the_upper_bound_returns_nan_with_warning`).

One tick outside either bound is arbitrage and raises `ValueError`
(`test_implied_vol_just_below_intrinsic_is_rejected`). A surface fit
should drop strikes whose vega is below a floor.

### 4.2 Volatilities above 500%

The bisection upper bracket starts at 5.0 and doubles until it spans the
target, so a 900%-vol quote round-trips to `rel=1e-5`
(`test_implied_vol_inverts_a_quote_above_five_hundred_percent`).

## 5. The volatility-smile experiment

If constant-vol lognormal GBM (A1) held, inverting prices at every
strike would recover one flat vol. This experiment prices a strike
ladder by Monte Carlo under a synthetic Student-t return distribution
(df=4, variance-matched to sigma=25%) and inverts each price through
the constant-vol closed form.

From the pipeline (2,000,000 synthetic paths, seed=3):

| strike | BS implied vol |
|---:|---:|
| K=70 (deep OTM put / deep ITM call) | 27.6% |
| K=105 (near ATM) | 22.9% |
| K=140 (deep OTM call / deep ITM put) | 28.6% |

Average wing-vs-ATM gap on this synthetic data: 5.2 vol points. Fat
tails make extreme outcomes more likely than the lognormal admits, so
tail-paying options are worth more than Black-Scholes at the ATM vol
says, and the inverted vol is higher in the wings. Listed equity and
index options have shown a smile or skew since 1987. The experiment
reproduces the mechanism, not the market shape, which is asymmetric and
driven by more than fat tails (§6.1).

`python examples/run_pipeline.py --out DIR --plot` writes
`DIR/figures/black_scholes_overview.png`; its bottom-right panel plots
the smile across strikes 70 to 140 against the flat BS assumption.

## 6. Known failure modes

Each failure mode below either points to a reproducible demonstration
or states what one would show and where it is built.

### 6.1 Non-constant or stochastic volatility

Demonstrated in §5: a 5.2-vol-point smile on synthetic fat-tailed
prices where constant-vol GBM predicts a flat line. Lowering the
Student-t degrees of freedom in `examples/run_pipeline.py` steepens the
smile; raising them flattens it toward the normal case.

### 6.2 Price jumps

Not built here. A demonstration would replace the terminal draw in
`mc_call_price` with Merton jump-diffusion
(`S_T = S0 exp((r - lambda*k - sigma^2/2)T + sigma sqrt(T) Z + sum J_i)`)
and invert through the jump-free closed form. The known result is
higher implied vol for short expiries than long ones at the same strike,
because jumps are a larger share of total variance over short horizons.
Jumps also break the hedging argument (A3): an overnight gap cannot be
delta-hedged at any rebalancing frequency.

### 6.3 Discrete-hedging frictions

Not built here. A demonstration would simulate a discretely rebalanced
delta hedge along GBM paths and show (a) hedge P&L noise that shrinks
like `1/sqrt(N)` in the number of rebalances but never reaches zero,
and (b) with proportional costs, an optimal finite rebalancing
frequency. The simulator is built and measured in
`python/equity/01-options-pricing` (`src/eq_options/hedging.py`, see
that project's `docs/VALIDATION.md` §5).

### 6.4 Early exercise

Not built here. The European closed form bounds an American option from
below and cannot represent early exercise. A CRR tree that compares
exercise with continuation at every node gives the early-exercise
premium. For American puts it is positive when the put is sufficiently
ITM. For American calls without dividends it is zero (Merton), which is
a sharp test that the exercise branch never fires. The tree, with a
measured 1y ATM put premium of 0.4234, is in
`python/equity/01-options-pricing` (`src/eq_options/binomial.py`, see
that project's `docs/VALIDATION.md` §4).

## 7. Numerical limits (each tested)

- **NaN and ±inf inputs:** `_d1_d2` checks `math.isfinite` on every
  input before the sign guards, because `NaN <= 0` is `False`. Every
  pricer, Greek, the implied-vol solver and the Monte Carlo pricer raise
  `ValueError` naming the argument (`tests/test_input_hardening.py`, 92
  parametrised cases).
- **T→0 and sigma→0:** `T<=0` or `sigma<=0` raises `ValueError`
  (`test_zero_or_negative_T_raises`, `test_zero_or_negative_sigma_raises`).
  At `T=1e-6` or `sigma=1e-6` the price converges to discounted
  intrinsic value within 1e-4.
- **Deep ITM and OTM:** delta saturates toward 0 or 1 (calls) and 0 or
  −1 (puts). Deep OTM prices keep full relative accuracy because the
  CDF uses `erfc`: `S=100, K=300, T=0.25, sigma=0.2` prices at 3.5e-28,
  not 0 (`test_deep_otm_call_is_positive_not_zero`;
  `test_norm_cdf_left_tail_keeps_relative_accuracy` checks `N(-x)` to
  rel 1e-8 for x from 8 to 37 against the asymptotic series).
- **Zero and negative rates:** supported. Parity, pricing and implied
  vol hold at `r=0` and `r=-1%` (`test_zero_rate_is_supported`,
  `test_negative_rate_is_supported`,
  `test_negative_rate_implied_vol_round_trips`).
- **Invalid inputs:** `sigma<=0`, `T<=0`, `S<=0` or `K<=0` raise
  `ValueError` stating the failed precondition and the values
  (`test_error_message_is_informative`). `implied_volatility` also
  raises on prices outside the no-arbitrage bounds, on `T<=0`, and on
  an unknown `option_type`.
- **Very long maturities (T = 10, 30, 50 years):** nothing overflows,
  parity holds to `1e-9`, Greeks stay finite and correctly signed. At
  `T=30, r=3%` the strike discounts to 40.66% of face, so an at-spot
  call is worth more than 60% of the stock. Gamma falls monotonically
  with maturity. Monte Carlo matches the closed form within 3 standard
  errors at `T=30` (`tests/test_extremes.py`).
- **Strikes from 1e-10 to 1e10:** `log(S/K)` spans ±23 and `d1` runs
  past ±100, where `N` saturates at 0 and 1. Prices stay finite and in
  bounds; parity holds to `rel=1e-12` (relative, since at `K=1e10` both
  sides are about 1e10). `K→0` makes the call the stock; `K→∞` makes
  the put the discounted strike minus the stock.
- **Discount-factor overflow:** `exp(-r·T)` overflows for very negative
  `r·T` (for example `r=-1000%`, `T=100y`). This raises a `ValueError`
  naming both inputs, not a bare `OverflowError`
  (`test_discount_factor_overflow_raises_informative_value_error`).
- **Saturation at extreme total vol:** at `sigma·sqrt(T)` of 8, a put's
  gap to its upper bound is about 1e-100, below double resolution next
  to a 40-point price, so the computed put equals the discounted strike.
  The bound holds as `<=`, not `<`
  (`test_put_bound_is_tight_and_saturates_at_extreme_vol`).
- **Put-side bounds** `max(K·e^{−rT} − S, 0) <= P <= K·e^{−rT}` hold on
  a 108-contract grid over three orders of moneyness, negative to
  positive rates and 4 days to 30 years
  (`test_put_bounds_hold_across_a_wide_grid`).
- **Monte Carlo input contract:** `mc_call_price` validates like the
  closed form, so the cross-check always compares the same contracts
  (`test_mc_input_validation_mirrors_the_closed_form`).
- **Pipeline side effects:** the pipeline runs without matplotlib and
  writes nothing unless `--out` is given
  (`tests/test_pipeline_side_effects.py`).
