# Validation — Black-Scholes Replication

This document answers documentation-contract items 3 and 4: **how the
implementation was validated** (analytic benchmarks, convergence
studies, cross-model consistency) and **where it fails** (known failure
modes, with reproducible numbers). Every number below is produced by
the committed code — `python examples/run_pipeline.py` regenerates the
report and figures under `output/`; `pytest -q` (200 tests, offline,
seeded) enforces the numeric claims permanently.

Reference contract unless stated otherwise: `S=100, K=105, r=3%,
sigma=25%, T=0.75y`.

---

## 1. Identities that must hold exactly

| Check | Result | Test |
|---|---|---|
| Put-call parity `C - P = S - K e^{-rT}` on the reference contract | error **0.00e+00** | `tests/test_pricing.py::test_put_call_parity` |
| Put-call parity across a 270-point grid (S, K, r ∈ {−2%, 0, 5%}, T, sigma) | max abs error < 1e-9 | `tests/test_pricing.py::test_put_call_parity_grid` |
| No-arbitrage bounds `max(S − K e^{-rT}, 0) ≤ C ≤ S` (and the put analogue) | holds on reference contract | `test_no_arbitrage_bounds`, `test_no_arbitrage_bounds_put` |
| Call strictly increasing in sigma (vega > 0); put likewise | monotone over sigma ∈ {10%, 20%, 30%, 40%} | `test_monotonic_in_vol`, `test_monotonic_in_vol_put` |
| Call monotone non-increasing in K; put monotone non-decreasing in K | monotone over K ∈ {80, 100, 120, 140} | `test_call_decreasing_in_strike`, `test_put_increasing_in_strike` |

Put-call parity is a **model-free** arbitrage relation (it follows from
static replication of a forward with a call minus a put, not from any
distributional assumption), so it catches implementation bugs
regardless of whether Black-Scholes itself is the right model for
reality. On the reference contract the parity residual from the
committed pipeline run is exactly `0.00e+00` (i.e. below float64
printing precision at 2 significant figures); the grid test enforces
< 1e-9 across 270 combinations including zero and negative rates.

## 2. Two independent implementations converging: Monte Carlo vs closed form

Closed-form call price on the reference contract: **7.467460**.

From `python examples/run_pipeline.py` (seed=7, antithetic variates):

| paths | MC price | std error | abs error | SE·√n |
|---:|---:|---:|---:|---:|
| 1,000 | 6.8840 | 0.3294 | 0.5835 | 10.42 |
| 10,000 | 7.4104 | 0.1113 | 0.0571 | 11.13 |
| 100,000 | 7.4314 | 0.0354 | 0.0361 | 11.21 |
| 1,000,000 | 7.4517 | 0.0112 | 0.0158 | 11.25 |

`SE·√n` is constant at **≈11.0** across three orders of magnitude of
sample size — the **O(n^{-1/2})** law, measured directly rather than
assumed. The fitted error-decay exponent over the full range is
**0.523** against a theoretical **0.5**. Every row's absolute error is
within 3 standard errors of zero (enforced for 10k/100k/1M paths by
`tests/test_monte_carlo.py::
test_mc_agrees_with_closed_form_within_3_standard_errors`; the SE-decay
law itself is enforced by `test_mc_standard_error_shrinks_like_inverse_sqrt_n`,
which checks `SE·√n` is constant to within 15% across the same three
sample sizes).

This is strong evidence for both implementations simultaneously: the
Monte Carlo pricer shares **no code** with the closed form (it draws
normal variates and evaluates the SDE's known terminal solution
directly; see `docs/METHODOLOGY.md` §1.2), so a shared bug would have
to corrupt both in exactly the same way to still agree at every sample
size — vanishingly unlikely to happen by coincidence across four
independent trials of increasing precision.

### 2.1 The antithetic standard error, and a bug that hid inside it

Antithetic variates measurably reduce variance at equal path count
(`tests/test_monte_carlo.py::test_mc_antithetic_reduces_variance` and
`tests/test_extremes.py::test_mc_antithetic_beats_plain_at_the_same_path_count`):
the antithetic standard error is strictly lower than the plain one, at no
extra RNG draws.

Getting that standard error *right* is subtler than it looks, and an
earlier version of this module got it wrong. With antithetic sampling the
estimator averages `2m` payoffs, but those payoffs come in mirrored pairs
that are **negatively correlated by construction** — that correlation is
the entire mechanism of the variance reduction. Computing the error as
`std(all 2m payoffs, ddof=1) / sqrt(2m)` treats them as `2m` independent
observations and therefore misstates the answer.

Measured directly: on the ATM reference contract at 20,000 paths, the
empirical standard deviation of the price estimate across 400 independent
seeds is **0.0750**. The old (naive) formula reported **0.0997** — a 33%
overstatement. The correct calculation treats the `m` *pair averages* as
the independent units, `std(pair means, ddof=1) / sqrt(m)`, and reports
**0.0743**, which matches the empirical dispersion.

The direction of the error is worth noting: the old formula was
*conservative*, reporting a wider error bar than the estimator deserved.
That made it invisible in testing — every "within 3 standard errors"
check still passed, just against a bar 33% looser than advertised. A test
suite that certifies accuracy in units of its own standard error is only
as good as that standard error, so this is pinned directly now:
`test_mc_antithetic_standard_error_is_honest_about_its_own_accuracy`
runs 60 independent seeds and requires the mean reported standard error
to match the empirical dispersion of the estimates to within 20%.

The same reasoning gives the degenerate cases their answers. With
antithetic sampling, `n_paths=2` (or 3) is **one** independent unit, not
two: `ddof=1` is undefined, so the standard error is `NaN` rather than a
falsely tight number (the old code reported a standard error numerically
equal to the price). And `n_paths=1` with antithetic pairing requested is
now a `ValueError` — it used to compute `1 // 2 == 0` draws and return
`(nan, nan)` after a stack of NumPy RuntimeWarnings.

## 3. Analytic Greeks vs finite differences

Central finite differences with `h = 1e-4` on the reference contract
(from the committed pipeline run):

| Greek | analytic | finite-difference | \|diff\| |
|---|---:|---:|---:|
| delta | 0.49474373 | 0.49474373 | 9.30e-14 |
| vega | 34.54641612 | 34.54641607 | 5.16e-08 |
| gamma | 0.01842476 | 0.01842579 | 1.04e-06 |

(rho and theta are additionally checked in `tests/test_greeks.py` at
the same tolerance.) These residuals are the expected O(h²) truncation
error of a central difference at `h = 1e-4` — they are numerical-method
noise, not model error, and shrinking `h` further shrinks them (down to
the point where float64 subtraction cancellation dominates). Both call
and put Greeks are checked this way; put Greeks are derived
algebraically from call Greeks via the put-call-parity relations
documented in `eq_bs_replication.black_scholes.put_greeks`, and
`tests/test_greeks.py::test_put_greeks_parity_relations` cross-checks
that derivation directly against the parity identities (all five
relations verified to < 1e-12), independent of the finite-difference
check.

## 4. Implied-vol round trips

Price → implied vol → price, reference contract, `S=100, K=105, r=3%,
T=0.75y` (from the committed pipeline run):

| input sigma | recovered | \|diff\| |
|---:|---:|---:|
| 0.08 | 0.08000000 | 7.26e-12 |
| 0.20 | 0.20000000 | 0.00e+00 |
| 0.55 | 0.55000000 | 4.44e-16 |
| 1.20 | 1.20000000 | 2.22e-16 |

Across low (8%) and high (120%) vol this exercises both the Newton-
Raphson stage and the bisection fallback (Newton is abandoned when
vega < 1e-10). `tests/test_implied_vol.py` additionally round-trips
across strikes 80–120 and expiries 1 week – 3 years to < 1e-5, and
checks that sub-intrinsic or super-spot prices are refused with
`ValueError` rather than silently inverted (no-arbitrage-bound guard).

**A documented exception:** deep ITM combined with low volatility
(e.g. `K=60` at `sigma=10%` on the reference spot/rate/expiry) makes
vega tiny, so the Newton step is ill-conditioned and the round trip
degrades to ~1e-2 precision rather than ~1e-8 — the option's price in
that corner is so close to intrinsic value that it simply does not
carry much information about sigma. This is expected numerical
behaviour, not a bug, and is tested explicitly rather than swept under
the main round-trip test:
`tests/test_edge_cases.py::test_deep_itm_low_vol_round_trip_is_imprecise`.

### 4.1 The two no-arbitrage boundaries: where implied vol stops existing

The deep-ITM case above is the mild version of a general property:
**implied volatility is only as well-determined as vega is large**, and
vega vanishes at both ends of the no-arbitrage interval. The two
endpoints are now tested directly (`tests/test_extremes.py`):

- **A call quoted at exactly its discounted intrinsic value**
  (`C = max(S − K·e^{−rT}, 0)`) has a true implied vol of **zero**. The
  routine returns **0.0092** — a sigma that reprices the option to within
  `1e-8` of the quote while being nowhere near the right answer, because
  every volatility below about 5% produces a price difference smaller
  than a cent (vega at the returned point is < 1.0 per unit of vol).
  The returned number satisfies the contract the function promises (a
  *price* residual below `tol`) and is meaningless as a *volatility*.
  Pinned by `test_implied_vol_at_exactly_intrinsic_is_ill_conditioned_not_wrong`,
  which asserts both halves: it reprices, and it is wrong.
- **A call quoted at exactly spot** (`C = S`, the upper bound) has a true
  implied vol of **infinity**. The routine returns a large finite number
  (~13 on the reference contract) — whatever value first matches to
  tolerance. Pinned by
  `test_implied_vol_at_exactly_the_upper_bound_returns_a_large_finite_number`.
- **One tick outside either bound** is arbitrage, not an extreme
  volatility, and raises `ValueError`
  (`test_implied_vol_just_below_intrinsic_is_rejected`).

The desk consequence, and the reason this is documented rather than
"fixed": there is no algorithm that recovers information the price does
not contain. A surface fit must **drop** strikes whose vega is below a
sensible floor (a few cents per vol point) rather than fitting them with
a wide error bar, because the implied vol there is not a noisy
measurement of a real quantity — it is an arbitrary solution to an
underdetermined equation.

### 4.2 Volatilities above 500%

The bisection fallback used a fixed `[1e-6, 5.0]` bracket. Any quote
implying a volatility above 500% — a distressed single name, a crypto
option, a very short-dated event straddle — would have been silently
pinned near the bracket edge whenever Newton also failed. The bracket now
**doubles upward** until it spans the target price, and a 900%-vol quote
round-trips to `rel=1e-5`
(`test_implied_vol_inverts_a_quote_above_five_hundred_percent`).

## 5. The volatility-smile experiment (where the model breaks, demonstrated)

If assumption A1 (constant-vol, lognormal GBM) held in reality, reading
option prices back through Black-Scholes at every strike would recover
the *same* implied vol everywhere — a flat line. This experiment prices
a strike ladder by Monte Carlo under a **Student-t** (df=4,
variance-matched to the reference `sigma=25%`) return distribution
instead of the model's own normal distribution, then inverts each price
through the (unchanged, constant-vol) closed form's `implied_volatility`.

From the committed pipeline run (2,000,000 simulated paths, seed=3):

| strike | BS implied vol |
|---:|---:|
| K=70 (deep OTM put / deep ITM call) | 27.6% |
| K=105 (near ATM) | 22.9% |
| K=140 (deep OTM call / deep ITM put) | 28.6% |

Average wing-vs-ATM gap: **5.2 vol points**. The wings trade at
noticeably higher implied vol than the middle — a **smile**, not the
flat line constant-vol GBM predicts — because a fat-tailed distribution
makes extreme terminal outcomes more likely than the lognormal admits,
so options that only pay off in the tails are worth more than
Black-Scholes (evaluated at the ATM-consistent vol) thinks, and
inverting that higher price through the constant-vol formula shows up
as a higher "implied" vol. Real listed equity- and index-option markets
have shown a smile/skew shape persistently since the 1987 crash, which
is direct market evidence against constant-vol GBM — this experiment
reproduces the *mechanism*, not the exact market shape (the real market
skew is asymmetric and driven by more than fat tails alone; see
§6.1 below).

Figure: `output/figures/black_scholes_overview.png` (bottom-right
panel) plots the full smile across strikes 70–140 in 5-point steps
against the flat BS assumption.

## 6. Known failure modes

Per CONVENTIONS.md item 4, each failure mode below either points to a
reproducible demonstration already in this codebase, or describes
precisely what a demonstration would show if built (and where it
*is* built, in the companion project).

### 6.1 Non-constant / stochastic volatility

**Demonstrated here.** §5 above is a direct, reproducible demonstration:
constant-vol GBM predicts a flat implied-vol curve; pricing under a
fat-tailed distribution and reading back through Black-Scholes produces
a 5.2-vol-point smile instead. Re-run with
`python examples/run_pipeline.py` (deterministic, seed=3) to reproduce
the exact numbers above; vary the Student-t degrees of freedom in
`examples/run_pipeline.py` to see the smile steepen as tails fatten
further (lower df) or flatten toward the model's own assumption as
df → ∞ (approaching a normal distribution again).

### 6.2 Price jumps

**Not built here; precisely specified.** A demonstration would replace
the continuous-diffusion terminal draw in `mc_call_price` with a
jump-diffusion terminal distribution (e.g. Merton's compound-Poisson
jumps superimposed on GBM: `S_T = S0 exp((r - lambda*k - sigma^2/2)T +
sigma sqrt(T) Z + sum of J_i jumps)`), price under that model by Monte
Carlo, and invert through the (jump-free) Black-Scholes closed form.
The expected result, well established in the literature this project
does not re-derive: implied vol comes out higher for *short-dated*
options than long-dated ones at the same strike (a downward-sloping
term structure of implied vol), because a jump's contribution to
terminal variance is a larger fraction of total variance the shorter
the horizon — jump risk cannot be diversified away by the passage of
time the way diffusive variance can. This project's own delta-hedging
argument (assumption A3) also breaks structurally under jumps: an
overnight gap cannot be delta-hedged continuously no matter how
frequently you rebalance during market hours, so the replication
argument that produces the Black-Scholes price in the first place does
not apply.

### 6.3 Discrete-hedging frictions

**Not built here; precisely specified.** A demonstration would simulate
a discretely-rebalanced delta hedge of a short option position along
simulated GBM paths (rebalance every `Delta t`, track hedge P&L) and
show two things: (a) even at the *correct* volatility, P&L has
irreducible standard deviation that shrinks like `1/sqrt(N)` in the
number of rebalances `N` but never reaches zero at any finite `N`
(this is exactly the same convergence-rate signature this project
already measures for Monte Carlo pricing error in §2, applied instead
to hedging error); and (b) adding proportional transaction costs turns
"rebalance more to reduce noise" into a trade-off against cost drag,
so there is an optimal finite rebalancing frequency rather than a
"more is strictly better" answer. This exact simulator, with measured
numbers (P&L std scaling, cost drag at various rebalance frequencies),
is built and tested in `python/equity/01-options-pricing`
(`src/eq_options/hedging.py`); see that project's
`docs/VALIDATION.md` §5 for the actual figures.

### 6.4 Early exercise

**Not built here; precisely specified.** The European closed form only
*bounds* an American option's value from below — it cannot represent
the early-exercise decision at all. A demonstration would price the
same contract with a Cox-Ross-Rubinstein binomial tree that checks, at
every node, whether immediate exercise beats continuation value, and
report the resulting **early-exercise premium** (American price minus
European price) as a function of moneyness and dividend yield. For
American puts (which can be optimal to exercise early even with zero
dividends, because receiving the strike early and earning interest on
it can outweigh remaining time value) that premium is strictly
positive whenever the put is sufficiently ITM; for American calls with
no dividends, Merton's theorem says the premium is exactly zero (never
optimal to exercise early), which is itself a sharp implementation test
(the early-exercise branch must fire *never*, despite being evaluated
at every tree node). This tree, with a measured 1y ATM put
early-exercise premium of 0.4234, is built and tested in
`python/equity/01-options-pricing` (`src/eq_options/binomial.py`); see
that project's `docs/VALIDATION.md` §4 for the actual convergence
numbers.

## 7. Numerical limits (documented and tested)

- **T→0 and sigma→0:** the closed form does *not* silently return NaN
  or a value at these boundaries — `_d1_d2` raises `ValueError` for
  `T<=0` or `sigma<=0` (tested:
  `test_edge_cases.py::test_zero_or_negative_T_raises`,
  `test_zero_or_negative_sigma_raises`). Approaching the boundary
  numerically (`T=1e-6` or `sigma=1e-6`) converges to the correct
  economic limit — discounted intrinsic value — to within 1e-4 or
  better (tested).
- **Deep ITM/OTM:** delta saturates toward 0 or 1 (calls) / 0 or −1
  (puts) as expected; deep OTM call/put prices are effectively zero
  (< 1e-8) at reasonable vol (tested).
- **Zero and negative rates:** fully supported — the formula has no
  positivity constraint on `r`, and parity, pricing and implied-vol
  round trips all hold exactly at `r=0` and `r=-1%` (tested:
  `test_zero_rate_is_supported`, `test_negative_rate_is_supported`,
  `test_negative_rate_implied_vol_round_trips`).
- **Invalid inputs:** `sigma<=0`, `T<=0`, `S<=0`, or `K<=0` all raise
  `ValueError` with a message stating which precondition failed and the
  offending values (tested, including a message-content check:
  `test_error_message_is_informative`). `implied_volatility` raises
  `ValueError` on prices outside the model-free no-arbitrage bounds
  rather than returning a meaningless vol (tested), and now also on
  `T<=0` (at expiry the price is intrinsic value and carries no
  volatility information at all).
- **Very long maturities (T = 10, 30, 50 years):** nothing overflows,
  put-call parity still holds to `1e-9`, and the Greeks stay finite and
  correctly signed. At `T=30, r=3%` the strike discounts to 40.66% of
  its face value, so a struck-at-spot call is worth more than 60% of the
  stock: most of a long-dated call's value is deferred payment, not
  optionality. Gamma falls monotonically with maturity — a 30-year option
  is nearly a static delta hedge. Monte Carlo still matches the closed
  form within 3 standard errors at `T=30`. All tested in
  `tests/test_extremes.py`.
- **Huge and tiny strikes (K from 1e-10 to 1e10):** `log(S/K)` spans
  ±23 and `d1` runs past ±100, where `math.erf` saturates cleanly at ±1.
  Prices stay finite and inside their no-arbitrage bounds; put-call
  parity holds to `rel=1e-12` (checked *relatively*, since at `K=1e10`
  both sides are ~1e10 and an absolute tolerance would be vacuous). The
  economic limits come out exactly right: `K→0` makes the call the stock
  itself and the put worthless; `K→∞` makes the put the discounted strike
  minus the stock and the call worthless.
- **Discount-factor overflow:** `exp(-r·T)` exceeds double precision for
  sufficiently negative `r·T` (e.g. `r=-10%` × `T=100y` is fine, but
  `r=-1000%` × `T=100y` is not). This used to surface as a bare
  `OverflowError: math range error` naming neither the input nor the
  reason; it is now a `ValueError` that names both
  (`test_discount_factor_overflow_raises_informative_value_error`).
- **Saturation at extreme total volatility:** at a total volatility
  `sigma·sqrt(T)` of 8, a put's remaining gap to its upper bound is of
  order 1e-100, which is below double-precision resolution next to a
  40-point price — so the computed put *equals* the discounted strike
  exactly. The bound is respected (`<=`, not `<`); code must not assume
  strict inequality there
  (`test_put_bound_is_tight_and_saturates_at_extreme_vol`).
- **Put-side no-arbitrage bounds** `max(K·e^{−rT} − S, 0) <= P <=
  K·e^{−rT}` are checked across a 108-contract grid spanning three orders
  of magnitude of moneyness, negative/zero/positive rates and maturities
  from 4 days to 30 years, at three volatility levels each
  (`test_put_bounds_hold_across_a_wide_grid`).
- **Monte Carlo input contract matches the closed form's.** The
  simulation would run happily at `sigma=0` or `T=0` (the terminal price
  is then deterministic), but the closed form refuses those inputs on
  purpose so the caller takes the intrinsic-value limit explicitly.
  Accepting them in one implementation and not the other would mean the
  two are no longer testing the same contract, so `mc_call_price` now
  validates identically (`test_mc_input_validation_mirrors_the_closed_form`).
