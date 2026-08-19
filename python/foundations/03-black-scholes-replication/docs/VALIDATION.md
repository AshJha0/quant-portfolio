# Validation — Black-Scholes Replication

This document answers documentation-contract items 3 and 4: **how the
implementation was validated** (analytic benchmarks, convergence
studies, cross-model consistency) and **where it fails** (known failure
modes, with reproducible numbers). Every number below is produced by
the committed code — `python examples/run_pipeline.py` regenerates the
report and figures under `output/`; `pytest -q` (44 tests, offline,
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
| 1,000 | 6.8840 | 0.3947 | 0.5835 | 12.48 |
| 10,000 | 7.4104 | 0.1337 | 0.0571 | 13.37 |
| 100,000 | 7.4314 | 0.0425 | 0.0361 | 13.45 |
| 1,000,000 | 7.4517 | 0.0135 | 0.0158 | 13.49 |

`SE·√n` is constant at **≈13.2** across three orders of magnitude of
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

Antithetic variates measurably reduce variance at equal path count
(`tests/test_monte_carlo.py::test_mc_antithetic_reduces_variance`,
200k paths, same seed): the antithetic standard error is strictly lower
than the plain (non-antithetic) one, at no extra RNG draws.

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
  rather than returning a meaningless vol (tested).
