# Methodology: Black-Scholes Replication

This document covers why Black-Scholes-Merton is the model replicated
here, against alternatives, and what it assumes, as a numbered register
with what breaks if each assumption fails.

Conventions: `r` is continuously compounded and annualised; `T` is time
to expiry in years (ACT/365F); `sigma` is annualised volatility of
log-returns; no dividends (q=0, see §1.3).

---

## 0. What this project is

This is a from-scratch, cross-validated replication, not a pricing
library. It shows by independent construction that a from-scratch
Black-Scholes reproduces the theory where theory makes exact
predictions, and shows with numbers where it fails to reproduce
markets. The vectorised, multi-model options engine with C++ and Rust
twins is `python/equity/01-options-pricing`. `docs/DESK_GUIDE.md`
describes the pricer-onboarding use.

## 1. Why Black-Scholes-Merton, and why replicate it this way

### 1.1 Why this model, against two alternatives

**Alternative 1: the Cox-Ross-Rubinstein (CRR) binomial tree.** A tree
backs out the price by risk-neutral expectation over `n` up/down steps
and converges to Black-Scholes at O(1/n). It needs no stochastic
calculus and extends naturally to American exercise. It is not the
primary model here because:

- It has no closed form to check against. A tree can only be validated
  by convergence to Black-Scholes, so building it first leaves no
  independent ground truth.
- It skips the normal CDF, which is the piece of mathematics this
  project is about: turning the replicating-portfolio argument into an
  integral of a normal density and evaluating it.
- CRR-to-BS convergence, with its odd/even oscillation, is already built
  and tested in `python/equity/01-options-pricing`.

**Alternative 2: local volatility (Dupire).** Local vol makes volatility
a deterministic function `sigma(S, t)` calibrated to reprice every
quoted vanilla. It is not chosen here because:

- It needs a vol surface as input, and the only check is the quotes it
  was fitted to. That is a weaker check than parity to machine
  precision.
- Its purpose is to fix the flat-vol assumption that this project
  demonstrates breaking (§2 A1 and the smile experiment). Fixing it is
  left to `python/equity/01-options-pricing` and the vol-surface
  projects.

**Why Black-Scholes-Merton.** It has (a) a closed form, so every
intermediate quantity has a known target; (b) a complete set of
textbook identities (parity, Greeks, monotonicity, limits) checkable
from theory alone; (c) an interpretable derivation (`N(d2)` is the
risk-neutral exercise probability, `N(d1)` weights the expected stock
receipt); and (d) a known failure mode, the smile, that the same tools
can demonstrate.

### 1.2 Why a Monte Carlo cross-check, not the closed form alone

Parity, monotonicity and Greeks-vs-finite-difference checks are
internal to the closed form. `call_price` and `put_price` share
`_d1_d2`, so a wrong sign in `d1`, a wrong `sigma^2/2` term or a wrong
`N(x)` would corrupt every internal check identically and pass them all.

The Monte Carlo pricer in `monte_carlo.py` shares no code with the
closed form. It draws standard normals, builds
`S_T = S0 exp((r - sigma^2/2)T + sigma sqrt(T) Z)` from the SDE's known
solution, and averages discounted payoffs. A sign or constant error in
the closed form would show up as a disagreement far larger than Monte
Carlo noise, across four sample sizes whose standard error shrinks at
the O(1/sqrt(n)) rate.

Two rules follow, and the code enforces both:

1. **The two pricers share one input contract.** The simulation would
   run at `sigma = 0` or `T = 0`, where the closed form refuses the
   input so the caller takes the intrinsic limit explicitly.
   `mc_call_price` validates identically, so the cross-check always
   compares the same contracts.
2. **The Monte Carlo error bar must be correct.** Antithetic payoffs are
   negatively correlated by construction. Treating the `2m` draws as
   independent overstates the error by about a third on an ATM
   contract, and every agreement test keeps passing against the looser
   bar. The standard error is computed from the `m` pair averages, and
   a test compares it with the estimator's dispersion across 60 seeds
   (`docs/VALIDATION.md` §2.1).

A tolerance expressed in units of your own error estimate is only as
good as that estimate.

### 1.3 Extending to dividends (not built)

A continuous dividend yield `q` replaces `S` with `S exp(-qT)` wherever
it stands for the current value of holding the stock:
`d1 = [ln(S/K) + (r - q + sigma^2/2)T] / (sigma sqrt(T))` and
`C = S e^{-qT} N(d1) - K e^{-rT} N(d2)`. It is left out to keep the
derivation minimal (A4). The dividend-aware version is in
`python/equity/01-options-pricing`.

### 1.4 The zero-scipy, `math.erfc` design

`scipy.stats.norm.cdf` would compute `N(x)` in one import. The CDF is
written out instead because it carries the model's economic content:
the exercise probability and the delta hedge ratio.

The implementation is `N(x) = 0.5 * erfc(-x / sqrt(2))`, using
`math.erfc` from the platform C library. `erfc` is a general special
function, not option-pricing machinery, so nothing model-specific is
hidden. The algebraically equal form `0.5 * (1 + erf(x / sqrt(2)))`
is not used: adding a tiny `erf` tail to 1 cancels, so its relative
error is 1.8% at x = -8 and it returns 0 below x = -8.3. A deep OTM
call (`S=100, K=300, T=0.25, sigma=0.2`) then prices at 0.0 instead of
3.5e-28. The `erfc` form keeps full relative accuracy in the tail
(`tests/test_tail_and_itm_iv.py`).

Everything downstream of `erfc` (`d1`/`d2`, discounting, Greeks,
implied vol) is built by hand. NumPy is used only in `monte_carlo.py`,
for batch path generation, and never in the closed-form path.

---

## 2. Assumptions register

### A1: The underlying follows geometric Brownian motion with constant volatility

`dS = mu S dt + sigma S dW`, so log-returns are i.i.d. normal with
constant variance rate `sigma^2`. This makes `S_T` lognormal and the
price an integral of the normal density.

**What breaks if violated.** Real volatility clusters (the stylised
fact GARCH models capture) and is itself stochastic (Heston, SABR). If
returns have fatter tails than the lognormal, a single flat `sigma`
underprices the wings relative to ATM. Reading prices back through
Black-Scholes then gives a smile or skew instead of a flat line. The
smile experiment (`docs/VALIDATION.md` §5) reproduces this mechanism on
synthetic Student-t prices.

### A2: Constant risk-free rate, continuous compounding

One constant `r`, discounting by `exp(-rT)`.

**What breaks if violated.** The rate curve is not flat, and desks fund
at a spread over it. A flat `r` misstates theta and rho as the curve
moves, more so for long-dated options. Stochastic rates correlated with
the underlying break the deterministic-discounting step of the
derivation. The code does not require `r >= 0`: negative rates work as
the formula predicts
(`tests/test_edge_cases.py::test_negative_rate_is_supported`).

### A3: Frictionless markets: continuous trading, no costs, unlimited shorting

Continuous, costless trading in any size makes a delta-hedged position
riskless at every instant, which forces the option price to equal the
cost of the hedge.

**What breaks if violated.** Hedging is discrete, so hedge P&L has
irreducible noise even at the correct vol. The noise shrinks with
rebalancing frequency, never to zero, and scales with gamma, so
near-expiry ATM options are hardest to hedge. Transaction costs turn
more frequent rebalancing into a trade-off against cost drag, so the
price becomes a band. Hard-to-borrow names break the put side of the
argument. The discrete-hedging simulator is in
`python/equity/01-options-pricing`, see `docs/VALIDATION.md` §6.3.

### A4: European exercise, no dividends

Exercise only at expiry; no dividends over the option's life.

**What breaks if violated.** American exercise is worth more whenever
early exercise can be optimal: for puts when the underlying is low
enough that interest on the strike outweighs time value; for calls only
with dividends (Merton: without dividends, early call exercise is never
optimal, so the European formula is correct for American calls). The
closed form cannot represent early exercise and understates an American
put by the exercise premium. A discrete dividend drops the stock price
on the ex-date, which a continuous yield only smears over the option's
life; this misprices short-dated options across an ex-date and distorts
the early-exercise boundary for American calls. Both are built in
`python/equity/01-options-pricing` (CRR tree), not here.

---

## 3. The validation this methodology implies

`docs/VALIDATION.md` follows from the choices above:

1. **Identities that must hold** (put-call parity). They are
   model-free, so they catch bugs whether or not Black-Scholes is the
   right model.
2. **Two independent implementations converging** (closed form vs Monte
   Carlo), because internal checks share the blind spots of the code
   they check.
3. **Analytic Greeks vs finite differences.** A Greek with a sign or
   constant error fails as soon as an input is perturbed.
4. **Implied-vol round trips.** They exercise the inverse direction
   with a different numerical method (Newton-Raphson with a bisection
   fallback, solved on the OTM side).
5. **Behaviour at the edges of the parameter space**: 30-year
   maturities, strikes from `1e-10` to `1e10`, prices saturating
   against a no-arbitrage bound, the smallest legal Monte Carlo sample,
   and quotes on a no-arbitrage bound where implied vol is not
   identified. At these edges a from-scratch pricer overflows, returns
   a silent `NaN`, or returns a number that meets its tolerance and
   means nothing. `docs/VALIDATION.md` §7.
