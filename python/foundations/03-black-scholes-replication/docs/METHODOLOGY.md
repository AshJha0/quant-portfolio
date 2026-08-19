# Methodology — Black-Scholes Replication

This document answers documentation-contract items 1 and 2: **why
Black-Scholes-Merton is the model replicated here** (against
alternatives, with trade-offs) and **what assumptions it makes** (a
numbered register, each with "what breaks if violated").

Conventions used throughout: `r` is continuously compounded and
annualised; `T` is time to expiry in years (ACT/365F); `sigma` is
annualised volatility of log-returns; no dividends (q=0 — see §1.3 for
the extension).

---

## 0. What this project is (and is not)

This is a **from-scratch, cross-validated replication exercise**, not a
pricing library. The deliverable is not "a fast option pricer" — it is
"proof, by independent construction and independent validation, that a
from-scratch implementation of Black-Scholes reproduces the theory
exactly where theory makes exact predictions, and reproduces reality
nowhere near as well." A production-grade, vectorised, multi-model
options engine (with C++/Rust performance twins) already exists in this
portfolio at `python/equity/01-options-pricing`; that project is where a
desk would actually get its pricing kernel. This project exists because
the discipline of independently re-deriving and cross-checking a known
model — before trusting a bigger one — is itself a skill worth
demonstrating in isolation, and is exactly what §"real-life scenario" in
`docs/DESK_GUIDE.md` describes happening on a real desk during
pricer-onboarding.

## 1. Why Black-Scholes-Merton, and why replicate it this way

### 1.1 Why this model, vs. at least two alternatives

**Alternative 1: the Cox-Ross-Rubinstein (CRR) binomial tree.**
A tree discretises the underlying's evolution into up/down steps over
`n` periods and backs out the price by risk-neutral discounted
expectation, converging to the Black-Scholes price as `n -> infinity`
at rate O(1/n). It is arguably the more *pedagogically primitive*
starting point (it needs only a discrete-probability argument, no
stochastic calculus), and it naturally extends to American exercise,
which the closed form cannot represent at all. It was **not** chosen as
the primary model to replicate here because:

- It has no closed form to check against — a tree can only be validated
  by convergence to *something else* (i.e. to Black-Scholes itself), so
  building a tree first would leave the project with no independent
  ground truth until Black-Scholes existed anyway.
- It doesn't exercise the piece of mathematics this project is actually
  about: turning a stochastic-calculus argument (Ito's lemma applied to
  a replicating portfolio) into a closed-form integral of a normal
  density, then implementing that integral (`math.erf`) from first
  principles. A tree sidesteps the normal CDF entirely.
- The tree/BS convergence relationship is already built, tested and
  documented in `python/equity/01-options-pricing` (CRR converging to
  BS at O(1/n), with the odd/even oscillation characteristic of the
  method) — duplicating it here would not add new validation insight,
  only new code.

**Alternative 2: a local-volatility model (Dupire).**
Local vol fits an entire implied-vol surface by allowing volatility to
be a deterministic function of spot and time, `sigma(S, t)`, calibrated
so the model exactly reprices every quoted vanilla. It is the natural
model to reach for once you *have* a real, skewed vol surface and want
self-consistent pricing across strikes. It was **not** chosen here
because:

- It requires calibration data (a real or synthetic vol surface) as an
  *input* — there is nothing to validate it against except the market
  quotes it was fit to, which makes "did I implement this correctly?"
  much harder to answer from first principles than "does put-call
  parity hold to machine precision?".
- Its entire raison d'être is to fix the flat-vol assumption that this
  project is precisely trying to demonstrate breaking (§3 below, the
  volatility-smile experiment). Building local vol here would blur the
  pedagogical point: this project shows *that* and *why* constant-vol
  GBM fails; fixing the failure is deliberately left to
  `python/equity/01-options-pricing`, whose local/stochastic-vol
  successors are the appropriate place for it.

**Why Black-Scholes-Merton wins for this specific project.** It is the
model with (a) a closed-form solution, so every intermediate quantity
has a known-correct target; (b) a small, complete set of textbook
identities (put-call parity, Greeks, monotonicity, limiting behaviour)
that are checkable from theory alone, with no reference implementation
needed; (c) an economically interpretable derivation (`N(d2)` is a
risk-neutral exercise probability, `N(d1)` weights the expected stock
receipt) that rewards being built by hand rather than imported; and (d)
a well-known, well-documented failure mode (the volatility smile) that
is itself demonstrable with the same tools used to validate the model,
making the project self-contained: build it, prove it right where
theory says it should be right, and prove it wrong where reality says
it should be wrong, with the same codebase.

### 1.2 Why closed-form-with-Monte-Carlo-cross-check, not closed-form alone

A closed-form implementation that is *wrong* can still look
self-consistent: put-call parity, monotonicity and the Greeks-vs-finite-
differences checks in this project are all checks *internal* to the
closed-form code (`call_price`/`put_price` share `_d1_d2`, so a bug in
`_d1_d2` would corrupt every internal check identically and still pass
all of them). What those checks cannot catch is a shared, systematic
error baked into the shared machinery — a wrong sign in `d1`, a
misremembered `+sigma^2/2` vs `-sigma^2/2`, or a subtly wrong `N(x)`.

The Monte Carlo pricer in `monte_carlo.py` shares **no code** with the
closed form: it draws standard normal variates, builds the terminal GBM
price `S_T = S0 exp((r - sigma^2/2)T + sigma sqrt(T) Z)` directly from
the SDE's known solution, and averages discounted payoffs. If the
closed form has a sign or constant error, the two will disagree by far
more than Monte Carlo noise — and they would have to be wrong in
*exactly the same way* to agree by coincidence, which is astronomically
unlikely across the convergence table in `docs/VALIDATION.md` (four
sample sizes, each within 3 standard errors, with the standard error
itself shrinking at the correct O(1/sqrt(n)) rate). That is why this
project ships both: the closed form for exactness and speed, the Monte
Carlo pricer as an *independent witness* that the closed form is
actually correct and not just internally consistent.

### 1.3 Extending to dividends (not built, but a documented one-liner)

A continuous dividend yield `q` requires only replacing `S` with
`S * exp(-q*T)` everywhere it appears as "current value of holding the
stock" — i.e. `d1 = [ln(S/K) + (r - q + sigma^2/2)T] / (sigma sqrt(T))`
and `C = S e^{-qT} N(d1) - K e^{-rT} N(d2)`. This project deliberately
leaves `q` out (see assumption A4 below) to keep the from-scratch
derivation minimal; the dividend-aware version is already built and
tested in `python/equity/01-options-pricing`.

### 1.4 The zero-scipy, `math.erf`-only design as a methodology decision

`scipy.stats.norm.cdf` would compute exactly the quantity this project
needs (`N(x)`) in one import. Not using it is deliberate: the point of
"replicating a known model" is to understand every link in the chain
from the SDE to the price, and the standard normal CDF is not a
peripheral detail of that chain — it *is* the model's exercise
probability and the delta hedge ratio. Depending on `scipy.stats.norm`
would let that piece stay a black box.

`math.erf` is the right place to stop, not `0`: it is a single
well-understood special function (`N(x) = 0.5*(1 + erf(x/sqrt(2)))`)
that is part of the Python standard library, backed by a numerically
stable, well-tested implementation, and it is *general-purpose* Fmath,
not option-pricing-specific machinery — using it does not hide any
model-specific logic the way `scipy.stats.norm.cdf` or a hypothetical
`black_scholes_call()` library function would. Everything downstream of
`erf` (assembling `d1`/`d2`, discounting, differentiating for Greeks,
inverting for implied vol) is built by hand in this project.

This has a real, stated cost: `monte_carlo.py` is the one place NumPy
is used (batch-generating and reducing arrays of a million paths
without it would be slow and unreadable), so the project is not
literally dependency-free — it is dependency-*minimal*, with the one
dependency it keeps justified in the module docstring of
`monte_carlo.py` and *not* used anywhere in the closed-form pricing
path.

---

## 2. Assumptions register

Each assumption states **what breaks if violated**, expanding the list
already identified during initial development.

### A1 — The underlying follows geometric Brownian motion with constant volatility

`dS = mu S dt + sigma S dW`, so log-returns are i.i.d. normal with
constant variance rate `sigma^2`. This is what makes the terminal
distribution of `S_T` lognormal in closed form and lets the entire
derivation collapse to an integral of the normal density.

**What breaks if violated.** Real volatility is neither constant
(it clusters — large moves are followed by large moves, a stylised fact
GARCH models exist specifically to capture) nor deterministic (it is
itself a stochastic process, motivating Heston, SABR, and other
stochastic-vol models). If you price a chain of options assuming one
flat `sigma` but the true return distribution has fatter tails than the
lognormal, every non-ATM option is mispriced *in the same direction*: a
Black-Scholes price with the "wrong" constant vol systematically
underprices deep OTM/ITM options relative to ATM ones. Reading market
prices back through Black-Scholes to recover an "implied" vol per
strike then reveals a **smile or skew** rather than a flat line — this
project reproduces that exact mechanism as a controlled experiment
(`docs/VALIDATION.md` §4): price a strike ladder with a Student-t
(fat-tailed) return distribution by Monte Carlo, then invert through
the (constant-vol) closed form. The measured implied vol comes out
higher at the wings than at the money (see the actual numbers in
`docs/VALIDATION.md`), which is qualitatively exactly what listed
equity- and index-option markets have shown persistently since the 1987
crash.

### A2 — Constant risk-free rate, continuous compounding

`r` is a single constant over the option's life, and discounting is
`exp(-rT)`.

**What breaks if violated.** A real term structure of rates is not
flat: financing a 1-week option and a 5-year option happens at
different rates, and both differ from the rate a desk actually funds at
(which includes a spread over the risk-free curve). Using one flat `r`
for a whole book misprices theta and rho relative to what actually
happens as the rate curve moves, and gets materially worse the longer
dated the option. A more serious violation is *stochastic* rates
correlated with the underlying — for long-dated equity options this
breaks the deterministic-discounting factorisation the derivation
relies on (the discount factor can no longer be pulled outside the
expectation as a constant). Note that this implementation does *not*
require `r >= 0` — the formula, the code (`_d1_d2` has no sign
constraint on `r`), and the tests (`tests/test_edge_cases.py::
test_negative_rate_is_supported`) all confirm negative rates work
exactly as the formula predicts, which matters in the low/negative-rate
regimes several major currencies have experienced.

### A3 — Frictionless markets: continuous trading, no transaction costs, unlimited shorting

This is the assumption that makes the whole no-arbitrage argument work:
if you can trade continuously, at zero cost, in unlimited size in
either direction, then a delta-hedged option position has *exactly*
zero risk at every instant, which is what forces the option's price to
equal the cost of running that hedge (the replication argument).

**What breaks if violated.** None of this holds in practice. Trading is
discrete (you rebalance at some finite frequency, not continuously),
which leaves irreducible hedging P&L noise even when your volatility
forecast is exactly right — the noise shrinks as you rebalance more
often but never to zero at any finite frequency, and it scales with
gamma, which is why hedging near-expiry ATM options (where gamma is
largest) is qualitatively harder than hedging further from expiry or
away from the money. Transaction costs turn "rebalance infinitely
often" from a free way to eliminate noise into a costly trade-off:
rebalancing more often reduces variance but increases cost drag, so the
"frictionless, unique arbitrage-free price" becomes, in practice, a
*band* rather than a point. Unlimited shorting fails for hard-to-borrow
names, which breaks the put side of the replication argument
specifically. This project does not attempt to quantify discrete-
hedging P&L (that simulator lives in `python/equity/01-options-pricing`);
it is named here, and in `docs/VALIDATION.md` §6, as a known,
documented failure mode.

### A4 — European exercise, no dividends

The option can only be exercised at expiry, and the underlying pays no
dividends over the option's life.

**What breaks if violated.** American exercise (the right to exercise
early) is worth strictly more than European exercise whenever early
exercise can ever be optimal — for puts, whenever the underlying is
low enough that immediate exercise's interest-on-strike benefit
outweighs the remaining time value; for calls, only in the presence of
dividends (with no dividends, Merton's theorem says early exercise of a
call is never optimal, so the European formula is exact for calls
regardless). The closed form here has no mechanism to represent early
exercise at all — it isn't an approximation that's *a bit* wrong for
American options, it is structurally the wrong model, and pricing an
American put with this formula understates its value by the exercise
premium. A discrete dividend (as opposed to the continuous yield `q`
mentioned in §1.3) causes a discontinuous jump down in the stock price
on the ex-date that a continuous-yield adjustment only smears out
smoothly across the option's life — mispricing short-dated options that
straddle an ex-date and distorting the early-exercise boundary for
American calls (real early exercise clusters immediately before
ex-dates; continuous `q` cannot represent that clustering at all).
Neither American exercise nor discrete dividends are implemented in
this project — both live in `python/equity/01-options-pricing`
(CRR tree for American exercise) — but the assumption and its
consequence are stated here in full because contract item 2 requires
it even for assumptions this project does not relax.

---

## 3. Summary: the validation discipline this methodology implies

Because the model is deliberately minimal and the implementation is
deliberately from-scratch, the validation in `docs/VALIDATION.md`
follows directly from this methodology:

1. **Identities that must hold exactly** (put-call parity) — because
   they are model-*free* arbitrage relations, they catch bugs
   regardless of whether Black-Scholes itself is the right model.
2. **Two independent implementations converging** (closed form vs Monte
   Carlo) — because internal self-consistency checks share the same
   blind spots as the code they check.
3. **Analytic Greeks vs finite differences** — because a Greek formula
   with a sign or constant error would still price correctly at a
   single point but fail as soon as you perturb an input, which finite
   differences catch directly.
4. **Implied-vol round trips** — because they exercise the *inverse*
   direction of the model (price -> vol) using a different numerical
   method (Newton-Raphson with a bisection fallback) than the forward
   direction, and are the exact operation a trading desk performs on
   every incoming quote.

Each of these is why this project exists in the shape it does: not to
build a pricer, but to demonstrate — with actual numbers, reproducibly
— that a from-scratch pricer can be trusted, and precisely where and
why it stops being trustworthy.
