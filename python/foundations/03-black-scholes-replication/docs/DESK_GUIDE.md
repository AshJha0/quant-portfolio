# Desk Guide: Black-Scholes Replication

This project is not a pricing library. No desk should point live risk
at `eq_bs_replication`. The vectorised, multi-model pricing engine
(Black-Scholes, CRR for American exercise, Black-76, Monte Carlo with
variance reduction, C++ and Rust twins) is
`python/equity/01-options-pricing` (`eq_options`). This guide covers
what `eq_bs_replication` is useful for, which is model validation and
teaching.

---

## 1. What this project is for: a model-validation reference

Every pricing model that goes live, whether bought, built in-house or a
new version of an existing one, must pass independent validation. That
validation is a repeatable set of checks against theory and against at
least one independently built alternative. This project runs that
checklist on the simplest model, small enough to read in one sitting:

1. Model-free identities hold (put-call parity to float64 precision,
   `docs/VALIDATION.md` §1).
2. An independently built second implementation agrees (Monte Carlo vs
   closed form, converging at the theoretical rate, §2).
3. Analytic sensitivities match numerical ones (Greeks vs central
   finite differences, §3).
4. The model inverts correctly (implied-vol round trips, §4).
5. The model's stated failure modes are demonstrated (the smile
   experiment, §5).
6. The edges of the parameter space are probed: 30-year maturities,
   strikes over 20 orders of magnitude, overflowing discount factors,
   quotes on a no-arbitrage bound, the smallest legal Monte Carlo
   sample (§7). A pricer that is right in the middle of its domain and
   returns a silent `NaN` at the edge passes checks 1 to 5 and still
   puts a wrong number on a risk report.

If an implementation cannot reproduce Black-Scholes from theory, it
cannot be trusted with a harder model.

## 2. Scenario: onboarding a new pricer

A desk brings in a new options pricer: a vendor library, an in-house
rewrite or a new model. Before it touches risk, model validation runs
it through a checklist that matches this project's sections:

1. **Reproduce known values.** Price the reference contract (`S=100,
   K=105, r=3%, sigma=25%, T=0.75y`) and compare prices and Greeks with
   an independent source. This project's functions were checked against
   theory, not copied from the code under review, so they do not share
   its bugs.
2. **Check the identities.** Put-call parity must hold to numerical
   precision at every strike and expiry the book trades. Parity is
   model-free, so a failure is an implementation bug.
3. **Cross-validate against a second method.** Price by Monte Carlo and
   confirm agreement within statistical error at several sample sizes,
   so the convergence rate is checked, not one lucky agreement.
4. **Check the Greeks.** Compare analytic Greeks with finite
   differences of the same pricer. Wrong Greeks with right prices
   misstate hedges and risk while passing a price comparison.
5. **Check the inverse.** Round-trip price to vol to price across the
   traded strikes, including deep ITM and OTM where vega is small
   (`docs/VALIDATION.md` §4).
6. **Document where it will be wrong.** Show with numbers what happens
   when the core assumption fails, as the smile experiment does
   (`docs/VALIDATION.md` §5).

## 3. Teaching use: what a new quant should take from this project

- **Read the derivation.** The module docstring in
  `src/eq_bs_replication/black_scholes.py` goes from the replicating
  portfolio to the closed form and states what `N(d1)` and `N(d2)` mean.
  A new hire should be able to explain both numbers, not only call the
  function.
- **Understand the normal CDF choice.** The CDF is written with
  `math.erfc` instead of imported from scipy (`docs/METHODOLOGY.md`
  §1.4). The `erfc` form keeps relative accuracy in the left tail,
  where the `erf` form cancels to zero below x = -8.3.
- **Treat the six checks in §1 as the minimum bar** for any pricing
  code. Know why put-call parity is model-free while a Greeks-vs-price
  check is not (`docs/METHODOLOGY.md` §1.2).
- **Check your own error bars.** The Monte Carlo standard error was once
  computed by treating antithetic pairs as independent draws, which
  overstated it by about a third (`docs/VALIDATION.md` §2.1). Every
  "within 3 standard errors" test still passed. A tolerance expressed in
  units of your own error estimate cannot detect an error in that
  estimate, so compare it at least once with the empirical dispersion.
- **Know where a number stops meaning anything.** A call quoted at its
  discounted intrinsic value has zero time value, so no vol is
  identified. The solver warns and returns `nan` instead of a sigma
  that matches the price only to tolerance (`docs/VALIDATION.md` §4.1).
  A surface fit should drop strikes whose vega is below a floor.
- **See a model fail.** The smile experiment (`docs/VALIDATION.md` §5)
  shows, with generated synthetic numbers, why option markets quote a
  smile.
- **Then read `python/equity/01-options-pricing`.** Its vectorisation,
  American exercise, Black-76, variance-reduced Monte Carlo Greeks and
  discrete-hedging simulator then read as extensions of a known model.

## 4. What this project does not provide

- No American exercise, dividends, rate or vol term structure,
  vectorised chain pricing, C++/Rust twin or market data. These exist in
  `python/equity/01-options-pricing`.
- No FX variant (Garman-Kohlhagen). The scope is a single equity asset.
- No control variate in the Monte Carlo pricer: it uses antithetic
  variates only.
- No governance or limits framework. If the checklist is used in a real
  onboarding review, limits, sign-off and change control belong to the
  model-risk function.
