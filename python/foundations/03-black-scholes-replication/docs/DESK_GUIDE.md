# Desk Guide — Black-Scholes Replication

Documentation-contract items 5 and 6. Read this first sentence twice:
**this project is not a production pricing library, and it is not meant
to be one.** No trading desk should point live risk at
`eq_bs_replication`. If you need a pricing kernel for real books —
vectorised, multi-model (Black-Scholes, CRR binomial for American
exercise, Black-76 for forwards, Monte Carlo with variance reduction),
with C++/Rust performance twins — that is
`python/equity/01-options-pricing` (`eq_options`), the flagship pricing
project in this portfolio. This document is about what
`eq_bs_replication` actually *is* useful for on a real desk, which is a
narrower and different thing than pricing.

---

## 1. What this project is for: a model-validation reference

Every pricing model that goes live on a desk — vendor-purchased,
built in-house, or a new version of an existing one — has to pass some
form of independent validation before it is trusted with real risk.
That validation is not "does it look reasonable"; it is a specific,
repeatable set of checks against theory and against at least one
independently-built alternative. This project *is* that checklist, run
against the simplest possible model, built to be small enough to read
end to end in one sitting:

1. **Model-free identities hold exactly** (put-call parity to
   float64 precision — §1, `docs/VALIDATION.md`).
2. **An independently-built second implementation of the same model
   agrees** (Monte Carlo vs closed form, converging at the correct
   theoretical rate — §2).
3. **Analytic sensitivities match numerical ones** (Greeks vs central
   finite differences — §3).
4. **The model inverts correctly** (implied-vol round trips — §4).
5. **The model's own stated failure modes are demonstrated, not just
   asserted** (the volatility-smile experiment — §5).

The logic new-hire training and model-governance reviews both lean on
is: *if an implementation cannot reproduce Black-Scholes cleanly from
theory — the simplest, most textbook model in the entire field — it
cannot be trusted with anything harder.* This project is a worked,
runnable example of exactly that bar, deliberately kept small enough
that every check in it can be read, understood and reproduced by a new
quant in an afternoon.

## 2. Concrete real-life scenario: onboarding a new pricer

A desk is bringing a new options pricer into production — perhaps a
vendor library replacing an old one, perhaps an in-house rewrite for
performance, perhaps a new model going live for the first time. Before
it is allowed anywhere near real risk, model validation (often a
function or committee independent of the desk that built or bought the
model) runs it through a checklist that is, in substance, this
project's five sections:

1. **Reproduce known values.** Feed the new pricer the same textbook
   contract this project uses as its reference (`S=100, K=105, r=3%,
   sigma=25%, T=0.75y`) and check the call/put prices and Greeks match
   a trusted independent source to a tight tolerance. This project's
   `call_price`/`put_price`/`call_greeks`/`put_greeks`, validated
   line-by-line against theory rather than against any other pricing
   library, is exactly the kind of *independent* trusted source that
   validation teams want — it was not built by copying the code under
   review, and it does not share any bugs the new pricer might have.
2. **Check the identities.** Put-call parity must hold to numerical
   precision on the new pricer's own outputs, at every strike/expiry
   the book will trade. If it does not, stop — this is a model-free
   check, so a failure here means an implementation bug, not a modeling
   choice.
3. **Cross-validate against a second method.** Price the same contracts
   by Monte Carlo (either the new pricer's own MC mode, if it has one,
   or an independent one like the one in this project) and confirm
   agreement within statistical error, at more than one sample size, so
   the convergence *rate* — not just one lucky agreement — is checked.
4. **Check the Greeks, not just the price.** Bump each input and
   compare the pricer's analytic Greeks to a finite-difference estimate
   of the *same pricer's* price function. A pricer that gets prices
   right but Greeks wrong will misprice hedges and misreport risk long
   before it misprices a trade ticket — this is a common and dangerous
   failure mode because it is invisible in a simple price-comparison
   test.
5. **Check the inverse direction.** If the new pricer offers implied
   vol, round-trip price → vol → price across the strikes/expiries the
   desk actually trades, including deep ITM/OTM where vega is small and
   precision genuinely degrades (this project documents and tests that
   exact degradation — `docs/VALIDATION.md` §4 — rather than pretending
   it doesn't happen).
6. **Understand where it will be wrong, on purpose.** No model is
   right everywhere; validation isn't about proving the new pricer is
   perfect, it's about documenting precisely where its assumptions
   break so risk limits and P&L attribution can account for it. This
   project's volatility-smile experiment (`docs/VALIDATION.md` §5) is a
   template for that kind of demonstration: show, with actual numbers,
   what happens when the model's core assumption (constant vol) is
   violated, rather than leaving it as an unquantified caveat in a
   README.

Only after a new pricer clears all six does it get trusted with real
risk. This project is small enough to *be* the training example a new
quant runs through by hand before they are asked to build or review
that checklist for something bigger.

## 3. Teaching use: what a new quant hire should take from this project

- **Read the derivation, not just the formula.** The module docstring
  in `src/eq_bs_replication/black_scholes.py` walks from the replicating-
  portfolio argument to the closed form and states what `N(d1)` and
  `N(d2)` mean economically. A new hire should be able to explain both
  numbers in an interview, not just type `bs_price(...)`.
- **Understand why `math.erf` and not `scipy.stats.norm.cdf`.** This is
  not a style choice — see `docs/METHODOLOGY.md` §1.4. It is a forcing
  function for understanding that the normal CDF *is* the model's
  economic content (exercise probability, hedge ratio), not an
  incidental library call.
- **Internalise the four-part validation discipline** (§1 above) as the
  minimum bar for trusting *any* pricing code, not just this one. A new
  hire who can explain why put-call parity is model-free while a
  Greeks-vs-price self-consistency check is not (§1.2,
  `docs/METHODOLOGY.md`) understands something that generalises well
  beyond Black-Scholes.
- **See a model fail on purpose.** The volatility-smile experiment
  (`docs/VALIDATION.md` §5) is the single most useful exhibit in this
  project for a new hire: it is the first time many people see, with
  their own generated numbers rather than as a stated fact, *why* real
  options markets quote a smile instead of one flat vol.
- **Then go read `python/equity/01-options-pricing`.** Once the from-
  scratch version is understood, the production version's extra
  machinery (vectorisation, American exercise, Black-76, variance-
  reduced Monte Carlo Greeks, the discrete-hedging P&L simulator) reads
  as an extension of a model already understood, not as a black box.

## 4. What this project deliberately does not provide

- No American exercise, no dividends, no term structure of rates or
  vol, no vectorised chain pricing, no C++/Rust performance twin, no
  market-data connectivity. All of these exist in
  `python/equity/01-options-pricing` and should be used for anything
  resembling real desk risk.
- No FX variant (Garman-Kohlhagen). The single-asset equity scope here
  is deliberate — see the top of `docs/METHODOLOGY.md`.
- No governance/limits framework of its own. If this project's
  validation checklist is adapted for an actual onboarding review, the
  limits, sign-off process and change-control record are the reviewing
  desk's / model-risk function's responsibility, not something this
  code provides.
