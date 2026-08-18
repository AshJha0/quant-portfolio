# Desk Guide

How a global FX risk function would run this engine day to day: the
follow-the-sun batch, per-currency limits, peg watchlists, backtest
governance, and the regression gate against the research stack. Written
for the risk manager consuming the numbers, not the developer.

---

## 1. Follow-the-sun operating model

FX is the one asset class with no close. The engine is designed to be run
as three regional sweeps plus continuous intraday recompute:

| Window (local close) | Run | Consumers |
| --- | --- | --- |
| Tokyo ~15:00 JST | Full sweep of APAC books: HS (plain + FHS), parametric, MC-normal 100k, stress library | APAC risk officer; global book handover to London |
| London ~17:00 GMT | Global consolidated sweep; EM peg watchlist refresh; 10-day scaled figures for capital | EMEA risk; end-of-day limit reports |
| New York ~17:00 EST | Official global EOD: VaR/ES for the regulatory backtest series, Basel traffic-light update | Head of market risk; regulatory reporting |

Because the engine is bitwise deterministic (seeded MC, library-owned RNG
transforms), Tokyo, London and New York produce *identical* numbers for
identical books — no "same book, different VaR" reconciliation meetings.
A regional sweep of a few hundred sub-books costs seconds of CPU
(≈ 3 ms per 250-position book for HS + parametric + stress; ≈ 0.8 s if the
100k MC is included), so the constraint is data readiness, not compute.

**Intraday**: the compiled book revalues in microseconds per scenario;
re-run `historical_var` + the stress library on every material fill or
every few minutes. The MC tier is re-run when exposures move more than a
tolerance or on demand ahead of a large ticket.

**Missing fixings**: the engine *refuses* NaNs rather than imputing.
Holiday calendars differ by region (Tokyo holiday ≠ London holiday); the
upstream data layer must forward-fill or drop the date explicitly, and the
decision is auditable because the engine will not do it silently.

## 2. The daily numbers and who reads them

- **VaR 99% 1d (HS-FHS)** — the desk headline and limit measure.
- **ES 97.5% 1d** — the FRTB-aligned measure; watched for divergence from
  VaR (divergence = tail mass the quantile does not see; typical on books
  with EM shorts or peg exposure).
- **Parametric exposure vector** (`ParametricResult::exposures`) — the
  per-factor P&L mapping in base ccy per unit factor move: this is the
  attribution the trader acts on ("your VaR is 70% FX:JPY").
- **Method disagreement** — HS vs parametric vs MC on the same book. The
  bench shows MC≈parametric on a linear normal book; a *widening* gap in
  production means fat tails, non-linearity or a broken history. It is a
  monitored diagnostic, not noise.
- **Stress ladder** (`run_stress`, worst-first) — the canned replays
  (Brexit-GBP, CHF depeg, JPY carry unwind), the broad-USD ±10% moves and
  the desk's peg-break add-ons. Stress consumes no history and is the
  complement to VaR, not a substitute.

## 3. Per-currency limits

The exposure vector prices a 1.00 log-return move per factor; divide by
100 for the 1% notional-equivalent exposure the limit framework uses.
A typical structure the engine feeds directly:

- **Per-ccy delta limits**: |exposure(FX:CCY)| ≤ limit_ccy, with
  tighter limits for flagged peg currencies (see §4) and EM.
- **Book VaR limit** with a soft-warning band (e.g. 85%): VaR from the
  official method (FHS), checked at every sweep.
- **Stress limits**: worst canned-scenario loss ≤ stress limit — this is
  the limit that actually binds on peg books, by design.
- **IR leg limits** on forward books: exposure(IR:CCY)/1e4 is the DV01 the
  rates desk recognises; forward-points risk shows up here (CIP legs), not
  in a separate "forward vol" bucket.

## 4. Peg watchlist workflow

The engine flags every `FX:*` factor whose realised daily vol is below
5e-4 (~0.8% annualised — HKD-in-band territory) in
`flagged_peg_factors` / `warnings` on every HS/parametric run. Desk
process, in order:

1. **Flag appears** → the currency goes on the peg watchlist. Its HS and
   parametric VaR contributions are treated as *unreliable-low*, not low.
2. **Mandatory stress add-on**: attach `peg_break_scenario(ccy, jump,
   contagion)` to the book's stress set — direction from the peg's
   economics (undervalued anchor → CHF-2015-style positive jump on the
   short side; classic EM peg → devaluation), size from the
   convertibility/NDF market if one exists, contagion for same-anchor
   neighbours.
3. **MC overlay for limits**: run the jump-mixture MC
   (`McDist::kJump`, e.g. prob 1–2%/day, jump = scenario size) — its VaR/ES
   is the number checked against the peg-adjusted limit. The validation
   suite demonstrates the point: on a pegged book the jump MC reports
   > 20× the HS VaR that the window alone produces.
4. **Exit**: a currency leaves the watchlist only by governance decision
   (regime change), not because the flag stopped firing for a week.

## 5. Backtesting and model governance

- **Daily**: append realised (hypothetical, static-book) P&L and the
  ex-ante VaR to the 250-day rolling series; `evaluate_var_backtest`
  yields the exception series, Kupiec, Christoffersen, conditional
  coverage and the Basel traffic light with the capital multiplier.
- **Reading the tests**: Kupiec failing high = too many exceptions
  (model too tight); Christoffersen failing with Kupiec passing =
  clustered exceptions — the classic signature of an unconditional method
  in a volatility-clustered market; the escalation is to the FHS variant,
  not to a bigger multiplier.
- **Traffic light**: yellow (5–9 exceptions at 250d/99%) triggers a model
  review with the multiplier add-on applied; red (10+) is an automatic
  escalation to model risk and a capital event. The engine computes the
  zones from the exact binomial, so non-standard windows (shortened
  history after a book migration) still get correct zones.
- **ES oversight**: the ES series is monitored alongside VaR; persistent
  realised-tail-mean above forecast ES on peg books is the peg-blindness
  signature even when the VaR backtest is green.

## 6. Reverse stress in the risk committee

`reverse_stress_for_loss(exposures, cov, loss_target)` answers the
committee's standing question — *"what is the most plausible market move
that costs us X?"* — as the worst direction at fixed Mahalanobis
plausibility. The output shock vector is readable factor by factor
("EUR −2.1%, JPY +1.4%, USD rates +8bp") and feeds a narrative scenario.
Caveat: it is the *linear* worst case on the *estimated* covariance; for
peg books the honest answer is the peg-break scenario, because the
covariance cannot represent the break (assumption A8).

## 7. Regression gates vs the research stack

The Python package `python/fx/03-var-es-engine` is the methodology source;
this engine is the production twin. The contract:

- **Golden gate (CI)**: `tests/test_golden_python.cpp` pins historical,
  parametric and backtest outputs to Python-generated constants
  (tolerances 1e-6 abs / 1e-8 rel). Any semantic change on either side
  breaks the C++ build — by design. A methodology change is rolled out by
  regenerating the goldens *in the same change* that implements it, with
  both diffs reviewed together.
- **Method-parity checks**: identities that must hold in both stacks
  (triangulation, CIP legs, ES ≥ VaR, Basel boundaries) are tested in
  both suites, so a divergence localises to whichever suite goes red.
- **Scope boundary**: options (Garman–Kohlhagen) are research-stack only;
  any book routed to the C++ engine must be linear-only, and the loader
  should enforce that upstream. The C++ engine will throw on factors it
  does not know rather than ignore risk it cannot see — except scenario
  shocks, which are filtered by convention (a global scenario library
  must apply to every book).
- **Determinism as an audit feature**: seed, code version and book
  snapshot fully determine every MC number; "reproduce yesterday's VaR"
  is a rerun, not an archaeology project.
