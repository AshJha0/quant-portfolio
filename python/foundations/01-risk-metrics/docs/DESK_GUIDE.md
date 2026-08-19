# Desk Guide — How Single-Asset Risk Metrics Are Used Day to Day

Documentation-contract items 5 and 6: who consumes these numbers, how
they're reported, what governance looks like, and realistic scenarios a
reviewer should imagine around this code. The library is the *metrics
kernel* for a single position; everything below is the operating context.

**Scope note, up front:** this project computes **single-asset** risk
metrics only. Real trading books hold many correlated positions, and
portfolio-level risk (correlation, diversification, marginal/component
VaR) is a materially different — and materially harder — problem than
running these functions once per position and adding the numbers up (in
fact, doing exactly that overstates risk for anything less than perfect
correlation, and VaR is not even guaranteed sub-additive across positions
in the first place — see `docs/METHODOLOGY.md` §4). The multi-asset
extension in this portfolio is `python/equity/03-var-es-engine`; treat
this project as the reference building block it calls per name.

---

## 1. Who consumes these numbers

- **Market/risk desk (independent risk management, not the trading
  desk):** owns the daily VaR/ES report, sets and monitors limits,
  escalates breaches. Wants the *conservative* number — typically
  historical VaR/ES over Gaussian, because the fat-tail understatement in
  §3.2 of `docs/METHODOLOGY.md` is exactly the kind of error a risk
  function cannot afford to make silently.
- **Portfolio manager (PM):** consumes the same numbers but with a
  different incentive — sizing positions against a risk budget, and
  legitimately pushing back when a number looks wrong for their specific
  view (§3.3 below has a concrete example). Also the primary consumer of
  Sharpe/Sortino, which the risk desk cares about far less (those measure
  reward-for-risk, not risk itself).
- **Compliance / model governance:** does not look at daily numbers, but
  owns sign-off on the *methodology* (which VaR method is the official
  one, what confidence/horizon is used for regulatory or internal capital
  purposes, when the model was last validated) and the audit trail behind
  limit breaches.
- **Senior management / board risk committee:** sees an aggregated,
  much-simplified version (typically a single VaR number per business
  line plus a trend chart), not per-position detail.

## 2. How they're reported

**Daily risk report (EOD batch).** Re-run `examples/run_pipeline.py`-style
logic against the previous close, per position: full-sample/rolling/EWMA
vol, historical + Gaussian + Cornish-Fisher VaR at 95%/99%, ES at the same
levels, max drawdown (rolling, e.g. trailing-year), Sharpe/Sortino,
Jarque-Bera diagnostics. The **historical VaR/ES numbers are what goes
into the limit-monitoring system**; Gaussian and Cornish-Fisher are shown
alongside as a sanity check and a fat-tail-gap indicator, not as the
binding number, per the methodology discussion.

**Limit checks.** A position (or desk, or book) has a VaR limit, e.g.
"1-day 99% historical VaR must not exceed $2m". The EOD batch computes
today's VaR against yesterday's close-of-book position and flags a
breach if it exceeds the limit. Breaches route to the risk desk and
(depending on severity/persistence) up to the PM's manager and
compliance — see the backtesting/governance discussion below for how the
limit itself gets validated, not just monitored.

**Real-time / intraday.** Full historical VaR recomputation is normally
an EOD batch job (it needs the full return history); intraday, desks more
commonly track a simpler proxy (delta times a fixed vol-scaled move, or a
"VaR utilization" estimate that scales yesterday's VaR by today's
realised P&L volatility) rather than recomputing the full pipeline on
every tick. This project's functions are cheap enough to call intraday if
needed (no optimization, no external calibration step), but the *design*
of an intraday risk system is a different problem from what's implemented
here.

## 3. Governance

**VaR backtesting.** The single most important governance control on any
VaR model is checking whether the stated confidence is actually being
honoured: at 99% VaR, you expect a loss exceeding the VaR estimate on
roughly 1% of days. The standard tools are:

- **Kupiec's POF (proportion of failures) test** — a likelihood-ratio
  test on whether the observed exception rate matches the nominal rate.
- **Christoffersen's independence test** — checks that exceptions aren't
  clustered in time (clustered exceptions mean the model is missing a
  volatility regime shift, not just miscalibrated on average).

**Neither is implemented in this project** — it is single-shot metric
computation on a fixed historical window, not a rolling backtest engine.
This is the single most important "next step" flagged in
`docs/METHODOLOGY.md` and worth stating plainly: **an unvalidated VaR
number is a number, not a risk control.** A production desk would not
accept this library's output as a binding limit without wrapping it in a
backtest that runs continuously and re-certifies the model periodically
(commonly quarterly, or immediately after any material market regime
change).

**Model sign-off.** Real model governance frameworks (e.g. under
SR 11-7-style expectations, or FRTB internal-models approval) require:
independent validation of the methodology (an independent team, not the
model's author, reviews `docs/METHODOLOGY.md`-equivalent documentation
and the backtest results); a documented model inventory entry (what it
does, what its assumptions are — exactly the register in
`docs/METHODOLOGY.md` §7); periodic re-validation; and a change-control
process so that switching, say, the default VaR method from historical to
filtered-historical-simulation is a signed-off model change, not a
silent code update. This project's test suite (`pytest -q`) is the
regression gate that any such change would need to keep passing, but it
is not a substitute for the independent-review process itself.

## 4. Realistic scenarios

### 4.1 A position breaches its VaR limit

A single-name position's 1-day 99% historical VaR jumps from $1.6m to
$2.3m overnight against a $2m limit — no new trade, just a repriced VaR
off yesterday's close. First question the risk desk asks: **is this a
position change or a vol change?** Checking `rolling_volatility`/
`ewma_volatility` against the full-sample figure (exactly the divergence
check in `docs/VALIDATION.md` §3.4) usually answers it immediately — if
EWMA vol has jumped materially, the breach is a genuine regime move and
the position needs to come down (or the limit needs a documented,
approved temporary increase); if vol is flat, the breach is a data or
position-sizing bug and gets investigated as one, not risk-managed as one.

### 4.2 A volatility regime shift makes yesterday's number stale

This is `docs/VALIDATION.md` §3.4 made operational: on the bundled
synthetic data, EWMA vol hit 33.4% against a 13.9% full-sample figure on
2022-03-03 — a 2.4x divergence on a single name. A risk report that only
shows full-sample vol and full-sample-parametrised VaR would have been
reporting a calm-market risk number *through* a regime that had already
turned stressed. Desk practice: EOD batch should flag any position where
`abs(ewma_vol - full_sample_vol) / full_sample_vol` exceeds a threshold
(a common convention is 50%) for manual review, and risk limits set off a
stale full-sample vol should be treated as provisional until re-based on
current vol.

### 4.3 A PM legitimately arguing with a Gaussian VaR number

A PM holding a position through an announced, dated catalyst (an earnings
release, an FDA decision, an index-rebalance date) will correctly point
out that Gaussian VaR — built on the *unconditional* mean/std of the
whole trailing sample — has no way to know that tomorrow's return
distribution is not this sample's typical day. This is a real and valid
objection, not just noise: none of the three VaR methods implemented here
model *known, dated, discrete* event risk; they all describe "a
day drawn from this history's overall or (for the EWMA-informed
comparison) recently-representative distribution," and a scheduled binary
event genuinely isn't that. The correct desk response is not to argue the
PM out of it, but to supplement — not replace — the historical/Gaussian/
Cornish-Fisher numbers with an explicit event-scenario stress test (e.g.
"P&L if the stock moves like it did on the last 4 earnings dates") for
the specific dated risk, while still using the standard VaR/ES pipeline
for the non-event tail risk that's genuinely present every other day.

### 4.4 Short-sample new listing / newly onboarded position

A newly listed name (IPO, spin-off) or a newly onboarded position with
only a few weeks of history is exactly the `docs/VALIDATION.md` §3.1
failure mode: every function still returns a number, but a 20-day
historical VaR is a handful of order statistics, not a risk estimate a
limit should be set against. Desk practice: either borrow volatility from
a comparable/peer name until enough own-history accumulates, or apply a
conservative multiplier/floor to the historical VaR of a short sample
rather than trusting it at face value. This project deliberately does
*not* build that guardrail in (see the methodology decision to let
functions run on whatever's passed to them) — enforcing a minimum sample
size before a number is allowed to reach a limit-monitoring system is a
desk-level control on top of this library, not a feature of it.

### 4.5 Sharpe ratio going degenerate on a low-vol position

A position with near-zero realised volatility (a pegged instrument, a
position that hasn't traded, a data feed repeating a stale close) can
push Sharpe to a huge, meaningless number rather than a clean error — see
`docs/VALIDATION.md` §3.2. A PM reporting a "Sharpe of 68 quadrillion" on
a performance summary is a data-quality bug, not investment skill; a
report generator consuming this library's output should sanity-bound
Sharpe/Sortino display (e.g. flag anything with `|value| > 100`) rather
than print it verbatim.

## 5. Out of scope — see the multi-asset extension

Correlation, diversification benefit, marginal and component VaR,
copula-based joint tail dependence, and portfolio-level backtesting are
all genuinely different problems from anything in this project — adding
single-asset VaRs together is not portfolio VaR, and getting the
correlation structure right (especially in the tails, where correlations
famously spike during a crash — "correlations go to 1") is the actually
hard part of portfolio risk. See `python/equity/03-var-es-engine` in this
portfolio for the multi-asset engine that builds on the single-asset
methodology documented here.
