# Desk Guide — Running a Stat-Arb Pairs Pod

How this library maps onto a real statistical-arbitrage pod: daily workflow,
who consumes which number, limits, and the scenarios that actually kill
pairs books (contract items 5 and 6).

---

## 1. Research → paper → size-up

**Research (weeks).** Universe = liquid names with tagged sectors/industries
(here: the panel's sector map; in production GICS + own factor buckets).
Run the funnel exactly as `run_pipeline` §2: same-sector candidates →
return-correlation or SSD screen → Engle-Granger at 5% with EG critical
values → OU fit gates (mean-reverting, half-life in [1, 126] days — a
half-life of hours is microstructure, a half-life of a year is a position,
not a trade). Deliverable per pair: β, α, κ, μ, σ, half-life, ADF stat, and
the formation-window backtest *with full costs*.

**Paper (1–3 months).** Trade the signal at target size on paper against
live closes. What you are actually testing: does live slippage match the
2bp assumption, does the borrow desk actually have the shorts, do fills at
the close exist in the size you want. Paper P&L within ~1 SE of backtest
expectation → promote.

**Size-up (quarters).** Start at 1/4 target gross per pair. Scale only on
realised cost per trade ≤ modelled cost and hit rate consistent with the
walk-forward number (67%, not the in-sample 96% — see VALIDATION §7 for why
the smaller number is the honest one). Every parameter refresh is a
walk-forward roll: re-fit on the trailing formation window, freeze, trade.

**Who consumes what:**

| Number | Consumer | Cadence |
|---|---|---|
| Per-pair z, half-life, days-in-trade vs time-stop | Trader | Intraday/EOD |
| Per-pair attribution (`PortfolioResult.attribution()`) | PM | Daily |
| Sharpe ± Lo SE, drawdown vs limits, turnover, cost drag | Risk | Daily/weekly |
| EG stat re-test on trailing window (cointegration health) | PM + risk | Weekly |
| Borrow fee & utilisation per short leg | Borrow desk ↔ trader | Daily |
| Funnel counts (candidates → survivors) drift | Research | Monthly |

## 2. Risk limits (numbers scaled to the demo book: $1mm gross/pair, 6 pairs)

- **Per pair:** gross ≤ $1mm; |z| stop at 4 (hard, no re-entry until |z|<2 —
  the arming rule); time stop 3× half-life; single-pair loss ≥ 1.5× the
  pair's stationary spread vol in dollars ⇒ trader review before re-entry.
- **Per sector:** ≤ 3 pairs sharing a sector leg; net sector beta of the
  book within ±5% of gross (the "hedged" book that is secretly long one
  sector is the A2 failure).
- **Book:** drawdown kill-switches at portfolio level: −1.5% of capital
  (≈ $90k here, ≈ the walk-forward max DD) ⇒ halve gross; −3% ⇒ flat and
  post-mortem. These are *mechanical* — the August 2007 lesson is that
  discretionary "it will snap back" overrides are how a 3-day event becomes
  a career event. Both thresholds sit above the backtest max drawdown
  ($128k on $6mm = 2.1%) deliberately: hitting them means the model is
  wrong, not unlucky.
- **Leverage:** cap gross/capital using *stressed* spread vol (3× formation
  vol), not trailing vol — the low-vol-spread leverage temptation in
  VALIDATION §9.
- **Cointegration health:** weekly EG re-test on the trailing formation
  window. Two consecutive failures at 10% ⇒ no new entries; failure at
  trailing 5% plus an open position beyond one half-life ⇒ exit, don't wait
  for the stop.

## 3. Capacity & crowding monitoring

Fixed-bps slippage is only honest while the book's child orders are a small
share of close-auction volume; capacity per pair ≈ the size at which modelled
cost drag (34bp/yr at 5bp legs, VALIDATION §8) doubles. Crowding telemetry:
short interest and days-to-cover on each short leg; borrow-fee trend
(rising fee = crowded short); correlation of book P&L with public
mean-reversion factors; and dispersion of entry z across the pod's own pairs
(everything entering at once = one factor, not 6 pairs). August 2007
playbook: when the book loses >2× daily vol on *no news* across many pairs
simultaneously, that is deleveraging by a co-holder — cut gross first, ask
questions second (the snap-back three days later rewards the survivor, but
only if solvent).

## 4. Borrow desk interaction

Before entry: locate confirmed and fee quoted for every short leg; fee into
`CostModel.borrow_bps` per pair, not a book-wide constant. Daily: fee drift
> 2× entry assumption ⇒ re-run the pair's economics (a 50bp assumption vs a
500bp reality turns CO1's +$106k into a loser at scale). Recall risk: names
on the threshold list or with utilisation > 90% are capped at half gross;
a recall is an *involuntary* exit — model it as the stop-loss firing at an
uncontrolled price.

## 5. Scenario playbook (real-life edge cases)

**Quant quake (Aug 2007).** Symptom: many pairs hit stops together, no
news, borrow fine. Action: mechanical kill-switch (halve, then flat); do not
re-lever until spreads re-tighten and the EG re-tests pass. The simulation
analogue: every pair's stop firing in the same week is a 6-sigma event under
A8 independence — treat its occurrence as A8 failing, not bad luck.

**Short squeeze (GME, Jan 2021).** Symptom: one short leg +30% in days,
borrow fee vertical, z screaming "add". Action: the z-signal is *wrong*
because the model has no state for "float cornered". Respect the |z|=4 stop
unconditionally (tested path `exit_stop`), buy in the short before the
recall does it for you, flag the sector. The regime-break simulation is the
tame version; a squeeze is A1+A4 breaking at once.

**Merger breaks a pair (target acquired).** The cash-target leg pins to the
offer price: κ collapses, spread jumps and *should not* revert. Detection:
half-life estimate explodes on the trailing window / EG re-test fails hard.
Action: pair retired same day (corporate action feed beats statistics —
statistics needs 60 days of post-announcement data to "confirm" what the
tape said in a minute). The regime-break case study (−$21k stopped vs
−$715k unstopped, VALIDATION §9) is exactly this in miniature.

**Halt / long data gap.** `align_pair` refuses gaps >5 days by design; a
halted leg means the pair is untradeable, not "unchanged price". Positions
through a halt are handled by the desk (news, not model).

**Delisting / split.** Splits are why the cointegrating regression carries
an intercept and why production data must be adjusted; a delisted leg ends
the pair (trailing trim in `align_pair` handles the history).

## 6. Model governance

The model file per pair: formation window hash, β/α/κ/μ/σ, ADF stat vs EG
critical values, chosen rules (entry/exit/stop/time-stop), cost assumptions.
Any parameter change = a new walk-forward run archived with seed — every
number in this repo regenerates from `examples/run_pipeline.py` (seed 7),
which is the audit trail pattern: no number in the deck that a fresh clone
cannot reproduce. Overrides (trading through a failed re-test, sizing past a
cap) require PM + risk sign-off in writing; the test suite is the regression
gate for any code change (191 tests must stay green).
