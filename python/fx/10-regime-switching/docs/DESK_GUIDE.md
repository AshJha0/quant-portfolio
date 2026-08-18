# Desk Guide — Using the RORO Regime Filter on a Macro / FX Desk

This is how a currency fund or bank FX desk would actually consume this
model.  The model's job is **not** to be a strategy by itself — it is a
*risk throttle and overlay driver* for books that already exist.

## 1. Who consumes what

| Output | Consumer | Cadence |
|---|---|---|
| Filtered regime probabilities (3 numbers) | PM, risk manager | daily close |
| Committed regime label (post-hysteresis) | carry book sizing, overlay desk | daily close, effective next open |
| Expected durations / transition row | PM (how long is this likely to last?) | at each flip |
| Per-regime P&L attribution | risk committee | weekly |
| Detection-lag & false-alarm report vs oracle | model governance | monthly |

## 2. Daily workflow

1. **T close**: features update (vol, correlation, carry/haven/EM/dollar
   baskets — all computable from spot closes and deposit rates by ~5pm
   NY).  Filtered probabilities refresh in seconds; the HMM refit runs
   monthly (expanding window, warm-started).
2. The hysteresis layer either keeps the committed regime or flips it
   (challenger ≥ 0.70 for 2 days, or incumbent < 0.30).
3. **T+1 open**: books move.  Risk-on → rank-carry basket at 10% vol
   target.  Risk-off → carry basket cut, defensive book on (long
   JPY/CHF vs AUD/NZD + EM).  Squeeze → long USD vs everything.
4. P&L ledger accrues spot + carry − pip costs, attributable line by
   line (the identity net = spot + carry − cost is enforced to 1e-15 in
   tests).

## 3. Realistic usage patterns

* **Carry-book risk throttle** (primary use).  Scale an existing carry
  book by p(risk_on): full size above 0.8, half below 0.6, flat below
  0.4.  The validation numbers say this is worth ~2% p.a. and, more
  importantly, converts the carry book's risk-off cell from
  −13% p.a. / Sharpe −0.7 to +20% p.a. / Sharpe +1.2 (VALIDATION.md §3).
* **Tail-hedge overlay**.  Do not wait for the filter for gap risk it
  cannot see (SNB-style breaks).  Standing JPY-call / AUDJPY-put ladder
  sized so the hedge budget ≈ expected false-alarm cost; the filter's
  p(risk_off) modulates the *delta* of the overlay, not its existence.
  Rationale: the filter's lag is ~1 day for diffusion-style unwinds and
  infinite for jumps — options cover the jumps.
* **Risk dashboard**.  The six features are themselves the dashboard:
  synthetic VXY, carry-basket momentum, haven relative strength,
  USD-pair correlation, EM spread, dollar factor — each as expanding
  z-scores.  A PM reads "vol +2.5σ, corr +1.7σ, havens bid" faster than
  any posterior probability.
* **Sizing new trades**.  Veto adding carry exposure when
  p(risk_on) < 0.6; require risk-off confirmation before adding
  defensive trades (avoid paying the flip twice).

## 4. Historical scenario playbook

How the states map to the episodes everyone remembers:

* **2008 GFC (Sep–Nov)**: textbook risk_off → usd_squeeze sequence.
  Carry unwind (AUDJPY −40%), then a dollar-funding squeeze in which
  even EUR and GBP fell vs USD while only JPY held.  A 3-state filter
  distinguishes the phases; the defensive → long-USD book switch is the
  P&L difference between the two.
* **March 2020 COVID**: 2-week usd_squeeze — *everything* including JPY
  and gold sold for dollars until Fed swap lines opened.  The squeeze
  label (dollar factor + correlation, haven bid failing) is built for
  exactly this; an ordinary risk-off book (long havens) underperformed
  long-USD by construction.
* **2013 taper tantrum**: EM-specific unwind — `em_g10` spread and EM
  vol move first, G10 carry only wobbles.  The filter flags risk-off
  late or not at all if G10 features stay calm: this is the documented
  "idiosyncratic vs systemic" failure mode; per-country limits must
  back it up.
* **2022 USD squeeze (Fed hiking)**: a *slow* dollar squeeze — months,
  not weeks.  Everything fell vs USD but at low daily vol.  The vol
  feature undersells it; the dollar-factor feature (`usd_str`) is what
  keeps the label honest.  Expect the filter to be structurally late in
  slow squeezes.
* **SNB Jan 2015**: not a regime — a discontinuity.  The filter reacts
  the day after.  Options and limits, not filters (VALIDATION.md §7).

## 5. Governance and controls

* **Hysteresis parameters are risk policy, not model internals.**  The
  enter/exit thresholds and confirmation days directly trade detection
  lag against false alarms.  Validation shows fast (0.70/0.30, 2d)
  dominates conservative (0.90/0.10, 5d) on synthetic RORO panels —
  lag cost ~10 bp/flip vs ~34 bp/flip, and false-alarm cost *rises*
  with symmetric conservatism because exits slow too.  Any change goes
  through the model-risk committee with the oracle-gap decomposition
  re-run.
* **Min-duration / flicker control**: the confirmation rule caps
  regime-driven turnover; tested to reduce switches vs raw argmax.
* **Human override protocol around scheduled events.**  The model
  assumes constant transition probabilities; the calendar does not.
  Standing desk rules: (i) freeze regime *upgrades* (defensive → risk-on)
  in the 24h before FOMC/ECB/BoJ and major CPI prints; (ii) a PM may
  impose `risk_off` ahead of a known event (referendum, election) —
  the override is logged with owner and expiry; (iii) the filter can
  always *downgrade* to defensive — overrides may never block the
  defensive direction.
* **Refit governance**: expanding refit monthly, warm-started; a refit
  that relabels history (state means migrating across the labeling
  boundary) is flagged for review before the new model trades.
* **Kill criteria**: detection rate on realised vol spikes < 60%
  rolling-yearly, or realised false-alarm cost > 2× the model-approved
  budget, sends the throttle to a neutral 50% carry scaling pending
  review.

## 6. P&L attribution the risk committee sees

The ledger separates **spot**, **carry**, and **cost** exactly, and the
risk module partitions by regime and by spell phase:

* per-regime table (are we being paid in the state we think we're in?),
* transition attribution: first-5-days vs steady-state P&L of each
  spell (the crash-capture metric — 93% of carry's risk-state losses
  land in the first 10 days on the validation panel),
* oracle-gap decomposition: how much of the gap to a true-state oracle
  came from lag vs false alarms this quarter.

## 7. Limits of responsibility

The filter manages *systemic* RORO risk in a diversified currency book.
It does not manage: idiosyncratic country risk (limits do), gap/peg
risk (options do), slow carry mediocrity (strategy review does), or
execution in gapping markets (the execution desk does — see the
fx-algo project).  A desk that treats a regime filter as a crystal
ball has misread this guide; the honest content of the model is the
oracle table: perfect state knowledge is worth ~4% p.a. over the filter,
and the filter is worth ~2% p.a. over doing nothing.
