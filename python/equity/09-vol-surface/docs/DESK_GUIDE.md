# Desk Guide — How an Equity Vol Desk Uses This Stack

Contract items 5 and 6: daily workflow, who consumes the numbers,
controls/limits, model governance, and realistic scenarios.

---

## 1. Daily workflow

### 07:30 — Marking the surface from listed quotes

1. Pull the listed chain (per underlier, per expiry), de-Americanise if
   needed, drop crossed/stale quotes.
2. `implied_vol_vector` on mids. Quotes that come back `nan` (sub-intrinsic,
   zero time value, dead wings) are *excluded by design* — the warning log is
   the audit trail of what was refused and why.
3. `fit_svi` per expiry (seeded restarts → reproducible marks). The fitter
   flags any slice violating the Durrleman condition: **a flagged slice is a
   quote problem or an arbitrage, either way a human looks before the mark
   goes out.**
4. `VolSurface` build with the calendar check. A calendar flag around an
   earnings date is usually *real event variance*, not bad quotes — see §4;
   enforcement (`enforce_calendar=True`) is a conscious override, never a
   default.
5. Publish: the surface object is the single source of truth for
   `vol(K, T)` used by cash desks, structured desks, and risk.

### 08:30 — Model calibration

`calibrate_heston` against the marked surface (multi-start, seeded). The
calibration report goes to the model-control log: RMSE per expiry, Feller
ratio, Jacobian condition number, and parameter deltas vs yesterday. Two
gates before the calibrated model is released to pricing:

* **Fit gate**: RMSE per pillar expiry below threshold (e.g. 0.5 vp beyond
  1m; the short end is *expected* to miss — VALIDATION.md §5.1 — and exotics
  inside the first pillar are not priced off Heston).
* **Stability gate**: parameter jump vs yesterday within band; κ/ξ moves get
  a wider band than v0/ρ/θ because of the documented ridge (VALIDATION.md
  §5.2). A big κ/ξ move with an unchanged surface is ridge noise → smooth it
  (carry yesterday's κ, refit the rest); a big ρ or v0 move is market.

### Intraday

* Vanilla quoting off the SVI surface (sticky-strike between rebuilds).
* Exotic pricing off calibrated Heston (Fourier for Europeans/digitals, QE
  Monte Carlo for path-dependents — 8–32 steps/year is enough, VALIDATION.md
  §3; **never** Euler on a Feller-violating fit).
* Greeks feed: `heston_greeks` (Richardson FD) for the model book,
  `bs_equivalent_greeks` for the BS-based firm risk system, and
  `smile_adjusted_delta` to publish both sticky-strike and sticky-moneyness
  deltas — the difference (≈ 0.08 delta ATM 6m in the pipeline run) is the
  hedging decision, and the desk head chooses the regime by market:
  range-bound → sticky-strike, trending/repricing → sticky-moneyness.

## 2. Who consumes what

| Consumer | Feed | Frequency |
|---|---|---|
| Cash/flow desk | `vol(K,T)`, smile slopes | continuous |
| Exotics desk | calibrated `HestonParams`, MC engine | per calibration |
| Market risk | Greeks + scenario ladders (spot × vol grid) | EOD + intraday snap |
| Product control | calibration report, fit RMSE, parameter history | EOD |
| Model validation | cross-method diffs, MC-vs-Fourier, recovery tests | monthly / on change |

P&L attribution runs off the Greeks feed: delta/gamma (spot), vega_v0 +
smile re-mark (vol), theta (carry), with the unexplained residual monitored —
persistent unexplained P&L on skew products is the symptom of marking
sticky-strike in a sticky-moneyness market (or vice versa).

## 3. Controls, limits, model governance

* **Arbitrage flags are blocking**: no surface publishes with an unreviewed
  butterfly or calendar violation.
* **Parameter-stability monitor**: time series of (v0, κ, θ, ρ, ξ) with the
  condition number; alert on jumps outside band. The ridge means κ/ξ pairs
  are monitored *jointly* (e.g. ξ²/κ, which vanillas actually pin) rather
  than individually.
* **Model reserve policy for unhedgeable parameters**: κ and ξ cannot be
  implied stably from vanillas (cond(J) ~ 1e3, day-over-day ±5–10 %).
  Exotics whose value loads on them individually (forward-starts, cliquets)
  carry a reserve sized by repricing across the ridge: revalue the trade at
  the day-1/day-2/day-3 parameter sets of VALIDATION.md §5.2 and reserve the
  spread. Short-dated skew products carry a separate reserve for the
  no-jumps assumption (the 2.8 vp 1-week miss of §5.1).
* **Scheme control**: MC pricing under Feller violation must use QE (or full
  truncation with ≥ 256 steps and a bias check vs Fourier); this is enforced
  in code review, and the bias table in VALIDATION.md is the evidence.
* **Change governance**: any change to pricers must keep the cross-method
  agreement tests (1e-6) and recovery tests green — they are the regression
  harness a validator reruns.

## 4. Realistic scenarios

### Earnings gap (single name, print tomorrow)

Tomorrow's expiry embeds one night of event variance: total variance jumps
between the pre- and post-earnings pillars. In smooth ACT/365F time this can
*look* like calendar arbitrage (post-event annualised vol far below
pre-event). **Do not** run `enforce_calendar=True` — the running-max would
smear event variance across the term structure. Correct handling: strip the
event (w = w_diffusive + w_event·1{T ≥ t_event}), mark the diffusive surface,
re-add the event. This stack flags the anomaly (the calendar check fires,
which is exactly the desired behaviour) and leaves event-time modelling to
the desk. The T→0 tests guarantee the machinery stays numerically sane at
overnight expiries.

### 2018 "Volmageddon" (Feb 5: short-vol unwind, VIX +115 % in a day)

* Marking: weekly wings gap 10+ vol points; many wing quotes go
  sub-intrinsic or crossed during the cascade — the robust inverter refuses
  them (`nan` + warning) instead of printing a 300-vol mark, and the SVI
  restarts keep slices fittable from the surviving body quotes.
* Calibration: v0 and ξ spike, ρ → −0.9, Feller deeply violated — all
  *allowed* by the bounds and warned, not rejected: the fit gate widens in
  crisis rather than the model silently failing. Euler-based risk runs are
  the hidden danger (bias grows exactly when ξ explodes — 40+ SE at 32
  steps): the QE requirement is a crisis control, not a nicety.
* Risk: the vega_v0 and gamma feeds reprice on the intraday snap; scenario
  ladders (spot −5 %/−10 % × vol +10/+20 pts) come straight from the pricer
  since parameters, not local slopes, drive the revaluation.

### Index skew in a crash (sticky what?)

In a fast sell-off the index smile empirically rides *with* the forward
(sticky-moneyness-like): realised delta of a call is the
`delta_sticky_moneyness` number, ~0.08 above sticky-strike ATM in the
pipeline run. A desk hedged on sticky-strike deltas into a crash is
under-hedged on the way down by vega × skew-slope / S per option. The dual
delta feed exists precisely so the hedging regime is an explicit desk
decision revisited when the regime shifts, with the gap between the two
deltas as the measured cost of being wrong.

### Dividend risk

The stack assumes continuous yield q. Around ex-div dates of large discrete
dividends the true forward is a step function: marking with smooth q shifts
forward moneyness, which shows up as a *spurious* skew/calendar kink at
expiries straddling the ex-date. Symptoms: butterfly/calendar flags
clustering at one expiry with clean quotes. Handling: mark those expiries on
dividend-adjusted forwards upstream (out of scope here, flagged in the
assumptions register); the sensitivity is quantifiable by bumping q — a 50 bp
dividend error on a 1y pillar moves the ATM strike's k by 50 bp and the
marked vol by skew-slope × 0.005 (~0.15 vp at the fitted 6m skew).

### Illiquid single names

Five good quotes per expiry is the documented hard floor (SVI has 5
parameters; `fit_svi` refuses fewer, single-strike expiries are rejected
informatively). Desk practice for thinner names: borrow the index smile shape
(fix b, ρ, σ from the index slice, fit only a, m) — the module's dataclass
parameterisation makes that a 5-line wrapper.
