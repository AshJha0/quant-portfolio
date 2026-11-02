# Desk Guide: how an equity vol desk would use this stack

Daily workflow, consumers of the numbers, controls, and scenarios. All
numbers quoted here come from `examples/run_pipeline.py` and
`examples/validation_tables.py` on synthetic data.

---

## 1. Daily workflow

### 07:30: marking the surface from listed quotes

1. Pull the listed chain per underlier and expiry, de-Americanise if
   needed, drop crossed or stale quotes.
2. Run `implied_vol_vector` on mids. Quotes that return `nan`
   (sub-intrinsic, zero time value, dead wings) are excluded. The warning
   log records what was refused and why.
3. Run `fit_svi` per expiry. The fit is constrained to be free of butterfly
   arbitrage and to respect Lee's wing bound. `active_constraints` lists any
   constraint that binds. A binding constraint means the quotes asked for
   an arbitrageable smile: a human looks before the mark goes out.
4. Build `VolSurface` with the calendar check. A calendar flag around an
   earnings date is usually event variance, not bad quotes (see section 4).
   `enforce_calendar=True` is an explicit override. After enforcement the
   surface re-checks butterfly arbitrage at pillars and midpoints
   (`butterfly_after_enforcement`).
5. Publish the surface object as the single source of `vol(K, T)` for cash
   desks, structured desks and risk.

### 08:30: model calibration

`calibrate_heston` against the marked surface (multi-start, seeded). The
report goes to the model-control log: RMSE per expiry, Feller ratio,
Jacobian condition number, and parameter changes against yesterday. Two
gates before the calibrated model is used for pricing:

* Fit gate: RMSE per pillar expiry below a threshold (for example 0.5 vol
  points beyond 1m). The short end is expected to miss (VALIDATION.md
  section 5.1), so exotics inside the first pillar are not priced off
  Heston.
* Stability gate: parameter change against yesterday within a band. Kappa
  and xi get a wider band than v0, rho and theta because of the ridge
  (VALIDATION.md section 5.2). A large kappa/xi move with an unchanged
  surface is ridge noise: carry yesterday's kappa and refit the rest. A
  large rho or v0 move is market.

### Intraday

* Vanilla quoting off the SVI surface between rebuilds.
* Exotic pricing off calibrated Heston: Fourier for Europeans and digitals,
  QE Monte Carlo (martingale-corrected, antithetic, control variate) for
  path-dependent payoffs. 8 to 32 steps per year is enough
  (VALIDATION.md section 3). Do not use Euler on a Feller-violating fit.
  No path-dependent payoff is implemented in this package.
* Greeks feed: `heston_greeks` (Richardson FD) for the model book,
  `bs_equivalent_greeks` for a BS-based risk system, and
  `smile_adjusted_delta`, which returns three deltas: sticky-strike,
  sticky-moneyness and a first-order local-vol rule. In the pipeline run
  (ATM, 6m) they are 0.538, 0.615 and 0.462, and the Heston delta is 0.616.

## 2. Who consumes what

| Consumer | Feed | Frequency |
|---|---|---|
| Cash and flow desk | `vol(K,T)`, smile slopes | continuous |
| Exotics desk | calibrated `HestonParams`, MC engine | per calibration |
| Market risk | Greeks and spot x vol scenario ladders | EOD and intraday snap |
| Product control | calibration report, fit RMSE, parameter history | EOD |
| Model validation | cross-method differences, MC against Fourier, recovery tests | monthly or on change |

P&L attribution uses the Greeks feed: delta and gamma (spot), vega_v0 plus
smile re-mark (vol), theta (carry). Persistent unexplained P&L on skew
products is the symptom of hedging with the wrong smile-dynamics regime.

## 3. Controls, limits, model governance

* Arbitrage flags block publication: no surface goes out with an unreviewed
  butterfly or calendar violation.
* Parameter-stability monitor: time series of (v0, kappa, theta, xi, rho)
  with the condition number. Kappa and xi are monitored jointly through
  xi^2/kappa, which vanillas pin, rather than individually.
* Model reserve for parameters vanillas cannot pin: kappa and xi move
  several percent day over day under quote noise with an unchanged surface
  (VALIDATION.md section 5.2). Exotics whose value loads on them
  individually (forward starts, cliquets) carry a reserve sized by
  repricing across the three parameter sets of that table. Short-dated skew
  products carry a separate reserve for the no-jumps assumption
  (section 5.1).
* Scheme control: Monte Carlo under Feller violation uses QE, or full
  truncation Euler with at least 256 steps and a bias check against
  Fourier. The bias table in VALIDATION.md section 3 is the evidence.
* Change control: any pricer change keeps the cross-method agreement tests
  (1e-6) and the recovery tests green.

## 4. Scenarios

### Earnings gap (single name, print tomorrow)

Tomorrow's expiry embeds one night of event variance, so total variance
jumps between the pre- and post-earnings pillars. In smooth ACT/365F time
this can look like calendar arbitrage. Do not run `enforce_calendar=True`:
the running maximum would spread event variance across the term structure.
Strip the event (`w = w_diffusive + w_event * 1{T >= t_event}`), mark the
diffusive surface, and add the event back. This package flags the anomaly
and leaves event-time modelling to the desk. The T -> 0 tests keep the
machinery numerically sane at overnight expiries.

### February 2018 short-vol unwind (VIX +115% in a day)

* Marking: weekly wings gap by 10+ vol points and many wing quotes go
  sub-intrinsic or crossed. The inverter refuses them (`nan` plus a
  warning). The SVI fit uses the surviving body quotes.
* Calibration: v0 and xi spike, rho goes towards -0.9, Feller is deeply
  violated. The bounds allow this and the code warns. The fit gate widens
  in a crisis instead of the model failing.
* Euler-based risk runs are the danger here: Euler bias grows with xi
  (tens of standard errors at 32 steps on the extreme set,
  VALIDATION.md section 3). QE is the crisis control.
* Risk: vega_v0 and gamma reprice on the intraday snap. Scenario ladders
  (spot -5%/-10% x vol +10/+20 points) come from the pricer.

### Index skew in a sell-off (which delta?)

Empirically, index implied vols at fixed strike rise as spot falls: the
smile moves against spot, closer to local-vol dynamics than to
sticky-moneyness. Under negative skew this puts the minimum-variance call
delta below the sticky-strike BS delta. A desk hedging on the
sticky-moneyness (or Heston at fixed v0) delta is over-hedged on the way
down. `delta_local_vol_approx` gives the first-order local-vol number; the
gap between the three deltas is the size of the regime decision. Heston
with a fixed v0 is a sticky-moneyness model, so its delta needs the same
regime adjustment.

### Dividend risk

The package assumes a continuous yield q. Around large discrete
dividends the true forward steps at the ex-date. Marking with a smooth q
shifts forward moneyness, which shows up as a skew or calendar kink at
expiries straddling the ex-date. Symptom: butterfly or calendar flags at
one expiry with clean quotes. Mark those expiries on dividend-adjusted
forwards upstream (out of scope here). A 50 bp dividend error on a 1y
pillar moves the ATM log-moneyness by 50 bp and the marked vol by the skew
slope times 0.005.

### Illiquid single names

Five valid quotes per expiry is the floor (SVI has five parameters;
`fit_svi` refuses fewer). For thinner names, borrow the index smile shape
(fix b, rho and sigma from the index slice and fit a and m only).

## 5. Round 3 models: when to use which

- SSVI for a quick, arbitrage-free surface across expiries from a handful of
  parameters; eSSVI when the skew flattens with maturity and one rho is not
  enough.
- Dupire local vol for path-dependent payoffs priced consistently with the
  vanilla surface. It reprices vanillas by construction, but its forward
  smile is too flat, so do not rely on it alone for forward-start or cliquet
  risk.
- Bates when short-dated skew is steeper than Heston can produce.
