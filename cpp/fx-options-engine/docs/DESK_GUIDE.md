# Desk Guide — where a C++ engine sits on an FX options desk

The Python twin's DESK_GUIDE covers the daily *model* workflow (marking,
P&L attribution, governance). This guide covers what is specific to the
**C++ engine**: the production roles it plays and the operational traps.

## Role 1: pricing tiles and streaming quotes

The engine is the inner loop of the market-making tile. On every spot tick
or curve/vol update the tile re-prices its ladder — per pair, both sides,
all quoted tenors and strikes — and republishes. Practical notes:

* At ~13M prices/s a 2,000-instrument tile refresh is ~0.15 ms
  single-threaded; pairs can simply be sharded across cores because every
  pricing function is a pure, thread-safe free function over immutable
  market data.
* Quotes are constructed in (delta, vol) space: `strike_from_delta` turns
  the quoted 10d/25d/ATM pillars into strikes, the smile interpolator (out
  of scope here) supplies the vol, `gk_price` the premium, and
  `analytic_greeks` the hedge ratios published alongside.
* The DNS ATM strike (`atm_dns_strike`) is what "ATM" means for most pairs
  — note it is *convention-dependent* (`F e^{+σ²T/2}` unadjusted vs
  `F e^{−σ²T/2}` premium-adjusted), which is the first place a
  mis-configured pair bites.

## Role 2: risk service

End-of-day and intraday risk runs call the same library — not a reimplementation:

* `analytic_greeks` gives the full ladder (both rhos, vanna, volga) per
  position; bucketed smile risk maps onto vanna (risk reversals) and volga
  (butterflies) directly.
* The templated `finite_difference_greeks` comparator lets risk control
  verify any pricer (tree, MC, or a new payoff) against bump-and-revalue
  with one line — the standard "analytic Greeks lie exactly once" check.
* Monte Carlo runs are bit-reproducible from the recorded seed
  (`mt19937_64` + inverse-CDF normals, single-threaded), so an overnight
  number can be reproduced to the last bit during a P&L dispute.
* Both rhos matter in FX: hedging only `rho_d` and ignoring `rho_f` leaves
  the book exposed to the foreign leg of every cross — the engine returns
  them separately and the tests pin their signs.

## Delta-convention pitfalls across counterparties

The single most common source of FX-desk breaks is two counterparties
agreeing a "25-delta" trade under *different* delta definitions. The engine
makes the convention an explicit argument everywhere; the desk still has to
choose the right one:

| Pitfall | Symptom | Guard in this engine |
|---|---|---|
| Spot vs forward delta (`Δ_f = Δ_s e^{r_f T}`) | strike breaks grow with tenor and `r_f` | explicit `DeltaConvention::{Spot,Forward}`; conversion helpers |
| Premium-adjusted vs unadjusted (premium paid in base ccy: USDJPY, most EM; unadjusted: EURUSD-style) | hedge off by `V/S`; strike breaks on the call wing | `SpotPa`/`ForwardPa` conventions; `premium_adjust_spot_delta` |
| PA call branch ambiguity (two strikes share one PA delta) | counterparty confirms the *other* strike | solver returns the market-standard high-strike branch; unattainable deltas throw instead of guessing |
| ATM definition (ATM-forward vs DNS, and DNS itself flips with premium adjustment) | ATM strike off by `F σ² T` | `atm_forward_strike` vs `atm_dns_strike(convention)` |
| Which currency's notional / premium | prices per unit *foreign* notional in *domestic* ccy — flipping the pair flips both | foreign–domestic symmetry identity is unit-tested; use it to re-express |

Rule of thumb encoded in the tests: for every quote confirm **(1) spot or
forward, (2) premium-adjusted or not, (3) which ATM** before touching a
strike. When in doubt, round-trip the counterparty's (delta, strike) pair
through `delta(...)` and see which convention reproduces it.

## Golden regression as the release gate

The golden-vector suite is not just a development aid; it is the release
control:

1. The Python reference is the model authority. Any intended behaviour
   change lands there first, is reviewed, and regenerates
   `tests/golden/golden_vectors.json`.
2. `tools/gen_golden_header.py` refreshes `tests/golden_vectors.hpp`; the
   diff of that generated header *is* the reviewable surface of the model
   change on the C++ side.
3. CI runs the full CTest suite with `-Werror`. A C++ change that moves any
   golden number by more than 1e-9 fails the build — so an "optimisation"
   can never silently change a price. An intentional model change without a
   regenerated golden file fails too, which forces the Python and C++
   engines to move in lockstep.
4. Release artefacts record the golden-file hash; production incident
   triage starts by confirming the running binary passed the golden gate
   for the model version risk signed off.

This is the cross-language analogue of a model-validation sign-off: fast
code is only trusted because, case by case, it is provably the slow,
reviewed code.
