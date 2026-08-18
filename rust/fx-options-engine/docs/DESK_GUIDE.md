# Desk Guide — where a Rust engine sits on an FX options desk

The Python twin's DESK_GUIDE covers the daily *model* workflow (marking,
P&L attribution, governance). This guide covers what is specific to the
**Rust engine**: the production roles it plays and the operational traps —
the same roles the C++ engine plays, with the differences called out where
they matter.

## Role 1: pricing tiles and streaming quotes

The engine is the inner loop of the market-making tile. On every spot tick
or curve/vol update the tile re-prices its ladder — per pair, both sides,
all quoted tenors and strikes — and republishes. Practical notes:

* At ~8.4M prices/s a 2,000-instrument tile refresh is well under a
  millisecond single-threaded; pairs can be sharded across cores because
  every pricing function is a pure function over immutable market data —
  and here that is a *compiler-checked* claim, not a code-review
  convention: nothing in this crate uses interior mutability (`Cell`,
  `RefCell`, `Mutex`, raw pointers), so every public type is `Send + Sync`
  automatically, and a stray shared-mutable-state bug that would need a
  thread-sanitizer run to catch in C++ simply does not compile here.
* Quotes are constructed in (delta, vol) space: `strike_from_delta` turns
  the quoted 10d/25d/ATM pillars into strikes, the smile interpolator (out
  of scope here) supplies the vol, `gk_price` the premium, and
  `analytic_greeks` the hedge ratios published alongside.
* The DNS ATM strike (`atm_dns_strike`) is what "ATM" means for most pairs
  — note it is *convention-dependent* (`F e^{+sigma^2 T/2}` unadjusted vs
  `F e^{-sigma^2 T/2}` premium-adjusted), which is the first place a
  mis-configured pair bites.

## Role 2: risk service

End-of-day and intraday risk runs call the same library — not a
reimplementation:

* `analytic_greeks` gives the full ladder (both rhos, vanna, volga) per
  position; bucketed smile risk maps onto vanna (risk reversals) and volga
  (butterflies) directly.
* The generic `finite_difference_greeks` comparator lets risk control
  verify any pricer (tree, MC, or a new payoff) against bump-and-revalue
  with one call — the standard "analytic Greeks lie exactly once" check.
* Monte Carlo runs are bit-reproducible from the recorded seed
  (xoshiro256** + inverse-CDF normals, single-threaded, no OS entropy, no
  platform-dependent float reduction order), so an overnight number can be
  reproduced to the last bit during a P&L dispute — `mc_price` even has a
  dedicated test (`same_seed_is_bitwise_reproducible`) asserting
  `.to_bits()` equality, not just closeness.
* Both rhos matter in FX: hedging only `rho_domestic` and ignoring
  `rho_foreign` leaves the book exposed to the foreign leg of every cross —
  the engine returns them separately and the tests pin their signs.
* Errors surface as `Result<T, FxError>`, never a panic on bad market data
  (a stale/garbled feed produces a strike of `-1.0` or a vol of `NaN`
  sometimes — the engine rejects it with a message, it does not abort the
  process it's embedded in).

## Delta-convention pitfalls across counterparties

The single most common source of FX-desk breaks is two counterparties
agreeing a "25-delta" trade under *different* delta definitions. The engine
makes the convention an explicit argument everywhere; the desk still has to
choose the right one:

| Pitfall | Symptom | Guard in this engine |
|---|---|---|
| Spot vs forward delta (`delta_fwd = delta_spot * e^{r_f T}`) | strike breaks grow with tenor and `r_f` | explicit `DeltaConvention::{Spot,Forward}`; conversion helpers |
| Premium-adjusted vs unadjusted (premium paid in base ccy: USDJPY, most EM; unadjusted: EURUSD-style) | hedge off by `V/S`; strike breaks on the call wing | `SpotPa`/`ForwardPa` conventions; `premium_adjust_spot_delta` |
| PA call branch ambiguity (two strikes share one PA delta) | counterparty confirms the *other* strike | solver returns the market-standard high-strike branch; unattainable deltas return `Err` instead of guessing |
| ATM definition (ATM-forward vs DNS, and DNS itself flips with premium adjustment) | ATM strike off by `F sigma^2 T` | `atm_forward_strike` vs `atm_dns_strike(convention)` |
| Which currency's notional / premium | prices per unit *foreign* notional in *domestic* ccy — flipping the pair flips both | foreign–domestic symmetry identity is unit-tested; use it to re-express |

Rule of thumb encoded in the tests: for every quote confirm **(1) spot or
forward, (2) premium-adjusted or not, (3) which ATM** before touching a
strike. When in doubt, round-trip the counterparty's (delta, strike) pair
through `delta(...)` and see which convention reproduces it.

## A wing-vol trap specific to this engine's implied-vol solver

`docs/VALIDATION.md` documents a genuine IEEE-754 precision floor
(`~1e-8`) for implied vol recovered from a very short-dated, meaningfully
ITM/OTM premium. On a live desk this maps directly onto the well-known
"don't quote implied vol from the ITM wing" rule: a short-dated deep-ITM
premium is dominated by intrinsic value and carries almost no information
about vol (vega near zero), so both this engine and a human trader should
back vol out from the *liquid, near-the-money or OTM* side of the market
instead. If a downstream vol-surface builder ever needs an ITM-wing vol
anyway (e.g. to sanity-check a quote), treat anything from `implied_vol` on
such an input as accurate to `~1e-8`, not the usual `1e-10`, and do not
feed it into a calibration that assumes source-tolerance-level precision.

## Golden regression as the release gate

The golden-vector suite is not just a development aid; it is the release
control:

1. The Python reference is the model authority. Any intended behaviour
   change lands there first, is reviewed, and regenerates
   `tests/golden/golden_vectors.json`.
2. `tools/gen_golden_rs.py` refreshes `src/golden.rs`; the diff of that
   generated module *is* the reviewable surface of the model change on the
   Rust side.
3. CI runs `RUSTFLAGS="-D warnings" cargo test --release`. A Rust change
   that moves any golden number by more than 1e-9 fails the build — so an
   "optimisation" can never silently change a price. An intentional model
   change without a regenerated golden module fails too, which forces the
   Python, C++ and Rust engines to move in lockstep.
4. Release artefacts record the golden-file hash; production incident
   triage starts by confirming the running binary passed the golden gate
   for the model version risk signed off.

This is the cross-language analogue of a model-validation sign-off: fast
code is only trusted because, case by case, it is provably the slow,
reviewed code.
