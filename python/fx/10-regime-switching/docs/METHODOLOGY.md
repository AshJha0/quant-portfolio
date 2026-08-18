# Methodology — FX Regime Switching (Risk-On / Risk-Off)

## 1. Why RORO regimes are *the* FX regime story

Currency markets do not cycle through abstract "states" — they cycle
through **risk-on / risk-off**, and the asset that lives or dies by it is
the **carry trade**:

* **Risk-on**: funding is cheap, vol is low, investors borrow JPY/CHF and
  lend AUD/NZD/EM.  Carry pairs grind higher, safe havens bleed, the
  interest differential accrues day after day.  Returns look like a
  short-vol position: steady premium, small daily moves.
* **Risk-off**: the same crowded position unwinds *violently*.  Carry and
  EM currencies gap lower against USD, JPY and CHF are bid as funding
  trades are repaid, realised vol multiplies, and — crucially — the
  cross-section collapses onto one factor: every USD pair trades the
  same "one trade" and pairwise correlations spike.
* **USD squeeze** (the 2008 / March-2020 variant): a dollar-funding
  scramble in which *everything* falls against USD, safe havens included.
  Distinct from ordinary risk-off precisely because the haven bid fails.

This is the empirical regularity behind the carry-crash literature —
Brunnermeier, Nagel & Pedersen's "carry trades and currency crashes"
story qualitatively: carry returns are negatively skewed because
crash risk is the compensation; unwinds coincide with funding-liquidity
stress and vol spikes; "up the stairs, down the elevator."  A regime
model is the natural formalisation: two (or three) persistent states
with different means, vols and **correlation structure**, and
state-dependent optimal books.

The economic punchline this project quantifies: **carry earns steadily
in risk-on and gives it back violently in risk-off; the value of a
regime filter is avoiding the unwind; its honesty metric is how many
days late it flags the flip — and what those days cost versus an oracle
that knows the true state.**

## 2. Pipeline

```
returns + deposit rates (synthetic RORO generator, known truth)
    │
    ▼
features.py     6+1 FX-native features, expanding-window (PIT) z-scores
    │
    ▼
pca.py          from-scratch PCA: PC1 of the currency panel = RORO axis
    │
    ▼
gmm.py/hmm.py   from-scratch EM: GMM (static clusters, BIC) and
    │           Gaussian HMM (Baum-Welch, log-space forward-backward)
    ▼
detection.py    expanding refit, FILTERED probabilities only,
    │           economic labeling, hysteresis + confirmation
    ▼
strategy.py     regime-conditional baskets, carry accrual, vol target,
    │           pip costs
    ▼
backtest.py     walk-forward ledger (net = spot + carry − cost, exact)
    │
    ▼
risk.py         per-regime stats, oracle comparison, detection-lag cost
```

## 3. Why an HMM (vs at least two alternatives)

| Model | What it captures | What it misses | Verdict |
|---|---|---|---|
| **Vol-threshold rule** ("risk-off if realised vol > x") | The single most informative symptom; trivially explainable | One symptom only: misses correlation spikes and haven bids; threshold is arbitrary and needs constant recalibration; binary output with no persistence model → flickers at the boundary | Good benchmark, not a model |
| **Correlation-threshold rule** ("risk-off if avg pairwise corr > y") | The "one-trade market" signature | Same arbitrariness; corr estimates need windows → lag; ignores vol and direction | Same class of problem |
| **GMM (static mixture)** | Full multivariate signature (vol + corr + direction) via cluster means/covariances | **No dynamics**: classifies each day independently, so it has no notion of regime *persistence*; posterior flickers day to day; cannot produce transition probabilities or expected durations | Used here for initialisation and BIC evidence |
| **Gaussian HMM** (chosen) | Everything the GMM sees **plus** a transition matrix: persistence, expected durations, and a filtered probability that correctly blends today's evidence with yesterday's belief | Gaussian emissions understate FX tails; constant transition probabilities; k must be chosen | **Chosen** — the persistence prior is exactly what separates a regime from a bad day |

The HMM's decisive advantage is the **filtered recursion**: p(sₜ|x₁..ₜ)
∝ emission × (transition-weighted prior).  A single bad day moves the
posterior a little; three bad days move it a lot.  Threshold rules and
GMMs cannot express that without ad-hoc smoothing — and ad-hoc smoothing
is just an unprincipled HMM.

Trade-offs accepted: EM finds local optima (mitigated by k-means-style
restarts and warm-started refits); Gaussian emissions are wrong in the
tails (see assumptions); k is treated as an economic choice with BIC as
evidence (see §6).

## 4. Filtered vs smoothed — the honesty rule

* **Filtered** p(sₜ | x₁..ₜ): uses data up to *t* only.  Tradeable.
* **Smoothed** p(sₜ | x₁..T): uses the whole sample, including the
  future.  Research only.  Smoothed probabilities make any regime model
  look clairvoyant — the flip is "detected" days before it happens,
  because the algorithm read the following week.

This project trades **only** filtered probabilities, produced by models
fitted on expanding windows, and enforces it with mutation tests:
perturbing data after *t* must leave every output at ≤ *t* bit-for-bit
unchanged (`tests/test_detection.py::test_filtered_only_past_mutation`,
`tests/test_backtest.py::test_no_lookahead_mutation`).  The suite also
verifies filtered ≠ smoothed except at the final observation.

## 5. Features (FX-native, point-in-time)

| Feature | Construction | Regime signature |
|---|---|---|
| `avg_vol` | mean rolling realised vol of G10-vs-USD legs, annualised (synthetic VXY analog) | ↑↑ in risk-off / squeeze |
| `carry_ret` | rolling return of the rank-carry basket (long top-3 yielders, short bottom-3, spot + accrual, PIT weights) | ↓↓ in unwinds |
| `haven_rs` | rolling (JPY+CHF)/2 − (AUD+NZD)/2 return | ↑ in risk-off; weaker in squeeze |
| `usd_corr` | average pairwise rolling correlation of the risk-block USD pairs | ↑↑ in unwinds ("one trade") |
| `em_g10` | rolling EM basket − G10 basket return | ↓↓ in unwinds |
| `usd_str` | − rolling average return of all currencies vs USD (dollar factor) | ↑↑ in a squeeze specifically |
| `fwd_ts` | CIP forward-point proxy: average annualised forward discount (r_ccy − r_USD) of the carry longs | slow-moving carry-availability gauge |

All features are standardised with **expanding-window z-scores**
(mean/std computed from data up to and including *t*) — no full-sample
statistics ever touch the past.  Windows are deliberately short (8/5/12
days): FX risk episodes last days-to-weeks, and longer windows smear
transitions into a spurious "transitional" state (found and documented
in VALIDATION.md §6).

`fwd_ts` is **excluded from the HMM input by default**: its expanding
z-score is near-unit-root, and an HMM offered a wandering feature will
happily carve the sample into meaningless early/late epochs.  It is
retained in the feature block as a desk-level diagnostic.

## 6. Economic labeling and the number of regimes

HMM state indices are arbitrary.  Each refit maps them to economics
from the state means in feature space:

1. `risk_on` = lowest `avg_vol` mean (with high-corr states this is
   equivalently the low-vol-low-corr state; the mapping "high-vol +
   high-corr ⇒ risk_off" is unit-tested);
2. with k=3, the remaining two split on `haven_rs`: havens rallying ⇒
   `risk_off`, havens falling too ⇒ `usd_squeeze`.

**Number of regimes.**  On raw iid returns, BIC recovers the true k
exactly (tested).  On rolling-window features — autocorrelated and
fat-tailed — HMM-BIC systematically over-selects k (the pipeline shows
GMM-BIC → 3, HMM-BIC → 4 on a true 3-state panel).  k is therefore an
**economic choice** (2 = RORO, 3 = RORO + squeeze) with BIC presented
as evidence, not verdict.

## 7. Regime-conditional strategy

| Regime | Book | Rationale |
|---|---|---|
| `risk_on` | rank-carry: long top-3 yielders, short bottom-3, dollar-neutral | harvest the differential while vol is low |
| `risk_off` | long JPY+CHF vs the risk block (AUD, NZD + EM) | carry is CUT; own the unwind |
| `usd_squeeze` | long USD vs everything | the haven bid fails; only USD works |

All books are **vol-targeted** at 10% annualised using a trailing
63-day covariance (PIT), leverage capped at 4×, with a vol floor
guarding degenerate (pegged) covariances.  P&L accrues
`w·(r_ccy − r_USD)/252` of **carry** daily on top of spot, and pays
**pip costs** on turnover (half-spreads per leg: ~0.5–2.5 pips G10,
6–12 pips EM; 1 pip = 1e-4 of notional).

**Hysteresis + confirmation**: a regime switch requires the challenger's
filtered probability ≥ 0.70 (or the incumbent's < 0.30) for 2
consecutive days.  This throttles turnover and flicker at the price of
detection lag — a governed, measured trade-off (VALIDATION.md §4).

## 8. Assumptions register

Each assumption states *what breaks if violated*.

1. **Gaussian emissions.**  FX daily returns are fat-tailed and jumpy
   (peg breaks, flash crashes).  *If violated*: a single 6σ day can be
   "explained" only by the high-vol state, causing false flips; tail
   risk inside a state is understated.  Mitigation: hysteresis absorbs
   one-day shocks; the failure is demonstrated for SNB-style jumps in
   VALIDATION.md §7.
2. **Constant transition probabilities.**  Real flip hazards are
   event-driven (FOMC, crises), not homogeneous.  *If violated*:
   expected durations and filtered priors are wrong around scheduled
   events; the desk override protocol (DESK_GUIDE.md) exists precisely
   because the model cannot see the calendar.
3. **Regime count k is known (2 or 3).**  *If violated*: with too-small
   k, distinct crash types merge (observed: risk_off and squeeze merge
   at realistic sample sizes); with too-large k, spurious transitional
   states appear and the labeler maps noise to books.
4. **Markov property (state depends only on yesterday's state).**
   *If violated* (e.g. duration-dependent hazards): duration estimates
   bias, but filtered inference degrades gracefully.
5. **Stationary feature distribution under expanding standardisation.**
   *If violated* (secular vol decline, structural break): z-scores
   drift and state means migrate; expanding refits partially adapt but
   with the full memory of the past.
6. **Deposit rates ≈ tradable forwards (CIP holds).**  Carry accrual is
   computed as (r_ccy − r_USD)/252, ACT/252.  *If violated* (cross-
   currency basis blowouts — exactly the squeeze state): realised carry
   on EM/funding legs is overstated in the worst state, so reported
   carry P&L in crises is an upper bound.
7. **Costs are constant half-spreads in pips.**  *If violated*: spreads
   widen 3–10× in risk-off, so switching costs at the flip are
   understated; hysteresis (fewer switches) limits the damage.
8. **Returns are currency-vs-USD legs with USD as numeraire and
   funding.**  BASE/QUOTE conventions follow the repo standard
   (EURUSD = USD per EUR); a "long AUD" leg means long AUDUSD.
9. **Liquidity is always available at the close.**  *If violated*
   (gapping markets): one-day execution delay understates slippage at
   exactly the flips the model is designed to catch — the oracle
   comparison partially prices this since the oracle pays the same
   delay.

## 9. Conventions

* Returns: daily log returns of currency vs USD; positive = currency
  appreciates.
* Rates: annualised decimals; daily accrual r/252 (ACT/252
  simplification, documented).
* Vol: annualised with √252; realised vol uses population ddof=0.
* All randomness flows through explicit `numpy.random.Generator`
  seeds; every number in the docs is reproducible from
  `examples/run_pipeline.py`.
