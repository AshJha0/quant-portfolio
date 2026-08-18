# Desk Guide — How a Real Desk Uses a Regime Model

Documentation contract item 5: daily workflow, consumers, controls,
P&L attribution, governance, and real-life scenarios.

## 1. What funds actually do with regime models

Regime models are rarely traded as standalone strategies. They are
**overlays and dashboards**:

* **Vol-targeting overlays** — the most common use. The regime-conditional
  vol forecast (bear vol ≈ 3–4x bull vol) feeds the denominator of a
  vol-target rule; the regime probability decides *how fast* to scale down
  (bear entries justify faster de-leveraging than a trailing-window vol
  estimate alone would).
* **Tail-hedge activation** — buying index puts/VIX calls systematically is
  a −2–3% p.a. drag; funds instead *activate* the hedge program when the
  filtered bear probability crosses a band (the same 0.70/0.30 hysteresis
  logic as here) and deactivate it in confirmed bulls.
* **Factor rotation** — momentum and quality carry different premia by
  regime; the filtered probabilities become weights in a factor-blend
  (bull: momentum/size; bear: quality/low-vol; transition: blend).
* **Risk-committee dashboards** — the daily regime probability, expected
  duration, and days-in-regime chart is a standard page in weekly risk
  packs; consumed by the CRO and PMs, not by an execution engine.

## 2. Daily workflow with this codebase

1. **Overnight batch**: append yesterday's close, rebuild the feature row
   (expanding z-scores are O(1) to extend), run the forward filter one step
   with the frozen model → today's `p_bull / p_transition / p_bear`.
2. **Refit schedule**: Baum–Welch refit every 63 trading days on the
   expanding window (`refit_every=63`); refit dates and log-likelihoods
   are logged for model-drift monitoring. Labels are re-derived from state
   vol means at every refit, so label switching never reaches the book.
3. **Signal**: hysteresis state (enter bear > 0.70, exit < 0.30) → target
   weight (bull 1.0 / transition 0.5 / bear 0.0) → vol-target scale
   (10% target, 1.5x cap) → orders for the next open/close.
4. **Reporting**: the ledger, per-regime stats, transition attribution and
   flip-aftermath tables (`eq_regime.risk`) go into the daily P&L note; the
   filtered-probability path with regime shading goes to the risk pack.

**Who consumes what**: execution consumes only the weight; PMs consume the
probabilities and durations; the risk committee consumes the per-regime
attribution ("did we make our money by not losing in bears?"); model
validation consumes the refit log-likelihood series and the null-data
guard results.

## 3. Controls, limits and governance

* **Min-duration & hysteresis to avoid churn** (measured −67% to −82%
  turnover vs a naive threshold): a regime flip inside the band or shorter
  than 5 days never reaches the order pipe. Every flip that *does* trade is
  logged with its triggering probability path.
* **Hard limits**: weight ∈ [0, 1.5]; one regime flip per week maximum
  (further flips need PM sign-off); costs monitored vs the 5 bps model.
* **Model governance**: K is fixed (3) between annual reviews; a K change
  requires BIC evidence *and* economically distinct state vols *and*
  out-of-sample confirmation. Refit parameters are diffed: a transition
  matrix whose diagonal moves > 0.05 in one refit triggers review.
* **Kill criteria** (pre-agreed, mechanical):
  - regime flip-flop rate > 10% over a trailing quarter (model has become
    a noise amplifier) → revert to the 200d-MA benchmark rule;
  - strategy trails buy-and-hold by more than its historical worst
    12-month gap → de-activate overlay, keep dashboard;
  - null-guard failure at re-validation (model "finds" regimes in
    permuted/simulated null data) → full model review;
  - realized slippage > 3x modelled cost on two consecutive flips →
    suspend automatic trading of flips.

## 4. Realistic scenarios

* **COVID, March 2020 — fast crash.** Peak-to-trough took 23 trading days;
  a filter needs several days of evidence, and on this project's synthetic
  fast transitions the mean bear-entry lag is ~1.5 days *with well-separated
  states* — on real 2020 data the first de-risk would realistically have
  triggered around Feb 27–Mar 2, after roughly −8–12% of drawdown. **The
  detection lag is the price of admission**; the flip-aftermath report
  exists precisely to keep this cost visible (worst 10-day post-flip P&L
  here: −3.2%). The model then keeps you out of the −34% trough and — the
  harder part — the hysteresis exit band avoids re-entering during the
  April bounce whipsaws.
* **2022 — slow grinding bear.** This is where regime models shine: vol and
  correlation stayed elevated for ~10 months, so the filtered bear
  probability stays pinned high and the overlay sits de-risked through the
  whole grind. The synthetic-panel analogue: in detected bears the strategy
  returned +0.9% ann. vs −15.8% for buy-and-hold with a 50% drawdown
  (VALIDATION §7). A 200d-MA rule also handles 2022 well — the HMM's edge
  is the *scaled* transition book and the earlier, probability-weighted
  re-entry.
* **2015-Q3 / 2018-Q4 — sharp corrections, fast recoveries.** The model's
  weak spot: vol spikes trigger bear entries near the local low, and the
  recovery re-entry lags. These are the "false alarms" the hysteresis and
  min-duration machinery is tuned for: each false alarm costs
  (cost × 2 turns + missed rebound days), measured by the flip-aftermath
  table. Governance response is *not* to widen the band ad hoc — that
  invalidates the backtest — but to log the episode and review bands
  annually.

## 5. What this project deliberately does not claim

* Synthetic-panel performance (Sharpe 1.22 vs 0.18 buy-and-hold) is an
  **upper bound**: the generator's regimes are Gaussian and genuinely
  separated. Live regimes are less obliging.
* No shorting, no options: in a real book the bear state would activate a
  hedge sleeve rather than pure cash.
* Costs are linear 5 bps; crisis liquidity is worse exactly when the model
  trades (assumption 6 in the register).
