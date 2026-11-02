# Desk guide: equity regime-switching overlay

This guide describes how a regime overlay like this one is used and which
controls the code implements. Every result in this project comes from
synthetic data. The negative drift of the bear state is planted by the
generator, and no real-data result exists yet (real data belongs to the research platform in
`research/`).

## 1. Typical uses of a regime model

Regime models are usually overlays and dashboards, not standalone
strategies.

* Vol-targeting overlays. The regime-conditional vol forecast feeds the
  denominator of a vol-target rule, and the regime probability sets how fast
  to scale down.
* Tail-hedge activation. A hedge programme is switched on when the filtered
  bear probability crosses a band and off in confirmed bulls.
* Factor rotation. Filtered probabilities become weights in a factor blend.
* Risk dashboards. The daily regime probability, expected duration and
  days-in-regime chart go into the risk pack.

On synthetic data with the same drift in every state, the overlay does no
better than plain vol targeting (VALIDATION section 7). Until a real-data
test shows a negative conditional mean in the detected high-vol state, treat
the overlay as a vol-timing tool.

## 2. Daily workflow with this codebase

1. Append the latest close, rebuild the feature row, and run the forward
   filter one step with the frozen model to get `p_bull`, `p_transition`,
   `p_bear`.
2. Refit by Baum-Welch every 63 trading days on the expanding window
   (`refit_every=63`). Labels are re-derived from state vol means at every
   refit, so label switching never reaches the weights.
3. Map the hysteresis state (enter bear above 0.70, exit below 0.30) to a
   target weight (bull 1.0, transition 0.5, bear 0.0), apply the vol-target
   scale (10% target, 1.5x cap) and, if set, the causal minimum-duration
   confirmation (`walk_forward_backtest(min_duration=5)`).
4. Signals use data up to the close of day t. Orders fill at a modelled t+1
   price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
   variant fills at p_t and pays an auction cost.
5. Report the ledger, per-regime statistics, transition attribution and
   flip-aftermath tables from `eq_regime.risk`. Per-regime tables label each
   P&L day with the regime known at the decision date
   (`regimes_known_at_decision`).

## 3. Controls

Implemented in code:

* Hysteresis band on `p_bear` (0.70 / 0.30), tested in `test_strategy.py`.
* Causal minimum-duration confirmation, `min_duration` on
  `walk_forward_backtest` and `build_weights`, tested in
  `test_execution.py::TestCausalMinDuration`.
* Weight cap of 1.5x through the vol-target scale.
* Leverage above 1.0 pays cash rate plus `borrow_spread_bps` (default 50 bp a
  year); cash below 1.0 earns `cash_rate` (default 0).
* Flip-flop rate diagnostic, `detection.flip_flop_rate`.

Not implemented (governance a desk would add before trading):

* A limit of one regime flip per week, or any flip-rate limit.
* Logging of each traded flip with its probability path.
* Kill criteria: flip-flop rate above a threshold, trailing loss against
  buy-and-hold, null-guard failure at re-validation, slippage above model.
* Annual review of K, of the hysteresis band and of transition-matrix drift.

## 4. Scenarios the synthetic tests cover

* Late detection. The filter needs several days of evidence. The
  flip-aftermath table measures what that costs on synthetic paths.
* Short false alarms. A vol spike can trigger a bear entry near a local low.
  Hysteresis and the minimum-duration confirmation reduce the churn; the
  cost shows in the flip-aftermath table.
* No regimes. On GBM data the overlay does not beat random timing at equal
  exposure (`test_null_guard.py`).

This project has not been run on 2008, 2020 or 2022 market data, and makes no
statement about how it would have behaved in those episodes.

## 5. Limitations

* The synthetic Sharpe ratios are properties of the generator. The bear
  state's negative drift is an assumption of the generator, and it is the
  question a real-data study has to answer.
* No shorting and no options. A real book would add a hedge sleeve.
* Costs are linear (5 bp per unit traded). Crisis liquidity is worse exactly
  when the overlay trades.
