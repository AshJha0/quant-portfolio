# Desk guide: how a systematic equity desk would run this

This guide describes the workflow the code is built to support. Only the
controls marked "coded" exist in this repository; the rest are the process
a desk would add. All numbers quoted are from synthetic data.

Owners: researchers own `features`, `signals`, `evaluation` and
`research_stats`; portfolio managers own `backtest` outputs and capacity;
execution traders own `intraday`, `benchmarks` and `almgren_chriss`; the
best-execution committee owns `tca`.

---

## 1. Research to production

1. Research. A new feature lands with a point-in-time mutation test and a
   hand-computed unit test before any IC is looked at. Every configuration
   evaluated goes into a `TrialLog`. Promotion needs an IC t-stat above 3,
   a deflated Sharpe above 0.9 at the logged trial count and a PBO below
   0.2. On synthetic data the combined book has a deflated Sharpe of 0.59
   at ten trials and would not be promoted.
2. Simulation. The signal runs through `run_backtest` at the target AUM
   with the default t+1 fill and the fitted cost coefficients, across many
   seeds or a block bootstrap, then through `capacity_backtest` at Y = 0.3,
   0.7 and 1.0.
3. Paper trading for 4-8 weeks. Live IC is compared with the backtest's
   Newey-West band. Two SE of divergence for 20 days triggers a review.
   Not coded.
4. Production. Code hash, parameters, trial count and approval date are
   recorded; a parameter change restarts paper trading. Not coded.

## 2. Order lifecycle

1. Close of day t: the alpha layer emits target weights from the frozen
   signal (`freeze_signal`, coded). The close is stamped as the decision
   price for Perold delay cost.
2. Pre-trade checks: participation cap (`pov_schedule` raises if the
   parent cannot complete, coded), per-name cap and dollar neutrality
   (`long_short_weights` invariants, coded), gross and net limits.
3. Scheduling on day t+1: VWAP for default flow, AC with lambda from the
   frontier for urgent orders, POV at the cap for illiquid names. The
   frontier on synthetic parameters is (15.5, 99.0), (21.1, 73.8) and
   (53.9, 36.5) bp expected cost and std for lambda 1e-6, 5e-6 and 5e-5.
4. Child execution and post-trade `tca_report`: delay, trading and
   opportunity cost. Unfilled tails are booked at the close.

## 3. TCA feedback (weekly)

- `aggregate_tca` cut by strategy, side, size bucket and schedule.
- Regress realised trading cost on sigma sqrt(Q/ADV) to re-estimate Y. The
  backtest and capacity runs use the new value. A move of more than 25% in
  a quarter triggers a capacity review.
- Compare realised schedule slippage with `evaluate_schedules`
  (model-implied IS and Monte Carlo std; schedule-level cost model on a
  bucket simulator). A schedule outside its band points at the impact
  model.
- Positive delay cost on buys means the alpha decays overnight. Raise
  urgency or move the fill earlier.

## 4. Kill switches

None of these is coded beyond the simulator's input checks.

- Participation: a child above the cap times bucket volume is rejected
  (the simulator raises exactly this).
- Book limits: gross, net and per-name breaches block new opening trades.
- Cost: realised daily slippage above three times the model halts the
  algo.
- Data: stale prices or a spike in feature NaNs freeze the signal; a
  stale-signal counter escalates after two days.
- Drawdown: strategy drawdown beyond the backtest's plus a buffer halves
  gross pending review.

## 5. Scenarios

- Flash crash. Volume explodes and a POV algo speeds up into the move. Cap
  child sizes in shares from expected volume as well as in percent.
  Rehearse with `execute(..., market_volumes=...)` and a crash tape.
- Earnings days. A large overnight gap shows up as delay cost. Tag these
  orders and cut them separately in TCA.
- Index rebalance days. The close bucket's share of volume doubles; pass a
  custom profile to the schedulers.
- Capacity reviews. Re-run `capacity_backtest` with current Y, spread and
  realised ADV. On synthetic seed 42 the net Sharpe is already negative at
  $1bn for Y >= 0.7; a real small-cap book would fail far earlier.
