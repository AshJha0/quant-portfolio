# Desk Guide — how a systematic equity desk runs this

Who consumes what: **researchers** own `features`/`signals`/`evaluation`,
**portfolio managers** own `backtest` outputs and capacity, **execution
traders / algo desk** own `intraday`/`benchmarks`/`almgren_chriss`, and
**TCA / best-execution committee** owns `tca`. This guide walks the daily
workflow, the order lifecycle, the feedback loops, and the controls.

---

## 1. Research -> sim -> paper -> prod pipeline

1. **Research.** New feature lands in `features.py` with (a) a PIT mutation
   test, (b) a hand-computed unit test, before any IC is even looked at.
   IC studies use `ic_summary` (NW t-stats) and `signal_decay`; anything
   promoted must clear IC t > 3 *and* a DSR > 0.9 at the honest trial
   count N logged in the research diary. N is incremented for every variant
   tried — that log is the input to `deflated_sharpe_ratio`, not a guess.
2. **Simulation.** Candidate signal runs through `run_backtest` at the
   target AUM with the desk's fitted cost coefficients (linear bps +
   sqrt-impact k re-fit weekly from TCA, see §4). Gross-vs-net gap and the
   capacity curve decide whether the signal is worth wiring up: the demo
   numbers (709 bps/yr drag at 20% turnover, $200m) are exactly the format
   the PM sees.
3. **Paper.** The signal trades in shadow for 4-8 weeks: daily target
   weights are generated live, executions simulated with `IntradayMarket`
   calibrated to the names' spread/vol/ADV, and live IC is compared to the
   backtest's NW confidence band. Divergence beyond 2 SE for 20 days
   triggers a research review, not a quiet re-fit.
4. **Production.** Weights flow to the execution layer (§2). Model
   governance: the signal's code hash, parameter set, trial count N, and
   approval date are recorded; any parameter change restarts step 3.

## 2. Order lifecycle: portfolio target -> broker algo

1. **T close - decision.** Alpha layer emits target weights from the frozen
   signal (`freeze_signal`, 0.25-z band — the band is the first turnover
   control, worth 10 pts of daily turnover in the demo). The *decision
   price* (close) is stamped on every parent order — this timestamp is
   what makes Perold delay cost measurable later.
2. **Netting & limit checks (pre-trade).** Orders net across sleeves; each
   parent gets: participation cap (default 10% ADV; `pov_schedule` raises
   an informative error if the parent cannot complete — the desk's cue to
   split across days), per-name position cap (`max_weight`, enforced in
   construction), gross/net book limits (tested invariants of
   `long_short_weights`).
3. **Scheduling.** The trader (or auto-router) picks urgency:
   - default flow -> **VWAP schedule** over the U-profile;
   - risk-reducing / high-alpha-decay orders -> **AC** with lambda from the
     efficient frontier — the frontier table (15.5/99.0, 21.1/73.8,
     53.9/36.5 bps E/std in the demo) is literally the menu shown to the PM;
   - illiquid names or uncertain volume days -> **POV** at the cap.
4. **Child execution.** Children go to broker algos per bucket; fills come
   back with timestamps and prices. The arrival mid at order release is
   stamped (second TCA anchor).
5. **Post-trade.** `tca_report` per parent: delay / trading / opportunity.
   Unfilled tails are booked at the close (opportunity cost) — never
   silently rolled.

## 3. TCA feedback loop (weekly)

- `aggregate_tca` over the week's parents, cut by: strategy, side, size
  bucket (%ADV), and schedule type. The demo's aggregate table (delay 20.0,
  trading 35.8 ± 96.2, total 55.8 bps) is the per-order-cohort format.
- Fit realised trading cost vs sigma·sqrt(Q/ADV): the slope re-estimates
  the sqrt-law coefficient k feeding both the backtest cost model and the
  capacity curve; the intercept re-estimates the effective spread. Drift in
  k of > 25% quarter-on-quarter triggers a capacity review (§5).
- Schedule attribution: realised AC-vs-VWAP slippage compared against the
  simulator's predicted distribution (`evaluate_schedules` with current
  calibration). A schedule consistently outside its predicted band means
  the impact model, not the trader, is wrong.
- Delay cost trending positive on buys and negative on sells = the alpha is
  decaying intraday -> raise urgency (higher lambda) for that signal's
  orders; this is the standard alpha-decay/urgency trade.

## 4. Kill-switches and limit checks

Hard, automated, checked pre-trade and intra-day:

- **Participation kill**: child > cap·bucket-volume rejected (the simulator
  raises exactly this); parent > day capacity refused at scheduling with a
  multi-day split proposal.
- **Book limits**: gross > target, |net| > tolerance, |w_i| > cap -> no new
  opening trades (construction-level invariants here; production re-checks
  against real-time marks).
- **Cost kill**: realised day slippage > 3x modelled (e.g. > ~3·(spread/2 +
  k·sigma·sqrt(part)) per order on average) -> halt algo, route residuals
  manually.
- **Data kill**: stale prices, missing volume prints (zero-volume buckets),
  or feature NaN-rate spikes -> signal not refreshed (the frozen signal
  keeps yesterday's value by design; a stale-signal counter escalates after
  2 days).
- **Drawdown kill**: strategy MDD beyond backtest MDD + buffer -> de-gross
  to half, PM review.

## 5. Realistic scenarios

- **Flash-crash participation.** Volume explodes, so a naive POV algo
  *accelerates* into the dislocation. Control: cap child sizes by *shares*
  (from expected, not realised, volume) as well as by percentage, and the
  cost kill above. Rehearse with the simulator: raise `vol_noise` and
  sigma, replay a schedule via `execute(..., market_volumes=...)` with a
  crash tape.
- **Earnings-day execution.** Overnight gap = large delay cost, elevated
  sigma, distorted (front-loaded) profile. Desk practice: exclude earnings
  names from the band-freeze exemption (force a fresh signal), execute
  with higher lambda (front-load before the vol builds is wrong — the AC
  answer with 2-3x sigma is *slower only if* alpha allows; the frontier
  quantifies it), and tag the orders so TCA cuts them separately.
- **Index rebalance days.** The close bucket's volume share doubles;
  `u_shaped_profile` under-weights the close, so VWAP schedules lag the
  tape. Swap in the rebalance-day profile (pass custom `market_volumes` /
  profile to the schedulers) and expect crowding: impact coefficient on
  those names is temporarily higher, not lower, despite the volume.
- **Capacity reviews (quarterly).** Re-run the capacity curve with current
  k, spread, turnover and the *realised* ADV of the traded book (median,
  not mean — ADV is lognormal-skewed). The strategy's capacity is quoted at
  the AUM where net Sharpe hits an agreed floor (e.g. 2/3 of gross); the
  demo curve (4.54 at $10m -> 3.94 at $5bn on deep synthetic liquidity)
  shows the format; real small-cap books cross the floor orders of
  magnitude earlier.
- **P&L attribution (daily).** Ledger splits gross return vs cost; TCA
  splits execution cost vs benchmarks. The PM's daily sheet is: signal
  P&L (gross), cost drag (ledger), execution quality (IS vs VWAP), each
  with its own owner — which is the entire point of separating the layers.
