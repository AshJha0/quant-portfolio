# Desk Guide: Running a Systematic Signal Like This One

How a systematic desk would review and run a signal of this kind. None of
the monitoring or kill controls below is implemented in this repository;
they are process, described for a reader who would build them. Every
number quoted is from synthetic data. No real-data result exists yet
(see `research/`).

---

## 1. Review before capital

- **Researcher** owns the backtest: signal, fill model, costs, split,
  walk-forward, sensitivity, trial log, multi-seed or multi-asset
  evidence.
- **Research committee** reviews the method: is the cost model right for
  the instrument and size, was the out-of-sample window touched only once,
  how many configurations were tried, where does it lose money
  (`docs/VALIDATION.md` §8).
- **Risk** re-runs `pytest` and `examples/run_pipeline.py` from a clean
  checkout, confirms the numbers in `docs/VALIDATION.md`, and checks the
  5 bp cost against the instrument's quoted spread and volume.
- **PM** signs off on size, correlation to the book, capacity and the
  re-selection process (walk-forward cadence).

## 2. Pre-launch checklist

**Gate 0: does the strategy pass its own validation?** On the synthetic
generator it does not. Over 100 synthetic seeds its out-of-sample Sharpe
minus buy and hold is -0.11 [-0.21, -0.00] and its walk-forward Sharpe
minus buy and hold is -0.05 [-0.11, 0.00] (`docs/VALIDATION.md` §6). The
decision there is do not allocate, and the checklist stops.

Every extra configuration tried against an out-of-sample window turns it
into an in-sample window. Log every trial (`TrialLog`) so the deflated
Sharpe uses the true count.

For a signal that clears Gate 0:

1. **Walk-forward, not one split.** The stitched out-of-sample Sharpe must
   hold across re-selections.
2. **Deflated Sharpe and PBO.** The grid here has 49 pairs. On synthetic
   seed 32 the deflated Sharpe of the selected in-sample result is 0.892,
   below the usual 0.95 bar, and PBO is 0.45.
3. **Capacity.** The flat 5 bp cost (`docs/METHODOLOGY.md` A2) is for a
   liquid ETF at small size. Re-run with a size-dependent cost model
   before any size that is a meaningful share of daily volume.
4. **Fill model.** The backtest assumes a fill at p_t + 0.5 (p_{t+1} - p_t).
   Paper trading must show which fill is achievable: next open, an
   early-session VWAP, or the closing auction with its cost.
5. **Paper trading.** Promotion criterion: paper P&L within about one
   standard error of the backtest's expected P&L over the paper period.
6. **Kill criteria written before launch** (§3).

## 3. Live monitoring (process, not implemented)

| Signal | Meaning | Action |
|---|---|---|
| Live Sharpe well below walk-forward Sharpe over about a year | Noise within the expected dispersion, or an uncovered regime | Check realised costs and fills against the assumptions first |
| Trade count diverging from the backtest | Wrong parameters deployed, or data gaps | Reconcile against `ma_crossover_signal` on the production prices |
| Drawdown well beyond the walk-forward drawdown | Possibly the choppy regime of `docs/VALIDATION.md` §8 | Decide on size; this is the documented failure mode |
| Realised costs above 5 bp | Cost assumption too low for the size | Re-run the backtest at the realised cost |

The engine is a pure function of prices, signal, cost and fill model, so a
live run should reconcile day by day with `run_backtest` on the same
closes. A divergence that survives reconciliation is an implementation
bug.

A rolling realised-vs-backtest Sharpe gap beyond a multiple of the Sharpe
standard error (`lo_sharpe_se`) is a reasonable review trigger. Example
kill criteria for a small sleeve: drawdown beyond 1.5 times the
walk-forward drawdown halves size; beyond 2.5 times goes flat pending a
review.

## 4. Scenarios

**Live underperforms the backtest.** Check in order:

1. **Costs and fills.** Compare realised fill prices with the modelled t+1
   price and the 5 bp cost. On synthetic seed 32, moving the (10, 50) pair
   from 0 to 20 bp lowers its Sharpe from 0.719 to 0.647
   (`docs/VALIDATION.md` §7).
2. **Regime.** Count recent crossovers. Frequent small crossovers with no
   trend match the documented whipsaw failure.
3. **Implementation drift.** Confirm the deployed parameters are the ones
   the walk-forward process last selected.

**A flat cost at size.** At many times the backtest notional, impact
dominates the spread and grows roughly with the square root of
participation. A size-aware cost such as
`base_bps + k sqrt(size / ADV)` lowers net Sharpe at scale with the same
signal. That is a capacity limit; the fix is smaller size or slower
execution. This project's flat cost does not model it.

## 5. Model governance

Every number in `docs/VALIDATION.md` regenerates from
`examples/run_pipeline.py` and `examples/validation_tables.py`. The test
suite (133 tests) is the regression gate. A change that alters a
documented number updates `docs/VALIDATION.md` in the same change.
