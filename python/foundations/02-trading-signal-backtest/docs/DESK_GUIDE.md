# Desk Guide — Running a Systematic Signal Like This One

How a real systematic trading desk actually evaluates a signal of this
kind day to day: governance, pre-launch checklist, live monitoring, and
concrete scenarios (contract items 5 and 6).

---

## 1. Who reviews a backtest like this before capital is allocated

No systematic signal goes live off the strength of one script's output.
The typical review chain:

- **Quant researcher** builds and owns the backtest (this project's role):
  signal definition, cost model, in-sample/out-of-sample split,
  walk-forward validation, sensitivity map. Presents the equivalent of
  `docs/VALIDATION.md` — not just headline numbers, but *how* they were
  produced and what would make them wrong.
- **Strategy research committee** (senior researchers + desk PMs) reviews
  the methodology, not just the P&L: is the cost model realistic for the
  intended instrument and size, is the train/test split honest, does the
  parameter sensitivity map show a plateau or a spike, has the researcher
  tried to break it (this is where `docs/VALIDATION.md` §8's choppy-
  regime stress test earns its keep — a committee will ask "where does
  this lose money" before "how much does it make").
- **Risk** independently re-derives the key numbers (or re-runs the code)
  rather than trusting the researcher's report — model risk sign-off
  means someone who did not build the model checks it. For this project
  that means re-running `pytest` and `examples/run_pipeline.py` from a
  clean checkout and confirming the numbers in `docs/VALIDATION.md`
  actually reproduce, and independently sanity-checking the cost
  assumption (5 bps) against the instrument's real quoted spread and
  average daily volume.
- **PM / capital allocator** signs off on size relative to the strategy's
  Sharpe, correlation to the existing book, and capacity — and, at a
  systematic pod, on the entire *process* by which parameters get
  re-selected over time (walk-forward cadence, re-fit triggers), not just
  today's parameter choice.

## 2. Pre-launch checklist

1. **Walk-forward validation, not a single split.** `walk_forward_backtest`
   is the minimum bar: does the stitched out-of-sample Sharpe survive
   several independent (as independent as one asset's history allows)
   re-selections, or does it depend on having picked the lucky 70/30
   boundary? A signal that only "works" for one particular train/test
   split does not clear this bar.
2. **Capacity / liquidity check.** The 5 bps cost assumption
   (`docs/METHODOLOGY.md` A2) is calibrated for a liquid large-cap ETF at
   modest size. Before any real allocation: check average daily volume
   at the intended instrument and holding period, and re-run the backtest
   with a cost model that scales with size (this project's flat-bps model
   does not — see `docs/METHODOLOGY.md` §4) if the intended notional is a
   meaningful fraction of ADV.
3. **Deflated Sharpe for multiple testing.** The grid search in this
   project tries `len(fast_range) * len(slow_range)` combinations (up to
   ~7×7=49 in `run_pipeline.py`); the reported in-sample Sharpe is the max
   of ~49 noisy estimates, which is optimistic even before out-of-sample
   decay is measured. A pre-launch checklist item this project does *not*
   implement (documented in `docs/METHODOLOGY.md` §4) is computing the
   Bailey & López de Prado deflated Sharpe ratio, which explicitly
   penalises the number of trials.
4. **Paper-trading period.** Before real capital: run the signal live,
   at target size, against live closes, without funding it. What this
   actually tests that a backtest cannot: does the assumed 5 bps cost
   match realised slippage, are fills at the close actually achievable in
   the intended size, does the data feed used in production match the
   (adjusted) closes the backtest was built on. Promotion criterion: paper
   P&L within roughly one standard error of the backtest's expected daily
   P&L over the paper period, not "paper P&L is positive" (a short paper
   period is itself a small, noisy sample).
5. **Kill-criteria defined in writing before launch**, not improvised
   after a drawdown (see §4 below).

## 3. Live monitoring against the backtest

| Signal | What it means | Action |
|---|---|---|
| Live Sharpe well below the walk-forward OOS Sharpe over a rolling window comparable in length to one walk-forward trading window (here, ~1 year) | Either bad luck within the expected dispersion of a Sharpe-0.6-0.8 strategy, or a regime the backtest didn't cover | Compare live trade-by-trade cost realisation against the 5 bps assumption first (§4, scenario 1) before concluding the *signal* is broken |
| Live trade count diverging sharply from the backtest's trade count over the same window | The live (fast, slow) parameters may not match what's actually deployed, or live data has gaps/adjustments the backtest data didn't | Reconcile the live signal computation against `ma_crossover_signal` on the exact same price series used in production |
| Realised max drawdown exceeds the walk-forward stitched OOS max drawdown (-18.53% on the bundled data) by a wide margin | The live regime may be range-bound/choppy — the documented failure mode (`docs/VALIDATION.md` §8) | Check whether recent price action resembles the choppy synthetic case study (frequent small crossovers, no sustained trend); if so, this is the strategy behaving as documented, not a bug — the question is whether to reduce size, not whether to panic |
| Realised transaction costs consistently above the 5 bps assumption | Slippage assumption was too optimistic for the traded size/liquidity | Re-run the backtest with the realised cost figure; if the edge disappears at realistic costs, the strategy was never viable at this size (§4, scenario 2) |

**Tracking error vs. the backtest.** Because this project's engine is a
pure function of (prices, signal, cost_bps), a live implementation should
be *reconcilable* day by day: given the same closes the live system saw,
`run_backtest` should reproduce the live day's position and P&L exactly.
Any divergence that survives this reconciliation is a live-implementation
bug (wrong close used, wrong execution time, wrong parameters), not a
market phenomenon, and should be fixed before drawing any conclusion about
the signal itself.

**Regime-change triggers.** A rolling (e.g. 60-day) realised-vs-backtest
Sharpe gap beyond some multiple of the backtest's own Sharpe standard
error (Lo, 2002) is a standard trigger for a formal review — not an
automatic kill, but a forcing function to re-run the walk-forward process
on the latest data and ask whether the originally-selected parameters are
still anywhere near the current grid's optimum.

**Kill criteria (illustrative, scaled to a small systematic sleeve):**
drawdown exceeding 1.5× the walk-forward stitched max drawdown ⇒ halve
size; 2.5× ⇒ flat and formal post-mortem before re-entry. These should sit
above the backtest's own max drawdown deliberately (as in the `eq_pairs`
project's risk limits) — hitting them means the live regime is
meaningfully different from anything validated, not that the strategy had
a bad week.

## 4. Scenario playbook

**Scenario: the strategy underperforms its backtest live. First three
things to check, in order.**

1. **Transaction costs.** Pull the actual fill prices and compare realised
   slippage to the assumed 5 bps. This is the single most common gap
   between a backtest and live P&L for a signal this simple — a naive
   live implementation that market-orders at the open instead of
   executing at the close the backtest assumed, or that trades a less
   liquid contract than the one back-tested, can easily realise 15-30 bps
   instead of 5. §7 of `docs/VALIDATION.md` shows exactly how much this
   matters: at the fast (10,50) parameters, Sharpe falls from 0.368
   (0 bps) to 0.222 (20 bps) — a cost miss of this size can turn a
   marginal signal negative.
2. **Regime.** Compare recent realised volatility and trendiness (e.g.
   count of crossovers per unit time) against both the backtest's
   historical regime and the choppy-regime case study in
   `docs/VALIDATION.md` §8. If the market has been range-bound, the live
   underperformance may be the *documented* failure mode operating
   exactly as expected, not a broken model.
3. **Implementation drift.** Reconcile the live position series against
   `ma_crossover_signal`/`run_backtest` run offline on the exact prices
   the live system saw (see "tracking error" above). Check the parameters
   actually deployed match the ones the walk-forward process most
   recently selected — a stale parameter pair from an old fit is a
   common, boring, and fixable cause of underperformance.

**Scenario: a proposed implementation and a naive implementation get
materially different costs because of naive slippage assumptions.** A
naive implementation might assume the historical 5 bps flat cost applies
regardless of order size — fine at the backtest's implied notional, wrong
at 50-100x that notional, where price impact (not spread) dominates and
scales roughly with the square root of participation rate, not linearly.
The proposed (correct) implementation prices trades using a size-aware
cost model (spread + impact, e.g. `cost_bps(size) = base_bps + k *
sqrt(size / ADV)`) and will show materially worse net Sharpe at scale even
though the *signal* is identical — this is a capacity problem, not a
signal problem, and the fix is smaller size or slower execution
(participation-rate-limited orders over the day), not a different signal.
This project's flat-bps cost model (`docs/METHODOLOGY.md` A2, §4) is the
naive version by design — it is adequate for illustrating the accounting
and the cost/Sharpe trade-off, but a real capacity study before sizing up
must replace it.

## 5. Model governance

Every number in `docs/VALIDATION.md` regenerates from
`examples/run_pipeline.py` on a fixed seed — the audit trail is "clone the
repo, run the script, get the same numbers", not a static report that can
drift from the code. Any change to the signal, cost model, or split logic
is a new run archived with its seed and diffed against the previous
numbers; the test suite (`pytest -q`, 64 tests) is the regression gate —
it must stay green for any change, and a change that alters a documented
number (e.g. the walk-forward Sharpe) requires updating
`docs/VALIDATION.md` in the same change, not as a follow-up.
