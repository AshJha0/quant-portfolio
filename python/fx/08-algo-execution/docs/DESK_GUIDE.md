# Desk guide: how an FX e-trading desk would use this stack

This file describes where each component would sit on an FX e-trading desk
and which workflow it would serve. Every number it cites is from
VALIDATION.md and is computed on synthetic data; no real-data result exists
yet (see `research/`). Reproduce with `python examples/run_pipeline.py`.

---

## 1. Where this sits

- Client algo suite (agency execution). Banks sell TWAP, POV,
  liquidity-seeking and fix-targeting algos to asset managers and
  corporates. `execution/schedulers.py` and `execution/optimal.py` are the
  scheduling layer; `execution/simulator.py` is a pre-trade cost model.
- Central risk book. The same schedules can work out internalisation
  residuals, with the AC lambda set from the book's risk budget.
- TCA and best execution. `execution/tca.py` produces the post-trade
  decomposition and venue scorecards.

The signal layer (features, signals, backtest) is an intraday overlay. Its
edge in this project is planted by the synthetic generator.

## 2. Workflow

07:00 London, profile check. The `PairProfile` tables hold session spread,
depth and vol. In this project they are stylised inputs; a desk would
re-estimate them from its own streams. That estimation is not implemented.

Pre-trade, per parent order (for example 500mm EURUSD):

1. Choose horizon (24h or London only) and benchmark (arrival, TWAP, fix).
2. Run the scheduler set and show the cost/risk frontier. On synthetic data,
   liquidity weighting saves about 11% of controllable cost over 24h, and
   piecewise AC at lambda = 1e-5 pays 0.22 pips more to cut IS sd from about
   31 to about 8 pips. The impact and AC parameters are not calibrated.
3. Check participation: 500mm in one 5-minute bucket violates the 30% depth
   cap and the simulator raises.

Intraday, venue routing. `venue_comparison` gives quoted half-spread,
effective cost, rejection rate and rejection cost per venue. On synthetic
data, benign flow is 0.018 pips cheaper on last look and alpha-bearing flow
0.275 pips dearer. The informed-flow result follows from the hold-window
assumption (METHODOLOGY.md A4).

17:10 London, fix debrief. `fix_print` and `fix_self_impact_pips` give the
tracking error against an independent print and the order's own impact on
it.

Post-trade. IS decomposition (spread plus temporary, permanent, drift),
slippage against arrival, TWAP and fix, venue attribution.

## 3. Controls

Implemented in code:

- Participation cap per bucket (`max_participation`, 30% of modelled depth);
  a child order above it raises `ValueError`.
- Blackout windows through the `tradeable` mask: schedulers place zero there
  and the simulator refuses fills.
- Input validation on every scheduler, the AC solver and `ExecutionConfig`.
- IS components sum to the total at 1e-10 (tested).

Not implemented (a desk would need them): parent-notional limits per pair
and session, recalibration of `k_temp` and `k_perm` from child-order
markouts, per-dealer last-look parameter estimation, kill switches.

FX Global Code. Signatories commit to transparent last-look disclosure and
no trading on hold-window information (Principle 17), and to fix-order
handling without front-running (Principles 9-12). The TCA outputs here are
the kind of evidence such a review uses.

## 4. Scenarios

Month-end fix. Index rebalancing concentrates large flow into the 5-minute
window. On synthetic data the fix schedule pays about 1 pip and carries
about 0.5 pip of tracking-error sd against an independent print, and its own
permanent impact moves the print by 0.035 pips (VALIDATION section 5). Pre-positioning ahead of the
window is a compliance decision, not a trading one.

Central bank blackout. An ECB-day band with no child orders is a
`tradeable=False` range, the same machinery as the weekend gap. The AC
schedule re-optimises around the hole.

GBP flash crash (7 Oct 2016). The Gaussian simulator cannot produce an 8%
gap. With `vol_scale = 5` on GBPUSD, synthetic controllable cost is more
than 2x and execution risk more than 3x calm conditions. GBPUSD Asia depth
is a fifth of overlap depth in the profile, so liquidity-aware schedules
place little there.

EM local close (USDMXN). Liquidity is NY-centred and Asia spreads are about
30x wider. A 20mm USDMXN order costs more than 20x the equivalent EURUSD
participation in the synthetic simulator.

## 5. What to run

```bash
python -m pytest tests -q          # 185 tests, about 30-45 s, offline
python examples/run_pipeline.py    # full pipeline, about 25-30 s, writes nothing
```

Read METHODOLOGY.md section 9 (assumptions register) before using any
number, especially A5 to A7 and A10 to A12.
