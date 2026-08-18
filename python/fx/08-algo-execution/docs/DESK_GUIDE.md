# Desk Guide — how an FX e-trading / algo desk runs this

This is the practitioner's view: who uses each component, in which daily
workflow, under which controls. The numbers referenced are from
VALIDATION.md (reproduce with `python examples/run_pipeline.py`).

---

## 1. Where this sits on a desk

An FX e-trading business has three consumers of exactly this stack:

- **Client algo suite** (agency execution): the bank sells TWAP, POV,
  liquidity-seeking and fix-targeting algos to asset managers and corporates.
  `execution/schedulers.py` + `execution/optimal.py` are the scheduling
  brain; `execution/simulator.py` is the pre-trade cost model behind the
  "expected cost / risk" numbers shown to the client before they press go.
- **Central risk book / principal desk**: uses the same schedules to work out
  of internalisation residuals, with the AC λ set by the book's risk budget.
- **TCA & best execution** (`execution/tca.py`): post-trade reports to
  clients and to the internal best-execution committee.

The signal layer (`features/signals/backtest`) is the systematic overlay:
intraday momentum/reversion/breakout with a carry gate, sized to a vol
target, used either as a prop overlay or to time the execution of known
parent flow.

## 2. Daily workflow

**07:00 London — profile check.** Recalibrate/eyeball session spread, depth
and vol profiles against yesterday's streams (the `PairProfile` tables).
Event calendar loaded: CB decisions, US data, month-end fix flags.

**Pre-trade, per parent order.** Client wants 500mm EURUSD:
1. Choose horizon (24h vs London-only) and benchmark (arrival, TWAP, fix).
2. Run the scheduler set; show the client the cost/risk frontier — the
   VALIDATION.md §2 table *is* that pitch: liquidity-weighted saves ~11% of
   controllable cost over 24h; piecewise-AC at λ=1e-5 pays +0.22 pips to cut
   IS risk from ~31 to ~8 pips. The client's risk preference picks the point.
3. Check participation caps: 500mm in one 5-min bucket violates the 30%
   depth cap (the tool refuses); the schedule must span sessions.

**Intraday — venue routing.** Child orders route across firm ECNs and
last-look streams. The router's dealer scorecard is `venue_comparison`:
quoted ½-spread, *effective* cost, rejection rate, rejection cost. Rule of
thumb from §3 of VALIDATION.md: benign flow may use last-look pools
(−0.018 pips); alpha-bearing flow must pay up for firm liquidity (last-look
would cost +0.275 pips). Flow is tagged accordingly.

**17:10 London — fix debrief** (fix days): tracking error of fix orders vs
the WM/R print, window participation, any impact footprint inside the window
(compliance reviews this — see §4).

**Post-trade / weekly.** TCA pack per client: IS decomposition
(spread+temporary vs permanent vs drift), slippage vs arrival/TWAP/fix,
venue attribution, rejection-cost league table per dealer. Dealers with
rising reject rates on adverse moves get a Global-Code-Principle-17
conversation or a routing downweight.

## 3. Controls and governance

- **Best execution (FX Global Code).** The Code (2017, updated 2021) is a
  voluntary-but-expected standard; signatories commit inter alia to
  transparent last-look disclosure (Principle 17), no trading on hold-window
  information (Principle 17), and fix-order handling without front-running
  (Principles 9–12). The TCA outputs here are the evidence pack: effective
  cost per venue, rejection attribution, fix tracking error.
- **Limits.** Participation cap per bucket (30% of modeled depth here),
  parent-notional limits per pair/session, blackout windows around CB
  announcements and major data (the simulator deliberately contains no such
  events — assumption A6 — so blackouts are a *control*, not a model output).
- **Model governance.** The AC optimiser is anchored to closed form
  (VALIDATION.md §1) — that is the recurring model-validation test; impact
  parameters `k_temp, k_perm` are recalibrated from realised child-order
  markouts; the last-look parameters per dealer come from reject-rate
  regressions on hold-window moves (the monotone logistic is the model
  validation team's null).
- **P&L attribution.** The backtest ledger separates gross, spread cost,
  carry — reviewed daily; execution desks attribute IS into the same exact
  components the TCA produces (they must sum — enforced at 1e-10).

## 4. Realistic scenarios

**Executing around the 4pm fix at month-end.** Month-end index rebalancing
concentrates billions into the 5-minute window. The fix algo (TWAP inside
the window) gives ~zero tracking error at 0.51 pips of controllable cost
(VALIDATION.md §4). The desk's judgment call: a client benchmarked to the
fix accepts that cost; pre-positioning ahead of the window to "save" it is
exactly the behaviour that produced the 2013 scandal and is now a
compliance, not a trading, decision. Impact inside the window moves the
print itself — the simulator reproduces this, which is why fix-window
participation is monitored.

**CB announcement blackout.** ECB day: no child orders 13:40–14:05 London.
Operationally a `tradeable=False` band, identical to the weekend-gap
machinery (schedulers place zero, simulator refuses fills). The AC schedule
re-optimises around the hole; the cost of the blackout is the difference in
the objective — quotable to the client.

**GBP flash crash (7 Oct 2016, ~00:07 London).** Cable lost ~8% in minutes
in Asia hours: stop cascades, one large misfiring algo, liquidity withdrawal.
Desk lessons encoded here: (i) GBPUSD Asia depth is 1/5 of overlap depth in
the profile, so liquidity-aware schedules place little there anyway;
(ii) the `vol_scale=5` regime shows controllable cost >2× and execution risk
>3× (test-locked) — pre-trade models must be re-run, not interpolated, in
stress; (iii) last-look reject rates spike exactly when you need fills —
firm-liquidity share should rise in stress, not fall.

**EM local close (USDMXN).** Liquidity is NY-centred; the Asia session is
30× wider in spread. A 20mm USDMXN order costs >20× the equivalent EURUSD
participation (test-locked); anything left to execute after the local close
should wait for the next NY morning rather than cross late-session spreads —
in schedule terms, the liquidity-weighted schedule already allocates ~0 to
those hours, and the depth cap refuses what a naive TWAP would attempt.

**Informed-flow routing.** A prop momentum signal (the signal layer here)
produces flow with positive short-horizon markout — precisely the flow
last-look dealers reject. The routing table in VALIDATION.md §3 is the
quantitative version of "alpha pays for firm liquidity".

## 5. What a reviewer should run

```bash
python -m pytest tests -q          # 121 tests, ~1 s, offline
python examples/run_pipeline.py    # full pipeline with the tables above
```

Then read METHODOLOGY.md §8 (assumptions register) before trusting any
number in production — especially A5–A7 (no event gaps, no internalisation).
