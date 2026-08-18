# FX Algorithmic Trading & Execution Modeling (`fx_algo`)

Intraday FX signal generation and parent-order execution, built for the OTC
dealer market as it actually is — **no consolidated tape, 24h session
liquidity, last-look dealer streams, and the WM/R 4pm London fix** — rather
than renamed equity market structure.

```
Signal Generation → Feature Engineering → Backtesting → Transaction Costs
      → Slippage → TWAP / fix benchmarks → Optimal Execution (piecewise AC)
      → Performance Analysis (TCA)
```

## What is genuinely FX here

- **No VWAP.** FX has no volume tape, so benchmarks are arrival price,
  interval TWAP and the **WM/R fix** (modeled post-2015: 5-minute window
  TWAP; the 2013 fix-rigging scandal and why benchmark gaming matters are
  documented). "POV" participates in *modeled* session depth (POV-analog).
- **Session liquidity.** Spread/depth/vol are hour-of-day step functions
  (Asia / London / **London–NY overlap** / NY / late); the overlap is
  tightest (0.2 pip EURUSD) and deepest — and the schedulers exploit it.
- **Dealer market.** A last-look venue quotes 40% tighter than the firm ECN
  but holds orders and rejects with probability monotonically increasing in
  the move against the dealer; rejects resubmit at worse prices. The trap is
  measured, both sides.
- **Impact.** Square-root temporary impact scaled by session vol and depth +
  small linear permanent impact; internalisation explicitly discussed as an
  ignored upper-bound assumption.
- **Pips everywhere.** Costs in pips by pair/session (EURUSD 0.2–1 pip; EM
  10–250 pips), P&L per unit base notional in quote ccy, pips and base ccy;
  overnight carry accrued at the 21:00-London rollover, ACT/365F.

## Layout

```
src/fx_algo/
├── sessions.py              # sessions, pair profiles, time grids, fix window
├── features.py              # bars from ticks-lite, momentum/reversion/breakout/carry (PIT-enforced)
├── signals.py               # combination, vol targeting, session filter, carry gate
├── backtest.py              # event-driven intraday backtester (session spreads, carry, no-lookahead)
├── data/synthetic.py        # seeded multi-session generator, planted intraday alpha, daily carry panel
└── execution/
    ├── simulator.py         # 24h session sim: sqrt+permanent impact, last-look & firm venues
    ├── schedulers.py        # TWAP, liquidity-weighted, POV-analog, fix-targeting
    ├── optimal.py           # piecewise Almgren-Chriss (KKT/QP + active set) vs closed form
    └── tca.py               # exact IS decomposition, benchmarks, venue scorecards
```

## Quickstart

```bash
pip install -e .                   # numpy, scipy, pandas
python -m pytest tests -q          # 121 tests, ~1 s, offline, seeded
python examples/run_pipeline.py    # full pipeline, ~1 s
```

## Headline numbers (from `examples/run_pipeline.py`, all seeded)

**Signals** (60 days, planted hourly AR(1) φ=0.25): 1h-momentum IC 0.251
(t = 9.8), ≈ 0 on noise (t = 0.12). Session-filtered vol-targeted backtest:
net **+3490 pips** under EURUSD costs; the *same* alpha nets **−974 pips**
under EM-style spreads — costs flip profitability.

**500mm EURUSD parent order** (200 replications, 5-min buckets, firm ECN):

| schedule (24h day) | ctrl cost (pips) | IS std (pips) |
|---|---|---|
| TWAP | 0.578 | 31.2 |
| liquidity-weighted | **0.517** (−11%) | 33.5 |
| piecewise-AC (λ=1e-5) | 0.733 | **7.7** |

**Last-look trap** (paired, common random numbers): quoted ½-spread 0.107 vs
0.178 pips (firm), but effective cost **0.290 vs 0.309** for uninformed flow
(last-look wins) and **0.584 vs 0.309** for informed flow (rejects 41%,
last-look loses by 0.275 pips, SE 0.0024).

**Fix targeting** (100mm, 1-min grid): tracking error to the fix print
0.51 ± **0.00** pips vs 0.24 ± **9.84** pips for a 3h TWAP.

**Model anchors**: piecewise-AC = closed-form AC at constant liquidity
(<1e-9); λ→0 ⇒ liquidity-weighted schedule (exact); IS components sum to
total at 1e-10.

## Documentation contract

- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — why these benchmarks/models
  vs alternatives; 11-item assumptions register with "what breaks".
- [docs/VALIDATION.md](docs/VALIDATION.md) — analytic anchors, scheduler and
  venue tables, failure modes (fix-gaming history, GBP flash regime,
  liquidity mirage, EM cost regimes) each with a test.
- [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md) — client algo suite, dealer
  scorecards, TCA reviews, FX Global Code obligations, month-end fix / CB
  blackout / flash-crash / EM-close scenarios.
