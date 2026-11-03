# FX algorithmic trading and execution modelling (`fx_algo`)

Intraday FX signal research and parent-order execution for the OTC dealer
market: no consolidated tape, 24h session liquidity, last-look dealer
streams and the WM/R 4pm London fix.

Every result in this project is computed on synthetic data. The signal edge
is planted by the generator (hourly AR(1) increments with phi = 0.25, far
stronger than any real intraday FX autocorrelation). No real-data result
exists yet; real data arrives in the research platform of the repository plan.

```
Signal generation -> features -> backtest -> transaction costs
  -> slippage -> TWAP and fix benchmarks -> optimal execution (piecewise AC)
  -> TCA
```

## What is FX-specific

- No VWAP. FX has no volume tape, so the benchmarks are arrival price,
  interval TWAP and the WM/R fix. The fix print is computed from market
  observations inside the 5-minute window that are independent of the
  order's fills, and the order's own impact on the print is reported.
- Session liquidity. Spread, depth and vol are hour-of-day step functions
  (Asia, London, London-NY overlap, NY, late). The overlap is tightest and
  deepest. The profile values are stylised, not estimated.
- Dealer market. A last-look venue quotes 40% tighter than the firm ECN but
  holds orders and rejects with probability increasing in the move against
  the dealer.
- Impact. Square-root temporary impact scaled by session vol and depth, plus
  small linear permanent impact. The coefficients, `k_eta` and the AC risk
  aversion are set by hand and are not calibrated.
- Pips. Costs in pips by pair and session; P&L per unit base notional in
  quote currency, pips and base currency; carry at the 21:00 London
  rollover, ACT/365F.

## Execution convention

Signals use data up to the close of period t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost. Set it with
`IntradayBacktester.run(bars, positions, execution=ExecutionConfig(...))`.
Spot FX has no auction; here `close_auction` means a fill at the decision
bar's close mid with an extra immediacy cost (default 0.5 bp, about 0.55
pips on EURUSD).

## Layout

```
src/fx_algo/
  sessions.py          sessions, pair profiles, time grids, fix window
  features.py          bars, momentum, reversion, breakout, carry (causal)
  signals.py           combination, vol targeting, session filter, carry gate
  backtest.py          intraday backtester, ExecutionConfig
  research_stats.py    Lo/Newey-West SE, deflated Sharpe, PBO, block bootstrap
  data/synthetic.py    seeded generator with planted intraday alpha
  execution/
    simulator.py       24h session simulator, impact, last-look and firm venues
    schedulers.py      TWAP, liquidity-weighted, POV analog, fix-targeting
    optimal.py         piecewise Almgren-Chriss (KKT/QP, active set)
    tca.py             IS decomposition, benchmarks, fix print, venue scorecards
```

## Quickstart

```bash
pip install -e ../../core   # qp-core, the shared package (not on PyPI)
pip install -e .[dev]
python -m pytest tests -q          # 185 tests, about 30-45 s, offline, seeded
python examples/run_pipeline.py    # full pipeline, about 25-30 s, writes nothing
```

## Results on synthetic data

All numbers from `python examples/run_pipeline.py`. Signal results are the
mean [95% CI] over 30 synthetic seeds of 60 days each; execution results are
over 200 synthetic replications.

Signal layer, session-filtered vol-targeted mom_1/mom_4 strategy, EURUSD
costs:

| execution variant | net pips per 60 days | annualised Sharpe |
|---|---:|---:|
| next, f = 0.5 (default) | +2079 [+1893, +2266] | +9.42 [+8.68, +10.16] |
| next, f = 1.0 | +803 [+600, +1006] | +3.16 [+2.39, +3.93] |
| close auction, 0.5 bp | +2896 [+2666, +3126] | +10.63 [+9.84, +11.41] |
| EM spreads, f = 0.5 | -3246 [-3489, -3004] | -13.42 [-14.49, -12.34] |

These Sharpe ratios are a property of the planted generator, not a market
claim. The pipeline logs 9 trials; the deflated Sharpe ratio of the seed-0
default run at 9 trials is 0.606 and the PBO over the weight grid is 0.06.

500mm EURUSD parent order, full 24h day, firm ECN:

| schedule | controllable cost (pips) | IS sd (pips) |
|---|---:|---:|
| TWAP | 0.578 | 31.3 |
| liquidity-weighted | 0.517 | 33.5 |
| POV 5% (analog) | 0.709 | 11.5 |
| piecewise AC (lambda = 1e-5) | 0.733 | 7.7 |

Mean IS is not ranked: its 95% CIs overlap for every schedule (VALIDATION.md
section 3).

Last look, paired: effective cost 0.290 vs 0.309 pips for uninformed flow
(last look cheaper by 0.018 [0.016, 0.020]) and 0.584 vs 0.309 for informed
flow (dearer by 0.275 [0.271, 0.280], 41% rejects).

Fix targeting, 100mm: tracking error to an independent fix print +1.015
[+0.940, +1.090] pips with sd 0.54, against sd 9.9 for a 3h TWAP. The
order's own impact moves the print by 0.035 pips. (v2.0.0 reported +0.498
and 0.078: on the 1-minute grid impact was scaled by the bucket vol; see
docs/METHODOLOGY.md section 4.)

Model anchors: piecewise AC equals closed-form AC at constant liquidity
(< 1e-9); lambda -> 0 gives the liquidity-weighted schedule; IS components
sum to the total at 1e-10.

## Documentation

- [docs/METHODOLOGY.md](docs/METHODOLOGY.md): benchmark, impact, AC,
  last-look, execution and statistics choices; assumptions register.
- [docs/VALIDATION.md](docs/VALIDATION.md): anchors, full result tables with
  CIs, failure modes, each with a test.
- [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md): where each component would sit
  on a desk, which controls exist in code and which do not.
