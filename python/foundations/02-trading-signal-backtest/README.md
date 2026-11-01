# Trading Signal Backtest: MA Crossover (`eq_signal_backtest`)

A long/flat moving-average crossover strategy, backtested with a modelled
t+1 fill, transaction costs, an in-sample/out-of-sample split, a
walk-forward variant, a parameter-sensitivity map, a deflated Sharpe ratio,
PBO and a multi-seed distribution of every headline number.

The strategy is simple on purpose. The project is about the evaluation
around it: where it worked, where it failed, what was assumed, and what
would break it.

Every result below comes from a synthetic price generator. No real-data
result exists yet; real data is Phase 4 of the repository plan.

```
close prices ──► ma_crossover_signal (fast MA vs slow MA, 0/1)
                    ──► run_backtest (t+1 fill, cost_bps per trade)
                          ──► performance_stats (CAGR, vol, Sharpe, maxDD)
                                ──► select_best_params (grid search, TRAIN ONLY)
                                      ──► out-of-sample backtest (TEST ONLY)
                                      ──► walk_forward_backtest (rolling
                                          re-selection, stitched OOS curve)
                                      ──► research_stats (Lo SE, deflated
                                          Sharpe, PBO, bootstrap, seeds)
```

## Fill timing

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost. The position held before
the fill earns p_fill / p_t - 1 of day t+1 and the new position earns
p_{t+1} / p_fill - 1. Set it with `ExecutionConfig(mode, fill_fraction,
auction_cost_bp)`; `ExecutionConfig("next", 1.0)` is a next-close fill and
`ExecutionConfig("close_auction", auction_cost_bp=0.0)` reproduces the
arithmetic of version 1.0.0 of this engine.

## Quickstart

```bash
cd python/foundations/02-trading-signal-backtest
pip install -e .[dev,plots]
pytest -q                          # 133 tests, offline, about 40-60 s
python examples/run_pipeline.py    # every number below, about 280 s (100 seeds, 4 workers)
python examples/validation_tables.py   # cost and choppy-regime tables, about 2 s
```

The pipeline writes `report.txt` and figures to a new temp directory, or
to `--out DIR`. It never writes into the repository.

```python
from eq_signal_backtest import ExecutionConfig, run_backtest
from eq_signal_backtest.data.synthetic import generate
from eq_signal_backtest.signals import ma_crossover_signal
from eq_signal_backtest.split import train_test_split, select_best_params

prices = generate().set_index("Date")["Adj Close"]        # synthetic, seed 32, offline
split = train_test_split(prices, train_frac=0.7)
fast, slow, grid = select_best_params(                     # TRAIN ONLY
    split.train, fast_range=range(10, 71, 10), slow_range=range(100, 251, 25)
)
sig = ma_crossover_signal(split.test, fast, slow)
res = run_backtest(split.test, sig, cost_bps=5.0, execution=ExecutionConfig())
print(res.stats["sharpe"], res.stats["benchmark"]["sharpe"])
```

For real market data (needs `pip install eq-signal-backtest[live]` and a
network connection; no result in this repository uses it):

```python
from eq_signal_backtest.data.live import load_prices
prices = load_prices("SPY", start="2016-01-01")
```

## Headline results (synthetic, 100 seeds)

Command: `python examples/run_pipeline.py`. Each seed is a fresh 10-year
synthetic path from the bundled two-regime generator. On each seed the
pipeline selects (fast, slow) on the first 70% by grid search over 49
pairs, trades it on the last 30%, and runs a 7-window walk-forward
(756-day formation, 252-day trading). Costs are 5 bp per trade. Values are
mean [95% CI] across seeds.

| Sharpe (synthetic, 100 seeds) | mean [95% CI] | sd |
|---|---:|---:|
| In-sample, selected pair | 0.67 [0.60, 0.75] | 0.36 |
| Out-of-sample strategy | 0.32 [0.18, 0.47] | 0.73 |
| Out-of-sample buy & hold | 0.43 [0.27, 0.59] | 0.80 |
| Out-of-sample strategy minus buy & hold | -0.11 [-0.21, -0.00] | 0.52 |
| Walk-forward strategy | 0.39 [0.29, 0.49] | 0.51 |
| Walk-forward buy & hold | 0.44 [0.34, 0.54] | 0.50 |
| Walk-forward strategy minus buy & hold | -0.05 [-0.11, 0.00] | 0.28 |

On this synthetic generator the crossover does not beat buy & hold on
Sharpe. It beats it out of sample on 34% of seeds and in walk-forward on
42% of seeds. The in-sample Sharpe of the selected pair overstates the
out-of-sample Sharpe by 0.35 on average, which is the selection bias of
taking the best of 49 noisy estimates. One seed in 100 had a window where
the selected strategy never invested; its undefined Sharpe is scored 0.

### One path in detail (synthetic, seed 32)

Parameters selected on the training window: fast=40, slow=100.

| synthetic, seed 32 | strategy | buy & hold |
|---|---:|---:|
| In-sample CAGR / Sharpe / maxDD | +13.14% / 0.86 / -31.87% | +15.73% / 0.82 / -46.57% |
| Out-of-sample CAGR / Sharpe / maxDD | -3.08% / -0.19 / -27.96% | -7.25% / -0.21 / -46.93% |
| Walk-forward (7 windows) CAGR / Sharpe / maxDD | -1.70% / -0.02 / -45.15% | +0.73% / 0.15 / -46.93% |

| synthetic, seed 32 | value |
|---|---:|
| Out-of-sample Sharpe, Lo/Newey-West SE | 0.56 |
| Out-of-sample Sharpe, block-bootstrap 95% CI | [-1.37, 0.85] |
| Trials logged in the grid search | 49 |
| Deflated Sharpe of the selected in-sample result | 0.892 |
| PBO (CSCV, 16 blocks, 49 configurations) | 0.45 |
| Full-period Sharpe: t+1 fill f=0.5 / next close f=1 / close auction 10 bp | 0.61 / 0.62 / 0.59 |
| Cost drag at the selected pair, 5 bp (24 trades in 10 years) | 0.13% CAGR |

On seed 32 the out-of-sample bootstrap interval contains both zero and the
in-sample 0.86, so one path cannot tell fitted noise from a real edge. The
100-seed table above is the evidence. The deflated Sharpe of 0.892 is
below the usual 0.95 bar, and a PBO of 0.45 means the in-sample winner of
the grid lands at or below the out-of-sample median in 45% of CSCV splits.

The strategy has smaller drawdowns than buy & hold on seed 32 (-27.96%
against -46.93% out of sample) because it steps aside in the synthetic
drawdown regime. A cash allocation gives drawdown control more cheaply.

On a synthetic mean-reverting path (seed 99) the same signal loses -8.13%
CAGR against +1.31% for buy & hold, from whipsaws
(`docs/VALIDATION.md` §8).

## Layout

```
src/eq_signal_backtest/
├── signals.py        # ma_crossover_signal
├── engine.py         # ExecutionConfig, strategy_returns (the one P&L
│                     # definition), run_backtest, performance_stats
├── sensitivity.py    # parameter_grid (Sharpe surface, fast x slow)
├── split.py          # train_test_split, select_best_params,
│                     # walk_forward_windows, walk_forward_backtest
├── research_stats.py # Lo/NW Sharpe SE, deflated Sharpe, PBO, bootstrap,
│                     # seed distributions (shared verbatim across projects)
└── data/
    ├── synthetic.py  # seeded two-regime price generator (tests and examples)
    └── live.py       # optional yfinance loader, import-guarded, unused in tests
tests/                # 133 tests: fill-timing detector, statistics, choppy-regime
                      # failure mode, data-quality and extreme-cost guards
examples/run_pipeline.py      # selection, OOS, walk-forward, DSR, PBO, 100 seeds
examples/validation_tables.py # cost-sensitivity and choppy-regime tables
docs/                 # METHODOLOGY, VALIDATION (all numbers), DESK_GUIDE
```

## Three design decisions

1. **Fill timing is tested, not asserted** (`tests/test_execution.py`). A
   perfect-foresight signal that knows the sign of the next day's return
   earns nothing under a next-close fill. The same signal filled at the
   decision close earns about 37 standard errors on the test path, so the
   test fails on any engine that fills at the close of day t.
2. **Selection bias is measured** (`docs/METHODOLOGY.md` A5). Every grid
   cell is logged as a trial. The deflated Sharpe uses that count, PBO uses
   the grid's return matrix, and the multi-seed run measures the in-sample
   to out-of-sample gap directly.
3. **Costs are always on** (`docs/VALIDATION.md` §7). A cost-free grid
   search favours over-trading pairs. `parameter_grid` and
   `select_best_params` default to 5 bp.

Limitations: one synthetic asset, a flat cost model with no market
impact, long/flat only, cash earns zero.
