# Trading Signal Backtest — MA Crossover (`eq_signal_backtest`)

A long/flat moving-average crossover strategy, backtested with next-day
execution, transaction costs, an in-sample/out-of-sample split, a
walk-forward variant, and a parameter-sensitivity map — built from first
principles.

The strategy itself is deliberately simple. The point of the project is
the *evaluation discipline*: showing where it worked, where it failed,
what was assumed, and what would break it.

```
close prices ──► ma_crossover_signal (fast MA vs slow MA, 0/1)
                    ──► run_backtest (t-1 execution lag, cost_bps per trade)
                          ──► performance_stats (CAGR, vol, Sharpe, maxDD)
                                ──► select_best_params (grid search, TRAIN ONLY)
                                      ──► out-of-sample backtest (TEST ONLY)
                                      ──► walk_forward_backtest (rolling
                                          re-selection, stitched OOS curve)
                                            ──► parameter_grid (sensitivity
                                                heatmap: plateau vs spike)
```

## Quickstart

```bash
cd python/foundations/02-trading-signal-backtest
pip install -e .[dev,plots]
pytest -q                        # 99 tests, offline, ~3 s
python examples/run_pipeline.py  # every number below, ~3 s
```

```python
from eq_signal_backtest.data.synthetic import generate
from eq_signal_backtest.signals import ma_crossover_signal
from eq_signal_backtest.engine import run_backtest
from eq_signal_backtest.split import train_test_split, select_best_params

prices = generate().set_index("Date")["Adj Close"]        # bundled, seeded (32), offline
split = train_test_split(prices, train_frac=0.7)
fast, slow, grid = select_best_params(                     # TRAIN ONLY
    split.train, fast_range=range(10, 71, 10), slow_range=range(100, 251, 25)
)
sig = ma_crossover_signal(split.test, fast, slow)
res = run_backtest(split.test, sig, cost_bps=5.0)          # TEST ONLY — the honest number
print(res.stats["sharpe"], res.stats["benchmark"]["sharpe"])
```

For real market data (needs `pip install eq-signal-backtest[live]` and a
network connection):

```python
from eq_signal_backtest.data.live import load_prices
prices = load_prices("SPY", start="2016-01-01")
```

## Headline results (bundled synthetic data, seed=32, 10y)

Parameters selected on the training window only: **fast=10, slow=125**.

| | strategy | buy & hold |
|---|---:|---:|
| In-sample CAGR / Sharpe / maxDD | +12.39% / 0.87 / -30.06% | +15.73% / 0.82 / -46.57% |
| **Out-of-sample** CAGR / Sharpe / maxDD | **-2.27% / -0.12 / -26.50%** | -7.25% / -0.21 / -46.93% |
| Walk-forward stitched OOS (7 windows) | -2.26% / -0.06 / -43.93% | +0.73% / 0.15 / -46.93% |

**The headline is a negative result, and it is the point of the project.**
An in-sample Sharpe of 0.87 became **-0.12** out-of-sample, and a
walk-forward run that re-selected parameters every year did no better
(-0.06). The evaluation machinery works: it caught a strategy that looked
good in the window it was fitted to and did not survive contact with the
window it wasn't. A project that reported only the in-sample number would
have shown a "0.87 Sharpe trend-following strategy" — which is exactly the
failure mode this repository exists to demonstrate.

**The four numbers that matter (see `docs/VALIDATION.md` for full detail
and reproduction):**

1. **In-sample vs out-of-sample Sharpe: 0.87 → -0.12.** Substantial
   decay: essentially all of the in-sample edge was fitted noise. Picking
   the `argmax` of a grid of noisy Sharpe estimates is itself a mild form
   of overfitting (`docs/METHODOLOGY.md` A5), and here it is the whole
   result.
2. **Strategy vs buy & hold: drawdown, not just return.** Max drawdown
   -30.06% in-sample / -26.50% out-of-sample (strategy) against -46.57% /
   -46.93% (buy & hold). This is the one thing the signal does reliably in
   both windows: it steps aside during sustained declines. Notably the
   out-of-sample *Sharpe* is still better than buy & hold's (-0.12 vs
   -0.21) — over a losing period, losing less is a real property, just not
   one you can call alpha.
3. **Transaction cost drag is parameter-dependent.** At the selected
   parameters (32 trades over 10 years), costs shave 0.17pp off CAGR. At
   the fastest grid pair (10, 50) — 55 trades — the drag is 1.21pp at
   20 bps (`docs/VALIDATION.md` §7). A cost-free grid search
   systematically flatters over-trading parameters.
4. **A plateau in the in-sample grid is not evidence of an edge.** 39% of
   the in-sample grid's cells sit within 25% of the best Sharpe, which
   *looks* like the classic robustness signature — and is nonetheless
   consistent with the whole plateau being over-fitted, as the
   out-of-sample number shows. The pipeline now says this explicitly in
   its own output rather than letting the heatmap imply otherwise.

**Regime dependence, stated plainly, and reproduced with a seeded
example:** this strategy class does well in sustained trends and
prolonged bear markets (it steps aside) and **loses money in choppy,
range-bound markets** — `docs/VALIDATION.md` §8 constructs a
mean-reverting synthetic path (seed 99) where the same signal loses
-8.59% CAGR against a roughly flat buy & hold, purely from whipsaws.

## Layout

```
src/eq_signal_backtest/
├── signals.py       # ma_crossover_signal
├── engine.py         # strategy_returns (the one P&L definition), run_backtest,
│                     # BacktestResult, performance_stats
├── sensitivity.py    # parameter_grid (Sharpe surface, fast x slow)
├── split.py           # train_test_split, select_best_params,
│                       # WalkForwardWindow/walk_forward_windows,
│                       # WalkForwardResult/walk_forward_backtest
└── data/
    ├── synthetic.py  # deterministic two-regime price generator, calibrated to
    │                 # realistic equity statistics (used by tests & examples)
    └── live.py       # optional yfinance loader, import-guarded, never used in tests
tests/                # 99 tests incl. the no-lookahead detector, the choppy-regime
                      # failure mode, and the data-quality/extreme-cost guards
examples/run_pipeline.py  # data -> selection -> OOS backtest -> walk-forward -> sensitivity -> report + figures
docs/                 # METHODOLOGY, VALIDATION (all numbers), DESK_GUIDE
```

## The three design decisions worth reading about

1. **No-look-ahead by construction** (`docs/VALIDATION.md` §1): the
   engine executes yesterday's signal at today's close
   (`position = signal.shift(1)`), and a detector test engineers a price
   jump where same-day execution is profitable and the honest engine must
   not capture it.
2. **Selection bias is contained, not eliminated**
   (`docs/METHODOLOGY.md` A5-A6): `select_best_params` only ever sees the
   training slice it is given; the out-of-sample and walk-forward numbers
   exist specifically because the in-sample argmax cannot be trusted on
   its own.
3. **A cost model that changes which parameters win**
   (`docs/VALIDATION.md` §7): fixed-bps transaction costs are cheap to
   implement and easy to get wrong by omission — running the parameter
   grid cost-free (as a first draft of this project once did, informally)
   silently favours over-trading parameter pairs that lose to realistic
   costs.

Docs answer the full portfolio contract: model choice vs. alternatives
(momentum, mean-reversion, ML classifier), an assumptions register with
what-breaks-if-violated for every design decision, validation evidence
with reproducible numbers, failure modes (choppy regime, cost
misspecification, selection bias) with a worked example, and desk
workflow (pre-launch checklist, live monitoring, kill criteria, real
scenarios).
