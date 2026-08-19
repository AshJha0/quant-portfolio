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
pytest -q                        # 64 tests, offline, ~2 s
python examples/run_pipeline.py  # every number below, ~3 s
```

```python
from eq_signal_backtest.data.synthetic import generate
from eq_signal_backtest.signals import ma_crossover_signal
from eq_signal_backtest.engine import run_backtest
from eq_signal_backtest.split import train_test_split, select_best_params

prices = generate().set_index("Date")["Adj Close"]        # bundled, seeded, offline
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

## Headline results (bundled synthetic data, seed=2, 10y)

Parameters selected on the training window only: **fast=20, slow=200**.

| | strategy | buy & hold |
|---|---:|---:|
| In-sample CAGR / Sharpe / maxDD | +7.12% / 0.70 / -11.74% | +7.92% / 0.62 / -26.68% |
| **Out-of-sample** CAGR / Sharpe / maxDD | **+6.20% / 0.64 / -9.77%** | +4.21% / 0.37 / -27.11% |
| Walk-forward stitched OOS (7 windows) | +8.05% / 0.69 / -18.53% | +10.56% / 0.79 / -27.11% |

**The four numbers that matter (see `docs/VALIDATION.md` for full detail
and reproduction):**

1. **In-sample vs out-of-sample Sharpe: 0.70 → 0.64.** Moderate decay —
   picking the `argmax` of a grid search is itself a mild form of
   overfitting (`docs/METHODOLOGY.md` A5), so some decay is expected and
   this is the honest read, not the training number.
2. **Strategy vs buy & hold: drawdown, not just return.** Max drawdown
   -11.74% (strategy) vs -26.68%/-27.11% (buy & hold) across every
   window. This trend-following signal earns its keep by *avoiding deep
   drawdowns*, not by out-compounding a bull market.
3. **Transaction cost drag is parameter-dependent.** At the selected
   slow-turnover parameters, costs shave CAGR by only 0.05pp — but a
   faster (10, 50) pair that looks best cost-free (Sharpe 0.368) falls
   behind the selected pair once realistic 5-20bps costs are applied
   (`docs/VALIDATION.md` §7). A cost-free grid search would pick the
   wrong parameters.
4. **The sensitivity map is a plateau, not a spike.** 37% of the
   in-sample grid's cells sit within 25% of the best Sharpe — evidence
   against pure curve-fitting (see `output/figures/backtest_overview.png`,
   bottom-right panel, after running the pipeline).

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
├── engine.py         # BacktestResult, run_backtest (t-1 lag + costs), performance_stats
├── sensitivity.py    # parameter_grid (Sharpe surface, fast x slow)
├── split.py           # train_test_split, select_best_params,
│                       # WalkForwardWindow/walk_forward_windows,
│                       # WalkForwardResult/walk_forward_backtest
└── data/
    ├── synthetic.py  # deterministic two-regime price generator (used by tests & examples)
    └── live.py       # optional yfinance loader, import-guarded, never used in tests
tests/                # 64 tests incl. the no-lookahead detector and the choppy-regime failure mode
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
