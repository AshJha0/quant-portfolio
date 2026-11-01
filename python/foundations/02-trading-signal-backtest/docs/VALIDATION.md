# Validation: MA-Crossover Signal Backtest

How the implementation was checked. Every result here is from synthetic
data; no real-data result exists yet (Phase 4).

Commands:

| command | produces | runtime (measured) |
|---|---|---|
| `pytest -q` | 133 tests, offline | 40-60 s |
| `python examples/run_pipeline.py` | §6 and §6a (100 seeds, 4 worker processes) | about 280 s |
| `python examples/validation_tables.py` | §7 and §8 tables | about 2 s |

Runtimes were measured on a shared Windows machine with other jobs running.

---

## 1. Fill timing

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost.

`tests/test_execution.py`:

1. **Hand ledger.** Closes 100, 100, 110, 121 and a buy decision at the
   close of day 1: day 2 returns exactly `110 / (100 + f x 10) - 1` for
   f = 0.25, 0.5 and 1, and day 3 returns 10%. An exit decided at 110 and
   filled at f = 0.5 keeps `115.5 / 110 - 1` of day 3.
2. **Perfect-foresight detector.** A signal equal to the sign of the next
   day's return, 3,000 synthetic days. With f = 1 its mean gross return is
   within 4 standard errors of zero. The same signal filled at the decision
   close earns about 37 standard errors, and the test asserts it would earn
   more than 20. Version 1.0.0 of this engine filled at the decision close;
   run against that engine the detector gives t = 36.8 and the test fails.
3. **Exact auction cost.** Close-auction mode at 10 bp costs exactly
   `trades x 10 / 10_000` more than at 0 bp, on every day.
4. **Input checks.** Unknown mode, f outside (0, 1], non-finite f, and
   negative or infinite auction cost raise `ValueError`.
5. **Property test (hypothesis).** An always-long signal ends at exactly
   `p_T / p_fill` for any f in [0.05, 1] and any seed.

`tests/test_engine.py::TestNoLookAhead` keeps the older structural checks:
the executed position equals the previous day's signal, the first position
is 0.0 and never `NaN`, a 30% jump on the signal day is not captured, and a
signal that fires only on the last day never produces a position.

## 2. Transaction-cost accounting

`tests/test_engine.py::TestTransactionCosts`: a single position change
costs exactly `cost_bps / 10_000` (`abs=1e-12`); no change means identical
costed and cost-free curves; a 3-trade round trip charges 3 costs. With an
always-on signal and zero cost, close-auction mode at 0 bp reproduces buy
and hold exactly, and the default t+1 fill equals buy and hold divided by
`1 + 0.5 r_1` (it misses half of the first day).

## 3. performance_stats against hand-computed curves

| Statistic | Construction | Check |
|---|---|---|
| CAGR | 252 days of a constant return solved from a 20% CAGR | `rel=1e-9` |
| Max drawdown | equity `[1.0, 1.2, 0.9, 1.1]` gives -0.25 | `1e-9` |
| Sharpe | alternating `mu+d` / `mu-d` returns, closed-form mean and sd | `rel=1e-9` |
| Sharpe, degenerate | all-zero returns and a single observation | `NaN` |

A repeated non-zero decimal such as `[0.001] * 20` does not give exactly
zero variance in floating point, so the zero-variance test uses literal
`0.0` returns, which is what an all-flat signal produces.

## 4. Research statistics

`tests/test_research_stats.py` (shared verbatim across the alpha projects):

| check | tolerance and why |
|---|---|
| Lo SE with lag 0 equals `sqrt((1 + s^2/2)/n)` on a Gaussian sample | 1% relative, n = 200,000 |
| Newey-West SE matches the Monte Carlo sd of the Sharpe under AR(1) phi = 0.4 | 10% relative; MC sd has 2.2% relative SE at 1,000 reps |
| Expected maximum Sharpe matches simulation at N = 50 | 3% relative (asymptotic formula) |
| Deflated Sharpe is uniform under the null: P(DSR > 0.95) = 5% | within 2 points; 1,500 draws, SE 0.56 points |
| PBO averages near 0.5 on noise and near 0 with one skilled column | mean of 30 noise matrices in [0.38, 0.62] |
| Block-bootstrap 95% interval covers the true mean | at least 89% of 200 trials |
| Seed-distribution interval equals scipy's t interval | exact |
| Non-finite and constant inputs raise `ValueError` | |
| Sharpe and its SE are scale invariant (hypothesis) | `rel=1e-9` |

## 5. Parameter grid and walk-forward hygiene

`tests/test_sensitivity.py` checks the grid's index and columns, the
handling of `fast >= slow` cells, and that a cell equals a direct
`run_backtest` call.

`tests/test_split.py`:

- Formation and trading windows cannot overlap; default trading windows
  are contiguous.
- `test_frozen_parameters_reproduce_window_returns_exactly` rebuilds the
  first trading window's daily returns from the recorded `(fast, slow)` and
  the price context with an explicit loop over the fill formula, and
  matches the engine to `1e-10`.
- A too-short sample raises `ValueError` naming the sizes.
- The stitched curve has every out-of-sample day once, and its final
  equity equals the product of the per-window gross returns.
- A formation window too short for any slow average raises instead of
  selecting an all-`NaN` grid.

`tests/test_pipeline_hygiene.py`: the pipeline's default output directory
is outside the repository, and its data label names the seed the generator
actually uses (32).

## 6. Headline: multi-seed distribution (synthetic, 100 seeds)

`python examples/run_pipeline.py`, section 6. Seeds 1 to 100 of the
bundled two-regime generator, 2,520 days each, 70/30 split, grid
`fast` in {10, ..., 70}, `slow` in {100, ..., 250} (49 pairs), 5 bp, default
t+1 fill, walk-forward 756/252 days. Mean [95% CI] across seeds.

| Sharpe (synthetic, 100 seeds) | mean [95% CI] | sd |
|---|---:|---:|
| In-sample, selected pair | 0.67 [0.60, 0.75] | 0.36 |
| Out-of-sample strategy | 0.32 [0.18, 0.47] | 0.73 |
| Out-of-sample buy & hold | 0.43 [0.27, 0.59] | 0.80 |
| Out-of-sample strategy minus buy & hold | -0.11 [-0.21, -0.00] | 0.52 |
| Walk-forward strategy | 0.39 [0.29, 0.49] | 0.51 |
| Walk-forward buy & hold | 0.44 [0.34, 0.54] | 0.50 |
| Walk-forward strategy minus buy & hold | -0.05 [-0.11, 0.00] | 0.28 |

| synthetic, 100 seeds | share |
|---|---:|
| Out-of-sample strategy Sharpe above buy & hold | 34% |
| Walk-forward strategy Sharpe above buy & hold | 42% |

On this synthetic generator the crossover does not beat buy and hold. The
selected in-sample Sharpe exceeds the out-of-sample Sharpe by 0.35 on
average. One seed had a window where the selected strategy never invested;
its undefined Sharpe is scored 0, which is what it earned.

## 6a. One path: seed 32 (synthetic)

`python examples/run_pipeline.py`, sections 1 to 5. Seed 32 is the
default of `generate()`. Its buy-and-hold path is near the generator's
central case. It is a worked example, not evidence.

Parameters selected on the training window: fast=40, slow=100.

| synthetic, seed 32 | strategy | buy & hold |
|---|---:|---:|
| In-sample CAGR / Sharpe / maxDD | +13.14% / 0.86 / -31.87% | +15.73% / 0.82 / -46.57% |
| Out-of-sample CAGR / Sharpe / maxDD | -3.08% / -0.19 / -27.96% | -7.25% / -0.21 / -46.93% |
| Full period, t+1 fill f = 0.5 (24 trades) | +9.05% / 0.61 / -31.87% | |
| Full period, zero cost | +9.18% / 0.62 / -31.80% | |
| Full period, next-close fill f = 1 | +9.13% / 0.62 / -32.26% | |
| Full period, close auction 10 bp | +8.70% / 0.59 / -31.62% | |
| Walk-forward, 7 windows (22 trades) | -1.70% / -0.02 / -45.15% | +0.73% / 0.15 / -46.93% |

| synthetic, seed 32 | value |
|---|---:|
| Out-of-sample Sharpe, Lo/Newey-West SE | 0.56 |
| Out-of-sample Sharpe, block-bootstrap 95% CI | [-1.37, 0.85] |
| Trials logged | 49 |
| Deflated Sharpe of the selected in-sample result | 0.892 |
| PBO (CSCV, 16 blocks, full sample) | 0.45 |
| In-sample grid Sharpe range | [0.18, 0.86] |
| Grid cells within 25% of the best | 39% |

Walk-forward window Sharpes on seed 32 were -0.83, 1.65, 1.14, -1.59,
0.43, -0.53 and -0.88, with the selected slow window moving between 100 and
200. The out-of-sample bootstrap interval contains both zero and the
in-sample 0.86. A plateau in the in-sample grid (39% of cells within 25%
of the best) says the result does not hang on one cell. It says nothing
about out-of-sample performance.

## 7. Cost sensitivity (synthetic, seed 32)

`python examples/validation_tables.py`. Full period, default t+1 fill.

| (fast, slow) | trades (10y) | Sharpe 0 bp | Sharpe 5 bp | Sharpe 20 bp | CAGR 0 bp | CAGR 5 bp | CAGR 20 bp |
|---|---:|---:|---:|---:|---:|---:|---:|
| (10, 50) | 55 | 0.719 | 0.701 | 0.647 | 10.47% | 10.16% | 9.26% |
| (10, 100) | 40 | 0.598 | 0.585 | 0.545 | 8.27% | 8.06% | 7.41% |
| (40, 100), selected | 24 | 0.622 | 0.614 | 0.592 | 9.18% | 9.05% | 8.66% |
| (60, 250) | 12 | 0.062 | 0.058 | 0.046 | -0.24% | -0.30% | -0.48% |

Going from 0 to 20 bp costs the 55-trade pair 1.21 points of CAGR and the
12-trade pair 0.24 points. On this synthetic path costs do not change the
ranking. A cost-free grid search is still biased toward over-trading, so
the grid always runs with costs.

## 8. Failure mode: choppy, range-bound prices (synthetic)

`tests/test_edge_cases.py::TestChoppyRegimeFailureMode` and
`python examples/validation_tables.py`. A mean-reverting synthetic log
price, `x_t = x_{t-1} - 0.05 x_{t-1} + 0.012 eps_t`, seed 99, 750 days,
traded with `ma_crossover_signal(fast=20, slow=100)` and 5 bp:

| synthetic, seed 99 | strategy | buy & hold |
|---|---:|---:|
| Trades | 14 | |
| CAGR | -8.13% | +1.31% |
| Sharpe | -0.57 | 0.16 |
| Max drawdown | -24.40% | -18.16% |

Each crossover enters just before the move reverses. At zero cost the
strategy still loses (-7.91% CAGR), so this is a signal failure, not a cost
failure. The tests assert that the strategy trades at least 5 times,
underperforms buy and hold on CAGR and Sharpe, and is worse with costs.

## 9. Failure modes summary

- **Selection bias:** the selected in-sample Sharpe exceeds out-of-sample
  by 0.35 on average over 100 synthetic seeds (§6); deflated Sharpe and
  PBO quantify it on one path (§6a).
- **Regime dependence:** loses on range-bound synthetic prices (§8).
- **Cost sensitivity:** fast pairs lose more to costs (§7).
- **One synthetic asset:** nothing here says how the signal behaves on
  real data or across assets.

## 10. Data-quality and degenerate-input guards

Each is a test in `tests/test_edge_cases.py`:

| Input | Behaviour |
|---|---|
| `NaN` or `inf` in prices | `ValueError`; a gap must be resolved before backtesting |
| A zero or negative price | `ValueError` ("strictly positive") |
| Signal index different from price index | `ValueError` ("share the exact index") |
| Signal containing `NaN`, `0.5`, `-1` or `2` | `ValueError` (long/flat only) |
| `cost_bps` negative or `NaN` | `ValueError` |
| `fast`/`slow` below 1 or non-integer | `ValueError` naming the window |
| Bad `ExecutionConfig` | `ValueError` (§1) |

**Extreme costs** (`TestExtremeTransactionCosts`): the daily return is
floored at -1.0, so a 200%-per-leg cost wipes equity to zero and zero is
absorbing; a 50%-per-leg cost leaves equity positive; final equity is
non-increasing in `cost_bps` from 0 to 20,000 bp.

**Windows longer than the sample** give an all-flat signal and constant
equity. An all-flat signal produces zero trades; a single-observation
series does not crash.
