# Equity Algorithmic Trading and Execution Modelling (`eq_algo`)

A two-layer equity research stack on synthetic data. The alpha layer decides
what to hold each day. The execution layer decides how to trade into it
during the next session.

```
ALPHA LAYER (daily)
  data/synthetic.py --> features.py --> signals.py --> backtest.py --> evaluation.py
  planted-alpha panel   PIT features    IC-weighted     L/S deciles     IC / NW t-stats
                        mom, reversal,  combination,    t+1 fills,      research_stats:
                        vol, RSI, ...   decay, deciles  linear + sqrt   Lo SE, DSR, PBO,
                                                        impact, borrow  bootstrap, seeds

EXECUTION LAYER (intraday)
  intraday.py ---------> benchmarks.py ---------> almgren_chriss.py --> tca.py
  U-shaped volume,       VWAP/TWAP/arrival,       closed-form optimal   Perold IS
  sqrt temp impact on    TWAP/VWAP/POV            trajectories, cost/   decomposition,
  bucket volume, linear  schedulers               variance frontier     attribution
  permanent impact
```

## What the results mean

The synthetic generator plants next-day returns on z(momentum 12-1) and
z(reversal 21d) (`src/eq_algo/data/synthetic.py`). The signal the book
trades combines those same regressors. Every alpha number below is therefore
a property of the generator, not evidence of a tradable edge. The planted
daily IC of about 0.03 is far larger than anything published for real
momentum. No real-data result exists yet; that work is Phase 4 of the
repository plan. The stage of this code is a study portfolio.

## Execution convention

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost.

`ExecutionConfig(mode="next", fill_fraction=0.5, auction_cost_bp=10.0)` is
the default for `run_backtest`. `mode="close_auction"` is the alternative.
`tests/test_execution_convention.py` fails if a decision at t is filled at
t's close in the default mode.

## Quickstart

```bash
cd python/equity/08-algo-execution
pip install -e .[dev]
pytest -q                          # 213 tests, 47-97 s measured, offline, seeded
python examples/run_pipeline.py    # full demo, 175-247 s measured, writes nothing
```

## Results (synthetic data, `python examples/run_pipeline.py`)

### Headline book across 20 seeds

Combined signal (IC-weighted momentum 12-1, momentum 6-1, reversal 1m),
150 synthetic stocks x 1,250 days, $200m AUM, 5 bp linear cost, square-root
impact with Y = 0.7, 50 bp a year borrow on the short book, 0.25-z
rebalance band, default t+1 fill. Seeds 0 to 19. Mean [95% CI] across
seeds, synthetic data:

| statistic | mean [95% CI] | sd across seeds |
|---|---|---:|
| mom_12_1 daily IC | 0.0325 [0.0315, 0.0335] | 0.0021 |
| gross Sharpe | 5.05 [4.80, 5.31] | 0.55 |
| net Sharpe | 0.28 [-0.05, 0.60] | 0.70 |
| net Sharpe, close-auction fill (+10 bp) | -0.34 [-0.70, 0.01] | 0.76 |
| gross return (% a year) | 43.1 [41.0, 45.2] | 4.5 |
| net return (% a year) | 2.3 [-0.5, 5.1] | 6.0 |
| cost and borrow drag (bp a year) | 4,075 [3,852, 4,298] | 476 |
| daily one-way turnover (%) | 22.3 [21.2, 23.5] | 2.5 |

At the cited impact coefficient the planted 43% gross return is almost all
paid away in costs on synthetic data. The net Sharpe CI includes zero.

### Seed 42 in detail (synthetic)

| statistic | value |
|---|---:|
| net Sharpe | 0.93 |
| Lo / Newey-West SE | 0.48 |
| block-bootstrap 95% CI | [0.02, 1.84] |
| logged trials (8 signals, 2 execution variants) | 10 |
| deflated Sharpe, null spread 1/(n-1) | 0.59 |
| deflated Sharpe, spread of the logged trial Sharpes | 0.00 |
| PBO (CSCV, 16 blocks, 8 signal configurations) | 0.00 |

PBO is zero because momentum 12-1 is the planted regressor and wins in every
split of synthetic data. That says the generator is strong, not that the
selection is safe on real data.

### Capacity (synthetic, seed 42, full backtest per AUM)

Net Sharpe by AUM and impact coefficient Y, per-name trailing 21-day ADV:

| Y | $10m | $100m | $1bn | $5bn |
|---|---:|---:|---:|---:|
| 0.3 | 4.34 | 3.56 | 1.08 | -3.21 |
| 0.7 | 3.86 | 2.03 | -3.53 | -11.21 |
| 1.0 | 3.50 | 0.89 | -6.61 | -14.90 |

The 95th-percentile name-day participation is 0.05% of ADV at $10m and 24%
at $5bn. The square-root law is not reliable past roughly 10% participation.

### Execution race (synthetic simulator)

Buy 50,000 shares (5% of ADV) over 26 buckets, sigma 2% a day, 5 bp spread,
square-root temporary impact on bucket volume, linear permanent impact.
Expected IS comes from the same simulator with zero price and volume noise.
The spread comes from 500 Monte Carlo replications.

| schedule | expected IS (bp) | std IS (bp) | MC mean IS (bp) | SE of MC mean (bp) |
|---|---:|---:|---:|---:|
| VWAP | 13.7 | 109.1 | 19.0 | 4.9 |
| TWAP | 14.3 | 113.2 | 19.4 | 5.1 |
| AC (lambda 5e-6) | 15.1 | 74.2 | 17.5 | 3.3 |
| POV 10% | 17.2 | 60.9 | 18.2 | 2.7 |
| Aggressive (2 buckets) | 27.5 | 19.2 | 28.1 | 0.9 |

Rank schedules on expected cost and on std. The Monte Carlo mean is not a
ranking tool at this replication count: its SE is several bp. The
Almgren-Chriss frontier on synthetic parameters: lambda 1e-6 gives (15.5,
99.0) bp expected cost and std, 5e-6 gives (21.1, 73.8), 5e-5 gives (53.9,
36.5).

### TCA (synthetic)

The decision price is the prior close. The arrival mid follows it by a
seeded move with sd sigma_daily x sqrt(0.5). Over 200 synthetic TWAP orders
the mean delay is -4.3 bp (sd 124.7), trading 18.6 bp (sd 106.5) and total
IS 14.3 bp (sd 170.9). The Perold components sum to the total exactly
(tested to 1e-10).

## Layout

- `src/eq_algo/`: `features`, `signals`, `backtest`, `evaluation`,
  `research_stats` (alpha layer); `intraday`, `benchmarks`,
  `almgren_chriss`, `tca` (execution layer); `data/synthetic` (seeded
  generators), `data/live` (optional Yahoo loader, not used by any result).
- `tests/`: 213 tests. Point-in-time mutation tests, hand-computed values,
  execution-convention tests, Monte Carlo checks of the statistics, AC
  optimality recursion, IS identities, edge cases.
- `docs/`: [METHODOLOGY.md](docs/METHODOLOGY.md),
  [VALIDATION.md](docs/VALIDATION.md), [DESK_GUIDE.md](docs/DESK_GUIDE.md).

## Limitations

- Every result is on synthetic data whose generator contains the signal.
- The book is dollar neutral only. It is not beta, sector or factor neutral.
- Almgren-Chriss parameters are set by hand, not fitted to market data.
- TCA has no market-adjusted IS and no fees.
