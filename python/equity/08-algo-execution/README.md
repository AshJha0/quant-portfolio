# Equity Algorithmic Trading & Execution Modeling (`eq_algo`)

Two-layer systematic equity stack, built and validated end-to-end:

```
ALPHA LAYER (daily)
  data/synthetic.py ──> features.py ──> signals.py ──> backtest.py ──> evaluation.py
  planted-alpha panel   PIT features    IC-weighted     L/S deciles     IC / NW t-stats
  (known IC ~0.04)      mom, reversal,  combination,    linear + sqrt   deflated Sharpe
                        vol, RSI, ...   decay, deciles  impact costs    capacity curve

EXECUTION LAYER (intraday)
  intraday.py ─────────> benchmarks.py ─────────> almgren_chriss.py ──> tca.py
  U-shaped volume,       VWAP/TWAP/arrival,       closed-form optimal   Perold IS
  temp (sqrt) + perm     TWAP/VWAP/POV            trajectories, cost/   decomposition,
  (linear) impact sim    schedulers               variance frontier     attribution
```

The alpha layer decides *what* to hold each day; the execution layer decides
*how* to trade into it intraday. The synthetic data has **planted alpha of
known strength**, so every statistic the pipeline produces can be checked
against ground truth.

## Quickstart

```bash
cd python/equity/08-algo-execution
pip install -e .[dev]
pytest -q                     # 161 tests, ~4 s, offline, seeded
python examples/run_pipeline.py   # full demo, ~4 s
```

## Results (all reproducible from `examples/run_pipeline.py`, seed 42)

**Feature ICs** (150 stocks x 1250 days, planted momentum IC target 0.04):

| feature      | mean IC | NW t-stat | annual ICIR |
|--------------|--------:|----------:|------------:|
| mom_12_1     |  0.0332 |     13.36 |        6.47 |
| mom_6_1      |  0.0279 |     12.37 |        5.55 |
| reversal_1m  |  0.0161 |      6.73 |        2.94 |
| ma_20_100    |  0.0132 |      5.84 |        2.63 |
| rsi_14       |  0.0129 |      5.37 |        2.49 |
| vol_63d      |  0.0020 |      0.86 |        0.38 |
| turnover_z   |  0.0004 |      0.17 |        0.08 |

Planted features are strongly significant; the two features with no planted
effect (low-vol, abnormal volume) correctly show t < 1. Decile portfolios
are perfectly monotone (Spearman rho = 1.000, Q1 = -10.9 bps/d ...
Q10 = +15.3 bps/d, L/S = 26.2 bps/d).

**Long-short backtest, gross vs net** ($200m AUM, 5 bps linear + sqrt
impact, 0.25-z rebalance band): gross Sharpe **4.86** -> net **4.06**;
annualised cost drag **709 bps** at 20% daily one-way turnover. Deflated
Sharpe demo (marginal MA-crossover strategy, SR 1.35): PSR vs 0 = 0.995,
but DSR = 0.89 after N=7 tried variants and **0.65 after N=45** — selection
bias eats most of the edge.

**Execution horse race** (buy 50k shares = 5% ADV, 26 buckets, 200 seeded
replications, sigma 2%/day, 5 bps spread, sqrt temporary + linear permanent
impact), IS vs arrival in bps:

| strategy            | mean IS | std IS |
|---------------------|--------:|-------:|
| TWAP                |   29.8  |  114.2 |
| VWAP                |   29.6  |  109.6 |
| POV 10%             |   25.2  |   63.6 |
| AC (lambda = 5e-6)  |   25.7  |   76.6 |
| Aggressive (2 bkts) |   37.6  |   19.3 |

The Almgren-Chriss frontier is a clean cost/variance dial: lambda 1e-6 ->
(15.5, 99.0) bps, 5e-6 -> (21.1, 73.8), 5e-5 -> (53.9, 36.5). In a 300-rep
paired test AC (lambda 5e-6) cuts IS std **113 -> 74 bps** vs TWAP for
+3.7 bps expected cost (Levene p = 1.2e-10).

**TCA** (one TWAP order, decision 99.80 / arrival 100.00): delay 20.0 bps +
trading 35.3 bps + opportunity 0.0 = total IS 55.3 bps — the Perold
decomposition sums exactly by construction (tested to 1e-10).

## Layout

- `src/eq_algo/` — `features` / `signals` / `backtest` / `evaluation`
  (alpha layer); `intraday` / `benchmarks` / `almgren_chriss` / `tca`
  (execution layer); `data/synthetic` (seeded generators), `data/live`
  (import-guarded optional Yahoo loader).
- `tests/` — 161 tests: point-in-time mutation tests, hand-computed exact
  values, statistical tests on planted alpha, AC optimality recursion,
  IS identities, edge cases.
- `docs/` — [METHODOLOGY.md](docs/METHODOLOGY.md) (model choices vs
  alternatives, assumptions register), [VALIDATION.md](docs/VALIDATION.md)
  (evidence + failure modes), [DESK_GUIDE.md](docs/DESK_GUIDE.md) (how a
  desk runs this).

## Key design points

- **Point-in-time discipline is test-enforced**: features and the full
  backtest pipeline are re-run after mutating all data strictly after a
  cutoff date; everything at or before the cutoff must be bit-identical.
- **Costs follow the empirical square-root law** in the backtest and the
  simulator's temporary impact; permanent impact is linear (the only
  no-arbitrage-consistent shape).
- **Deflated Sharpe (Bailey-Lopez de Prado)** is wired in as the standard
  guard against backtest overfitting.
- Every stochastic component takes an explicit seed; the suite runs offline.
