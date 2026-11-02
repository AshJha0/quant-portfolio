# FX statistical pairs: cross-rate mean reversion

An FX relative-value research pipeline for **pairs of currency pairs**
(AUDUSD vs NZDUSD, NOKSEK-style cross RV): Engle-Granger cointegration built
from scratch and cross-checked against statsmodels, OU spread dynamics, and a
backtest whose accounting is FX-native: **P&L = spot + carry (swap-point
roll) - pip costs**.

**Status.** Every result in this project comes from synthetic data. The
planted pair is cointegrated by construction, so the profit it shows is put
there by the generator and is not evidence of an edge in FX markets. The
first real-data study (`research/prereg/pairs-fx-g10.md`, signed) has a
training-period result on ECB data, 1999 to 2021, in `research/results/pairs_fx_g10_train.json`. It
is not out-of-sample evidence; the 2022-2025 hold-out has not been run.

Carry is part of the P&L (the pipeline builds a synthetic pair where
including carry flips the sign of P&L), the triangular no-arbitrage identity
is a machine-checked null case, costs are in pips with pair-specific spreads,
and an SNB-style floor-then-break is simulated end to end.

## Execution convention

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost. For daily FX the t+1 price is
the next daily close. Configure it with `ExecutionConfig(mode,
fill_fraction, auction_cost_bp)` on `run_backtest` and
`walk_forward_backtest`.

## Pipeline

```
USD legs (CCYUSD) -> crosses = exact ratios -> candidate pairs of pairs
        |                                              |
        |  triangular identity (null case) -> must be flagged DEGENERATE
        v                                              v
 correlation screen (drop pegs, warn) -> Engle-Granger (MacKinnon N=2 cvs)
                                                       v
                          OU spread fit (OLS+MLE, half-life), RLS hedge ratio
                                                       v
        z-score state machine (entry/exit/stop/time-stop, carry entry veto)
                                                       v
    daily backtest: spot + swap-point carry - pip costs, t+1 fills, walk-forward
                                                       v
     metrics: Sharpe (Lo/Newey-West SE), Sortino, MDD, hit rate, turnover,
     carry-vs-spot decomposition; deflated Sharpe, PBO, block bootstrap
```

## Quickstart

```bash
pip install -e ../../core   # qp-core, the shared package (not on PyPI)
pip install -e .[dev]              # numpy, scipy, pandas, statsmodels; pytest, hypothesis
python examples/run_pipeline.py    # seeded synthetic data, 60 seeds, about 200 s, writes no files
python examples/run_pipeline.py --seeds 10   # quicker, wider intervals
pytest -q                          # 286 tests, offline, about 110 s
```

## Headline numbers (synthetic)

From `python examples/run_pipeline.py`: mean [95% CI] over 60 synthetic seeds
(1000 to 1059) unless a seed is named.

| Check | Result |
|---|---|
| ADF / EG vs statsmodels | statistics match to 1e-10 / about 2e-15 |
| Spurious-regression size (200 random-walk pairs, 5% EG) | 4.5% rejections |
| Funnel finds the planted pair hidden among 8 market pairs | 60 of 60 seeds; 1.12 [0.80, 1.43] false positives per seed |
| OU recovery, seed 9 (true kappa 20, half-life 8.7 bd) | kappa 20.1, half-life 8.7 bd |
| Triangular identity | std 4e-16, flagged degenerate, untradable |
| In-sample Sharpe, major vs EM costs | 1.77 [1.70, 1.84] vs -0.44 [-0.50, -0.38] |
| Carry-flip pair P&L, spot only vs with carry | -0.143 [-0.153, -0.133] vs +0.120 [0.110, 0.129] |
| SNB floor-then-break, seed 3 | +0.053 over 750 days, then -0.150 in one day |
| Walk-forward Sharpe (252/63), costs + carry | 1.49 [1.39, 1.59] |
| Walk-forward Sharpe, vol-targeted to 10% | 1.46 [1.36, 1.56] |
| PBO over the 6-config entry/exit grid | 0.32 [0.26, 0.38] |
| Deflated Sharpe, seed 9 walk-forward, 9 logged trials | 0.83 |

## Layout

```
05-pairs-trading/
├── src/fx_pairs/
│   ├── universe.py        # USD legs, exact crosses, pip conventions, screen, funnel
│   ├── cointegration.py   # ADF + Engle-Granger from scratch, MacKinnon (2010),
│   │                      #   degenerate-spread detection
│   ├── spread.py          # log-spread, OU (OLS+MLE), half-life, RLS hedge
│   ├── carry.py           # CIP forwards, swap points, daily roll, carry ledger
│   ├── signals.py         # z-score state machine, vol targeting, carry veto
│   ├── backtest.py        # daily engine (spot+carry-costs), ExecutionConfig, walk-forward
│   ├── metrics.py         # Sharpe with Lo/NW SE, Sortino, MDD, decomposition
│   ├── research_stats.py  # Lo/NW SE, deflated Sharpe, PBO, block bootstrap, seeds
│   └── data/
│       ├── synthetic.py   # seeded generators: cointegrated pairs, two-block
│       │                  #   risk-on/off panel, SNB floor-break, carry-flip
│       └── live.py        # guarded Frankfurter loader (not used by any test or result)
├── examples/run_pipeline.py
├── tests/                 # 286 tests
└── docs/                  # METHODOLOGY.md, VALIDATION.md, DESK_GUIDE.md
```

## Where things are documented

* Model choice (EG+OU vs Johansen, distance method, Kalman), why FX-level
  cointegration is rarer than in equities, execution convention, assumptions
  register: `docs/METHODOLOGY.md`.
* Validation with numbers, failure modes (SNB 2015, policy divergence, EM
  devaluations, crowded carry unwinds), edge cases: `docs/VALIDATION.md`.
* Desk usage (forwards, PB credit, vol sizing, per-block limits,
  carry-book/RV-book separation): `docs/DESK_GUIDE.md`.

## Limitations

* All numbers in this package are synthetic. The ECB training-period
  result lives in `research/`, not here, and has no hold-out run yet.
* There is no market-impact model; costs are the half spread plus an optional
  auction charge.
* The `live.py` loader returns ECB reference rates, which are a fixing, not
  tradable prices.
