# FX Statistical Pairs — Cross-Rate Mean Reversion Trading

Flagship-quality implementation of an FX relative-value pipeline: **pairs of
currency pairs** (AUDUSD vs NZDUSD, NOKSEK-style cross RV), Engle–Granger
cointegration built from scratch and cross-validated against statsmodels,
OU spread dynamics, and a backtest whose accounting is FX-native:
**P&L = spot + carry (swap-point roll) − pip costs**.

This is not a renamed equity pairs project. Carry is first-class (a
mean-reversion signal on spot alone can be *systematically wrong-carry* — the
pipeline constructs a pair where including carry **flips the sign of P&L**),
triangular no-arbitrage is used as a machine-checked null case, costs are in
pips with pair-specific spreads (majors ~0.5–1 pip, EM 30–150), and the SNB
floor-then-break failure mode is simulated end-to-end.

## Pipeline

```
USD legs (CCYUSD) ──► crosses = exact ratios ──► candidate pairs of pairs
        │                                              │
        │  triangular identity (null case) ──► must be flagged DEGENERATE
        ▼                                              ▼
 correlation screen (drop pegs, warn) ──► Engle–Granger (MacKinnon N=2 cvs)
                                                       ▼
                          OU spread fit (OLS+MLE, half-life) · RLS hedge ratio
                                                       ▼
        z-score state machine (entry/exit/stop/time-stop, carry-aware filter)
                                                       ▼
    daily backtest: spot + swap-point carry − pip costs · walk-forward
                                                       ▼
     metrics: Sharpe (Lo SE), Sortino, MDD, hit rate, turnover,
                    carry-vs-spot P&L decomposition
```

## Quickstart

```bash
pip install -e .            # numpy, scipy, pandas, statsmodels
python examples/run_pipeline.py   # full pipeline, seeded synthetic data, ~1 s
pytest -q                         # 246 tests, offline, ~6 s
```

## Headline numbers (seeded synthetic data, from `run_pipeline.py`)

| Check | Result |
|---|---|
| ADF / EG vs statsmodels | stats match to 1e-10 / ~2e-15 |
| Spurious-regression size (200 random-walk pairs, 5% EG) | 4.5% rejections |
| Planted pair: EG stat / beta recovery | −7.76 (cv −3.34) / 0.999 vs true 1.0 |
| OU recovery (true kappa 20, hl 8.7 bd) | kappa 20.1, hl 8.7 bd (MLE ≡ OLS) |
| Triangular identity | std 4e-16 → flagged degenerate, untradable |
| Same signals, major vs EM costs | **+0.190** vs **−0.065** total P&L |
| Carry-flip pair: spot-only vs carry-inclusive | **−0.143** vs **+0.085** (carry +0.228) |
| SNB floor-then-break replay | +0.058 over 750d, then **−0.150 in one day** (2.6× gains) |
| Walk-forward (252/63), costs+carry | +0.085, Sharpe 0.77 ± 0.41 (Lo SE), 27 trades |

## Layout

```
05-pairs-trading/
├── src/fx_pairs/
│   ├── universe.py        # USD legs, exact crosses, pip conventions, corr screen
│   ├── cointegration.py   # ADF + Engle–Granger from scratch, MacKinnon (2010),
│   │                      #   degenerate-spread detection
│   ├── spread.py          # log-spread, OU (OLS+MLE), half-life, RLS hedge
│   ├── carry.py           # CIP forwards, swap points, daily roll, carry ledger
│   ├── signals.py         # z-score state machine, vol targeting, carry filter
│   ├── backtest.py        # daily engine (spot+carry−costs), walk-forward
│   ├── metrics.py         # Sharpe w/ Lo SE, Sortino, MDD, decomposition
│   └── data/
│       ├── synthetic.py   # seeded generators: cointegrated pairs, two-block
│       │                  #   risk-on/off panel, SNB floor-break, carry-flip
│       └── live.py        # guarded Frankfurter loader (never used in tests)
├── examples/run_pipeline.py
├── tests/                 # 246 tests: identities, cross-checks, scenarios
└── docs/                  # METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```

## Documentation contract

* **Why this model** (EG+OU vs Johansen, distance method, Kalman) and **why
  FX-level cointegration is rarer than in equities** (currencies are relative
  prices; common-USD-factor trap) — `docs/METHODOLOGY.md`
* **Assumptions register** (9 items, each with "what breaks") — `docs/METHODOLOGY.md`
* **Validation** incl. carry-flip numbers and the SNB depeg P&L path — `docs/VALIDATION.md`
* **Failure modes**: SNB 2015, policy divergence, EM devaluations, crowded
  carry unwinds — `docs/VALIDATION.md`
* **Desk usage**: forwards not cash, PB credit, vol sizing, per-block limits,
  carry-book/RV-book separation, real scenarios — `docs/DESK_GUIDE.md`
* **Edge cases**: pegged pair, triangular identity, zero trades, missing
  days, weekend accrual, EM costs — documented *and* unit-tested.
