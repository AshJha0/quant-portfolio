# 10 — Equity Regime-Switching Strategy

Regime detection for equities with a **from-scratch Gaussian HMM, GMM and
PCA** (cross-checked against `hmmlearn` / `scikit-learn`), traded through a
hysteresis-banded, vol-targeted allocation and validated with a strictly
causal walk-forward backtest.

```
multi-asset panel
      │
      ▼
features.py      realized vol (10/21/63d) · dispersion · avg pairwise corr ·
                 drawdown · trend vs 200d MA · credit proxy · term proxy
                 (all EXPANDING-window z-scored → no lookahead, test-enforced)
      │
      ▼
pca.py           correlation-matrix eigendecomposition, sign-fixed loadings,
                 rolling PCA with sign continuity  (vs sklearn to 1e-8)
      │
      ▼
hmm.py / gmm.py  scaled forward-backward · Baum-Welch EM (monotone, tested
                 every iteration) · Viterbi · stationary dist · durations ·
                 BIC k-selection  (vs hmmlearn / sklearn log-likelihood)
      │
      ▼
detection.py     expanding-window refits · ONLINE FILTERED probabilities
                 (never smoothed — the central honesty point) · economic
                 labels bull/transition/bear · flip-flop diagnostics
      │
      ▼
strategy.py      bull 1.0 / transition 0.5 / bear 0.0 · hysteresis band
                 enter p>0.70 exit p<0.30 (−67…−82% turnover) · 10% vol target
      │
      ▼
backtest.py      walk-forward, w_t earns r_{t+1}, 5 bps costs, exact ledger,
                 full-pipeline no-lookahead mutation test · benchmarks:
                 buy-and-hold and 200d-MA timing
      │
      ▼
risk.py          per-regime stats · transition P&L attribution (exact
                 identity) · flip-aftermath worst-case analysis
```

## Quickstart

```bash
cd python/equity/10-regime-switching
pip install -e .[dev]          # numpy/scipy/pandas + sklearn/hmmlearn for cross-checks
pytest -q                      # 130 tests, ~20 s, offline, seeded
python examples/run_pipeline.py   # full pipeline, ~55 s
```

## Headline results (seeded 3-state synthetic panel, 2 520 days, net of 5 bps)

| | strategy | buy & hold | 200d-MA rule |
|---|---|---|---|
| CAGR | **10.6%** | 1.6% | 1.6% |
| ann. vol | 8.5% | 15.1% | 11.0% |
| Sharpe | **1.22** | 0.18 | 0.20 |
| max drawdown | **10.2%** | 44.4% | 19.2% |

Where the edge comes from (per detected regime, annualised net):
strategy vs buy-and-hold is **+0.9% vs −15.8% in bears** (drawdown 7% vs
50%) — the overlay wins by not losing in bear markets, as designed.

Model validation highlights (see `docs/VALIDATION.md` for tables):

* BIC recovers the true state count (k=3) on regime data and k=1 on
  no-regime GBM null data; on null data the strategy does **not** beat
  buy-and-hold (mean excess CAGR −5.0% over seeds) — no spurious alpha.
* HMM recovers the true transition matrix (diagonal within 0.01) and state
  vols; Viterbi accuracy ≈ 99% on well-separated states.
* Filtered-vs-smoothed table around a real transition: the smoothed
  posterior is 74% bear the day *before* the flip (it sees the future);
  trading uses filtered probabilities only, enforced by mutation tests at
  the filter, detection and full-ledger level.
* Measured detection lag ~1.5 days on synthetic transitions — regime models
  are late by construction; `risk.flip_aftermath` keeps that cost visible.

## Layout

```
src/eq_regime/          features · pca · gmm · hmm · detection · strategy ·
                        backtest · risk · data/synthetic (seeded generators)
tests/                  130 offline tests incl. causality mutation tests,
                        EM monotonicity, sklearn/hmmlearn cross-checks,
                        null-data guard, edge cases
examples/run_pipeline.py  reproduces every number above
docs/                   METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```
