# FX Regime Switching (Risk-On / Risk-Off)

A study project, fully offline: PCA, GMM and a Gaussian HMM written from
scratch (scikit-learn and hmmlearn are used only as cross-checks), driving
a regime-conditional currency strategy. The books are carry in risk-on,
safe havens in risk-off, and long USD in a dollar squeeze, with carry
accrual, pip costs and point-in-time detection.

**Read this first.** Every result in this project comes from synthetic
data. The generator plants the edge: its risk-off and squeeze states have
large negative drifts for carry and EM currencies (-25% to -40% a year)
and a Gaussian HMM is the same model family that generated the data. On
the synthetic vol-clustering null with zero drift in every state, the
strategy has no edge. No real-data result exists yet; that is Phase 4 of
the repository plan.

```
synthetic RORO panel (known states, transition matrix, per-state
 means/vols/correlations, deposit-rate panel)
   |
   +- features.py    FX features (vol, carry basket, haven RS, USD-pair
   |                 correlation, EM spread, dollar factor, fwd points),
   |                 expanding-window z-scores, mutation-tested PIT
   +- pca.py         PC1 of the currency panel = the RORO axis
   +- gmm.py         EM from scratch, BIC/AIC, sklearn cross-check
   +- hmm.py         scaled forward/backward, Baum-Welch, Viterbi,
   |                 durations, stationary dist, hmmlearn cross-check
   +- detection.py   expanding refit, FILTERED probs only, economic
   |                 labeling, hysteresis, periodic cold restarts
   +- strategy.py    regime books, vol targeting, ACT/365 carry, pip costs
   +- backtest.py    walk-forward ledger (net = spot + carry - cost, exact)
   +- risk.py        per-regime stats, oracle gap, detection lag,
   |                 random-timing benchmark
   +- research_stats.py  Lo/Newey-West Sharpe SE, deflated Sharpe, PBO,
                     block bootstrap, multi-seed summaries
```

## Quickstart

```bash
cd python/fx/10-regime-switching
pip install -e ../../core   # qp-core, the shared package (not on PyPI)
pip install -e .[dev]
pytest                              # 196 tests, about 15 minutes
python examples/run_pipeline.py     # 5 to 7 minutes with 6 worker processes
```

The pipeline prints to the terminal and writes no files.

## Execution convention

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost (10 bp of traded notional by
default). `ExecutionConfig` sets the mode, f and the auction cost.
`tests/test_execution.py` fails if a decision at t fills at the close of t.

## How the configuration was chosen

Twelve configurations were tried during development: two feature-window
sets (21/10/21 and 8/5/12 days), three HMM input column sets (with
`fwd_ts`, with `usd_str`, five columns without either) and two risk-off
baskets (AUD/NZD, AUD/NZD plus MXN/ZAR/BRL). The pipeline runs all twelve
on design seeds 1001-1003 only, logs each in a `TrialLog`, and keeps the
best mean Sharpe: 8/5/12-day windows, the five-column set, and the basket
with EM legs. These are the library defaults. The probability of backtest
overfitting across the twelve (CSCV, 10 blocks) is 0.15 on the synthetic
design seeds. Evaluation then uses fresh seeds 2001-2008.

## Results on synthetic data

Produced by `python examples/run_pipeline.py`: 3-state synthetic panels
of 1,500 business days and 12 currencies, 8 evaluation seeds, net of
costs, carry included, 10% vol target. Each cell is the mean across seeds
with a 95% Student-t confidence interval.

| strategy (synthetic, 8 seeds) | Sharpe | ann. return | max drawdown (additive) |
|---|---|---|---|
| oracle (knows the true state) | 0.76 [0.43, 1.09] | 8.9% [4.9%, 12.9%] | -18.6% [-21.6%, -15.6%] |
| filtered HMM | 0.47 [-0.09, 1.02] | 5.2% [-1.0%, 11.4%] | -19.8% [-24.4%, -15.2%] |
| static carry | -0.12 [-0.65, 0.40] | -1.4% [-7.3%, 4.5%] | -34.9% [-49.5%, -20.3%] |
| same regime path at random times | 0.16 [-0.01, 0.34] | | |

Where the filtered strategy's edge comes from (Sharpe differences, same 8
synthetic seeds; the zero-drift panels share each seed's states and shocks
but plant no drift in any state):

| comparison (synthetic) | Sharpe difference |
|---|---|
| filtered minus static carry, planted drift | 0.59 [-0.19, 1.37] |
| filtered minus static carry, zero drift | -0.39 [-1.01, 0.23] |
| part due to the planted drift (paired) | 0.98 [0.77, 1.19] |
| filtered minus random timing, planted drift | 0.30 [-0.15, 0.75] |
| filtered minus random timing, zero drift | -0.11 [-0.53, 0.31] |

* The whole measured edge over static carry is the planted crash drift.
  With vol and correlation regimes but no drift link, the filter does not
  beat static carry or random timing on synthetic data.
* Pooled over the 8 synthetic seeds the filtered Sharpe is 0.47 with a
  block-bootstrap 95% CI of [0.13, 0.80]. The deflated Sharpe ratio at 12
  logged trials is 0.72, below the usual 0.95 bar.
* A single synthetic seed (2001) has filtered Sharpe 0.10 with a
  Lo/Newey-West standard error of 0.49. One seed says little.
* Fill model (synthetic, filtered Sharpe): 0.47 at f = 0.5, 0.40 at
  f = 1.0, 0.28 with a 10 bp close auction.
* Detection with fast hysteresis (0.70/0.30, 2 days) on synthetic data:
  95% of fresh flips detected, mean lag 2.7 days [2.0, 3.5]. The
  conservative setting (0.90/0.10, 5 days) detects 87% with a 5.3-day lag
  and a lower Sharpe.
* Static carry loses in risk states, but on synthetic data the losses are
  not concentrated in the first days of a spell: the first-10-day share is
  0.43 [-0.11, 0.97] across seeds.

## Tests

`pytest` runs 196 tests (see `docs/VALIDATION.md` for the inventory). The
regime guard is `tests/test_risk.py::test_vol_regime_null_no_timing_edge`:
on synthetic panels with vol clustering and zero drift in every state, the
filtered book must not beat random timing at equal exposure by more than
3 standard errors across 6 seeds. A companion test shows the same
statistic detects the planted edge when the true states drive the book.

## Conventions

* P&L is in units of notional: spot is the simple return of each
  currency against USD, carry accrues ACT/365 on the calendar gap, costs
  are half-spreads converted with each pair's pip size and quote level.
* Max drawdown is the most negative peak-to-trough of the additive
  cumulative sum of daily returns (no compounding).
* Sharpe ratios are annualised with sqrt(252).

## Limitations

* All data is synthetic and the edge is planted by the generator.
* The Gaussian HMM is the same model family as the generator.
* Spreads are constant in pips; they do not widen in risk-off.
* Carry uses deposit rates under covered interest parity; there is no
  cross-currency basis.

## Docs

* `docs/METHODOLOGY.md`: models, features, labeling, books, assumptions.
* `docs/VALIDATION.md`: cross-checks, recovery on known truth, the
  design/evaluation protocol, the null tests, the test inventory.
* `docs/DESK_GUIDE.md`: how a desk would use a regime filter, and which
  controls exist in code.
