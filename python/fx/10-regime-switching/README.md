# FX Regime-Switching Quant Strategy (Risk-On / Risk-Off)

A flagship-quality, fully offline FX regime project: **PCA, GMM and a
Gaussian HMM implemented from scratch** (scikit-learn / hmmlearn used
only as cross-checks), driving a regime-conditional currency strategy —
carry in risk-on, safe havens in risk-off, long USD in a
2008/2020-style dollar squeeze — with carry accrual, pip costs,
point-in-time discipline, and an **oracle comparison** as the honesty
metric.

```
synthetic RORO panel (known truth: states, transition matrix, per-state
 means/vols/correlations, deposit-rate panel)
   │
   ├─ features.py    FX-native features (VXY analog, carry basket, haven RS,
   │                 USD-pair correlation, EM spread, dollar factor, fwd points)
   │                 — expanding-window z-scores, mutation-tested PIT
   ├─ pca.py         PC1 of the currency panel = the RORO axis
   ├─ gmm.py         EM from scratch, BIC/AIC, sklearn cross-check
   ├─ hmm.py         log-space forward/backward, Baum-Welch, Viterbi,
   │                 durations, stationary dist, hmmlearn cross-check
   ├─ detection.py   expanding refit, FILTERED probs only, economic
   │                 labeling (high-vol-high-corr ⇒ risk_off), hysteresis
   ├─ strategy.py    regime books, vol targeting, carry accrual, pip costs
   ├─ backtest.py    walk-forward ledger (net = spot + carry − cost, exact)
   └─ risk.py        per-regime stats, transition attribution, oracle gap,
                     detection-lag cost, drawdown decomposition
```

## Quickstart

```bash
cd python/fx/10-regime-switching
pip install -e .[dev]          # or just have numpy/scipy/pandas/sklearn/hmmlearn
pytest -q                      # 125 tests, offline, ~35s
python examples/run_pipeline.py   # full pipeline, ~50s, all numbers below
```

## Headline numbers (seeded pipeline, 2,000 days, 12 currencies, 3 states)

Three-way comparison, net of pip costs, carry accrual included, 10% vol
target, one-day execution delay for everyone:

| strategy | ann. return | Sharpe | note |
|---|---|---|---|
| oracle (knows the true state) | 11.6% | 0.98 | upper bound |
| **filtered HMM (tradeable)** | **7.8%** | **0.69** | this project |
| static carry (never hedges) | 5.6% | 0.49 | baseline |

* The regime filter is worth **+2.2% p.a.** over static carry; perfect
  state knowledge would be worth another **+3.8% p.a.**
* Static carry earns **+11.2% p.a. (Sharpe 1.37)** in true risk-on and
  loses **−12.8% p.a. (Sharpe −0.69)** in true risk-off; the filtered
  strategy flips the crash cell to **+20.4% p.a. (Sharpe 1.16)**.
* **93%** of carry's risk-state losses land in the first 10 days of a
  spell — the crash happens at the flip.
* Detection lag: **~1 day** mean on fresh flips (94% detection rate),
  estimated cost **~10 bp per flip**; the dominant cost of imperfect
  detection is **false alarms** (gap to the oracle on true risk-on days
  ≈ 3,900 bp over the 6.5-year sample), not lag — quantified in
  `docs/VALIDATION.md`.
* PC1 of the standardised currency panel is the RORO axis: carry + EM
  load +0.26…+0.34, JPY/CHF load negative, 53% of variance.
* 2-state HMM recovery on known truth: max transition-matrix error
  0.027, Viterbi accuracy 98.4%.

## Honesty features

* **Filtered probabilities only** — smoothed (future-peeking) posteriors
  are computed for research but never trade; enforced by mutation tests
  that perturb the future and require the past to be bit-identical.
* **Oracle comparison** — the strategy is benchmarked against an agent
  with the true state sequence and identical costs/delay, so the price
  of imperfect detection is a reported number, not a vibe.
* **Null guards** — on a no-regime GBM panel the filter shows no
  spurious edge and BIC prefers one state on iid returns.
* **Failure modes documented with numbers** (`docs/VALIDATION.md`):
  near-unit-root features hijacking the HMM, phantom "transitional"
  states from long windows, HMM-BIC over-selecting k, crash-state
  merging at realistic sample sizes, pegs, SNB-style jumps, slow-bleed
  drawdowns.

## Docs

* `docs/METHODOLOGY.md` — why RORO is *the* FX regime story, HMM vs
  threshold/GMM alternatives, filtered-vs-smoothed, 9-item assumptions
  register with what-breaks-if-violated.
* `docs/VALIDATION.md` — cross-check tables, recovery on known truth,
  the oracle table, detection-lag and false-alarm quantification, null
  results, failure modes.
* `docs/DESK_GUIDE.md` — carry-book throttle, tail-hedge overlay,
  dashboards, hysteresis governance, event-override protocol, scenario
  playbook (2008, COVID-2020, taper tantrum, 2022 USD squeeze, SNB).
