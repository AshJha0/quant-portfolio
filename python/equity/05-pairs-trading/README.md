# Equity statistical pairs trading (`eq_pairs`)

A pairs-trading research pipeline: candidate pairs, a correlation or SSD
screen, Engle-Granger cointegration with MacKinnon N=2 critical values, an
Ornstein-Uhlenbeck spread model, z-score signals, a backtest with a
modelled next-period fill and a full cost model, walk-forward validation,
and research statistics (Lo/Newey-West Sharpe SE, deflated Sharpe, PBO,
block bootstrap, multi-seed distributions).

**Status.** Every result in this project is from synthetic data. The
generator plants cointegrated pairs with known half-lives, and the
strategy's P&L is that planted mean reversion. No real-data result exists
yet; that is Phase 4 of the repository plan.

## Execution convention

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost.

```python
from eq_pairs import ExecutionConfig
ExecutionConfig()                                         # next, f = 0.5
ExecutionConfig(fill_fraction=1.0)                        # next close
ExecutionConfig(mode="close_auction", auction_cost_bp=10) # same close, pays 10 bp
```

## Quickstart

```bash
cd python/equity/05-pairs-trading
pip install -e ../../core   # qp-core, the shared package (not on PyPI)
pip install -e .[dev]
pytest -q                          # 314 tests, offline
python examples/run_pipeline.py    # every number below; prints only
```

```python
import eq_pairs as ep
from eq_pairs.data import cointegrated_pair

df, truth = cointegrated_pair(n=1500, beta=1.5, kappa=0.08, sigma=1.0, seed=1)
eg = ep.engle_granger(df["Y"].to_numpy(), df["X"].to_numpy())
ou = ep.fit_ou_ols(eg.resid)
z = ep.zscore_ou(ep.compute_spread(df["Y"], df["X"], eg.beta, eg.alpha), ou)
target = ep.generate_signals(z, ep.SignalRules(max_holding=30))["position"]
res = ep.backtest_pair(df["Y"], df["X"], target, beta=eg.beta)  # fills at t+1, f = 0.5
print(res.net_pnl, ep.summary(res.daily, res.trades, res.ledger, 2e6)["sharpe"])
```

## Headline results (synthetic, 30 seeds)

30 synthetic panels of 20 names x 1500 days, each with 4 planted
cointegrated pairs, 3 correlated-random-walk traps and 1 regime-break pair.
Capital $6mm (6 pair slots x $1mm gross). Costs: 5 bp commission, 2 bp
slippage, square-root impact with Y = 0.7 on an assumed $20mm ADV, 50 bp
borrow, 50 bp long financing. Mean [95% CI] across seeds, from
`python examples/run_pipeline.py` section 9.

| metric (synthetic) | mean [95% CI] |
|---|---|
| planted pairs passing Engle-Granger (of 4) | 3.93 [3.84, 4.03] |
| trap pairs passing Engle-Granger | 0.20 [0.05, 0.35] |
| in-sample Sharpe (selection on the full sample) | 2.68 [2.54, 2.83] |
| walk-forward Sharpe | 1.33 [1.14, 1.53] |
| walk-forward Sharpe, close auction at 10 bp | 0.81 [0.61, 1.00] |
| walk-forward annual return on capital | 1.4% [1.1%, 1.6%] |
| walk-forward cost drag per year | 1.2% [1.1%, 1.3%] |
| regime-break pair after the break, with stops | -$20k [-$26k, -$15k] |
| regime-break pair after the break, without stops | -$472k [-$540k, -$404k] |

On the seed-7 synthetic path, 9 walk-forward configurations were logged;
the headline configuration has a deflated Sharpe ratio of 0.955 against
that trial count and the PBO across the nine is 0.27. At impact Y = 0.7 and
$10mm per pair the seed-7 synthetic walk-forward Sharpe falls to 0.10.
Details, single-path tables and the cost and timing sensitivities are in
`docs/VALIDATION.md`.

## Layout

```
src/eq_pairs/
  universe.py        candidates, correlation screen on returns, SSD screen
  cointegration.py   OLS hedge ratio, ADF with AIC lags, MacKinnon N=1/N=2, Engle-Granger
  spread.py          spread, OU fit (OLS and MLE), half-life, rolling OLS and RLS hedges
  signals.py         z-scores (OU-stationary and rolling), state machine, sizing
  backtest.py        ExecutionConfig, CostModel, pair and portfolio backtest, walk-forward
  metrics.py         Sharpe with iid and Lo SE, Sortino, drawdown, turnover, cost drag
  research_stats.py  Lo/Newey-West SE, deflated Sharpe, PBO (CSCV), block bootstrap, seeds
  data/synthetic.py  seeded cointegrated, trap, regime-break and mixed panels
tests/               314 tests
docs/                METHODOLOGY, VALIDATION, DESK_GUIDE
```

## Limitations

- All data is synthetic and the edge is planted by the generator.
- `data/live.py` exists but no test or result uses it.
- ADV and the impact coefficient are assumptions.
- Borrow recalls, hard-to-borrow fees and delisting returns are not modelled.
- Signal thresholds were set globally, not chosen inside the walk-forward.
- Risk limits and kill switches in `docs/DESK_GUIDE.md` are desk process,
  not code.

Runtimes measured on the development machine: test suite 110 s,
pipeline 567 s with 30 seeds (both measured while other test jobs shared the machine).
