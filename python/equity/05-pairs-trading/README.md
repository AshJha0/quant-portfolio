# Equity Statistical Pairs Trading (`eq_pairs`)

Statistical arbitrage pairs pipeline, built from first principles and
validated against `statsmodels`:

```
price panel ──► pair candidates (same sector)
                  ──► correlation screen (on RETURNS — prices are spuriously correlated)
                        ──► Engle-Granger two-step (from-scratch ADF, AIC lags,
                            MacKinnon *EG* critical values — not plain ADF)
                              ──► OU spread fit (OLS-AR1 + MLE, half-life)
                                    ──► z-score signals (entry/exit/stop/time-stop)
                                          ──► event-driven backtest (costs, slippage,
                                              borrow, strict t-1 execution)
                                                ──► walk-forward + metrics + attribution
```

## Quickstart

```bash
cd python/equity/05-pairs-trading
pip install -e .[dev]
pytest -q                        # 256 tests, offline, ~5 s
python examples/run_pipeline.py  # every number below, ~5 s
```

```python
import eq_pairs as ep
from eq_pairs.data import cointegrated_pair

df, truth = cointegrated_pair(n=1500, beta=1.5, kappa=0.08, sigma=1.0, seed=1)
eg = ep.engle_granger(df["Y"].to_numpy(), df["X"].to_numpy())
ou = ep.fit_ou_ols(eg.resid)
print(eg.beta, eg.stat, eg.crit["5%"], eg.cointegrated())   # hedge ratio + EG test
print(ou.half_life)                                          # ln 2 / kappa, days

z = ep.zscore_ou(ep.compute_spread(df["Y"], df["X"], eg.beta, eg.alpha), ou)
target = ep.generate_signals(z, ep.SignalRules(max_holding=30))["position"]
res = ep.backtest_pair(df["Y"], df["X"], target, beta=eg.beta)   # t-1 signal, t close
print(res.net_pnl, ep.summary(res.daily, res.trades, res.ledger, 2e6)["sharpe"])
```

## Headline results (seeded synthetic panel, 20 names × 1500 days)

**The selection funnel catches the classic trap** — 40 same-sector
candidates → 8 pass the return-correlation screen → **4** pass Engle-Granger
at 5%. The three correlated-random-walk trap pairs (return ρ ≈ 0.92!) and
the regime-break pair are exactly the ones rejected; the four truly
cointegrated pairs are exactly the ones accepted, with β and half-life
recovered (e.g. true HL 6.9d → est 7.0d; true β 1.718 → est 1.729).

| metric ($6mm capital, costs 5+2bp/leg, borrow 50bp) | in-sample | walk-forward (OOS) |
|---|---:|---:|
| net P&L (5.9y) | $2.92mm | $1.40mm |
| annualised return | 12.3% | 7.4% |
| Sharpe (± Lo SE) | 2.42 ± 0.57 | 1.59 ± 0.50 |
| hit rate | 95.6% | 67.2% |
| max drawdown | $92k | $128k |
| turnover / cost drag | 5.7x / 44bp-yr | 6.7x / 51bp-yr |

**Regime break (cointegration dies mid-sample):** +$81k pre-break becomes
−$21k post-break with stops and re-entry arming — and **−$715k without
stops**. The stop rules are the product.

## Layout

```
src/eq_pairs/
├── universe.py        # candidates, correlation screen (returns!), SSD screen
├── cointegration.py   # OLS hedge ratio, from-scratch ADF (AIC lags),
│                      # MacKinnon N=1/N=2 surfaces, Engle-Granger two-step
├── spread.py          # spread, OU fit (OLS + MLE), half-life, RLS (Kalman-lite)
├── signals.py         # z-scores (rolling / OU-stationary), state machine, sizing
├── backtest.py        # event-driven engine, costs/slippage/borrow, walk-forward
├── metrics.py         # Sharpe (+ Lo-adjusted SE), Sortino, MDD, turnover, attribution
└── data/synthetic.py  # seeded: cointegrated / trap / regime-break / mixed panel
tests/                 # 256 tests incl. the no-lookahead detector
docs/                  # METHODOLOGY, VALIDATION (all numbers), DESK_GUIDE
```

## The three design decisions worth reading about

1. **EG critical values, not ADF** (`docs/METHODOLOGY.md`): testing EG
   residuals against plain ADF critical values (−2.86 instead of −3.34 at
   5%) more than doubles the false-discovery rate — measured, tested, and
   the reason `mackinnon_crit` carries both surfaces.
2. **No-lookahead by construction** (`docs/VALIDATION.md` §5): the engine
   executes yesterday's decision at today's close and a detector test
   engineers a spread where cheating is profitable — the engine must lose.
3. **Correlation ≠ cointegration**: the correlation screen runs on returns
   and is only a pre-filter; ρ=0.92 pairs still fail the ADF (and would have
   bled money: the trap pair that slipped into one walk-forward window lost
   −$11k).

Docs answer the full contract: model choice vs alternatives, assumptions
with what-breaks, validation evidence, failure modes (quant quake, short
squeeze, merger break), desk workflow and limits.
