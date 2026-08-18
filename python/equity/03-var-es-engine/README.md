# Equity Market Risk — VaR & Expected Shortfall Engine

Historical, parametric and Monte Carlo VaR + Expected Shortfall for an
equity book (cash equities, index futures, options with delta-gamma-vega
*and* full revaluation), with the complete model-validation stack a real
risk desk runs on top: Kupiec / Christoffersen backtests, the Basel traffic
light, Acerbi-Székely ES backtesting, historical & hypothetical stress
scenarios, and closed-form + numerical reverse stress testing.

```
Portfolio ─ factor mapping (prices, index, implied vol)
   │
   ├── Historical VaR ──── plain / age-weighted (BRW) / filtered (FHS)
   ├── Parametric VaR ──── sample & EWMA Σ · normal / Student-t / Cornish-Fisher
   ├── Monte Carlo VaR ─── normal & t factor sim · full reval · SE + CI
   │
   ├── Expected Shortfall ─ empirical (exact tail integral) / closed forms
   ├── Backtesting ──────── Kupiec POF · Christoffersen IND/CC · AS Z₂
   ├── Basel traffic light ─ 0-4 / 5-9 / 10+ → k = 3.0 … 4.0
   └── Stress testing ───── 1987 / 2008 / 2020 replays · ladders · reverse stress
```

## Quickstart

```bash
cd python/equity/03-var-es-engine
pip install -e .[dev]
python -m pytest tests -q        # 183 tests, ~35 s, offline
python examples/run_pipeline.py  # full demo, ~95 s
```

```python
import eq_var as ev
from eq_var.data import demo_portfolio, demo_covariance, simulate_returns

pf, cov = demo_portfolio(), demo_covariance()
pnl = pf.pnl(simulate_returns(1000, cov, dist="t", seed=1))   # scenario P&L

ev.filtered_historical_var(pnl, alpha=0.01)                   # FHS 99% VaR
ev.parametric_var(pf.delta_exposures(), cov, 0.01, dist="t")  # var-covar
ev.monte_carlo_var(pf, cov, 0.01, n_paths=100_000, seed=7)    # MC full reval
ev.expected_shortfall(pnl, 0.025)                             # ES 97.5%
ev.kupiec_pof(n_obs=250, n_exceptions=5, alpha=0.01)          # backtest
ev.basel_traffic_light(5)                                     # zone + multiplier
```

## Results highlights (demo book: $1.76m equities + futures hedge + SPX puts)

- **Method disagreement is the point**: 99 % 1d VaR spans $39.0k (FHS,
  quiet current regime) → $46.9k (parametric normal) → $51.7k
  (parametric-t) → $54.3k (historical on fat-tailed data). Normal-family
  methods sit ~10 % low at 99 % — the missing kurtosis, visible.
- **The headline backtest** (500 days, GARCH-t data): unconditional-normal
  parametric VaR takes **14 exceptions vs 5 expected** (Kupiec p = 0.001,
  Basel **red** on the stress subwindow, k = 4.0) while **FHS passes green**
  (6 exceptions, p = 0.66, k = 3.0). Plain HS fails the Christoffersen
  clustering test (p = 0.02) where FHS passes (p = 0.48).
- **VaR is not a worst case**: 2020-replay stress loss $254k ≈ 4.7× the
  99 % VaR; delta-gamma misprices that scenario by $93k (37 %) — full
  revaluation matters for gap moves.
- **Reverse stress** names the worst joint 3σ direction (AAPL −4.7 %,
  JPM −3.6 %, SPX −1.9 %, IV +2.1pts → −$60k), closed form matching the
  numerical optimiser to 1e-6.
- Analytic identities to 1e-10 (normal ES φ(z)/α, Kupiec LR, reverse-stress
  loss = r·σ_p); MC converges to closed form within 3 SE; the classic VaR
  non-subadditivity counterexample is constructed and ES shown coherent on
  it.

## Layout

```
src/eq_var/
├── portfolio.py           # RiskFactor / Position / Portfolio, full & delta-gamma reval
├── historical_var.py      # plain, BRW age-weighted, FHS; sqrt-time & overlapping 10d
├── parametric_var.py      # sample/EWMA Σ, normal/t/Cornish-Fisher (+ domain check)
├── monte_carlo_var.py     # normal/t sim, Cholesky+jitter, VaR SE & order-stat CI
├── expected_shortfall.py  # empirical (exact), normal/t closed forms, bootstrap SE
├── backtesting.py         # Kupiec, Christoffersen, Basel zones, AS Z₂, rolling driver
├── stress_testing.py      # 1987/2008/2020 replays, ladders, reverse stress
└── data/                  # synthetic.py (seeded normal/t/GARCH, demo book), live.py (guarded)
```

Docs: [METHODOLOGY.md](docs/METHODOLOGY.md) (model choice + assumptions
register), [VALIDATION.md](docs/VALIDATION.md) (evidence + failure modes),
[DESK_GUIDE.md](docs/DESK_GUIDE.md) (daily workflow, capital link, stress
committee).
