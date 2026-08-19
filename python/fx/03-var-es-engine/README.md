# FX Market Risk — VaR & Expected Shortfall Engine

Market-risk engine for a **multi-currency FX book**: historical, parametric
and Monte Carlo VaR, coherent Expected Shortfall, Kupiec / Christoffersen /
Basel-traffic-light backtesting, and an FX-native stress-testing layer with
peg-break add-ons and reverse stress.

Built FX-first, not equities-renamed:

* **USD triangulation** — every currency is one USD factor; cross positions
  (EURJPY) decompose into USD legs, so scenario sets are arbitrage-consistent
  by construction (identity tested to machine precision).
* **CIP-consistent forwards** — a forward is spot + two deposit legs; forward
  points expose the book to interest-rate factors (small, present, tested).
* **Garman–Kohlhagen options** — internal GK pricer; full revaluation or
  delta–vega(–gamma) mapping, with the mapping error characterised in tests.
* **Pegged currencies** — near-zero-vol factors trigger a
  `PegBlindnessWarning`; a peg-break scenario generator and jump-mixture MC
  supply the loss the historical window cannot contain (CHF 2015 case study
  in `docs/VALIDATION.md`).
* **EM fat tails** — variance-matched Student-t and jump-mixture Monte Carlo;
  normal MC demonstrably underestimates the EM 99% tail (+12% / +140%).

```
pipeline:  book & market ──► factor history (synthetic G10/EM blocks,
           GARCH, regimes, pegs; guarded live loader)
                 │
                 ▼
   Historical VaR (plain / BRW age / FHS)     ┐
   Parametric VaR (sample/EWMA; N, t, C-F)    ├──► ES (Acerbi–Tasche) ──►
   Monte Carlo VaR (N / t / jump, full reval) ┘
                 │
                 ▼
   Backtesting: Kupiec, Christoffersen, Basel traffic light, A-S ES test
                 │
                 ▼
   Stress: historical replays, USD ±10%, peg breaks, ladders, reverse stress
```

## Quickstart

```bash
pip install -e .
python -m pytest tests -q         # 365 tests, offline, ~7s
python examples/run_pipeline.py   # full demo, ~4s
```

```python
import fx_var as fv
from fx_var.data.synthetic import demo_market, demo_book, simulate_history

market, book = demo_market(), demo_book()
hist = simulate_history(book, market, 1000, seed=42, garch=True)

fv.historical_var(book, market, hist, alpha=0.99, method="fhs")  # headline
fv.parametric_var(book, market, hist, dist="t", df=5)            # challenger
fv.monte_carlo_var(book, market, fv.sample_cov(hist), dist="jump",
                   jumps=fv.JumpSpec(0.02, {"FX:MXN": -0.10}), seed=1)
fv.run_stress(book, market, fv.historical_scenarios())
```

## Headline results (seeded demo book: G10 spots + 6m forward + 3m option + EM + HKD peg)

| 99% / 1d | HS | BRW | FHS | Param-N | Param-t5 | MC-N |
|---|---|---|---|---|---|---|
| VaR (USD) | 690,889 | 682,957 | 629,121 | 651,515 | 729,964 | 635,368 |

* **Backtests (500d rolling, GARCH + regime-switching data)**:
  parametric-normal 14 exceptions → Kupiec p=0.0009, independence p=0.0002,
  **Basel red (4.00)**; FHS 7 exceptions → all p>0.05, **green (3.00)**.
* **EM fat tails (long-EM book, equal covariance)**: normal MC $1.21m,
  t(5) $1.36m, jump-mixture $2.91m at 99%.
* **Peg blindness**: HS 99% VaR $0.69m vs HKD −30% peg-break loss $15.0m
  (21.7×) — flagged by the engine, priced by the stress add-on.
* **Reverse stress**: worst joint move at the 99% radius loses $651k
  (closed form, confirmed by full revaluation and numerical optimisation).

## Layout

```
src/fx_var/            book, gk, historical_var, parametric_var,
                       monte_carlo_var, expected_shortfall, backtesting,
                       stress_testing, common
src/fx_var/data/       synthetic.py (seeded G10/EM/peg generators, demo book)
                       live.py (guarded Frankfurter/ECB loader — opt-in only)
tests/                 231 offline, seeded tests
examples/run_pipeline.py
docs/                  METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```

See `docs/METHODOLOGY.md` for model choices vs alternatives and the
assumptions register, `docs/VALIDATION.md` for evidence and failure modes,
`docs/DESK_GUIDE.md` for the desk workflow, limits and scenario playbook.
