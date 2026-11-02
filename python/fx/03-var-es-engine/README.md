# FX Market Risk: VaR and Expected Shortfall Engine

Market-risk engine for a multi-currency FX book. It computes historical,
parametric and Monte Carlo VaR with Acerbi-Tasche Expected Shortfall. It
backtests with Kupiec, Christoffersen, the Basel traffic light and the
Acerbi-Szekely Z2 test. It stress tests with historical replays, peg breaks
and reverse stress.

FX-specific design:

* **USD triangulation.** Every currency is one USD factor. Cross positions
  (EURJPY) decompose into USD legs, so every scenario set is
  arbitrage-consistent. The identity is tested to machine precision.
* **CIP forwards.** A forward is spot plus two deposit legs, so forward
  points carry interest-rate risk.
* **Garman-Kohlhagen options.** Full revaluation by default, or a
  delta-vega(-gamma) mapping whose error is measured in the tests.
* **Pegged currencies.** Near-zero-vol factors raise a
  `PegBlindnessWarning`. A peg-break scenario and a jump-mixture Monte Carlo
  supply the loss the historical window cannot contain.
* **EM fat tails.** Variance-matched Student-t and compound-Poisson jump
  Monte Carlo.

## Conventions

* `alpha` is the tail probability: `alpha=0.01` is 99 % VaR. Values outside
  (0, 0.5) raise `ValueError`. The keyword `confidence=0.99` still works for
  one release with a `DeprecationWarning`.
* VaR is the order statistic `L(m)`, `m = ceil(n alpha)`, of losses sorted
  worst first. ES is the Acerbi-Tasche average over exactly `alpha` of mass.
* Inputs that are NaN, infinite, empty or out of range raise `ValueError`.
  An empty book raises. No function returns a number from bad input.

`docs/METHODOLOGY.md` section 0 lists every convention.

## Quickstart

```bash
pip install -e .[dev]
python -m pytest -q               # 867 tests, offline, about 90 s
python examples/run_pipeline.py   # full demo, about 30 s
```

Timings are from Windows 11, Python 3.12.10, numpy 2.5, scipy 1.18.

```python
import fx_var as fv
from fx_var.data.synthetic import demo_market, demo_book, simulate_history

market, book = demo_market(), demo_book()
hist = simulate_history(book, market, 1000, seed=42, garch=True)

fv.historical_var(book, market, hist, alpha=0.01, method="fhs")
fv.parametric_var(book, market, hist, alpha=0.01, dist="t", df=5)
fv.monte_carlo_var(book, market, fv.sample_cov(hist), alpha=0.01, dist="jump",
                   jumps=fv.JumpSpec(0.02, {"FX:MXN": -0.10}), seed=1)
fv.run_stress(book, market, fv.historical_scenarios())
```

## Demo results on synthetic data

Produced by `python examples/run_pipeline.py` on the seeded synthetic demo
book (G10 spots, a 6m forward, a 3m option, EM positions, an HKD peg).

| 99 % / 1d, USD | HS | BRW | FHS | Param-N | Param-t5 | MC-N |
|---|---|---|---|---|---|---|
| VaR | 690,889 | 682,957 | 629,121 | 651,515 | 729,964 | 635,368 |

| Synthetic 500-day backtest, 99 % | Exceptions | Kupiec p | Indep. p | Basel zone |
|---|---|---|---|---|
| Parametric normal | 14 | 0.0009 | 0.0002 | red (4.00) |
| Plain HS | 8 | 0.2149 | 0.1092 | yellow (3.40) |
| FHS | 7 | 0.3966 | 0.0790 | green (3.00) |

| Synthetic EM book, 99 % / 1d, USD | VaR | ES |
|---|---|---|
| Normal MC | 1,211,798 | 1,389,156 |
| Student-t(5) MC | 1,356,099 | 1,771,315 |
| Jump-mixture MC | 2,887,602 | 3,630,371 |

On the synthetic demo book the HKD -30 % peg break loses 15.0m USD, 21.7
times the 99 % HS VaR.

## Layout

```
src/fx_var/            book, gk, historical_var, parametric_var,
                       monte_carlo_var, expected_shortfall, backtesting,
                       stress_testing, common
src/fx_var/data/       synthetic.py (seeded generators, demo book)
                       live.py (opt-in Frankfurter/ECB loader)
tests/                 867 offline, seeded tests (pytest --co -q)
tests/golden/          generate_golden.py and golden_vectors.json
examples/run_pipeline.py
docs/                  METHODOLOGY.md, VALIDATION.md, DESK_GUIDE.md
```

The C++ (`cpp/fx-var-engine`) and Rust (`rust/fx-var-engine`) engines are
linear-book ports. They check their results against
`tests/golden/golden_vectors.json`.

## Round 3 additions (risk attribution, FRTB IMA, tails, revaluation, limits)

Eight new modules. They are asset-agnostic, ship as the same file in
`eq_var` and `fx_var`, and are imported from their submodule (they are not
re-exported from the package root yet):

| Module | What it gives you |
|---|---|
| `fx_var.risk_attribution` | Euler marginal, component and incremental VaR and ES, closed form and from scenarios |
| `fx_var.frtb_ima` | Liquidity-horizon ES cascade, stressed ES, IMCC, IMA capital, desk backtest, PLA test |
| `fx_var.stressed_var` | Basel 2.5 stress-window search, stressed VaR, the VaR + sVaR capital formula |
| `fx_var.evt` | Peaks-over-threshold GPD fit, POT VaR and ES, mean-excess function |
| `fx_var.copula_mc` | Student-t copula Monte Carlo VaR and ES with any marginals |
| `fx_var.revaluation` | Full revaluation of option books under scenarios, Greek approximations, P&L explain |
| `fx_var.limits` | Limit hierarchy with utilisation, warning and breach status |

```python
import numpy as np
from fx_var.risk_attribution import parametric_attribution
a = parametric_attribution([1e6, -4e5], np.array([[4e-4, 1.2e-4], [1.2e-4, 2.25e-4]]), 0.01)
a.total, a.component, a.percent   # components sum to the total (Euler)
```

The compiled VaR engines gain the option revaluation part only
(`option_reval` in C++ and Rust); see their READMEs.
