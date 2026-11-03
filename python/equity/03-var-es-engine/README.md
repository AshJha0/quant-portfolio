# Equity Market Risk: VaR and Expected Shortfall Engine

Historical, parametric and Monte Carlo VaR and Expected Shortfall for an
equity book (cash equities, index futures, options with delta-gamma-vega and
full revaluation). Kupiec and Christoffersen backtests, the Basel traffic
light for any window, the Acerbi-Szekely ES backtest with a simulated
p-value, historical and hypothetical stress scenarios, and reverse stress
testing.

```
Portfolio: factor mapping (prices, index, implied vol)
   |
   +-- Historical VaR      plain / age-weighted (BRW) / per-factor FHS
   +-- Parametric VaR      sample and EWMA covariance, normal / t / Cornish-Fisher
   +-- Monte Carlo VaR     normal and t factor simulation, full revaluation, SE and CI
   +-- Expected Shortfall  empirical (S2) / closed forms
   +-- Backtesting         Kupiec, Christoffersen, Acerbi-Szekely Z2
   +-- Basel traffic light any window and alpha
   +-- Stress testing      1987 / 2008 / 2020 replays, ladders, reverse stress
```

`alpha` is the tail probability everywhere (0.01 is 99 % VaR). VaR is the
order statistic `L(ceil(n alpha))` of the losses, with no interpolation. The
full list of conventions is in [METHODOLOGY.md](docs/METHODOLOGY.md) §2.

## Historical VaR rule (`rule=`)

Plain idea: with 250 daily P&Ls, "the 99 % VaR" is a loss exceeded on
about 2.5 days, so it must be read off between order statistics. Two
conventions exist and they differ by a few percent on short windows.

Formal statement: `rule="kth_worst"` (the default) returns the `m`-th
largest loss with `m = ceil(n * alpha)`, clamped to 1 (BCBS MAR33.4 and
MAR99 use the 3rd worst of 250 days at 99 %). `rule="interpolated"`
returns the type-7 (linear) quantile of the losses (Hyndman and Fan 1996).
Both match an independent mpmath oracle, and the C++ (`HistRule`) and
Rust (`HistRule`) engines implement the same two rules.

Worked example: losses 1, 2, ..., 250 at alpha = 1 %: `kth_worst` gives
248 (the 3rd worst); `interpolated` gives 247.51.

## Quickstart

```bash
cd python/equity/03-var-es-engine
pip install -e .[dev]
python -m pytest -p no:cacheprovider -q   # 676 tests, offline
python examples/run_pipeline.py           # full demo
```

```python
import eq_var as ev
from eq_var.data import demo_portfolio, demo_covariance, simulate_returns

pf, cov = demo_portfolio(), demo_covariance()
rets = simulate_returns(1000, cov, dist="t", seed=1)              # factor history
ev.factor_filtered_historical_var(pf, rets, alpha=0.01)           # per-factor FHS 99 % VaR
ev.parametric_var(pf.delta_exposures(), cov, 0.01, dist="t")     # var-covar
ev.monte_carlo_risk(pf, cov, 0.01, n_paths=100_000, seed=7)       # MC VaR, ES, SE
ev.expected_shortfall(pf.pnl(rets), 0.025)                        # ES 97.5 %
ev.kupiec_pof(n_obs=250, n_exceptions=5, alpha=0.01)              # backtest
ev.basel_traffic_light(n_obs=250, n_exceptions=5, alpha=0.01)     # zone and multiplier
ev.historical_var(pf.pnl(rets), 0.01)                             # k-th worst loss (default)
ev.historical_var(pf.pnl(rets), 0.01, rule="interpolated")        # type-7 quantile
```

## Results on the synthetic demo book

Produced by `python examples/run_pipeline.py` (seeded, synthetic t(6) and
GARCH-t factor data; $1.76m equities, an index future hedge and SPX puts).

| method | 99 % 1-day VaR | 99 % 1-day ES |
|---|---|---|
| historical | $54,681 | $67,530 |
| age-weighted (lam 0.98) | $49,402 | |
| FHS per factor (lam 0.94) | $41,337 | |
| parametric normal | $46,864 | $53,690 |
| parametric t(6) | $51,691 | $66,327 |
| MC normal (100k paths) | $45,238 | $51,535 |
| MC t(6) (100k paths) | $49,853 | $63,097 |

| 500-day backtest, 99 % | exceptions (expected 5) | Kupiec p | zone |
|---|---|---|---|
| parametric normal | 14 | 0.0009 | yellow |
| plain historical | 10 | 0.048 | yellow |
| FHS per factor | 4 | 0.64 | green |

The 2020 replay loses $254k, 4.6 times the historical 99 % VaR, and
delta-gamma misprices it by $93k. Full revaluation of 100,000 Monte Carlo
paths on the demo book takes 0.03 s (`monte_carlo_pnl(..., method="full")`).

## Test suite

| Command | Result | Wall time |
|---|---|---|
| `python -m pytest -p no:cacheprovider -q` | 579 passed | 28 s alone, 72 s with other jobs on the machine |

`tests/test_fuzz_public_api.py` sends NaN, infinities, empty arrays and
out-of-domain values to every name in `eq_var.__all__`. Golden vectors for
the C++ and Rust ports are in `tests/golden/`.

## Layout

```
src/eq_var/
  portfolio.py           RiskFactor, positions, Portfolio, vectorised full and delta-gamma revaluation
  historical_var.py      plain, BRW, per-factor FHS, EWMA (S8), sqrt-time and overlapping 10-day
  parametric_var.py      sample and EWMA covariance, normal / t / Cornish-Fisher with validity check
  monte_carlo_var.py     cholesky with capped jitter and Higham / clip repair, simulation, SE, CI
  expected_shortfall.py  empirical (S2), normal and t closed forms, bootstrap SE
  backtesting.py         Kupiec, Christoffersen, Basel, Acerbi-Szekely, rolling driver
  stress_testing.py      replays, ladders, reverse stress
  _tail.py               the shared S2 quantile and ES estimator
  _validation.py         shared input checks
  data/                  seeded synthetic generators and the demo book; live.py (guarded)
```

The C++ and Rust equity engines share the conventions and golden vectors.
Their historical and Monte Carlo VaR also take European options, repriced in
full in every scenario by `option_reval` (BSM), with parity to
`eq_var.revaluation` pinned at 1e-11 for historical VaR/ES. They hold vol
fixed per scenario in the VaR drivers (their `full_revaluation_pnl` takes
the same three vol modes as Python); Taylor approximations and P&L explain
stay in Python.

Docs: [METHODOLOGY.md](docs/METHODOLOGY.md) (conventions, model choice,
assumptions), [VALIDATION.md](docs/VALIDATION.md) (evidence and failure
modes), [DESK_GUIDE.md](docs/DESK_GUIDE.md) (workflow, capital link, stress
review).

## Choosing an engine

`eq_var.engine` wraps the main functions of this package with
`engine="python" | "cpp" | "rust"`. The default, `"python"`, is this
package unchanged and stays the reference. `"cpp"` and `"rust"` call the
compiled ports through the `qp_cpp` and `qp_rust` extensions (install
them from the repository root with `pip install ./bindings/cpp-nanobind`
or `pip install ./bindings/rust-pyo3`). An unknown name raises
`ValueError`; a missing extension raises `ImportError` with that install
command. The compiled results agree with the reference to the
cross-language tolerances listed in the module docstring, checked over
10,000 random inputs per engine by `tests/test_engine_differential.py`.

## Risk attribution, FRTB IMA, tails, revaluation and limits (risk attribution, FRTB IMA, tails, revaluation, limits)

Seven new public modules and one private helper (`_r3checks`). They are
asset-agnostic, ship as the same file in
`eq_var` and `fx_var`, and are imported from their submodule (they are not
re-exported from the package root yet):

| Module | What it gives you |
|---|---|
| `eq_var.risk_attribution` | Euler marginal, component and incremental VaR and ES, closed form and from scenarios |
| `eq_var.frtb_ima` | Liquidity-horizon ES cascade, stressed ES, IMCC, IMA capital, desk backtest, PLA test |
| `eq_var.stressed_var` | Basel 2.5 stress-window search, stressed VaR, the VaR + sVaR capital formula |
| `eq_var.evt` | Peaks-over-threshold GPD fit, POT VaR and ES, mean-excess function |
| `eq_var.copula_mc` | Student-t copula Monte Carlo VaR and ES with any marginals |
| `eq_var.revaluation` | Full revaluation of option books under scenarios, Greek approximations, P&L explain |
| `eq_var.limits` | Limit hierarchy with utilisation, warning and breach status |

```python
import numpy as np
from eq_var.risk_attribution import parametric_attribution
a = parametric_attribution([1e6, -4e5], np.array([[4e-4, 1.2e-4], [1.2e-4, 2.25e-4]]), 0.01)
a.total, a.component, a.percent   # components sum to the total (Euler)
```

The compiled VaR engines gain the option revaluation part only
(`option_reval` in C++ and Rust, used by their historical and Monte Carlo
VaR); see their READMEs.
