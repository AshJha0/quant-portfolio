# 04: Fixed Income Pricing and Risk Analytics (`fi_rates`)

Government and corporate bond pricing and rates risk:

```
Yield curve construction -> Bootstrapping -> Bond pricing -> Duration
       -> Convexity -> Key-rate duration -> Scenario analysis
```

* **Curves** (`curve.py`): discount curve with four interpolations:
  log-linear on discount factors (default, piecewise-constant forwards),
  linear on zeros (kept to show its forward sawtooth), PCHIP on zeros
  (C1 zeros, C0 forwards) and Hagan-West monotone convex (continuous,
  local forwards that keep the monotonicity of the inputs, with the
  paper's optional positivity collar). Long-end
  extrapolation is flat and warned.
* **Bootstrap** (`bootstrap.py`): sequential bootstrap from deposits, FRAs
  and par swaps (Brent per pillar), plus a coupon-bond bootstrap. Inputs
  reprice to 1e-10 (measured 4e-16) under local interpolation.
* **Dates** (`daycount.py`, `calendars.py`, built on `qp_core.daycount`
  and `qp_core.holidays`): ACT/365F, ACT/360, 30/360US,
  ACT/ACT-ISDA, ACT/ACT-ICMA; schedules with the end-of-month rule;
  rule-based business-day calendars (Good Friday, Thanksgiving, UK bank
  holidays and weekend substitutes included, checked against QuantLib
  2000-2030) with following, modified following and preceding.
* **Bonds** (`bond.py`): fixed coupon (clean, dirty, accrued, street YTM),
  government bonds on ACT/ACT-ICMA with exact c/f coupons, zero coupon,
  FRN (par at reset; mid-period with the current fixing), annuities,
  z-spread solver.
* **Risk** (`risk.py`): Macaulay and modified duration, convexity, DV01,
  analytic and bump-based; Taylor error table; portfolio aggregation.
* **Key rates** (`keyrates.py`): triangular-bump KRDs at 2/5/10/30y.
* **Scenarios** (`scenarios.py`): parallel, steepener, butterfly and
  historical episodes (2013 taper, 2022 hiking, 2008 from UST CMT
  year-end changes); full revaluation vs duration estimate; funded carry
  with a pull-to-par and curve roll-down split.
* **Data** (`data/synthetic.py`): seeded synthetic quote sets and a
  government plus corporate sample book; `data/live.py`: optional FRED CMT
  loader, never used by tests.

Limitations: the bond and risk pipeline is single-curve (OIS discounting
and tenor basis live in `multicurve.py`, below), curve
inputs in year fractions rather than dated instruments, partial holiday
calendars, KRDs in zero-rate space with no par-rate Jacobian.

## Quickstart

```bash
cd python/equity/04-fixed-income
pip install -e ../../core   # qp-core, the shared package (not on PyPI)
pip install -e .[test]
python -m pytest tests -q          # see the table below
python examples/run_pipeline.py    # writes nothing to disk
```

```python
import datetime as dt
import fi_rates as fr
from fi_rates.data import market_quotes, sample_portfolio

curve = fr.bootstrap_curve(market_quotes("upward", seed=42))
settle = dt.date(2026, 8, 18)
book = sample_portfolio(settle)
print(fr.portfolio_risk(book, settle, curve))
print(fr.krd_report(book, settle, curve))
print(fr.scenario_pnl_table(book, settle, curve,
                            list(fr.HISTORICAL_SCENARIOS.values())))
```

| Command | Result |
|---|---|
| `python -m pytest tests -q` | 430 passed, about 10 s |
| `python examples/run_pipeline.py` | about 0.5 s inside the script, 6 s with interpreter start-up |

## Headline numbers

`python examples/run_pipeline.py`, synthetic quotes, seed 42, settlement
2026-08-18:

| Quantity | Value |
|---|---|
| Zero curve, 3m to 30y | 3.06% to 4.55% |
| Max quote repricing error (11 quotes) | 4.0e-16 |
| Book market value | 1,739,957 |
| Book modified duration | 5.53 |
| Book DV01 per bp | 980.2 |
| KRD ladder 2/5/10/30y | 154 / 278 / 368 / 180 |
| KRD sum vs parallel DV01 | 5.3e-8 relative |
| 2022 hiking / 2013 taper / 2008 P&L | -228,406 / -94,037 / +205,552 |
| 100bp Taylor error, 10y govt, duration vs duration+convexity | -0.401 vs -0.014 per 100 |
| 1y funded carry + roll, 5y govt, repo 3.10% | 1.02 per 100 |

## Docs

* [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): bootstrap and
  interpolation choices, day counts and calendars, conventions,
  assumptions register.
* [`docs/VALIDATION.md`](docs/VALIDATION.md): round trips, identities,
  KRD residuals, failure modes with numbers.
* [`docs/DESK_GUIDE.md`](docs/DESK_GUIDE.md): daily workflow, ladder
  hedging, carry, scenarios, controls.

## November 2026 additions

- `multicurve.py`: OIS discount curve and a separate projection curve per
  index tenor, both bootstrapped exactly from par quotes.
- `rate_options.py`: Black-76 and Bachelier caps, floors and swaptions on
  the multi-curve setup, with implied-vol inversion.
- `hull_white.py`: one-factor Hull-White fitted to the curve; zero-bond
  options, caps and Jamshidian swaptions; an exact-OU Monte Carlo. The
  short rate drives the OIS discount curve and swaptions can project their
  floating leg on a separate forward curve (two-curve, deterministic
  spread). `calibrate_hull_white` fits `(a, sigma)`, or `sigma` with `a`
  fixed, to a grid of swaption vols (normal or Black).

Tests: `tests/test_round3_rates.py` (14 tests) and
`tests/test_round3_fix_rates.py` (monotone convex, two-curve Hull-White,
calibration).
