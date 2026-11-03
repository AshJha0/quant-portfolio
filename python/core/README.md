# qp-core

Shared Python code for the quant-portfolio projects. Distribution name
`qp-core`, import name `qp_core`. Pure Python; it depends on numpy, scipy and
pandas only.

```
pip install -e python/core
python -m pytest -p no:cacheprovider python/core
```

`qp-core` is also a member of the root uv workspace, so `uv sync --all-packages`
from the repository root installs it with every project that depends on it.

## Modules

| Module | What it holds | Used by |
|---|---|---|
| `qp_core.stats` | Sharpe ratio, Lo (2002) Sharpe standard error with a Newey-West long-run variance, deflated Sharpe ratio, PBO by CSCV, circular block bootstrap, seed distributions, `TrialLog` | eq_pairs, eq_algo, eq_regime, eq_signal_backtest, fx_pairs, fx_algo, fx_regime (through their `research_stats` modules) |
| `qp_core.execution` | `ExecutionConfig` and `fill_price`: the t+1 fill rule | the same seven strategy projects |
| `qp_core.calendars` | Partial business-day calendars (weekends and fixed-date holidays) and the four roll conventions | fi_rates, fx_rates |
| `qp_core.holidays` | Full rules-based settlement calendars for USD, EUR, GBP, JPY, CHF, CAD, AUD (2010 to 2099, moveable feasts and observance rules included), `is_business_day`, `adjust_following`, `add_business_days`, and runtime one-off closures (`add_closure`, test hook `extra_closures`) | eq_credit (`isda_cds`), fx_options (re-exported as `fx_options.holidays`) |
| `qp_core.normal` | `norm_cdf`, `norm_pdf` for scalar floats (`erfc` form, full lower-tail precision) | eq_options, fx_options |
| `qp_core.black_scholes` | Black-Scholes in cost-of-carry form (`b = r - q = r_d - r_f`): `carry_d1_d2`, `carry_price`, `carry_greeks` and the NumPy array forms, with a worked example in the module docstring | eq_options (prices, Greeks, arrays), fx_options (prices, d1/d2, arrays, gamma/vega/vanna/volga) |
| `qp_core.iv_core` | Implied volatility from normalised Black, safeguarded Halley (the same algorithm as `qp/iv_core.hpp` and `qp_core::iv`) | eq_options, fx_options |
| `qp_core.daycount` | ACT/365F, ACT/360, 30/360US, ACT/ACT-ISDA, ACT/ACT-ICMA; month arithmetic; coupon schedules; money-market day counts | fi_rates, fx_rates |
| `qp_core.validation` | `require_finite`, `finite_scalar`, `finite_series` | fx_rates, fx_pairs |
| `qp_core.metrics` | Downside deviation, drawdown (equity, returns, P&L), annualisation | eq_pairs, eq_port, eq_algo, eq_risk_metrics, fx_pairs, fx_port |

Every project keeps its old import paths. For example
`from eq_pairs.research_stats import lo_sharpe_se` still works; the module
re-exports `qp_core.stats`.

## Conventions

Each one is stated once here and in the module docstring.

**Downside deviation.** `sqrt(mean(min(r - mar, 0)^2))` over all `n`
observations, not only the ones below `mar`. Every project already used this.
What a Sortino ratio returns when the downside deviation is zero (NaN, inf or
an exception) stays a project choice.

**Drawdown of a return series.** A positive fraction of the running peak of
wealth, `1 - W_t / max W`, with wealth starting at 1 and that start included
in the peak (`include_start=True`). `eq_algo.evaluation.max_drawdown` and
`fx_port.metrics.max_drawdown` measure from the first post-return wealth
instead. They pass `include_start=False` so their numbers do not change.

**Drawdown of a P&L series.** Equity is `cumsum(pnl)` from 0, and the running
peak includes that 0.

**Day counts.** `30/360` and `30/360US` are the same rule (no end-of-February
adjustment). `year_fraction` allows `end < start` (negative fraction) unless
`allow_negative=False`. `fi_rates.daycount.year_fraction` allows it;
`fx_rates.daycount.year_fraction` and `money_market_year_fraction` do not.

**Fill model.** `mode="next"` fills at `p_t + f (p_{t+1} - p_t)` with
`f = fill_fraction` in (0, 1]. `mode="close_auction"` fills at `p_t` and pays
`auction_cost_bp`. The default auction cost is 10 bp; `fx_algo` subclasses
the config with 0.5 bp.

## Limitations

`qp_core.calendars` knows fixed-date holidays only; moveable feasts and
observance rules are missing there. `qp_core.holidays` has the full rules
for its seven centres, but only the closures known when the rules were
written. A newly announced one-off closure (a state funeral, say) goes in
two steps: `add_closure("GBP", date)` at runtime straight away, then a
dated line in the currency's rule function with its official notice cited
and a test that pins it. The module docstring spells out the procedure.
