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
| `qp_core.holidays` | Rule-based settlement calendars for USD, EUR, GBP, JPY, CHF, CAD, AUD and the SIFMA US government bond market `USD-GOVT` (alias `SIFMA`; every code in `CALENDAR_CODES`) (2000 to 2099, moveable feasts, observance rules and proclamation holidays included); `HolidayCalendar`, `get_calendar`, `joint_calendar` and the four roll conventions; `is_business_day`, `adjust_following`, `add_business_days`; FX spot dates (`spot_lag`, `fx_spot_date`); runtime one-off closures (`add_closure`, test hook `extra_closures`). Matches QuantLib for every year 2000-2030 | qp_core.daycount, fi_rates, fx_rates, eq_credit (`isda_cds`), fx_options (re-exported as `fx_options.holidays`) |
| `qp_core.vol` | Volatility maths shared by the surface and rates packages: `ssvi` (fit and no-arbitrage checks), `dupire` (local vol from total variance), `sabr`, `sabr_pde`, `essvi` | eq_surface, fx_surface, fi_rates, eq_options |
| `qp_core.rng` | One seed-to-generator rule (`numpy.random.default_rng`, with one validation of what counts as a seed) for the Python Monte Carlo routines | not yet imported by the packages; their own seed helpers move onto it in a later round |
| `qp_core.calendars` | Deprecated shim over `qp_core.holidays`; importing it warns, removal in v4.0.0 | none (a test fails on any import) |
| `qp_core.normal` | `norm_cdf`, `norm_pdf` for scalar floats (`erfc` form, full lower-tail precision) | eq_options, fx_options |
| `qp_core.black_scholes` | Black-Scholes in cost-of-carry form (`b = r - q = r_d - r_f`): `carry_d1_d2`, `carry_price`, `carry_greeks` and the NumPy array forms, with a worked example in the module docstring | eq_options (prices, Greeks, arrays), fx_options (prices, d1/d2, arrays, gamma/vega/vanna/volga) |
| `qp_core.iv_core` | Implied volatility from normalised Black, safeguarded Halley (the same algorithm as `qp/iv_core.hpp` and `qp_core::iv`) | eq_options, fx_options |
| `qp_core.daycount` | ACT/365F, ACT/360, 30/360US, ACT/ACT-ISDA, ACT/ACT-ICMA; month arithmetic; coupon schedules; money-market day counts | fi_rates, fx_rates |
| `qp_core.validation` | `require_finite`, `finite_scalar`, `finite_series` | fx_rates, fx_pairs |
| `qp_core.metrics` | Downside deviation, drawdown (equity, returns, P&L), annualisation | eq_pairs, eq_port, eq_algo, eq_risk_metrics, fx_pairs, fx_port |
| `qp_core.instruments`, `qp_core.market` | The trade layer (W10): `Instrument` (`pv(market)`, `cashflows()`, `risk(market)`), `Trade`, `Portfolio` in a reporting currency, `Money`; `MarketSnapshot` (one immutable object every `pv` reads) with curves and surfaces; `instruments.adapters` converts the older per-package trade classes | new code; old per-package classes keep working |
| `qp_core.store` | Append-only trade, snapshot and PV stores, so a past business date can be restated exactly | the trade layer |
| `qp_core.trees` | `crr_lattice`, the one Cox-Ross-Rubinstein kernel; FX passes `(r_d, r_f)` in place of `(r, q)` (Garman and Kohlhagen 1983) | eq_options (`binomial.crr_price`), fx_options (`binomial.binomial_price`) |
| `qp_core.exotics` | Barriers, touches, double barriers, vanna-volga, Asians, Longstaff-Schwartz Americans, TARFs, particle-method LSV; closed forms checked against published tables | eq_options, fx_options, research |
| `qp_core.risk` | `jacobian` (quote-to-node Jacobian, cross-gamma), `vega_grid` (bucketed vega), `pnl_explain`, `smile_dynamics` (sticky strike, sticky delta, sticky moneyness) | fi_rates, fx_options, eq_options |
| `qp_core.scenarios` | Historical stress scenarios; each shock reproduces its source-series move on its window to 1e-14 relative | fi_rates, VaR packages |
| `qp_core.capital` | `saccr` (SA-CCR EAD, BCBS 279 / CRE52, the Basel worked examples in the tests) and `frtb_sbm` (FRTB standardised sensitivities-based method) | research, xva |
| `qp_core.xva` | `hull_white` (one-factor model, Jamshidian swaptions, exact simulation), `exposure` (EE, ENE, PFE, effective EPE), `adjustments` (CVA, DVA, FVA); CVA of a swap is checked against a swaption strip | research; Rust (`qp_core::xva`) and C++ (`qp/xva`) parity |
| `qp_core.credit_index` | CDX and iTraxx on the ISDA CDS Standard Model: series and rolls, factor and defaults, quote conversion, intrinsic value and index basis; the `isda` kernel has Rust and C++ ports | eq_credit, research |
| `qp_core.hedging` | Hedging with transaction costs (W26): time, Whalley-Wilmott band and Leland rules, simulation and a hedge backtest | qp_exec (CRB), research |
| `qp_core.aad` | Reverse-mode adjoint differentiation: `tape`, applied to Black-Scholes, a pathwise Monte Carlo pricer and a swap PV off the dated OIS bootstrap; Rust (`rust/aad`) and C++ (`cpp/aad`) ports of the same tape | research, docs |
| `qp_core.eventlog` | Append-only, hash-chained event log with write-ahead ordering and an fsync policy (`EventLog`, `verify`) | qp_oms (journal, recovery) |
| `qp_core.model` | The `@model` tag and `python -m qp_core.model check` against the inventory in `docs/models/models.toml` | every tagged model |
| `qp_core.errors`, `qp_core.logconfig`, `qp_core.engines` | Typed errors (`QPError`, `InputError`, `ConvergenceError`, `DataError`); one structured log record per entry-point run; the `engine="python"/"cpp"/"rust"` dispatch rule | every package and both bindings |

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

`qp_core.holidays` has the full rules for its seven centres and USD-GOVT, but only the
closures known when the rules were written. Onshore calendars for NDF
currencies (BRL, INR, KRW, TWD) are not shipped; those currencies count
weekends only. A newly announced one-off closure (a state funeral, say) goes in
two steps: `add_closure("GBP", date)` at runtime straight away, then a
dated line in the currency's rule function with its official notice cited
and a test that pins it. The module docstring spells out the procedure.
