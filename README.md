# quant-portfolio

[![ci](https://github.com/AshJha0/quant-portfolio/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AshJha0/quant-portfolio/actions/workflows/ci.yml)

Standard quantitative-finance models written carefully, tested against
independent references, and used for a small amount of honest research on
free data. It is for anyone who wants to read or check a worked
implementation of a model, not for running a live book.

Three things are worth looking at first:

- **The research workflow.** Studies are registered and hashed before the
  hold-out data is touched, a checker refuses results that do not match a
  signed registration, and null results are published as they came out.
  See [`research/README.md`](research/README.md).
- **Cross-language checks.** The option and VaR libraries exist in Python,
  C++ and Rust. All three are tested against the same mpmath reference
  values and against each other on random inputs; a skipped comparison
  fails CI. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **One option kernel, checkable builds.** Equity and FX Black-Scholes
  share one cost-of-carry kernel per language, moved there with no output
  bit changed. CI installs every Python package with its hash checked
  against `uv.lock`, and flags a slowdown only when the same runner shows
  it in alternating runs ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

Quickstart (Python 3.11 to 3.13, about two minutes):

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv/Scripts/activate
pip install -e python/core -e python/equity/01-options-pricing
python python/equity/01-options-pricing/examples/run_pipeline.py
```

New here? [docs/START_HERE.md](docs/START_HERE.md) lists the five pages
worth reading in ten minutes.

## What is in it

quant-portfolio is a set of study implementations of standard
quantitative-finance models: option pricing, volatility forecasting, VaR and
expected shortfall, fixed income, pairs trading, credit scoring, portfolio
optimisation, execution scheduling, volatility surfaces and regime detection.
Each area is written in Python with equity conventions and again with FX
conventions. The option-pricing and VaR libraries also have C++ and Rust
ports, tested against the Python code and, for options, against an mpmath
oracle. All results come from seeded synthetic data unless a page names a
dataset. The code has not been used on a live book.

The plan has three stages. The repository is at stage one, and the
November 2026 release started parts of stages two and three.

| Stage | What it means | Status |
|---|---|---|
| 1. Study portfolio | Correct, tested implementations with plain documentation | In progress |
| 2. Research platform | Free public data, pre-registered experiments, reported results | Started: ECB and FRED fetchers; one study scored on the 2022-2025 hold-out (null result), one withdrawn before it; two signed registrations scored once on forward data from 2026-11-03, on or after 2028-11-13 |
| 3. Desk library | Shared core, bindings, option revaluation in the compiled VaR engines | Started: shared core with one cost-of-carry option kernel in Python, C++ and Rust; bindings and the `engine=` switch; full option revaluation with vol scenarios in all four compiled VaR engines; batch and NumPy array pricers; CMake package and CI-built wheels. Not yet: dated rates schedules, trade and portfolio objects, order management, execution venues |

**Status.** See the CI badge above. Test counts are not typed into the docs.
The CI `counts` job writes the executed pass counts to `counts.json` and shows
them in the run summary.

Further reading in [`docs/`](docs/README.md):
[ARCHITECTURE](docs/ARCHITECTURE.md) (layout, conventions, test pipeline),
[DIAGRAMS](docs/DIAGRAMS.md), [LEARN](docs/LEARN.md) (study guide and
self-test question bank), [COOKBOOK](docs/COOKBOOK.md) (recipes) and
[MARKET_RISK](docs/MARKET_RISK.md) (the VaR, ES and backtesting workflow).
Changes are listed in [CHANGELOG.md](CHANGELOG.md).

```
quant-portfolio/
├── CONVENTIONS.md     # engineering and documentation rules for every project
├── CONTRIBUTING.md    # setup, test gate, golden regeneration
├── CHANGELOG.md
├── docs/              # architecture, study guide, recipes, market-risk workflow
├── data/              # free-data fetch scripts; downloads go to data/raw/ (git-ignored)
├── research/          # pre-registrations, studies, results, the hash checker
├── pyproject.toml     # uv workspace: python/core and the 23 projects (uv.lock)
├── python/
│   ├── core/          # qp-core: shared Python code used by 16 projects
│   ├── equity/01-…10-…    # 10 equity projects
│   ├── fx/01-…10-…        # 10 FX projects (separate code from equity)
│   └── foundations/       # 3 single-asset projects
├── cpp/               # qp::core (header-only) + 4 engines, C++23, CMake presets, GoogleTest
├── rust/              # Cargo workspace: qp-core + the same 4 engines, Rust edition 2024
└── bindings/          # qp-cpp (nanobind) and qp-rust (PyO3) Python packages
```

## Areas

31 sub-projects: 23 Python projects, 4 C++ engines and 4 Rust crates. Three
shared libraries sit beside them and are not counted as sub-projects:
[`python/core`](python/core/README.md) (`qp-core`),
[`cpp/core`](cpp/core/README.md) (`qp::core`) and
[`rust/core`](rust/core/README.md) (the `qp-core` crate). Two binding
packages expose the compiled engines to Python:
[`qp-cpp`](bindings/cpp-nanobind/README.md) and
[`qp-rust`](bindings/rust-pyo3/README.md).

| # | Area | Equity | FX |
|---|------|--------|-----|
| 1 | Options pricing and Greeks | `python/equity/01-options-pricing` (+ C++/Rust) | `python/fx/01-options-pricing` (+ C++/Rust) |
| 2 | Volatility modelling and forecasting | `python/equity/02-volatility-modeling` | `python/fx/02-volatility-modeling` |
| 3 | Market risk: VaR and expected shortfall | `python/equity/03-var-es-engine` (+ C++/Rust ports, linear and option books) | `python/fx/03-var-es-engine` (+ C++/Rust ports, linear and option books) |
| 4 | Fixed income pricing and risk | `python/equity/04-fixed-income` | `python/fx/04-fixed-income` |
| 5 | Statistical pairs trading | `python/equity/05-pairs-trading` | `python/fx/05-pairs-trading` |
| 6 | Credit risk and PD modelling | `python/equity/06-credit-risk` | `python/fx/06-credit-risk` |
| 7 | Portfolio optimisation and risk allocation | `python/equity/07-portfolio-optimization` | `python/fx/07-portfolio-optimization` |
| 8 | Execution scheduling and TCA | `python/equity/08-algo-execution` | `python/fx/08-algo-execution` |
| 9 | Volatility surface and stochastic volatility | `python/equity/09-vol-surface` | `python/fx/09-vol-surface` |
| 10 | Regime switching | `python/equity/10-regime-switching` | `python/fx/10-regime-switching` |

Foundations projects are smaller: one asset, Python only, no FX version.

| # | Project | Path | What it is |
|---|---|---|---|
| F1 | Risk metrics | `python/foundations/01-risk-metrics` | Single-asset volatility, VaR, ES and Sharpe toolkit on bundled synthetic data. Three VaR methods side by side. No portfolio risk; see `python/equity/03-var-es-engine` for that. |
| F2 | Trading signal backtest | `python/foundations/02-trading-signal-backtest` | Moving-average crossover with fills modelled at t+1 as p_t + f(p_{t+1} - p_t), f = 0.5, transaction costs, in-sample and out-of-sample splits and walk-forward evaluation. |
| F3 | Black-Scholes replication | `python/foundations/03-black-scholes-replication` | Black-Scholes-Merton from scratch with `math.erfc` only, N(x) = 0.5 erfc(-x/sqrt 2), checked against a Monte Carlo pricer, analytic identities and finite-difference Greeks. `python/equity/01-options-pricing` is the fuller implementation. |

Every project has `src/`, `tests/`, `examples/run_pipeline.py` and
`docs/{METHODOLOGY,VALIDATION,DESK_GUIDE}.md`. See `CONVENTIONS.md`.

## What the November 2026 release added

Most of what follows is new code beside the old code. Two changes moved
existing numbers on purpose, each with a before/after table: the deprecated
P&L-level filtered historical VaR now uses the same quantile convention as
every other VaR path (one golden moved, [docs/MARKET_RISK.md](docs/MARKET_RISK.md)),
and the Python FX `analytic_greeks` now uses the shared normal CDF
(changes of at most 2.13e-14 relative to max(1, |value|), CHANGELOG).

### Engine switch

The four packages with compiled ports now have an `engine` module whose
functions take `engine="python" | "cpp" | "rust"`. `"python"` is the default
and calls the reference function unchanged. The rule for mapping a name to a
backend lives once, in `qp_core.engines` (`check_engine`, `load_engine`,
`engine_available`): an unknown name raises `ValueError`; a compiled engine
whose extension is not installed raises `ImportError` with the
`pip install ./bindings/...` command.

| Module | Functions |
|---|---|
| `eq_options.engine` | `bs_price`, `bs_greeks`, `implied_vol`, `crr_price`, `bs_price_batch` |
| `fx_options.engine` | `gk_price`, `analytic_greeks`, `implied_vol`, `binomial_price`, `gk_price_batch` |
| `eq_var.engine` | `historical_var`, `expected_shortfall`, `parametric_var` |
| `fx_var.engine` | `empirical_var_es`, `normal_var`, `normal_es`, `t_var`, `t_es`, `var_covar` |

`tests/test_engine_differential.py` checks each compiled engine against the
reference on 10,000 random inputs per group (smaller samples where the
reference is O(n^2) or uses pandas), at the tolerances the binding tests
already use. Without the extensions installed, the compiled cases skip.
Monte Carlo pricing, MC VaR and the backtests have no `engine=` wrapper yet,
and neither do the new Leisen-Reimer, Bachelier and higher-Greek ports.

### Research workflow and real data

Two new top-level folders:

- [`data/`](data/README.md) holds fetch scripts only. `data/fetch/qp_fetch.py`
  reads ECB reference rates and FRED series and writes each file to
  `data/raw/` (ignored by git) with a `.sha256` file beside it.
- [`research/`](research/README.md) holds pre-registrations, studies, results
  and the checker.

A study runs in four steps:

1. **Draft.** A file in `research/prereg/` fixes the hypothesis, data,
   parameters, costs, primary metric and decision rule. Its registration
   hash is the SHA-256 of the text above `## Signature`.
2. **Sign.** The signer fills `signed_by`, `signed_on` and `sha256`.
   Signing does not change the hash.
3. **Hash check.** `python research/tools/prereg.py check` refuses a result
   whose `prereg_sha256` does not match its registration, and any result
   whose period reaches 2022-01-01 or later unless the registration is
   signed with that hash.
4. **Run on the hold-out.** Only after signing. Until then every study calls
   `training_only`, which raises on any date from 2022-01-01.

| Registration | State |
|---|---|
| `pairs-fx-g10` | Signed on 2026-11-02, then withdrawn the same day before any hold-out data was read ([DR-R4-5](research/README.md#dr-r4-5-pairs-registration-withdrawn-2026-11-02)). Training result only; uninformative. |
| `regime-fx-basket` | Signed by the maintainer on 2026-11-02. Hold-out run once on 2026-11-02 (2022-2025): H0 not rejected. The hold-out is spent. |
| `pairs-fx-g10-v2` | Signed on 2026-11-02. Level-based z with a simulated power check. Scored once on forward data, 2026-11-03 to 2028-11-02, on or after 2028-11-13. |
| `momentum-fx-g10` | Signed on 2026-11-02. Monthly G10 spot momentum. Scored once on forward data, 2026-11-03 to 2028-11-02, on or after 2028-11-13. Low power: a fail is the expected outcome even if a modest effect exists. |
| Order-book execution study | Unsigned draft in `research/prereg/`. Not done: its data needs an account and terms of use the owner decided not to accept. |

Two studies are not done because their data needs an account, which the
owner decided not to open: equity pairs on a point-in-time US universe and
the order-book execution study. The free sources used instead are listed in
[data/README.md](data/README.md).

The training-period numbers (ECB data, 1999 to 2021) are in
`research/results/`. They are not out-of-sample evidence. The pairs
result is uninformative: the signal as registered almost never fires, even
on simulated pairs that are truly cointegrated, so the registration was
withdrawn before the hold-out (DR-R4-5 in
[`research/README.md`](research/README.md)). For the record, the pairs study
had a net Sharpe of 0.24 with a block-bootstrap 95% interval of
[-0.17, 0.46], and held a position on only 79 of 4,867 days. Its parameters
were changed twice on training data before signing, and the registration
counts all three variants as trials. The regime study did not support its
hypothesis in training: the high-volatility state had a higher mean
next-day return than the low state (HAC t = +0.27).

The regime hold-out (2022-01-03 to 2025-12-31, run once) agreed with
training. On the 161 of 1,019 days classified high-volatility the basket
earned +12.5% a year annualised, against -4.0% on the other days; HAC
t = +0.94 with 4 lags. The registered rule rejects H0 only if t < -1.645,
so H0 is not rejected. The effect has the opposite sign to the hypothesis,
but t = +0.94 is not evidence of a positive effect either. Result file:
`research/results/regime_fx_basket_holdout.json`.

Because the 2022-2025 window can be used only once, the two later
registrations are scored on data nobody had seen when they were signed:
ECB fixes from 2026-11-03 onward, collected by a scheduled job into a
forward log (`research/forward/`) and scored once, on or after 2028-11-13,
by `research/forward/evaluate.py`. Before that date the script prints
figures labelled INTERIM, decides nothing and writes no result: acting on
them would be optional stopping (LEARN Set 19). The
momentum test has low power. In a simulation with planted momentum
(`research/power/momentum-fx-g10.json`, 200 runs per cell) it rejects H0
about 13% of the time when the true Sharpe ratio is about 0.4, and needs a
Sharpe ratio of about 1.3 to reject half the time. A fail on the forward
data is therefore the expected outcome and will say little either way; the
registration states this before any forward data exists.

Every training, stress and calibration run now appends a line to
`research/results/trials.jsonl` (commit, registration hash, parameters,
metric), so trial counts come from a log. A calibration of the pairs
selection against a null that resamples whole days found 53 pairs in
2003-2021 against 22.2 expected by chance (p = 0.010), but only 4 of 19
years beat the null and Benjamini-Hochberg at 5% keeps 14 pairs. Both
forward strategies stay negative in training at 1, 3 and 5 bp costs
(`research/README.md`, "Robustness checks").

### New models and their oracles

Every new model is tested against an independent result: a closed form, a
published value, a different numerical method or a limiting case. The
modules are imported by path (for example `eq_surface.ssvi`); they are not
re-exported from the package `__init__`.

| Model | Module | Oracle |
|---|---|---|
| Leisen-Reimer tree | `eq_options.lr_tree` | Black-Scholes; error falls by a factor of 3 to 5 per doubling of steps |
| Crank-Nicolson PDE, American | `eq_options.pde` | Black-Scholes for European; Richardson-extrapolated LR for American; second order |
| Andersen-Lake-Offengeim American | `eq_options.alo` | Richardson-extrapolated LR (2001/4001 steps), four cases, 2e-5 |
| Discrete dividends (escrowed spot, tree) | `eq_options.dividends` | Haug (2007) Roll-Geske-Whaley value 4.3860 |
| Term-structure rates and vol | `eq_options.term_structure` | Merton averaged parameters; flat inputs equal Black-Scholes |
| Bachelier | `eq_options.bachelier` | mpmath quadrature and derivatives at 30 digits |
| SABR (Hagan, Obloj, West) | `eq_options.sabr` | Hagan ATM formula; beta = 1, nu = 0 limit; smoothness through ATM; density sign |
| Charm, speed, colour, zomma, veta | `eq_options.greeks_higher` | `mpmath.diff`, 1e-9 relative |
| Ladders and grids | `eq_options.risk_grids` | Direct repricing; vega buckets sum to book vega |
| FX smile from ATM, RR, BF | `fx_options.smile` | Quote recovery under four delta conventions |
| FX premium styles | `fx_options.premium` | Inverse-pair Garman-Kohlhagen |
| FX cuts, expiry and delivery | `fx_options.cuts` | Hand-checked dates, including DST changes |
| HAR-RV | `eq_vol.har_rv` | Noiseless recursion recovered exactly; simulated HAR within 4 HAC SE |
| OIS multi-curve | `fi_rates.multicurve` | Reprices all OIS and swap quotes to 1e-13; zero basis collapses to the OIS curve |
| Caps and swaptions (Black, Bachelier) | `fi_rates.rate_options` | Parity and ATM closed forms; implied-vol round trip |
| Hull-White 1F, Jamshidian swaptions | `fi_rates.hull_white` | Curve repricing; parity; exact-OU Monte Carlo (200k paths) |
| FX twins: Dupire, SSVI/eSSVI, Bates, HAR-RV, Euler attribution with currency mapping | `fx_surface.dupire`, `fx_surface.ssvi`, `fx_surface.bates`, `fx_vol.har_rv`, `fx_var.fx_attribution` | The equity module where the maths coincides (1e-12 or better); Bates against the Merton series via Garman-Kohlhagen; HAR-RV on 17:00 New York days |
| Dupire local vol | `eq_surface.dupire` | Flat and term-only cases; total-variance route equals price route; local-vol MC reprices the surface |
| SSVI and eSSVI | `eq_surface.ssvi` | Slice equals raw SVI to 1e-15; Gatheral-Jacquier conditions give no butterfly arbitrage |
| Bates | `eq_surface.bates` | No jumps equals Heston; no vol of vol equals the Merton (1976) series; Euler MC within 4 SE |
| Float-float cross-currency swap | `fx_rates.xccy_float` | Constant notional equals the existing `BasisSwap`; one-period MtM equals constant |
| LGD, EAD, margin of conservatism, migration, CDS | `eq_credit.lgd_ead`, `calibration`, `migration`, `cds` | Closed forms and hand-worked cases; see the package VALIDATION |
| Bilateral CVA/DVA, collateral, wrong-way risk; SA-CCR for FX | `fx_credit.xva`, `fx_credit.saccr` | Closed-form MPoR exposure; Monte Carlo within 4 SE; CRE52 steps |
| HRP, robust and resampled MVO, cost QP, factor model, tracking error, Sharpe tests | `eq_port.*` and `fx_port.*` (same code) | Closed forms where they exist (SLSQP agrees to about 1e-6); see the package VALIDATION |
| Euler marginal, component, incremental VaR and ES | `eq_var.risk_attribution`, `fx_var.risk_attribution` | Hand example; Euler sum; finite-difference gradient; MC converges to the closed form |
| FRTB IMA: liquidity-horizon ES, stressed ES, IMCC, capital, desk backtest, PLA | `eq_var.frtb_ima`, `fx_var.frtb_ima` | Hand cascade; sqrt(12) telescoping; table thresholds; scipy cross-check |
| Stressed VaR, Basel 2.5 capital | `eq_var.stressed_var`, `fx_var.stressed_var` | Planted crisis window found; order statistic by hand |
| Closed-form and historical VaR/ES, Kupiec, Christoffersen, Basel traffic light | `eq_var`, `fx_var` and the C++ and Rust VaR engines | An independent mpmath oracle (50 digits, no engine code; 196 values per project): closed forms to 1e-10, historical VaR exact, ES to 1e-12, engine Monte Carlo within 3 SE. It reproduces the BCBS (1996) traffic-light table and the Kupiec (1995) table, which the owner checked against the printed sources on 2026-11-03 |
| Optimal schedule under square-root impact | `eq_algo.power_law_trajectory` | Equals the Almgren-Chriss closed form to 1e-8 of the order at delta = 1; bucket cost converges to the continuous integral |
| EVT peaks over threshold (GPD) | `eq_var.evt`, `fx_var.evt` | Exact GPD quantile and integrated ES; MLE convergence; t(4) tail index |
| t-copula Monte Carlo VaR and ES | `eq_var.copula_mc`, `fx_var.copula_mc` | Kendall tau mapping; tail dependence; Gaussian limit |
| Option revaluation and P&L explain | `eq_var.revaluation`, `fx_var.revaluation` | Finite-difference Greeks; parity; the package `bs_*`/`gk_*` pricers; linear legs exact |
| Limit hierarchy | `eq_var.limits`, `fx_var.limits` | Exact cases |
| Johansen trace test | `research/studies/models.py` | statsmodels `coint_johansen`, 1e-9 relative |
| Kalman hedge ratio | `research/studies/models.py` | Recursive least squares when delta = 0 |
| Settlement calendars, business-day rolls and FX spot dates | `qp_core.holidays` (also Rust `qp_core::holidays`, C++ `qp/holidays.hpp`) | Published holiday dates per centre; QuantLib 1.44 calendars for USD, GBP, TARGET, JPY, CHF, CAD and AUD on every year 2000-2030; one-off closures through `add_closure` |
| FX forwards from the spot date, broken dates, NDFs | `fx_rates` (`MarketState(valuation_date=)`, `forward_for_date`, `ndf`) | Closed form S exp((r_d - r_f)(T - t_s)) to 1e-13; forward-points strip repriced to 1e-10 |
| Standard normal CDF and density | `qp_core.normal` | One shared definition for the option packages; `erfc` form keeps the lower tail |

The C++ and Rust equity options engines also gained Leisen-Reimer,
Bachelier and the higher-order Greeks, checked against the Python values to
1e-12 relative (1e-13 for Bachelier). All four compiled VaR engines gained
`option_reval` (BSM price, Greeks, full-revaluation and delta-gamma P&L),
checked against Python goldens to 1e-11, and their historical and Monte
Carlo VaR now reprice option legs with it in every scenario (historical
VaR/ES matched to Python at 1e-11). Each package's `docs/VALIDATION.md`
has the full oracle list and tolerances.

### Performance kernels

`cpp/core` and `rust/core` gained a Philox4x32-10 generator, O(n) quantile
selection, cache-blocked GEMM and `D^T D`, branch-free batch `exp`, `erfc`
and normal CDF, and (Rust) a persistent thread pool. C++ has a new
`bench_core` latency bench. The equity VaR sample covariance in C++ and Rust
now uses the blocked `D^T D` and gives the same bits as before. The engines
do not use Philox. The batch kernels are used by the vectorised C++
option batches described below, after `erfc_batch` was fixed: its sign
select had compiled to a branch, which is why it did not vectorise in the
table that follows.

Measured with `bench_core` on an Intel Core 5 120U under WSL2, gcc 15.2
`-O3`, default x86-64 target (SSE2 only); p50 in microseconds per call:

> **How these figures were timed.** Old harness: each call timed on its own with rdtscp, empty-call overhead subtracted. They are one laptop run (Intel Core 5 120U, WSL2) kept for the record; no results file was committed for it. Benches now use one method in C++ and Rust: a batch of back-to-back calls sized to at least 2 us per sample, median with p10/p90 ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)). Current figures, measured base against head on one CI runner, are in the `bench-results` artifact of the [CI bench job](https://github.com/AshJha0/quant-portfolio/actions/workflows/ci.yml).

| Kernel | Old | New | Ratio |
|---|---|---|---|
| quantile, n = 10,000 | 1169 (full sort) | 160 (select) | 7.3x faster |
| GEMM 128^3 | 6837 (naive) | 2109 (blocked) | 3.2x faster |
| erfc x1024 | 141 (`std::erfc`) | 212 (`vm::erfc_batch`) | 0.66x, slower |
| exp x1024 | 11.1 (`std::exp`) | 29.1 (`vm::exp_batch`) | 0.38x, slower |

The batch `erfc` and `exp` are slower than libm on this target. Without
AVX2 the loops do not vectorise, and the `erfc` also pays for a 25-step
Clenshaw recurrence and two `exp` calls. A wider target (x86-64-v3) or
`std::simd` might change that; it was not measured. Philox costs 2.7 times
as much per word as `mt19937_64`; it is there for reproducible parallel
streams, not speed. These are one laptop run, and the tails are noisy. See
[`cpp/core/README.md`](cpp/core/README.md).

### Shared option kernel and array pricing

The plain idea: equity Black-Scholes with a dividend yield and FX
Garman-Kohlhagen are one formula with a different cost of carry `b`, so
each language now has one kernel and the packages wrap it. Formally,
`qp_core.black_scholes` (Python), `qp/black_scholes.hpp` (C++) and
`qp_core::black_scholes` (Rust) price `phi [S e^{-qT} N(phi d1) - K e^{-rT}
N(phi d2)]` with `q = r - b`. Worked example: S = K = 100, T = 1, r = 5%,
q = 2%, sigma = 20% gives 9.227005508154036 from `bs_price`, from
`gk_price` with r_d = 5%, r_f = 2%, and from the kernel. On a 5,944-quote
grid no output bit changed in any language when the maths moved.

On top of it: NumPy entry points `eq_options.bs_price_array`,
`bs_greeks_array`, `fx_options.gk_price_array` and `gk_greeks_array`
(keyword-only after `K`; about 19x faster than a scalar loop on 10,000
options), and C++ `_fast` batches that vectorise (GCC 15: 94 to 23 ns per
option for equity prices, 110 to 27 for FX Greeks; tested within 1e-13 and
1e-12 of the exact batch). Recipes and the full table are in
[docs/COOKBOOK.md](docs/COOKBOOK.md), "Arrays, Packaging, CI Checks and the
Forward Log".

### Packaging and supply chain

- The C++ engines install as the CMake package `QuantPortfolio` (see
  "Using the C++ engines from another CMake project" below).
- CI builds wheels of `qp-cpp` and `qp-rust` for CPython 3.11 to 3.13 on
  Linux and Windows, as artefacts only.
- Every Python install in CI comes from `uv.lock` with sha256 hashes
  checked (`.github/scripts/hashed_install.sh`, `requirements/README.md`).
- Optional QuantLib cross-checks for Hull-White, Heston and the ISDA CDS
  run when QuantLib is installed; CI does not install it.
- The Rust crates move to edition 2024.

## How to run

Use a virtual environment outside the repository, one per project if you want
to match CI. Full commands for Python, C++ and Rust are in
[CONTRIBUTING.md](CONTRIBUTING.md).

```bash
python -m venv ~/.venvs/qp && . ~/.venvs/qp/bin/activate
cd python/equity/01-options-pricing
python -m pip install -c ../../../requirements/ci-constraints.txt -e ".[dev]"
python -m pytest -p no:cacheprovider -q
python examples/run_pipeline.py
```

A project declares either a `test` or a `dev` extra; install whichever it has.
Projects that import `qp_core` need `python/core` installed first
(`python -m pip install -e python/core` from the root). The alternative is
the uv workspace: `uv sync --all-packages` from the root installs
`python/core` and all 23 projects from `uv.lock`.

### Using the C++ engines from another CMake project

The four engines and `qp::core` install as one CMake package,
`QuantPortfolio`, with targets `qp::core`, `qp::eqopt`, `qp::eqvar`,
`qp::fxopt` and `qp::fxvar`:

```bash
cmake -S cpp -B build/cpp-install -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-install -j
cmake --install build/cpp-install --prefix "$HOME/.local/qp"
```

A consumer then needs only `find_package(QuantPortfolio 1.0 CONFIG REQUIRED)`
and `-DCMAKE_PREFIX_PATH=$HOME/.local/qp`. `cpp/tests/find_package_consumer`
is that consumer; the `cmake install` CI job builds it against a fresh
prefix with the source tree moved aside, so a build-tree path leaking into
the exported files fails the job. Python wheels for the bindings are built
by the `wheels` CI job and kept as artefacts (see `bindings/*/README.md`).

## How it is tested

CI (`.github/workflows/ci.yml`) runs on every push and pull request. The
single required check is `ci-ok`, which passes only when every job passed.

| Job | What it runs |
|---|---|
| `py core`, `py <project>` | pytest for `python/core` and each of the 23 projects, each in its own venv, on Python 3.11 to 3.13 |
| `lint` | `uv lock --check`, `ruff check python`, `mypy` |
| `goldens --check` | regenerates each C++ golden header and Rust golden module and fails on any difference from the committed JSON |
| `cpp <engine> (<preset>)` | each engine built and tested with the `gcc`, `clang`, `asan` and `tsan` presets, `gcc16-cxx26` in a GCC 16 container, and `msvc` on Windows |
| `clang-format`, `clang-tidy` | formatting check over `cpp/` and clang-tidy per engine with warnings as errors |
| `rust <crate>`, `rust fmt` | `cargo test --release --locked` and clippy with warnings denied for `qp-core` and the four engines; `cargo fmt --check` |
| `bind <package>` | builds `qp-cpp` and `qp-rust` and tests them against the Python reference packages |
| `bench` | runs the release benches and fails when a kernel whose committed p50 is above 1 µs is more than 2x slower |
| `cpp core (<preset>)` | builds and tests `cpp/core`, including the kernel tests and `bench_core` |
| `engine differential` | builds both bindings and runs `tests/test_engine_differential.py`; a skipped compiled case fails the job |
| `research (prereg check, tests)` | `python research/tools/prereg.py check` and the `research/tests` suite |
| `hygiene` | banned-word check, a voice report and the tests for the CI scripts |
| `counts` | sums executed pass counts into `counts.json` and runs `tests/test_docs_counts.py` |

### Cross-language tests

The C++ and Rust engines are ports of the Python design. They are not
independent derivations. Two kinds of test tie them together.

1. **Golden vectors.** For options, `tests/golden/generate_golden.py` computes
   reference values with an mpmath oracle and stores each output with its own
   tolerance in `tests/golden/golden_vectors.json`. The Python consumer is
   `tests/test_golden.py`. For VaR, the generator uses the Python reference
   with fixed inputs, checked from Python by `tests/test_golden_vectors.py`.
   `tools/gen_golden_header.py` (C++) and `tools/gen_golden_rs.py` (Rust)
   turn the JSON into committed source. The CI `goldens --check` job fails
   if the committed source has drifted from the JSON.
2. **Differential tests.** The options projects' `tests/test_differential.py`
   send random inputs to small JSONL shims (Rust `eq_jsonl`/`fx_jsonl`, C++
   `eqopt_jsonl`/`fxopt_jsonl`, found through `QP_RS_EQ_JSONL`,
   `QP_RS_FX_JSONL`, `QP_CPP_EQ_JSONL`, `QP_CPP_FX_JSONL`) and compare the
   answers with Python. The tests skip when the binaries are not built.

Monte Carlo results do not match path by path across languages, because each
language uses its own random number generator. Agreement there is
statistical, within the tolerance stated in each engine's `docs/VALIDATION.md`.

The compiled VaR engines (C++ and Rust, equity and FX) revalue the linear
part of a book as a dot product and reprice each European option leg in
full, by Black-Scholes, in every historical or Monte Carlo scenario. Implied
vol is held at today's level in every scenario, FX option books must report
in USD, and historical option VaR needs a one-day horizon. The historical
numbers match the Python full revaluation to 1e-11 relative.

### Engine latency

How latency is measured now, and where current figures are, is in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md): one harness for C++ and Rust, and
a CI job that compares base and head on the same runner (`bench-results`
artifact). The tables below are older: one laptop run with the previous
harnesses, kept for the record. Intel Core 5 120U (12 threads) under WSL2 on
Windows 11, 2026-11-01. C++ figures are GCC 15.2 at `-O3 -DNDEBUG` (Clang 21.1
figures are in each engine README). Rust figures are rustc 1.99.1, release
profile with `lto = "fat"` and `codegen-units = 1`. WSL2 adds scheduler
jitter, so read p99 as an upper bound. CI does not produce these numbers.
The current harness reports tails of batch means; see
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) for what that label means.

The two tables use different timing methods and should not be compared
with each other. The C++ harness times every call separately (lfence,
rdtscp, timer overhead subtracted). The Rust bench times batches of calls
with `Instant` and divides, which amortises the timer and keeps the
pipeline full, so it reads lower for the same work. Run to run, the
committed C++ `bs_price` p50 moves between about 98 and 149 ns on the same
machine and compiler, roughly plus or minus 25%; treat single numbers
accordingly.

C++:

| Engine | Kernel | p50 | p99 (per call, old harness) |
|---|---|---|---|
| [equity-options](cpp/equity-options-engine/README.md#latency) | `bs_price` | 135 ns | 385 ns |
| [equity-options](cpp/equity-options-engine/README.md#latency) | `try_mc_price`, 1M paths, 4 threads | 23.95 ms | 29.75 ms |
| [fx-options](cpp/fx-options-engine/README.md#latency) | `gk_price` | 113 ns | 329 ns |
| [fx-options](cpp/fx-options-engine/README.md#latency) | `try_mc_price`, 1M paths, 4 threads | 39.60 ms | 46.21 ms |
| [equity-var](cpp/equity-var-engine/README.md#latency) | `try_historical_var`, n = 1000 | 4.5 µs | 8.1 µs |
| [equity-var](cpp/equity-var-engine/README.md#latency) | `try_matvec`, 10000 x 50 scenario P&L | 376.2 µs | 1.02 ms |
| [fx-var](cpp/fx-var-engine/README.md#latency) | `try_pnl`, 1 scenario, 250 positions | 2.0 µs | 2.6 µs |
| [fx-var](cpp/fx-var-engine/README.md#latency) | `var_standard_error`, n = 10000 | 317.6 µs | 700.2 µs |

Rust:

| Engine | Kernel | p50 | p99 of B-call batch means |
|---|---|---|---|
| [equity-options](rust/equity-options-engine/README.md#latency) | `bs_price` | 44 ns | 116 ns |
| [equity-options](rust/equity-options-engine/README.md#latency) | `mc_price`, 1M paths, antithetic + CV | 16.16 ms | 24.82 ms |
| [fx-options](rust/fx-options-engine/README.md#latency) | `gk_price` | 64 ns | 72 ns |
| [fx-options](rust/fx-options-engine/README.md#latency) | `mc_price`, 100k paths, antithetic + CV | 5.79 ms | 7.89 ms |
| [equity-var](rust/equity-var-engine/README.md#latency) | historical VaR (`fast`), n = 250 | 541 ns | 1.79 µs |
| [equity-var](rust/equity-var-engine/README.md#latency) | MC run, 20 factors x 10k paths | 7.08 ms | 19.2 ms |
| [fx-var](rust/fx-var-engine/README.md#latency) | `CompiledBook::pnl`, one scenario | 2.03 µs | 4.1 µs |
| [fx-var](rust/fx-var-engine/README.md#latency) | `monte_carlo_var`, 100k normal | 346 ms | 437 ms |

## Conventions to know

- `alpha` is the tail probability in (0, 0.5) in every engine: 0.01 means 99%
  VaR. FX functions used to take a confidence level; passing 0.99 now raises.
- Historical VaR is the order statistic L(ceil(n alpha)) of the sorted losses.
  ES is the Acerbi-Tasche estimator with the fractional atom.
- `kupiec_pof(*, n_obs, n_exceptions, alpha)` takes keywords only.
- `basel_traffic_light(*, n_obs, n_exceptions, alpha=0.01)` works for any
  window and tail probability, using the binomial CDF cut-offs 0.95 and 0.9999.
- The Monte Carlo VaR standard error is a bootstrap estimate. The density
  (KDE) estimate is a diagnostic only.
- Backtests in areas 05, 08, 10 and F2 fill at t+1 by default, as in F2 above.

Old entry points whose signature or convention changed stay for one release
with a `DeprecationWarning`. CHANGELOG.md lists them.

## Limitations

- All strategy results are on synthetic data with planted structure. A
  strategy that finds the planted signal is a recovery test, not evidence of
  an edge.
- The Kupiec test uses the chi-squared(1) approximation. At 250 days and 99%
  its actual size is about 9.5%, not 5% (`TestKupiecExactSize`).
- Not implemented: structural (Merton/KMV) PD, FRTB NMRF/SES and the
  default risk charge. CHANGELOG.md lists the other known issues.
- The ISDA CDS module has not been compared with ISDA's published test
  grid (a licence click-through). It is checked against QuantLib 1.44
  `IsdaCdsEngine` in an optional test (`test_isda_cds_quantlib.py`, skipped
  unless QuantLib is installed; not run in CI): protection legs to 1e-12,
  risky annuity and par spread to 1e-7 relative. Business-day calendars are optional (`calendar=`); the default is
  weekends only.
- `fx_surface.ssvi` misses five-point FX smiles by up to about one vol
  point on USDJPY (about 0.5 on EURUSD); eSSVI does not close the gap. See
  fx/09-vol-surface VALIDATION, finding F10.
- One result here is out-of-sample: the regime hold-out, which did not
  reject H0. Everything else in `research/results/` is training-period
  only. The two forward registrations cannot be scored before 2028-11-13.
- The bindings (`qp-cpp`, `qp-rust`) cover pricing, Greeks, implied vol,
  binomial, Monte Carlo and the main VaR, ES and backtest calls. They do not
  bind the FX VaR Monte Carlo or book-level VaR, deltas and strikes from
  delta, Black-76, Christoffersen or Acerbi-Szekely. See each binding
  README.
- The `engine=` switch covers closed-form pricing, Greeks, implied vol,
  the binomial tree, batch pricing and the main VaR and ES calls. Monte
  Carlo, the backtests and the newer compiled ports are not behind it.
- Python dependencies are locked with hashes in `uv.lock` and checked in
  CI. `requirements/ci-constraints.txt` mirrors the pins for local `pip -c`
  installs, without hashes.

## References

The sources each area is built from and checked against.

**Options pricing & Greeks (areas 01, F3)**

1. Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy* 81(3), 637–654.
2. Merton, R. C. (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics and Management Science* 4(1), 141–183.
3. Garman, M. B. & Kohlhagen, S. W. (1983). "Foreign Currency Option Values." *Journal of International Money and Finance* 2(3), 231–237.
4. Reiswich, D. & Wystup, U. (2010). "A Guide to FX Options Quoting Conventions." *Journal of Derivatives* 18(2), 58–68.
5. Wystup, U. (2006). *FX Options and Structured Products.* Wiley.
6. Clark, I. J. (2011). *Foreign Exchange Option Pricing: A Practitioner's Guide.* Wiley, ch. 3.
7. Black, F. (1976). "The Pricing of Commodity Contracts." *Journal of Financial Economics* 3(1–2), 167–179.
8. Cox, J. C., Ross, S. A. & Rubinstein, M. (1979). "Option Pricing: A Simplified Approach." *Journal of Financial Economics* 7(3), 229–263.
9. Broadie, M. & Glasserman, P. (1996). "Estimating Security Price Derivatives Using Simulation." *Management Science* 42(2), 269–285.
10. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering.* Springer, ch. 7.
11. Brent, R. P. (1973). *Algorithms for Minimization without Derivatives.* Prentice-Hall, ch. 4.
12. Jäckel, P. (2015). "Let's Be Rational." *Wilmott* 2015(75), 40–53.

**Volatility modelling & forecast evaluation (area 02)**

13. Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics* 31(3), 307–327.
14. Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993). "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *Journal of Finance* 48(5), 1779–1801.
15. Nelson, D. B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New Approach." *Econometrica* 59(2), 347–370.
16. J.P. Morgan / Reuters (1996). *RiskMetrics: Technical Document*, 4th ed. (EWMA λ = 0.94.)
17. Hansen, P. R. & Lunde, A. (2005). "A Forecast Comparison of Volatility Models: Does Anything Beat a GARCH(1,1)?" *Journal of Applied Econometrics* 20(7), 873–889.
18. Patton, A. J. (2011). "Volatility Forecast Comparison Using Imperfect Volatility Proxies." *Journal of Econometrics* 160(1), 246–256.
19. Diebold, F. X. & Mariano, R. S. (1995). "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics* 13(3), 253–263.

**Market risk: VaR, ES, backtesting (areas 03, F1)**

20. Boudoukh, J., Richardson, M. & Whitelaw, R. (1998). "The Best of Both Worlds." *Risk* 11(5), 64–67.
21. Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). "VaR without Correlations for Portfolios of Derivative Securities." *Journal of Futures Markets* 19(5), 583–602.
22. Hull, J. & White, A. (1998). "Incorporating Volatility Updating into the Historical Simulation Method for Value-at-Risk." *Journal of Risk* 1(1), 5–19.
23. Acerbi, C. & Tasche, D. (2002). "On the Coherence of Expected Shortfall." *Journal of Banking & Finance* 26(7), 1487–1503.
24. Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). "Coherent Measures of Risk." *Mathematical Finance* 9(3), 203–228.
25. McNeil, A. J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management*, 2nd ed. Princeton University Press, Ex. 2.14–2.15.
26. Cornish, E. A. & Fisher, R. A. (1938). "Moments and Cumulants in the Specification of Distributions." *Revue de l'Institut International de Statistique* 5(4), 307–320.
27. Maillard, D. (2012). "A User's Guide to the Cornish Fisher Expansion." SSRN 1997178.
28. Kupiec, P. H. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models." *Journal of Derivatives* 3(2), 73–84.
29. Christoffersen, P. F. (1998). "Evaluating Interval Forecasts." *International Economic Review* 39(4), 841–862.
30. Basel Committee on Banking Supervision (1996). *Supervisory Framework for the Use of "Backtesting" in Conjunction with the Internal Models Approach to Market Risk Capital Requirements.* BIS.
31. Acerbi, C. & Székely, B. (2014). "Backtesting Expected Shortfall." *Risk*, December, 76–81.
32. Basel Committee on Banking Supervision (2019). *Minimum Capital Requirements for Market Risk* (FRTB). BIS.
33. Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." *Annals of Statistics* 7(1), 1–26.
34. Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis.* Chapman & Hall.
35. Acklam, P. J. (2003). "An Algorithm for Computing the Inverse Normal Cumulative Distribution Function."
36. Cody, W. J. (1969). "Rational Chebyshev Approximations for the Error Function." *Mathematics of Computation* 23(107), 631–637.
37. Moro, B. (1995). "The Full Monte." *Risk* 8(2), 57–58.
38. Blackman, D. & Vigna, S. (2021). "Scrambled Linear Pseudorandom Number Generators." *ACM Transactions on Mathematical Software* 47(4), art. 36. (xoshiro256++.)
39. Matsumoto, M. & Nishimura, T. (1998). "Mersenne Twister: A 623-Dimensionally Equidistributed Uniform Pseudo-Random Number Generator." *ACM TOMACS* 8(1), 3–30.
40. O'Neill, M. E. (2014). "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation." Harvey Mudd College, HMC-CS-2014-0905.

**Fixed income (area 04)**

41. Macaulay, F. R. (1938). *Some Theoretical Problems Suggested by the Movements of Interest Rates, Bond Yields and Stock Prices in the United States since 1856.* NBER.
42. Tuckman, B. & Serrat, A. (2011). *Fixed Income Securities: Tools for Today's Markets*, 3rd ed. Wiley, ch. 4–5.
43. Ho, T. S. Y. (1992). "Key Rate Durations: Measures of Interest Rate Risks." *Journal of Fixed Income* 2(2), 29–44.
44. Hagan, P. S. & West, G. (2006). "Interpolation Methods for Curve Construction." *Applied Mathematical Finance* 13(2), 89–129.
45. Du, W., Tepper, A. & Verdelhan, A. (2018). "Deviations from Covered Interest Rate Parity." *Journal of Finance* 73(3), 915–957.

**Statistical pairs trading (area 05)**

46. Engle, R. F. & Granger, C. W. J. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing." *Econometrica* 55(2), 251–276.
47. Dickey, D. A. & Fuller, W. A. (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74(366), 427–431.
48. Said, S. E. & Dickey, D. A. (1984). "Testing for Unit Roots in Autoregressive-Moving Average Models of Unknown Order." *Biometrika* 71(3), 599–607.
49. MacKinnon, J. G. (1996). "Numerical Distribution Functions for Unit Root and Cointegration Tests." *Journal of Applied Econometrics* 11(6), 601–618.
50. MacKinnon, J. G. (2010). "Critical Values for Cointegration Tests." Queen's Economics Department Working Paper 1227.
51. Schwert, G. W. (1989). "Tests for Unit Roots: A Monte Carlo Investigation." *Journal of Business & Economic Statistics* 7(2), 147–159.
52. Ng, S. & Perron, P. (2001). "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica* 69(6), 1519–1554.
53. Gatev, E., Goetzmann, W. N. & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies* 19(3), 797–827.
54. Uhlenbeck, G. E. & Ornstein, L. S. (1930). "On the Theory of the Brownian Motion." *Physical Review* 36(5), 823–841.
55. Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal* 58(4), 36–52.

**Credit risk (area 06)**

56. Hosmer, D. W. & Lemeshow, S. (1980). "Goodness of Fit Tests for the Multiple Logistic Regression Model." *Communications in Statistics: Theory and Methods* A9(10), 1043–1069.
57. Hanley, J. A. & McNeil, B. J. (1982). "The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve." *Radiology* 143(1), 29–36.
58. Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.* Wiley. (WoE/IV, PSI.)
59. Vasicek, O. (2002). "The Distribution of Loan Portfolio Value." *Risk* 15(12), 160–162.
60. Gordy, M. B. (2003). "A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules." *Journal of Financial Intermediation* 12(3), 199–232.
61. Basel Committee on Banking Supervision (2005). *An Explanatory Note on the Basel II IRB Risk Weight Functions.* BIS.
62. Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital Standards.* BIS, §272–273.
63. Pykhtin, M. & Zhu, S. (2007). "A Guide to Modelling Counterparty Credit Risk." *GARP Risk Review* 37, 16–22.
64. Gregory, J. (2015). *The xVA Challenge: Counterparty Credit Risk, Funding, Collateral and Capital*, 3rd ed. Wiley.
65. Basel Committee on Banking Supervision (2013). *Supervisory Guidance for Managing Risks Associated with the Settlement of Foreign Exchange Transactions.* BIS.

**Portfolio optimisation & risk allocation (area 07)**

66. Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance* 7(1), 77–91.
67. Black, F. & Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal* 48(5), 28–43.
68. He, G. & Litterman, R. (1999). "The Intuition Behind Black-Litterman Model Portfolios." Goldman Sachs Investment Management Research.
69. Idzorek, T. (2005). "A Step-by-Step Guide to the Black-Litterman Model." Ibbotson Associates.
70. Ledoit, O. & Wolf, M. (2003). "Improved Estimation of the Covariance Matrix of Stock Returns with an Application to Portfolio Selection." *Journal of Empirical Finance* 10(5), 603–621.
71. Jorion, P. (1986). "Bayes-Stein Estimation for Portfolio Analysis." *Journal of Financial and Quantitative Analysis* 21(3), 279–292.
72. Higham, N. J. (1988). "Computing a Nearest Symmetric Positive Semidefinite Matrix." *Linear Algebra and its Applications* 103, 103–118.
73. Maillard, S., Roncalli, T. & Teïletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." *Journal of Portfolio Management* 36(4), 60–70.
74. Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights." SSRN 2297383.
75. Griveau-Billion, T., Richard, J.-C. & Roncalli, T. (2013). "A Fast Algorithm for Computing High-dimensional Risk Parity Portfolios." SSRN 2325255.

**Algorithmic execution & TCA (area 08)**

76. Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions." *Journal of Risk* 3(2), 5–39.
77. Perold, A. F. (1988). "The Implementation Shortfall: Paper versus Reality." *Journal of Portfolio Management* 14(3), 4–9.
78. Bertsimas, D. & Lo, A. W. (1998). "Optimal Control of Execution Costs." *Journal of Financial Markets* 1(1), 1–50.
79. Kissell, R. (2013). *The Science of Algorithmic Trading and Portfolio Management.* Academic Press.
80. Oomen, R. (2017). "Last Look." *Quantitative Finance* 17(7), 1057–1070.
81. Bank for International Settlements (2021). *FX Global Code*, Principle 17.

**Volatility surfaces & stochastic volatility (area 09)**

82. Gatheral, J. (2004). "A Parsimonious Arbitrage-Free Implied Volatility Parameterization with Application to the Valuation of Volatility Derivatives." Global Derivatives & Risk Management, Madrid.
83. Gatheral, J. & Jacquier, A. (2014). "Arbitrage-Free SVI Volatility Surfaces." *Quantitative Finance* 14(1), 59–71.
84. Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *Review of Financial Studies* 6(2), 327–343.
85. Albrecher, H., Mayer, P., Schoutens, W. & Tistaert, J. (2007). "The Little Heston Trap." *Wilmott*, January, 83–92.
86. Carr, P. & Madan, D. (1999). "Option Valuation Using the Fast Fourier Transform." *Journal of Computational Finance* 2(4), 61–73.
87. Andersen, L. (2008). "Simple and Efficient Simulation of the Heston Stochastic Volatility Model." *Journal of Computational Finance* 11(3), 1–42.
88. Lord, R., Koekkoek, R. & van Dijk, D. (2010). "A Comparison of Biased Simulation Schemes for Stochastic Volatility Models." *Quantitative Finance* 10(2), 177–194.
89. Castagna, A. & Mercurio, F. (2007). "The Vanna-Volga Method for Implied Volatilities." *Risk*, January, 106–111.
90. Hagan, P. S., Kumar, D., Lesniewski, A. S. & Woodward, D. E. (2002). "Managing Smile Risk." *Wilmott*, September, 84–108. (SABR: implemented in `eq_options.sabr`.)

**Regime switching (area 10)**

91. Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica* 57(2), 357–384.
92. Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE* 77(2), 257–286.
93. Baum, L. E., Petrie, T., Soules, G. & Weiss, N. (1970). "A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains." *Annals of Mathematical Statistics* 41(1), 164–171.
94. Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm." *Journal of the Royal Statistical Society B* 39(1), 1–38.
95. Viterbi, A. J. (1967). "Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm." *IEEE Transactions on Information Theory* 13(2), 260–269.
96. Ang, A. & Bekaert, G. (2002). "International Asset Allocation with Regime Shifts." *Review of Financial Studies* 15(4), 1137–1187.

**Backtest discipline (areas F2, 05, 08, 10)**

97. Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2014). "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance." *Notices of the AMS* 61(5), 458–471.
98. López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley, ch. 7, 11–12.
99. Merton, R. C. (1974). "On the Pricing of Corporate Debt: The Risk Structure of Interest Rates." *Journal of Finance* 29(2), 449–470. (Structural PD: not implemented in area 06; listed for orientation.)

## Data

Every project uses synthetic data from seeded generators in
`src/*/data/synthetic.py` or similar. Some projects have optional loaders for
free public data (Yahoo Finance, FRED, ECB). Tests never need the network.

The research studies use real data: ECB euro reference rates fetched by
`data/fetch/qp_fetch.py`, with a SHA-256 file per download. See
[`data/README.md`](data/README.md) for sources, terms and what is blocked.

## License

MIT License, Copyright (c) 2026 Ashish Jha. See [LICENSE](LICENSE).
