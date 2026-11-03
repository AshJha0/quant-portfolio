# Architecture Reference

This page answers three questions for each capability: where it lives, what it
depends on, and where its tests are. [DIAGRAMS.md](DIAGRAMS.md) has the same
material as Mermaid diagrams: the directory layout, the golden-vector pipeline,
the differential-test pipeline, two per-area data flows, the CI jobs, the
`engine=` switch, the pre-registration workflow and the newer model
dependencies.

Each project's own `docs/METHODOLOGY.md`, `docs/VALIDATION.md` and
`docs/DESK_GUIDE.md` is the source for that project's maths and assumptions.
This file is the map of how the 31 sub-projects fit together.

## Stage of the repository

The repository is a study portfolio. The November 2026 release started
the research platform (`data/` and `research/`), added the newer models
and the performance kernels, and put an `engine=` switch over
the bindings. All data is synthetic unless a page names a real source; the
research studies use ECB data. One study has a hold-out result (a null);
two signed registrations are scored on forward data from 2026-11-03, once,
on or after 2028-11-13. None
of this code runs on a live book.

## The shape of the portfolio

The portfolio has 10 project areas. Each area has an equity project and an FX
project, written as two separate Python codebases. Two areas, options pricing
and market-risk VaR/ES, also have C++ and Rust ports of the Python design for
both asset classes. That makes four compiled engines per language. The VaR
engines revalue linear books and European options (full repricing per
scenario; the VaR drivers and the revaluation kernel take the same
sticky-strike, sticky-delta and historical-ATM vol modes as Python, with vol
shocks as extra factors on the spot axis);
P&L explain stays in Python. A smaller
3-project foundations tier predates the 10-area build. It covers the same core
techniques for a single asset in Python only.

```
31 sub-projects
= 10 areas x 2 asset classes (equity, FX)          = 20 Python projects
+ 3 foundations projects (equity-only, Python)     =  3 Python projects
+ 4 engines x 2 languages (C++, Rust)              =  8 compiled projects
```

Three shared libraries and two binding packages sit beside the 31 and are not
counted in them:

| Path | Name | What it holds |
|---|---|---|
| `python/core` | `qp-core` (import `qp_core`) | Statistics, the t+1 fill rule, calendars, day counts, validation and metrics used by 16 Python projects; the shared normal CDF (`normal`), the Black-Scholes kernel in cost-of-carry form (`black_scholes`, scalar and NumPy) and the implied-vol solver (`iv_core`) behind both option packages |
| `cpp/core` | `qp::core` (header-only CMake target) | Normal CDF, implied-vol core, RNG, special functions, matrix and linear algebra, Brent, thread pool; the carry-form Black-Scholes kernel `qp/black_scholes.hpp` and its vectorised twin `qp/black_scholes_vm.hpp`; the latency harness `qp/bench/latency.hpp`; since November 2026 `qp/philox.hpp`, `qp/quantile.hpp`, `qp/gemm.hpp`, `qp/vmath.hpp` and the `bench_core` latency bench |
| `rust/core` | `qp-core` crate | RNGs, special functions, implied-vol core, `brentq`, Cholesky, thread partitioning; the carry-form kernel `black_scholes` and the latency harness `bench`; since November 2026 `philox`, `quantile`, `gemm`, `vmath` and the persistent thread pool `pool` |
| `bindings/cpp-nanobind` | `qp-cpp` (import `qp_cpp`) | nanobind bindings over the four C++ engines |
| `bindings/rust-pyo3` | `qp-rust` (import `qp_rust`) | PyO3 bindings over the four Rust crates |

Since November 2026 the four packages with ports (`eq_options`, `fx_options`,
`eq_var`, `fx_var`) each have an `engine` module whose functions take
`engine="python" | "cpp" | "rust"`. The mapping from name to backend is in
one place, `python/core/src/qp_core/engines.py`: an unknown name raises
`ValueError`, a missing extension raises `ImportError` with the install
command, and `"python"` calls the reference unchanged. The rest of each
package does not call the bindings. `tests/test_engine_differential.py`
checks the compiled engines against the reference on 10,000 random inputs
per group.

Test counts are not typed here. The CI `counts` job aggregates the executed
counts from every job into `counts.json`.

Equity and FX conventions differ. Equity uses a continuous dividend yield. FX
uses a domestic and foreign rate pair, BASE/QUOTE quoting, four delta
conventions and pair-dependent day counts. The two codebases keep those
differences explicit in their public functions.

The Black-Scholes maths underneath is shared. The plain idea: a stock with
dividend yield `q` and a currency with foreign rate `r_f` are the same
contract to a pricer, because only the cost of carry `b` differs. Formally,
one kernel per language prices `V = phi [S e^{-qT} N(phi d1) - K e^{-rT}
N(phi d2)]` with `q = r - b`, and `eq_options`/`fx_options` (and the C++
and Rust option engines) are wrappers that keep their own names, argument
order, validation and boundary rules. Worked example: S = K = 100, T = 1,
r = 5%, q = 2%, sigma = 20% gives 9.227005508154036 from `bs_price`, from
`gk_price` with `r_d = 5%`, `r_f = 2%`, and from the kernel itself. Moving
the maths into the kernel changed no output bit on a 5,944-quote grid in
any language; `tests/test_carry_twin.py` pins it. The C++ and Rust kernels
take the normal CDF as a type parameter, because the equity and FX engines
use different implementations. The Python FX `analytic_greeks` moves to
the shared normal CDF in this release, which changes its outputs by at most
2.13e-14 relative to max(1, |value|); see the CHANGELOG. Diagram 10 in
[DIAGRAMS.md](DIAGRAMS.md) shows the layout.

Packaging. The C++ engines and `qp::core` install as the CMake package
`QuantPortfolio` 1.0.0 (`find_package(QuantPortfolio 1.0 CONFIG REQUIRED)`,
targets `qp::core`, `qp::eqopt`, `qp::eqvar`, `qp::fxopt`, `qp::fxvar`).
CI builds wheels of `qp-cpp` and `qp-rust` for CPython 3.11 to 3.13 on
Linux and Windows and keeps them as artefacts. The Rust crates move to
edition 2024 in this release.

## Directory layout

```
quant-portfolio/
├── CONVENTIONS.md         # Engineering and documentation contract for every project
├── CHANGELOG.md           # Release history and fixes
├── README.md              # Portfolio overview and build/test instructions
├── docs/                  # Portfolio-wide architecture, learning path, cookbook
│   ├── README.md
│   ├── ARCHITECTURE.md    # This file
│   ├── DIAGRAMS.md
│   ├── LEARN.md
│   ├── COOKBOOK.md
│   └── MARKET_RISK.md
├── pyproject.toml, uv.lock # uv workspace: python/core and the 23 projects; mypy config
├── tests/                 # repository-level tests: docs counts, engine differential harness
├── data/                  # free-data fetchers (data/fetch/qp_fetch.py); downloads in data/raw/, git-ignored
├── research/              # research platform
│   ├── prereg/            # pre-registrations (hash over the text above "## Signature")
│   ├── studies/           # models.py (Johansen, Kalman, training_only guard), fx_studies.py
│   ├── results/           # result JSON, each naming its registration hash
│   ├── tools/prereg.py    # hash and check
│   └── tests/
├── python/
│   ├── core/              # qp-core, shared Python code
│   ├── equity/01-…10-…    # 10 equity projects, src/-layout Python packages
│   ├── fx/01-…10-…        # 10 FX projects, same layout, separate code
│   └── foundations/        # 3 single-asset Python projects
│       ├── 01-risk-metrics/
│       ├── 02-trading-signal-backtest/
│       └── 03-black-scholes-replication/
├── cpp/                   # 4 engines, C++23, CMake presets, tests use GoogleTest
│   ├── core/              # qp::core, header-only
│   ├── equity-options-engine/
│   ├── fx-options-engine/
│   ├── equity-var-engine/
│   └── fx-var-engine/
├── rust/                  # Cargo workspace (Cargo.toml, Cargo.lock, rust-toolchain.toml)
│   ├── core/              # qp-core crate
│   ├── equity-options-engine/
│   ├── fx-options-engine/
│   ├── equity-var-engine/
│   └── fx-var-engine/
└── bindings/
    ├── cpp-nanobind/      # qp-cpp
    └── rust-pyo3/         # qp-rust
```

Every Python project uses the same layout: `src/<package>/`, `tests/`,
`examples/run_pipeline.py` and `docs/{METHODOLOGY,VALIDATION,DESK_GUIDE}.md`.
`CONVENTIONS.md` describes it in full. Every C++ and Rust engine uses a matching
layout: `include/<ns>/` or `src/*.rs` for the library, `tests/` (GoogleTest or
`cargo test`), `bench/` (C++) or `src/bin/bench.rs` (Rust) for the latency
benchmark, `tools/gen_golden_header.py` (C++) or `tools/gen_golden_rs.py` (Rust)
for the golden-vector generator, and `docs/{METHODOLOGY,VALIDATION,DESK_GUIDE}.md`.

The C++ engines set `CMAKE_CXX_STANDARD 23` in each `cpp/*/CMakeLists.txt`,
or 26 when `QP_CXX26=ON`. They need CMake 3.25 and link `qp::core`. They have
no runtime dependencies beyond the standard library. Their tests use
GoogleTest. Each Rust engine's only `[dependencies]` entry is the workspace
`qp-core` crate, which itself has none.

## The 10 project areas

| # | Area | Equity | FX | C++/Rust port? |
|---|------|--------|-----|---|
| 1 | Options Pricing & Greeks | `python/equity/01-options-pricing` | `python/fx/01-options-pricing` | yes |
| 2 | Volatility Modeling & Forecasting | `python/equity/02-volatility-modeling` | `python/fx/02-volatility-modeling` | no |
| 3 | Market Risk VaR / Expected Shortfall | `python/equity/03-var-es-engine` | `python/fx/03-var-es-engine` | yes (linear book plus European options) |
| 4 | Fixed Income Pricing & Risk | `python/equity/04-fixed-income` | `python/fx/04-fixed-income` | no |
| 5 | Statistical Pairs Trading | `python/equity/05-pairs-trading` | `python/fx/05-pairs-trading` | no |
| 6 | Credit Risk / PD Modeling | `python/equity/06-credit-risk` | `python/fx/06-credit-risk` | no |
| 7 | Portfolio Optimization & Risk Allocation | `python/equity/07-portfolio-optimization` | `python/fx/07-portfolio-optimization` | no |
| 8 | Algorithmic Trading & Execution | `python/equity/08-algo-execution` | `python/fx/08-algo-execution` | no |
| 9 | Volatility Surface & Stochastic Vol | `python/equity/09-vol-surface` | `python/fx/09-vol-surface` | no |
| 10 | Regime-Switching Quant Strategy | `python/equity/10-regime-switching` | `python/fx/10-regime-switching` | no |

Options pricing and VaR/ES are the two workloads that run many times per
revaluation cycle. Both have a closed-form or Monte Carlo core that is worth
porting once the Python semantics are fixed. The other eight areas are
calibration, backtesting and optimisation workflows, so they stay in Python.
[LEARN.md](LEARN.md) Part II has the longer argument. [DIAGRAMS.md](DIAGRAMS.md)
diagram 1 shows the layout.

## Engine latency

How latency is measured now is in [BENCHMARKS.md](BENCHMARKS.md): C++ and
Rust use one harness (each sample a batch of calls lasting at least 2 us,
at least 31 samples, median with p10/p90), and the CI `bench` job compares
base and head on the same runner. Current figures are in that job's
`bench-results` artifact.

The tables below are older. They are one laptop run (Intel Core 5 120U,
WSL2, 2026-11-01; GCC 15.2 at `-O3 -DNDEBUG`, rustc 1.99.1 with
`lto = "fat"`) with the previous harnesses, kept for the record. Then the
C++ bench timed each call separately with rdtscp and subtracted the timer
overhead, and the Rust bench timed batches with `Instant`, so the two
tables are not comparable with each other. The C++ `bs_price` p50 moved
between about 98 and 149 ns across runs on the same machine.

For batch pricing, the C++ `_fast` batches measured 23-29 ns per option
with GCC 15 against 91-110 ns for the exact batches, and the Python array
functions about 19x faster than a scalar loop; the COOKBOOK has the table.

C++:

| Engine | Kernel | p50 | p99 |
|---|---|---|---|
| [equity-options](../cpp/equity-options-engine/README.md#latency) | `bs_price` | 135 ns | 385 ns |
| [equity-options](../cpp/equity-options-engine/README.md#latency) | `try_mc_price`, 1M paths, 4 threads | 23.95 ms | 29.75 ms |
| [fx-options](../cpp/fx-options-engine/README.md#latency) | `gk_price` | 113 ns | 329 ns |
| [fx-options](../cpp/fx-options-engine/README.md#latency) | `try_mc_price`, 1M paths, 4 threads | 39.60 ms | 46.21 ms |
| [equity-var](../cpp/equity-var-engine/README.md#latency) | `try_historical_var`, n = 1000 | 4.5 µs | 8.1 µs |
| [equity-var](../cpp/equity-var-engine/README.md#latency) | `try_matvec`, 10000 x 50 scenario P&L | 376.2 µs | 1.02 ms |
| [fx-var](../cpp/fx-var-engine/README.md#latency) | `try_pnl`, 1 scenario, 250 positions | 2.0 µs | 2.6 µs |
| [fx-var](../cpp/fx-var-engine/README.md#latency) | `var_standard_error`, n = 10000 | 317.6 µs | 700.2 µs |

Rust:

| Engine | Kernel | p50 | p99 |
|---|---|---|---|
| [equity-options](../rust/equity-options-engine/README.md#latency) | `bs_price` | 44 ns | 116 ns |
| [equity-options](../rust/equity-options-engine/README.md#latency) | `mc_price`, 1M paths, antithetic + CV | 16.16 ms | 24.82 ms |
| [fx-options](../rust/fx-options-engine/README.md#latency) | `gk_price` | 64 ns | 72 ns |
| [fx-options](../rust/fx-options-engine/README.md#latency) | `mc_price`, 100k paths, antithetic + CV | 5.79 ms | 7.89 ms |
| [equity-var](../rust/equity-var-engine/README.md#latency) | historical VaR (`fast`), n = 250 | 541 ns | 1.79 µs |
| [equity-var](../rust/equity-var-engine/README.md#latency) | MC run, 20 factors x 10k paths | 7.08 ms | 19.2 ms |
| [fx-var](../rust/fx-var-engine/README.md#latency) | `CompiledBook::pnl`, one scenario | 2.03 µs | 4.1 µs |
| [fx-var](../rust/fx-var-engine/README.md#latency) | `monte_carlo_var`, 100k normal | 346 ms | 437 ms |

## Foundations projects

| # | Project | Path | Relationship to the 10-area projects |
|---|---|---|---|
| F1 | Risk Metrics | `python/foundations/01-risk-metrics` | Single-asset vol/VaR/ES/Sharpe toolkit on synthetic data. It compares three VaR methods side by side. It has no correlation or portfolio risk. `python/equity/03-var-es-engine` is the multi-asset, backtested version. |
| F2 | Trading Signal Backtest | `python/foundations/02-trading-signal-backtest` | A moving-average crossover backtest with transaction costs. Fills are modelled at t+1 as p_t + f(p_{t+1} - p_t) with f = 0.5 by default, plus a close-auction variant. The subject is backtest mechanics, not the signal. `python/equity/08-algo-execution` and `python/equity/10-regime-switching` hold the fuller strategy code. |
| F3 | Black-Scholes Replication | `python/foundations/03-black-scholes-replication` | Black-Scholes rebuilt without scipy. The normal CDF uses `math.erfc`. It is checked against Monte Carlo and finite-difference Greeks. It is a model-validation reference. `python/equity/01-options-pricing` and its C++/Rust ports are the main pricer. |

## The engine package map

| Engine | Language | Public entry point | Key modules | Golden vectors from |
|---|---|---|---|---|
| Equity options/Greeks | Python | `eq_options` (`black_scholes.py` incl. implied vol, `black76.py`, `greeks.py`, `binomial.py`, `monte_carlo.py`, `hedging.py`, `comparison.py`) | Closed-form BS, finite-difference and analytic Greeks, Newton plus bisection implied vol, Black-76, CRR binomial (American), MC pricer | `tests/golden/generate_golden.py` -> `tests/golden/golden_vectors.json` |
| | C++ | `eqopt::` (`include/eqopt/`) | `black_scholes.hpp`, `black76.hpp`, `greeks.hpp`, `implied_vol.hpp`, `binomial.hpp`, `monte_carlo.hpp` | `tools/gen_golden_header.py` -> `tests/golden_vectors.hpp` |
| | Rust | `eq_options_engine` crate (`src/`) | `black_scholes.rs`, `black76.rs`, `greeks.rs`, `implied_vol.rs`, `binomial.rs`, `monte_carlo.rs` | `tools/gen_golden_rs.py` -> `src/golden.rs` |
| FX options/Greeks | Python | `fx_options` (`garman_kohlhagen.py` incl. implied vol, `deltas.py`, `forwards.py`, `black76.py`, `greeks.py`, `binomial.py`, `monte_carlo.py`, `hedging.py`) | Garman-Kohlhagen (BS with q = r_f), four delta conventions and strike-from-delta, CIP forwards, implied vol, binomial, MC | `tests/golden/generate_golden.py` -> `tests/golden/golden_vectors.json` |
| | C++ | `fxopt::` (`include/fxopt/`) | `garman_kohlhagen.hpp`, `deltas.hpp`, `forwards.hpp`, `black76.hpp`, `greeks.hpp`, `implied_vol.hpp`, `binomial.hpp`, `monte_carlo.hpp` | `tools/gen_golden_header.py` -> `tests/golden_vectors.hpp` |
| | Rust | `fx_options_engine` crate | mirrors the C++ module set | `tools/gen_golden_rs.py` -> `src/golden.rs` |
| Equity VaR/ES | Python | `eq_var` (`historical_var.py`, `parametric_var.py`, `monte_carlo_var.py`, `expected_shortfall.py`, `backtesting.py`, `portfolio.py`, `stress_testing.py`) | Historical, age-weighted and FHS; parametric (delta-normal, Student-t, Cornish-Fisher); Monte Carlo (normal, t, jump); Kupiec, Christoffersen, Acerbi-Szekely and Basel traffic-light backtests | `tests/golden/generate_golden.py` -> `tests/golden/golden_vectors.json` (locked by `tests/test_golden_vectors.py`) |
| | C++ | `eqvar::` (`include/eqvar/`) | `historical.hpp`, `parametric.hpp`, `monte_carlo.hpp`, `expected_shortfall.hpp`, `backtest.hpp`, `matrix.hpp`, `stats.hpp` | `tools/gen_golden_header.py` -> `tests/golden_vectors.hpp` |
| | Rust | `eq_var_engine` crate | mirrors the C++ module set | `tools/gen_golden_rs.py` -> `tests/golden/mod.rs` |
| FX VaR/ES | Python | `fx_var` (`book.py`, `historical_var.py`, `parametric_var.py`, `monte_carlo_var.py`, `expected_shortfall.py`, `backtesting.py`, `stress_testing.py`) | Same methods as equity VaR plus a `Book`/`Market` full-revaluation layer that reprices spot, forward and option positions per scenario with Garman-Kohlhagen | `tests/golden/generate_golden.py` -> `tests/golden/golden_vectors.json` (locked by `tests/test_golden_vectors.py`) |
| | C++ | `fxvar::` (`include/fxvar/`) | `book.hpp`, `historical.hpp`, `parametric.hpp`, `monte_carlo.hpp`, `expected_shortfall.hpp`, `backtest.hpp`, `stress.hpp` | `tools/gen_golden_header.py` -> `tests/golden_vectors.hpp` |
| | Rust | `fx_var_engine` crate | mirrors the C++ module set | `tools/gen_golden_rs.py` -> `tests/golden/mod.rs` |

The November 2026 release added these modules. They are imported by path and are not
re-exported from the package `__init__`. Each has oracle tests listed in the
package `docs/VALIDATION.md` and in the README table.

| Package | New modules |
|---|---|
| `eq_options` | `engine`, `lr_tree`, `pde`, `alo`, `dividends`, `term_structure`, `bachelier`, `sabr`, `greeks_higher`, `risk_grids` |
| `fx_options` | `engine`, `smile`, `premium`, `cuts` |
| `eq_var`, `fx_var` (same files) | `engine`, `risk_attribution`, `frtb_ima`, `stressed_var`, `evt`, `copula_mc`, `revaluation`, `limits`, `_r3checks` |
| `eq_vol` | `har_rv` |
| `fi_rates` | `multicurve`, `rate_options`, `hull_white` |
| `fx_rates` | `xccy_float` |
| `eq_surface` | `ssvi`, `dupire`, `bates` |
| `fx_surface` | `ssvi`, `dupire`, `bates` (FX twins: forward from r_d and r_f, delta conventions; checked against the equity modules) |
| `fx_vol` | `har_rv` (realised variance on 17:00 New York days) |
| `fx_var` | `fx_attribution` (currency mapping and option exposure for Euler attribution) |
| `qp_core` | `normal` (shared normal CDF and density), `holidays` (settlement calendars and business-day rolls, moved from `fx_options`) |
| `eq_credit` | `lgd_ead`, `calibration`, `migration`, `cds` |
| `fx_credit` | `xva`, `saccr` |
| `eq_port`, `fx_port` (same files) | `hrp`, `robust_mvo`, `cost_qp`, `factor_model`, `tracking_error`, `sharpe_tests` |
| C++ `eqopt`, Rust `eq-options-engine` | `lr_tree`, `bachelier`, `greeks_higher` (matched to Python to 1e-12 relative, 1e-13 for Bachelier) |
| C++ and Rust VaR engines (all four) | `option_reval`: BSM price, Greeks, full-revaluation and delta-gamma P&L, matched to Python to 1e-11; called by historical and Monte Carlo VaR for option legs |

Rates curves now build in two layers: `fi_rates.multicurve` (OIS discount
curve, then projection curves) feeds `fi_rates.rate_options` and
`fi_rates.hull_white`. On the surface side `eq_surface.ssvi` feeds
`eq_surface.dupire`, and `eq_surface.bates` sits beside `heston`.

The C++ headers and Rust golden modules are generated files. Every
`tools/gen_golden_*.py` script accepts `--check`, which exits non-zero when the
committed file no longer matches the Python JSON. The CI `goldens --check` job
runs it for all eight engines.

## Conventions shared by the VaR engines

- `alpha` is the tail probability, in (0, 0.5), in every function and language.
  A 99% VaR is `alpha=0.01`.
- Historical VaR is the order statistic L(ceil(n * alpha)), the m-th largest
  loss with m = ceil(n * alpha). There is no interpolation.
- Expected Shortfall is the exact Acerbi-Tasche estimator, including the
  fractional atom at the VaR rank.
- The Basel traffic light works for any window length and any `alpha`. Zone
  edges come from the binomial CDF at 0.95 and 0.9999.
- `kupiec_pof` takes keyword arguments only (`n_obs`, `n_exceptions`, `alpha`).
- The Monte Carlo VaR standard error is a bootstrap estimate. The KDE
  density-at-quantile estimate is kept as a diagnostic.
- EWMA volatility is seeded with the mean of r^2 over the first min(30, n)
  observations. The equity EWMA covariance is seeded with mean(r r') over the
  first min(30, T) rows (S8) in Python, C++ and Rust.
- Cholesky adds a diagonal jitter capped at 1e-6 times the mean diagonal and
  raises above that cap. Callers can ask for a repair (`clip` or `higham`).

## Data flow at a glance

```
                          synthetic generators (deterministic, seeded)
                          or import-guarded live loaders (Yahoo/FRED/ECB)
                                          │
                                 src/<pkg>/data/*.py
                                          ▼
   pricing/vol/rates path         risk/backtest path            research path
   ┌────────────────────┐   ┌───────────────────────┐   ┌───────────────────────────┐
   │ eq_options/         │   │ eq_var/, fx_var/      │   │ eq_pairs/{cointegration,  │
   │  black_scholes.py   │   │  historical_var.py    │   │   spread,signals,backtest}│
   │ fx_options/         │──▶│  parametric_var.py    │   │ eq_credit/{model,         │
   │  garman_kohlhagen.py│   │  monte_carlo_var.py   │   │   validation,portfolio_risk}│
   │  deltas.py           │   │  expected_shortfall.py│   │ eq_port/{mvo,risk_parity, │
   │ eq_surface/{smile,   │   │  backtesting.py       │   │   returns_est}            │
   │   surface,heston}    │   │  (fx_var: book.py)    │   │ eq_algo/{almgren_chriss,  │
   │ eq_vol/{garch,gjr,   │   │                       │   │   tca,benchmarks}         │
   │   egarch,ewma}       │   │                       │   │ eq_regime/{hmm,detection, │
   │ fi_rates/, fx_rates/ │   │                       │   │   strategy}  (+ fx_*)     │
   └────────────────────┘   └───────────────────────┘   └───────────────────────────┘
             │                          │                            │
             ▼                          ▼                            ▼
     examples/run_pipeline.py   examples/run_pipeline.py    examples/run_pipeline.py
             │
             ▼ (options + VaR/ES only)
   tests/golden/generate_golden.py ──▶ tests/golden/golden_vectors.json
        ├──▶  cpp/*/tools/gen_golden_header.py  ──▶  cpp/*/tests/golden_vectors.hpp
        └──▶  rust/*/tools/gen_golden_rs.py     ──▶  rust/*-options-engine/src/golden.rs
                                                     rust/*-var-engine/tests/golden/mod.rs
```

Each area reads left to right: synthetic data, then the model, then the area's
own validation, then the summary that `examples/run_pipeline.py` prints. The
options and VaR/ES areas add the golden-vector step described next.

## Cross-language testing

The C++ and Rust engines are ports of the Python design. Two mechanisms check
that they agree with Python.

**Golden vectors.** Each of the four Python reference projects has a
`tests/golden/generate_golden.py` that writes `tests/golden/golden_vectors.json`
(committed). `tools/gen_golden_header.py` turns the JSON into a `constexpr` C++
header and `tools/gen_golden_rs.py` turns it into a Rust module. Both outputs
are committed and checked by the CI `goldens --check` job.

- Options goldens: inputs are evaluated with an mpmath oracle at high
  precision. Each output carries its own tolerance, computed by the rules in
  the JSON's `tolerance_rules` block. There is no single flat tolerance.
- VaR/ES goldens: expected values come from the Python reference engine
  through the same JSON generator pipeline. The Python side is locked by
  `tests/test_golden_vectors.py`.

**Differential tests.** `python/equity/01-options-pricing/tests/test_differential.py`
and `python/fx/01-options-pricing/tests/test_differential.py` generate input
grids, send them as JSON lines to a compiled shim, and compare the replies with
the Python result. The shims are:

| Engine | Rust bin | C++ target | Env var (Rust) | Env var (C++) |
|---|---|---|---|---|
| Equity options | `eq_jsonl` | `eqopt_jsonl` | `QP_RS_EQ_JSONL` | `QP_CPP_EQ_JSONL` |
| FX options | `fx_jsonl` | `fxopt_jsonl` | `QP_RS_FX_JSONL` | `QP_CPP_FX_JSONL` |

When an env var is unset the test looks for the default build path
(`rust/target/release/<bin>`, the workspace target directory, or
`cpp/<engine>/build/<target>` and then `cpp/<engine>/build/gcc/<target>`). A missing binary
skips its test. Set `QP_DIFF_VERBOSE=1` to print the largest deviations.
[DIAGRAMS.md](DIAGRAMS.md) diagram 3 shows the flow.

Agreement with Python does not prove Python is right. Each project's analytic
identity and convergence tests carry that burden. Monte Carlo paths differ
across languages because the RNGs differ (NumPy PCG64, xoshiro256++ in Rust,
`std::mt19937_64` in C++). Cross-language MC agreement is statistical, within
a stated number of standard errors. A fixed seed reproduces the same paths on
the same engine and platform.

## Continuous integration

`.github/workflows/ci.yml` runs these jobs:

| Job | What it does |
|---|---|
| `discover` | Lists the Python projects and their test extra (`.github/scripts/discover_projects.py`). |
| `py core` | pytest for `python/core` on Python 3.11 to 3.13. |
| `py <project>` | One job per project and Python version (3.11, 3.12, 3.13). Installs `python/core` and the project in its own venv and runs pytest. Uploads JUnit XML. |
| (all Python jobs) | Third-party packages come from `uv.lock` with sha256 hashes checked (`.github/scripts/hashed_install.sh`; see `requirements/README.md`). |
| `lint` | `uv lock --check`, `uv sync --all-packages --locked --group test --group lint`, a check that `requirements/ci-constraints.txt` matches `uv.lock`, `ruff check python`, `mypy`. The `lint` group holds `pandas-stubs` and `scipy-stubs` for Python 3.12 and later. |
| `goldens --check` | Runs every `gen_golden_header.py --check` and `gen_golden_rs.py --check`. |
| `cpp <engine> (<preset>)` | Each C++ engine with the `gcc`, `clang`, `asan` and `tsan` presets: configure, build, ctest. |
| `cpp <engine> (gcc16-cxx26)` | The same in a `gcc:16` container with `QP_CXX26=ON`. |
| `cpp <engine> (msvc)` | The same on Windows with MSVC. |
| `clang-format`, `clang-tidy <engine>` | Format check over `cpp/`; clang-tidy with warnings as errors. |
| `rust <crate>` | `cargo test --release --locked` and clippy with warnings denied, per workspace crate including `qp-core`. |
| `rust fmt` | `cargo fmt --check` over the workspace. |
| `bind <package>` | Builds `qp-cpp` or `qp-rust` and runs its tests against the Python reference packages. |
| `cmake install (<toolchain>)` | Installs the `QuantPortfolio` CMake package with GCC 15 or Clang 21, moves the source tree aside, and builds and runs `cpp/tests/find_package_consumer` against the prefix. |
| `wheels <package> (<os>)` | cibuildwheel wheels of `qp-cpp` and `qp-rust`, CPython 3.11 to 3.13, manylinux_2_34 x86_64 and Windows amd64, smoke-tested by `bindings/wheel_smoke.py`; artefacts `wheels-*`, never published. |
| `bench` | Builds base and head on one runner and runs their C++ and Rust benches in three alternating rounds pinned to CPU 1; fails a kernel only when the head median is more than 25% above base and head p10 is above base p90 ([BENCHMARKS.md](BENCHMARKS.md)). Artefact `bench-results`. |
| `hygiene` | Fails on a banned word anywhere in the repo. Prints a voice report (report-only). Runs `tests/test_ci_scripts.py`. |
| `cpp core (<preset>)` | Builds and tests `cpp/core` (including the kernel tests); the gcc job runs `bench_core --quick` pinned to one CPU and uploads the JSON, report-only. |
| `engine differential` | Builds both bindings and runs `tests/test_engine_differential.py`; fails if any compiled case is skipped. |
| `research (prereg check, tests)` | `python research/tools/prereg.py check` and `pytest research/tests`. |
| `counts` | Aggregates executed test counts from the JUnit and cargo artefacts into `counts.json`, uploads it, and runs `tests/test_docs_counts.py` against it. |
| `ci-ok` | Succeeds only when every job above succeeded. This is the required check. |

## Determinism

Every stochastic component takes an explicit seed. No code reads a global RNG
state. This is a portfolio-wide rule in `CONVENTIONS.md` §"Code quality".

## Design invariants

- **Golden values change only through the generator.** A changed expected value
  needs a regenerated JSON, header and module, and a CHANGELOG entry.
- **Entry points check finiteness explicitly.** `if x <= 0: raise` lets NaN
  through, because `NaN <= 0` is false. The VaR/ES, backtest, options, bond and
  Black-Scholes entry points use `isfinite` checks. `test_input_hardening.py`
  and `test_nan_guards.py` cover them.
- **Equity and FX share the option maths, not the conventions code.**
  `CONVENTIONS.md` states the rate, day-count and quoting rules once and
  each codebase implements them separately. The Black-Scholes kernel, the
  implied-vol solver and the normal CDF are shared through the core library
  of each language.
- **The compiled engines use only the standard library.** The RNG, normal
  CDF/PPF and Cholesky factorisation are written in this repository, once per
  language, in `cpp/core` and `rust/core`.

Fix history is in [CHANGELOG.md](../CHANGELOG.md).

## Further reading

- [DIAGRAMS.md](DIAGRAMS.md): the same architecture as diagrams.
- [LEARN.md](LEARN.md): a curriculum through every topic, with a 532-question
  self-test bank organised by round.
- [COOKBOOK.md](COOKBOOK.md): recipes for common tasks in every area.
- [MARKET_RISK.md](MARKET_RISK.md): the VaR/ES and backtesting workflow the
  risk engines support.
- Each project's `docs/METHODOLOGY.md`, `docs/VALIDATION.md` and
  `docs/DESK_GUIDE.md` for its maths, assumptions and usage notes.
- `CONVENTIONS.md`: the engineering and documentation contract for all 31
  sub-projects.
