# Architecture Reference

One page per question: where does a capability live, what does it depend on, and
where are its tests. For rendered visual versions of the flows below — the
directory layout, the cross-language golden-vector validation pipeline, the
per-area data flows, and the two numerical-robustness bug case studies — see
**[DIAGRAMS.md](DIAGRAMS.md)** (Mermaid, renders on GitHub).

Every project's own `docs/METHODOLOGY.md` / `docs/VALIDATION.md` /
`docs/DESK_GUIDE.md` is the authoritative source for that project's math and
assumptions — this file and [DIAGRAMS.md](DIAGRAMS.md) are the map that shows
how the 31 sub-projects fit together, not a replacement for their own docs.

## The shape of the portfolio

quant-portfolio is **10 project areas, each built twice** — once for equities,
once for FX, as two fully independent, non-shared codebases — **in Python
throughout**, with **C++ and Rust performance twins** for the four areas that
are actually latency-sensitive on a real desk (options pricing/Greeks, market
risk VaR/ES). A separate, smaller **3-project foundations tier** predates the
10-area buildout and demonstrates the same core techniques from scratch,
single-asset, single-language, with no compiled twin.

```
31 sub-projects
= 10 areas × 2 asset classes (equity, FX)         = 20 Python projects
+ 3 foundations projects (equity-only, Python)     =  3 Python projects
+ 4 engines × 2 languages (C++, Rust)              =  8 compiled projects
                                                     ────────────────────
                                                      31 total, 5,963 tests
```

Why two fully separate codebases per area instead of one asset-class-agnostic
engine? Equity and FX are different enough in convention (continuous dividend
yield vs. domestic/foreign rate differential; single spot vs. BASE/QUOTE pairs
with four delta conventions; ACT/365F vs. currency-pair-dependent day counts)
that a shared abstraction would either leak asset-class-specific branches
everywhere or hide real differences behind a false unification. A real desk
keeps its equity derivatives and FX derivatives libraries separate for the
same reason — see each project's `docs/METHODOLOGY.md` §"why not a shared
abstraction" for the area-specific version of this argument.

## Directory layout

```
quant-portfolio/
├── CONVENTIONS.md         # The engineering & documentation contract every project follows
├── README.md              # Portfolio overview, test-coverage tables, how to build/test
├── docs/                  # THIS folder — portfolio-wide architecture, learning path, cookbook
│   ├── README.md
│   ├── ARCHITECTURE.md    # This file
│   ├── DIAGRAMS.md
│   ├── LEARN.md
│   ├── COOKBOOK.md
│   └── MARKET_RISK.md
├── python/
│   ├── equity/01-…10-…    # 10 equity projects, src/-layout Python packages
│   ├── fx/01-…10-…        # 10 FX projects — same structure, independent codebase
│   └── foundations/       # 3 standalone single-asset, single-language projects
│       ├── 01-risk-metrics/
│       ├── 02-trading-signal-backtest/
│       └── 03-black-scholes-replication/
├── cpp/                   # 4 performance-critical engines, C++20, GoogleTest
│   ├── equity-options-engine/
│   ├── fx-options-engine/
│   ├── equity-var-engine/
│   └── fx-var-engine/
└── rust/                  # The same 4 engines, Rust 2021, zero external dependencies
    ├── equity-options-engine/
    ├── fx-options-engine/
    ├── equity-var-engine/
    └── fx-var-engine/
```

Every Python project directory follows the identical internal layout —
`src/<package>/`, `tests/`, `examples/run_pipeline.py`,
`docs/{METHODOLOGY,VALIDATION,DESK_GUIDE}.md` — laid out in full in
`CONVENTIONS.md`. Every C++/Rust engine follows the mirror-image layout:
`include/<ns>/` or `src/*.rs` for the library, `tests/` (GoogleTest /
`cargo test`), `bench/` for the latency benchmark, `tools/gen_golden_*.py`
for the cross-language vector generator, `docs/{METHODOLOGY,VALIDATION,
DESK_GUIDE}.md`.

## The 10 project areas

| # | Area | Equity | FX | Compiled twin? |
|---|------|--------|-----|---|
| 1 | Options Pricing & Greeks | `python/equity/01-options-pricing` | `python/fx/01-options-pricing` | ✅ C++/Rust |
| 2 | Volatility Modeling & Forecasting | `python/equity/02-volatility-modeling` | `python/fx/02-volatility-modeling` | — |
| 3 | Market Risk VaR / Expected Shortfall | `python/equity/03-var-es-engine` | `python/fx/03-var-es-engine` | ✅ C++/Rust |
| 4 | Fixed Income Pricing & Risk | `python/equity/04-fixed-income` | `python/fx/04-fixed-income` | — |
| 5 | Statistical Pairs Trading | `python/equity/05-pairs-trading` | `python/fx/05-pairs-trading` | — |
| 6 | Credit Risk / PD Modeling | `python/equity/06-credit-risk` | `python/fx/06-credit-risk` | — |
| 7 | Portfolio Optimization & Risk Allocation | `python/equity/07-portfolio-optimization` | `python/fx/07-portfolio-optimization` | — |
| 8 | Algorithmic Trading & Execution | `python/equity/08-algo-execution` | `python/fx/08-algo-execution` | — |
| 9 | Volatility Surface & Stochastic Vol | `python/equity/09-vol-surface` | `python/fx/09-vol-surface` | — |
| 10 | Regime-Switching Quant Strategy | `python/equity/10-regime-switching` | `python/fx/10-regime-switching` | — |

**Why only these four get compiled twins**: options pricing/Greeks and VaR/ES
are the two workloads on a real desk that are (a) called millions of times per
revaluation cycle (full-book Greeks, intraday VaR) and (b) sit on a
numerically well-understood, closed-form-or-Monte-Carlo core that is worth
hand-optimizing once semantics are locked down in Python. The other six areas
are research/modeling workflows (calibration, backtesting, optimization)
where Python iteration speed matters more than microsecond latency, so they
stay Python-only — see [LEARN.md](LEARN.md) Part II for the fuller version of
this argument, and [DIAGRAMS.md](DIAGRAMS.md) diagram 1 for the picture.

## Foundations projects

| # | Project | Path | Relationship to the flagship equivalent |
|---|---|---|---|
| F1 | Risk Metrics on Real Data | `python/foundations/01-risk-metrics` | Single-asset vol/VaR/ES/Sharpe toolkit, three VaR methods compared side by side. No correlation/portfolio risk — see `python/equity/03-var-es-engine` for the multi-asset, backtested extension. |
| F2 | Trading Signal Backtest | `python/foundations/02-trading-signal-backtest` | A moving-average crossover, backtested with strict no-look-ahead execution and transaction costs. Point is backtest discipline, not the signal — see `python/equity/08-algo-execution` and `python/equity/10-regime-switching` for production-grade strategy machinery. |
| F3 | Black-Scholes Replication | `python/foundations/03-black-scholes-replication` | Zero-scipy (`math.erf` only) rebuild of Black-Scholes, validated against Monte Carlo and finite-difference Greeks. Model-validation reference, not a production pricer — `python/equity/01-options-pricing` (+ C++/Rust) is the production version. |

## The engine package map

| Engine | Language | Public entry point | Key modules | Golden vectors from |
|---|---|---|---|---|
| Equity options/Greeks | Python | `eq_options` (`black_scholes.py`, `greeks.py`, `implied_vol.py`, `binomial.py`, `monte_carlo.py`) | Closed-form BS, finite-diff & analytic Greeks, Newton+bisection implied vol, CRR binomial (American), MC pricer | — (reference) |
| | C++ | `eqopt::` (`include/eqopt/`) | `black_scholes.hpp`, `greeks.hpp`, `implied_vol.hpp`, `binomial.hpp`, `monte_carlo.hpp` | `tools/gen_golden_header.py` → `tests/golden_vectors.hpp` |
| | Rust | `eq_options_engine` crate (`src/`) | `black_scholes.rs`, `greeks.rs`, `implied_vol.rs`, `binomial.rs`, `monte_carlo.rs` | `tools/gen_golden_rs.py` → `src/golden.rs` |
| FX options/Greeks | Python | `fx_options` (`garman_kohlhagen.py`, `greeks.py`, `implied_vol.py`, `binomial.py`, `monte_carlo.py`) | Garman-Kohlhagen (BS with q=r_f), four-delta-convention Greeks, implied vol, binomial, MC | — (reference) |
| | C++ | `fxopt::` (`include/fxopt/`) | mirrors the Python module set | `tools/gen_golden_header.py` |
| | Rust | `fx_options_engine` crate | mirrors the Python module set | `tools/gen_golden_rs.py` |
| Equity VaR/ES | Python | `eq_var` (`historical_var.py`, `parametric_var.py`, `monte_carlo_var.py`, `expected_shortfall.py`, `backtesting.py`, `portfolio.py`, `stress_testing.py`) | Historical/age-weighted, parametric (delta-normal + Cornish-Fisher), Monte Carlo (normal/t/jump), Kupiec + Basel traffic-light backtesting | — (reference) |
| | C++ | `eqvar::` (`include/eqvar/`) | `historical.hpp`, `parametric.hpp`, `monte_carlo.hpp`, `expected_shortfall.hpp`, `backtest.hpp` | `tools/gen_golden_header.py` |
| | Rust | `eq_var_engine` crate | mirrors the Python module set | `tools/gen_golden_rs.py` |
| FX VaR/ES | Python | `fx_var` (`book.py`, `historical_var.py`, `parametric_var.py`, `monte_carlo_var.py`, `expected_shortfall.py`, `backtesting.py`) | Same methods as equity VaR plus `Book`/`Market` full-revaluation layer (spot, forward, option positions repriced scenario by scenario via Garman-Kohlhagen) | — (reference) |
| | C++ | `fxvar::` (`include/fxvar/`) | `book.hpp`, `historical.hpp`, `parametric.hpp`, `monte_carlo.hpp`, `expected_shortfall.hpp`, `backtest.hpp`, `stress.hpp` | `tools/gen_golden_header.py` |
| | Rust | `fx_var_engine` crate | mirrors the Python module set | `tools/gen_golden_rs.py` |

## Data flow at a glance

```
                          synthetic generators (deterministic, seeded)
                          or import-guarded live loaders (Yahoo/FRED/ECB)
                                          │
                                 src/<pkg>/data/*.py
                                          ▼
   pricing/vol/rates path         risk/backtest path            research path
   ┌────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
   │ black_scholes.py /  │   │ historical_var.py /   │   │ pairs_trading.py /    │
   │ garman_kohlhagen.py │   │ parametric_var.py /   │   │ credit_risk.py /      │
   │ implied_vol.py      │──▶│ monte_carlo_var.py    │   │ portfolio_opt.py /    │
   │ greeks.py            │   │ expected_shortfall.py │   │ regime_switching.py   │
   │ vol_surface / GARCH  │   │ backtesting.py        │   │ algo_execution.py     │
   └────────────────────┘   └───────────────────────┘   └───────────────────────┘
             │                          │                            │
             ▼                          ▼                            ▼
     examples/run_pipeline.py   examples/run_pipeline.py    examples/run_pipeline.py
     (end-to-end: data → model → validation → decision, reproduces the README numbers)
             │
             ▼ (options + VaR/ES only)
   tests/golden/generate_golden.py  ──▶  tools/gen_golden_header.py  ──▶  cpp/*/tests/golden_vectors.hpp
                                    └──▶  tools/gen_golden_rs.py     ──▶  rust/*/src/golden.rs
```

The research lane above is read left to right within each area: raw/synthetic
data → model/engine → the area's own validation (analytic identities,
convergence, statistical backtests) → the desk-facing decision the
`examples/run_pipeline.py` script prints. The four compiled-twin areas add a
fourth stage, the golden-vector bridge, described next.

## Cross-language validation — the technical backbone

Each of the four engine pairs is validated **three ways**, not just tested in
isolation:

1. **Python reference** — the original model, fully tested against analytic
   identities (put-call parity, Greeks-vs-finite-difference, tree→BS
   convergence, martingale checks) and statistical backtests (Kupiec,
   Basel traffic-light).
2. **C++ engine** — mirrors the Python semantics exactly. A
   `tests/golden/generate_golden.py` script in the Python project (or, for
   the VaR engines, direct `PYTHONPATH=src python3` invocation) produces
   reference values; `tools/gen_golden_header.py` turns them into a
   `constexpr` C++ header (`tests/golden_vectors.hpp`). GoogleTest validates
   every case to 1e-9 (options) or 1e-6–1e-8 (VaR/ES) tolerance.
3. **Rust engine** — the same golden values, generated into `src/golden.rs`
   by `tools/gen_golden_rs.py`, validated by `cargo test` to the same
   tolerances.

The result: for every priced option or computed VaR number, you can trace
identical (to floating-point-noise) results through three independently
implemented, independently tested codebases — the same discipline a real
market-risk function uses when a new pricing library has to be proven against
the incumbent before it is allowed to touch P&L. See
[DIAGRAMS.md](DIAGRAMS.md) diagram 2 for the pipeline picture, and
[LEARN.md](LEARN.md) Round 15 for the engineering theory behind why this
catches bugs that single-language testing does not.

**What the cross-language checks do *not* prove**: Monte Carlo paths are
never bit-identical across languages — NumPy's PCG64, a custom xoshiro256++ in
Rust, `std::mt19937_64` in C++ are three different streams — so MC agreement
is statistical (within a documented number of standard errors), not exact.
Determinism *within* a language is still bitwise: a fixed seed reproduces the
exact same path set on every run of the same engine.

## Determinism, by design

Every stochastic component across all 31 projects takes an explicit seed —
never an implicit global RNG state. This is a portfolio-wide convention
(`CONVENTIONS.md` §"Code quality"), not a per-project choice, because a
non-reproducible test is worse than no test: a flaky Monte Carlo test that
occasionally fails teaches you to ignore red CI, which is the actual failure
mode a real quant-risk team is trying to avoid.

## Design invariants worth knowing

- **No golden-pinned value ever changes silently.** Two portfolio-wide review
  passes (documented in the top-level `README.md`) hardened input validation
  and fixed numerical-robustness bugs across every engine without moving a
  single golden-vector number — a numerics fix that also happened to change a
  pinned reference value would be indistinguishable, from the outside, from a
  regression, so every fix in this portfolio's history was verified against
  that constraint before being accepted.
- **Every public entry point validates its inputs with explicit finiteness
  checks**, not bare comparisons. `if x <= 0: raise` looks like it rejects bad
  input, but `NaN <= 0` is `False` in IEEE 754 — the guard silently passes a
  NaN through. This was found as a portfolio-wide defect class and fixed
  everywhere; see [LEARN.md](LEARN.md) Round 16 and
  [DIAGRAMS.md](DIAGRAMS.md) diagram 9 for the full story, including the most
  dangerous manifestation (`max(NaN, 0.0)` silently returning a plausible
  zero VaR instead of propagating the NaN).
- **Two estimators, not one, wherever a single estimator has a known bias
  regime.** The Monte Carlo VaR standard error is a case in point: a local
  density-at-the-quantile estimate (Gaussian KDE in Python/FX-C++/FX-Rust, an
  order-statistic finite difference in equity-C++/equity-Rust) systematically
  underestimates the true sampling error at deep tails or modest scenario
  counts. Rather than tune the bandwidth and hope, every Monte Carlo VaR
  engine now also exposes a distribution-free bootstrap estimator as a
  cross-check — see [LEARN.md](LEARN.md) Round 18.
- **Equity and FX share conventions documentation, not code.** Both asset
  classes' rate/day-count/quoting conventions are written once, in
  `CONVENTIONS.md`, but every formula that uses them is implemented twice —
  see "The shape of the portfolio" above.
- **The C++/Rust engines have zero business-logic dependencies.** The Rust
  engines in particular take zero external crates by convention — the RNG,
  the normal-CDF/PPF, the Cholesky factorization, everything is written from
  first principles in this repository, specifically so that a reviewer never
  has to trust an opaque dependency's numerics to trust this portfolio's
  numbers.

## Further reading

- [DIAGRAMS.md](DIAGRAMS.md) — the same architecture, as pictures.
- [LEARN.md](LEARN.md) — a structured curriculum through every topic this
  portfolio covers, plus an ~450-question self-test bank organized by round.
- [COOKBOOK.md](COOKBOOK.md) — copy-pasteable recipes for the most common
  "how do I..." tasks across every project area.
- [MARKET_RISK.md](MARKET_RISK.md) — the VaR/ES/backtesting workflow this
  portfolio's risk engines are built to support, end to end.
- Each project's own `docs/METHODOLOGY.md`, `docs/VALIDATION.md`, and
  `docs/DESK_GUIDE.md` for that project's specific math, assumptions, and
  real-desk usage.
- `CONVENTIONS.md` at the repository root — the engineering and documentation
  contract every one of the 31 sub-projects follows.
