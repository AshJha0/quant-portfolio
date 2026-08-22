# Quant Finance Portfolio

A flagship, end-to-end quant finance portfolio: **10 dual-asset-class project
areas**, each built **twice** (equity and FX as fully separate, self-contained
projects), in **Python** throughout and with **C++ and Rust performance
twins** for the four highest-value pricing/risk engines — plus **3 standalone
foundations projects** that build and validate core techniques from scratch,
single-asset, single-language. 31 sub-projects and **5,963 passing tests** in
total, every one with its own tests, and documentation answering *why this
model*, *what assumptions it makes*, *how it was validated*, *where it
fails*, and *how a real desk would use it*.

```
quant-portfolio/
├── CONVENTIONS.md        # The engineering & documentation contract every project follows
├── README.md              # This file
├── python/
│   ├── equity/01-…10-…    # 10 equity projects
│   ├── fx/01-…10-…        # 10 FX projects (fully separate from equity)
│   └── foundations/       # 3 standalone, single-asset foundations projects
│       ├── 01-risk-metrics/
│       ├── 02-trading-signal-backtest/
│       └── 03-black-scholes-replication/
├── cpp/                   # 4 performance-critical engines, C++20
│   ├── equity-options-engine/
│   ├── fx-options-engine/
│   ├── equity-var-engine/
│   └── fx-var-engine/
└── rust/                  # The same 4 engines, Rust 2021, zero dependencies
    ├── equity-options-engine/
    ├── fx-options-engine/
    ├── equity-var-engine/
    └── fx-var-engine/
```

## The 10 project areas

| # | Area | Equity | FX |
|---|------|--------|-----|
| 1 | Options Pricing & Greeks | `python/equity/01-options-pricing` (+ C++/Rust) | `python/fx/01-options-pricing` (+ C++/Rust) |
| 2 | Volatility Modeling & Forecasting | `python/equity/02-volatility-modeling` | `python/fx/02-volatility-modeling` |
| 3 | Market Risk VaR / Expected Shortfall | `python/equity/03-var-es-engine` (+ C++/Rust) | `python/fx/03-var-es-engine` (+ C++/Rust) |
| 4 | Fixed Income Pricing & Risk | `python/equity/04-fixed-income` | `python/fx/04-fixed-income` |
| 5 | Statistical Pairs Trading | `python/equity/05-pairs-trading` | `python/fx/05-pairs-trading` |
| 6 | Credit Risk / PD Modeling | `python/equity/06-credit-risk` | `python/fx/06-credit-risk` |
| 7 | Portfolio Optimization & Risk Allocation | `python/equity/07-portfolio-optimization` | `python/fx/07-portfolio-optimization` |
| 8 | Algorithmic Trading & Execution | `python/equity/08-algo-execution` | `python/fx/08-algo-execution` |
| 9 | Volatility Surface & Stochastic Vol | `python/equity/09-vol-surface` | `python/fx/09-vol-surface` |
| 10 | Regime-Switching Quant Strategy | `python/equity/10-regime-switching` | `python/fx/10-regime-switching` |

Every project directory follows the same layout (`src/`, `tests/`, `examples/run_pipeline.py`,
`docs/{METHODOLOGY,VALIDATION,DESK_GUIDE}.md`) — see `CONVENTIONS.md` for the
full contract. Run any project's pipeline end-to-end with:

```bash
cd python/equity/01-options-pricing
pip install -e . && pytest -q && python examples/run_pipeline.py
```

## Foundations projects

Three additional, standalone projects that predate the 10-area buildout above
and are kept deliberately smaller in scope: single asset (equity), single
language (Python), no FX twin, no compiled engine. Each still follows the
full CONVENTIONS.md documentation contract. They exist to demonstrate core
technique from first principles rather than to duplicate the flagship
engines above — cross-references to the relevant flagship project are called
out explicitly in each one's `docs/DESK_GUIDE.md`.

| # | Project | Path | What it is, and how it differs from the flagship equivalent |
|---|---|---|---|
| F1 | Risk Metrics on Real Data | `python/foundations/01-risk-metrics` | Single-asset volatility/VaR/ES/Sharpe toolkit with three VaR methods compared side by side. Single-asset only — no correlation/portfolio risk; see `python/equity/03-var-es-engine` for the multi-asset, backtested extension. |
| F2 | Trading Signal Backtest | `python/foundations/02-trading-signal-backtest` | A moving-average crossover strategy, backtested with strict no-look-ahead execution, transaction costs, in-sample/out-of-sample and walk-forward evaluation. The point is backtest discipline, not the signal itself. |
| F3 | Black-Scholes Replication | `python/foundations/03-black-scholes-replication` | A from-scratch, zero-scipy (`math.erf` only) rebuild of Black-Scholes-Merton, validated by an independent Monte Carlo pricer, analytic identities, and Greeks-vs-finite-differences — used as a model-validation reference, not a production pricer. `python/equity/01-options-pricing` (with C++/Rust twins) is the production-grade version. |

```bash
cd python/foundations/01-risk-metrics
pip install -e . && pytest -q && python examples/run_pipeline.py
```

## Why C++ and Rust twins, and why only these four

Options pricing/Greeks and VaR/ES are the two workloads on a real desk that
are (a) called millions of times per revaluation cycle (full-book Greeks,
intraday VaR) and (b) sit on a numerically well-understood, closed-form-or-MC
core that is worth hand-optimizing once semantics are locked down in Python.
The other six areas are research/modeling workflows (calibration, backtesting,
optimization) where iteration speed in Python matters far more than
microsecond latency, so they stay Python-only. This mirrors how a real quant
desk actually splits its stack: Python/research on top, a compiled pricing
kernel underneath for the paths that are actually hot.

## Cross-language validation — the technical backbone

Each of the four engine pairs is validated **three ways**, not just tested
in isolation:

1. **Python reference** — the original model, fully tested against analytic
   identities (put-call parity, Greeks-vs-finite-difference, tree→BS
   convergence) and statistical backtests.
2. **C++ engine** — mirrors the Python semantics exactly. A `tests/golden/generate_golden.py`
   script in the Python project (or, for the VaR engines, direct
   `PYTHONPATH=src python3` invocation) produces reference values; a
   `tools/gen_golden_header.py` script turns them into a `constexpr` C++
   header (`tests/golden_vectors.hpp`). GoogleTest validates every case to
   1e-9 (options) or 1e-6–1e-8 (VaR/ES) tolerance.
3. **Rust engine** — the same golden values, generated into `src/golden.rs`
   by `tools/gen_golden_rs.py`, validated by `cargo test` to the same
   tolerances.

The result: for every priced option or computed VaR number, you can trace
identical (to floating-point-noise) results through three independently
implemented, independently tested codebases. This is the same discipline a
real market-risk function uses when a new pricing library has to be proven
against the incumbent before it's allowed to touch P&L.

**What the cross-language checks do *not* prove**: Monte Carlo paths are
never bit-identical across languages (each uses its own RNG — NumPy's PCG64,
a custom xoshiro256++ in Rust, `std::mt19937_64` in C++), so MC agreement is
statistical (within a few standard errors), not exact. This is documented
explicitly in each engine's `docs/VALIDATION.md`.

## Build & test everything

```bash
# Python (all 23 projects: 20 dual-asset-class + 3 foundations)
for d in python/equity/*/ python/fx/*/ python/foundations/*/; do (cd "$d" && pip install -e . -q && pytest -q); done

# C++ (4 engines)
for d in cpp/*/; do
  cmake -S "$d" -B "/tmp/build-$(basename "$d")" && \
  cmake --build "/tmp/build-$(basename "$d")" -j && \
  ctest --test-dir "/tmp/build-$(basename "$d")"
done

# Rust (4 engines)
for d in rust/*/; do (cd "$d" && RUSTFLAGS="-D warnings" cargo test --release); done
```

## Test coverage

**5,963 tests across the portfolio**, all passing, all independently
reproduced from a clean build (not taken from build logs, not taken from
agent self-reports — every count below was reproduced by a fresh
`pip install -e .` + `pytest` / `cmake --build` + `ctest` / `cargo test
--release` run):

| Area | Projects | Tests |
|---|---|---|
| Python — equity | 10 | 2,080 |
| Python — FX | 10 | 2,713 |
| Python — foundations | 3 | 462 |
| C++ engines | 4 | 324 |
| Rust engines | 4 | 384 |
| **Total** | **31** | **5,963** |

Per-engine, showing the three-language cross-validation:

| Engine | Python ref | C++ | Rust |
|---|---|---|---|
| Equity options/Greeks | `python/equity/01-options-pricing` — 291 | 58 | 88 |
| FX options/Greeks | `python/fx/01-options-pricing` — 414 | 89 | 99 |
| Equity VaR/ES | `python/equity/03-var-es-engine` — 266 | 89 | 98 |
| FX VaR/ES | `python/fx/03-var-es-engine` — 371 | 88 | 99 |

See each project's `docs/VALIDATION.md` for the exact commands, tolerances,
and the documented failure modes each suite pins.

### On input validation

A portfolio-wide hardening pass found the same defect class in project after
project, in all three languages: **a guard written as `if x <= 0: raise`
silently accepts NaN**, because every comparison with NaN is false. The
consequences were not cosmetic — a NaN spot reaching a pricer returned a NaN
"price"; a NaN covariance reaching a VaR aggregator returned, in the Rust and
C++ VaR engines, a portfolio VaR of *exactly zero* (because `max(NaN, 0.0)`
propagates the non-NaN operand), which is more dangerous than a NaN because it
is a plausible number for a hedged book; and a NaN in a P&L feed caused
VaR-exception counters to record zero breaches, so a broken model passed its
Kupiec and Basel traffic-light backtests green.

Every public entry point across the portfolio now validates with explicit
finiteness checks, and each project's `docs/VALIDATION.md` documents its
validation contract and what the pre-fix symptom was.

### On numerical robustness

A second, deeper review pass (benchmarked against the standard a top-tier
options/risk desk would hold a new pricing library to before it touches P&L)
went past input validation into the solvers' own numerics, and found two
defect classes repeated across every engine that shared the affected
algorithm:

- **Implied-vol solver plateau.** All six options-pricing engines (Python,
  C++, and Rust, equity and FX) used a Newton-Raphson implied-vol solver
  that exited as soon as the price residual stopped improving. Near a flat
  region of the price-vs-vol curve (deep ITM/OTM, long-dated, high-vol) this
  let the solver stop one or more Newton steps early, so the returned
  implied vol silently lost precision instead of continuing to bisection.
  In the FX engines this could saturate the solver at the model's
  no-arbitrage upper bound and return a vol that did not actually reprice
  the input — a materially wrong, not just imprecise, answer. Fixed in all
  six engines by adding a bracket-bisection refinement stage that always
  runs to convergence rather than exiting on stalled Newton progress.
- **Cornish-Fisher domain-check grid resolution.** All six VaR/ES engines
  checked the Cornish-Fisher expansion's monotonicity domain by scanning a
  finite grid of points, which can miss a non-monotone region that falls
  between two grid nodes and silently accept an invalid quantile. Fixed in
  all six engines by replacing the grid scan with the exact closed-form
  location of the expansion's stationary point, which cannot miss a
  non-monotone region regardless of grid spacing.

Neither fix changed any golden-vector reference value — both are corrections
to numerical *robustness* (how the solver behaves in the tail of its input
domain), not to the pricing/risk formulas the golden vectors pin.

A third defect, found in the equity and FX Monte Carlo VaR engines: the
standard-error estimate attached to the VaR figure is a local
density-at-the-quantile estimate (a Gaussian KDE in Python and the FX C++/Rust
engines, an order-statistic finite-difference estimate in the equity C++/Rust
engines) — the same defect class as before, just one level removed from the
price itself. A fixed, bulk-tuned bandwidth under-resolves the tail density,
so the reported SE systematically underestimates the true sampling
variability by roughly 9-17% in deep tails or with modest scenario counts —
directionally overconfident, not just noisy, exactly where a desk would rely
on it most. All four Monte Carlo VaR engines (Python, C++, and Rust, equity
and FX) now expose a second, distribution-free bootstrap standard-error
estimator as a cross-check (`var_standard_error_bootstrap` in Python,
`mc_bootstrap_se`/`var_standard_error_bootstrap` in C++, `var_bootstrap_se`/
`var_standard_error_bootstrap` in Rust): resample the scenario P&L with
replacement, recompute VaR on each resample with the exact same quantile rule
as the point estimate, and take the standard deviation across resamples. No
bandwidth to choose, so it doesn't share the local-density estimator's bias
(at the cost of higher trial-to-trial variance in the SE estimate itself
unless the bootstrap count is generous) — recommended whenever `alpha >=
0.995` or scenario counts are modest. This also did not touch any
golden-pinned value; it adds a second estimator, it does not change the
first.

One narrower, single-engine finding from the same pass: a latent
antithetic-pairing bug in the FX vol-surface Heston Monte Carlo (an odd
`n_paths` could silently break the antithetic invariant and understate the
reported standard error — now rejected explicitly rather than silently
mishandled).

One statistical caveat was investigated and documented rather than "fixed":
Kupiec's proportion-of-failures test uses a chi-squared(1) asymptotic
reference distribution that is provably oversized at the Basel
250-day/99% backtesting window (empirically closer to a 9.5% actual
rejection rate than its nominal 5%), which is a property of the classical
test itself, not a bug in this portfolio's implementation — pinning an exact
alternative would trade one textbook convention for a nonstandard one, so
this is called out explicitly in `docs/VALIDATION.md` wherever Kupiec is
used instead of silently changed.

## Data

All projects use synthetic data (deterministic generators, explicit seeds,
in `src/*/data/synthetic.py` or equivalent) or free/downloadable public data
(Yahoo Finance, FRED, ECB — import-guarded, never required for tests to
pass offline). No paid data dependency anywhere in the portfolio.

## Documentation contract

Every project's `docs/` answers, in writing: why this model was chosen over
at least two alternatives; a numbered list of assumptions and what breaks if
each is violated; how it was validated (analytic, convergence, statistical,
cross-model); where it fails, with reproducible examples; and how a real
desk would actually use it (daily workflow, controls, governance). See
`CONVENTIONS.md` for the full contract and asset-class conventions
(equity: continuous dividend yield, ACT/365F; FX: BASE/QUOTE pairs,
Garman–Kohlhagen, four delta conventions, CIP forwards).
