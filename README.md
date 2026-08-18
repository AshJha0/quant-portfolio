# Quant Finance Portfolio

A flagship, end-to-end quant finance portfolio: **10 project areas**, each built
**twice** (equity and FX as fully separate, self-contained projects), in
**Python** throughout and with **C++ and Rust performance twins** for the four
highest-value pricing/risk engines. 28 sub-projects in total, every one with
its own tests, benchmarks, and documentation answering *why this model*,
*what assumptions it makes*, *how it was validated*, *where it fails*, and
*how a real desk would use it*.

```
quant-portfolio/
├── CONVENTIONS.md        # The engineering & documentation contract every project follows
├── README.md              # This file
├── python/
│   ├── equity/01-…10-…    # 10 equity projects
│   └── fx/01-…10-…        # 10 FX projects (fully separate from equity)
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
# Python (all 20 projects)
for d in python/equity/*/ python/fx/*/; do (cd "$d" && pip install -e . -q && pytest -q); done

# C++ (4 engines)
for d in cpp/*/; do
  cmake -S "$d" -B "/tmp/build-$(basename "$d")" && \
  cmake --build "/tmp/build-$(basename "$d")" -j && \
  ctest --test-dir "/tmp/build-$(basename "$d")"
done

# Rust (4 engines)
for d in rust/*/; do (cd "$d" && RUSTFLAGS="-D warnings" cargo test --release); done
```

## Engine status

| Engine | Python ref | C++ | Rust |
|---|---|---|---|
| Equity options/Greeks | `python/equity/01-options-pricing` | 48/48 tests | 71 tests (45 integration + 26 doctests) |
| FX options/Greeks | `python/fx/01-options-pricing` | 79/79 tests | 88 tests (69 integration + 19 doctests) |
| Equity VaR/ES | `python/equity/03-var-es-engine` | 77/77 tests | 86 tests (77 integration + 9 doctests) |
| FX VaR/ES | `python/fx/03-var-es-engine` | 72/72 tests | 84 tests |

All counts independently reproduced (`ctest` / `cargo test`) from a clean
build, not taken from build logs — see each engine's `docs/VALIDATION.md`
for the exact commands and tolerances.

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
