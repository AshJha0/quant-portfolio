# Engineering and documentation conventions

Every project in this repository, Python, C++ or Rust, follows these rules.
They give every project the same layout and the same set of questions
answered in writing.

## Directory layout (Python projects)

```
<NN>-<project-name>/
├── README.md              # Overview, quickstart, results summary
├── pyproject.toml         # Installable package, minimum dependency versions, test extra
├── src/<package_name>/    # Library code (import <package_name>)
│   ├── __init__.py        # Public API re-exports with __all__
│   └── *.py               # One module per pipeline stage
├── tests/                 # pytest suite: offline, seeded
│   └── test_*.py
├── examples/              # Scripts that reproduce the README results
│   └── run_pipeline.py    # Data, model, validation, decision; writes to a temp dir or --out
└── docs/
    ├── METHODOLOGY.md     # The maths and why this model was chosen over alternatives
    ├── VALIDATION.md      # How it was validated, failure modes, edge cases
    └── DESK_GUIDE.md      # Intended use, inputs, outputs, limitations
```

## Documentation contract

Every project answers these in writing:

1. Why this model. Compare it with at least two alternatives and state the
   trade-offs (METHODOLOGY.md).
2. What it assumes. A numbered assumptions register, each with what breaks if
   it is violated (METHODOLOGY.md).
3. How it was validated: analytic benchmarks, convergence studies,
   statistical backtests, cross-model checks (VALIDATION.md).
4. Where it fails: known failure modes, regime dependence, numerical limits,
   with reproducible examples (VALIDATION.md).
5. How it is meant to be used: inputs, outputs, controls and limitations
   (DESK_GUIDE.md). Claims about market practice cite a source.
6. Edge cases: deep ITM/OTM, T to 0, zero and negative rates, vol to 0 and to
   infinity, missing data, crisis regimes. An edge case that the docs mention
   has a unit test.

Writing rules: short declarative sentences, no em-dashes, numbers in tables
with the command that produced them, every result from synthetic data says
"synthetic" in the same sentence, no narration of review history (that goes in
CHANGELOG.md). CONTRIBUTING.md has the full list.

## Code quality

- Type hints on public functions; NumPy-style docstrings stating units and
  conventions (for example, rates continuously compounded, annualised).
- Pure functions for pricing and maths; classes only where there is real
  state (calibrated models, backtest engines).
- No network access in code paths used by tests. Loaders for public data
  (Yahoo, FRED, ECB) live in `data/live.py` and are import-guarded. Tests use
  seeded generators in `data/synthetic.py`.
- Every stochastic component takes an explicit `seed` or
  `numpy.random.Generator`.
- Every public entry point checks its inputs for finiteness and raises
  `ValueError` (Python), returns `Err` (Rust) or throws
  `std::invalid_argument` (C++). Bad input never produces a number.
- Tolerances: a tolerance in a test states why it is the right size (ulps,
  condition number or Monte Carlo standard errors).

## Testing contract

- `python -m pytest -q` passes offline from the project root with the
  project's test extra installed. Every `pyproject.toml` sets
  `pythonpath = ["src"]` or ships a root `conftest.py`.
- Tests cover happy paths, edge cases, analytic identities (put-call parity,
  Greeks against finite differences, martingale checks), convergence (tree to
  Black-Scholes, Monte Carlo to Black-Scholes) and failure behaviour.
- Property tests use `hypothesis` (Python) or `proptest` (Rust) where the
  property is natural.
- A bug fix comes with a regression test that fails on the old code.

## Asset-class conventions

- **Equity**: continuous dividend yield q; day count ACT/365F; vol quoted on
  log returns, annualised.
- **FX**: pairs quoted BASE/QUOTE (EURUSD = USD per 1 EUR). Domestic rate r_d
  is the quote-currency rate, foreign rate r_f is the base-currency rate.
  Garman-Kohlhagen is Black-Scholes with q = r_f. Spot settles on business
  days. Delta conventions (spot, forward, premium-adjusted) and ATM/RR/BF
  quoting are documented where used; the FX surface uses each pair's native
  convention by default.

## Risk conventions (all VaR engines)

- `alpha` is the tail probability in (0, 0.5): 0.01 means 99% VaR. Values
  outside that interval raise.
- Historical VaR is the order statistic L(ceil(n alpha)) of the sorted
  losses. ES is the Acerbi-Tasche estimator with the fractional atom.
- `kupiec_pof(*, n_obs, n_exceptions, alpha)` takes keywords only in both
  packages (Rust: `KupiecInput`, C++: `KupiecInput` struct).
- `basel_traffic_light(*, n_obs, n_exceptions, alpha=0.01)` works for any
  window and alpha.
- EWMA variance is seeded with the mean squared return of the first
  min(30, n) observations.
- Cholesky adds jitter up to 1e-6 times the mean diagonal, warns when it does,
  and raises above that unless `repair="clip"` or `repair="higham"` is passed.

## C++ and Rust engines

- C++20, CMake 3.20 or later, GoogleTest for tests. Warnings are errors when
  the `<ENGINE>_WERROR` option is ON (EQOPT, FXOPT, EQVAR, FXVAR); CI sets it.
  No runtime dependencies beyond the standard library.
- Rust 2021 edition, standard library only. CI runs
  `cargo test --release --locked` and clippy with warnings denied.
- The engines are ports of the Python design. The golden pipeline: the Python
  project's `tests/golden/generate_golden.py` writes
  `tests/golden/golden_vectors.json`. For options the reference values come
  from an mpmath oracle with a tolerance per output, consumed in Python by
  `tests/test_golden.py`. For VaR they come from the Python reference and are
  checked by `tests/test_golden_vectors.py`. The engine's
  `tools/gen_golden_header.py` (C++) and `tools/gen_golden_rs.py` (Rust) emit
  the committed constants; `--check` fails if they drift from the JSON. The CI
  `goldens --check` job runs all of them.
- Equity and FX packages are separate code, so argument orders differ
  (`eq_options.bs_price(S, K, T, r, sigma, q, option_type)` against
  `fx_options.gk_price(S, K, T, r_d, r_f, sigma, option_type)`). Pass
  everything after the market inputs by keyword.
