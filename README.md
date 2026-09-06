# Quant Finance Portfolio

A flagship, end-to-end quant finance portfolio: **10 dual-asset-class project
areas**, each built **twice** (equity and FX as fully separate, self-contained
projects), in **Python** throughout and with **C++ and Rust performance
twins** for the four highest-value pricing/risk engines — plus **3 standalone
foundations projects** that build and validate core techniques from scratch,
single-asset, single-language. 31 sub-projects and **6,446 passing tests** in
total, every one with its own tests, and documentation answering *why this
model*, *what assumptions it makes*, *how it was validated*, *where it
fails*, and *how a real desk would use it*.

**New here?** [`docs/`](docs/README.md) is the portfolio-wide learning and
reference layer: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/DIAGRAMS.md`](docs/DIAGRAMS.md) for how everything fits together,
[`docs/LEARN.md`](docs/LEARN.md) for a structured tour of every topic plus a
481-question self-test bank, [`docs/COOKBOOK.md`](docs/COOKBOOK.md) for
verified-runnable "how do I..." recipes, and
[`docs/MARKET_RISK.md`](docs/MARKET_RISK.md) for the VaR/ES/backtesting
workflow end to end.

```
quant-portfolio/
├── CONVENTIONS.md        # The engineering & documentation contract every project follows
├── README.md              # This file
├── docs/                  # Portfolio-wide architecture, learning guide, cookbook, market-risk workflow
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

(`pytest -q` also works from any project root *without* the editable
install — every `pyproject.toml` sets `pythonpath = ["src"]` or ships a root
`conftest.py` — so the install is only needed for `examples/run_pipeline.py`.)

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
2. **C++ engine** — mirrors the Python semantics exactly. A
   `tests/golden/generate_golden.py` script in each of the four Python
   reference projects writes `tests/golden/golden_vectors.json` (committed,
   and locked from the Python side by `tests/test_golden_vectors.py`); the
   engine's `tools/gen_golden_header.py` turns that JSON into a `constexpr`
   C++ header (`tests/golden_vectors.hpp`, committed). GoogleTest validates
   every case to 1e-10 (options) or 1e-9 relative / 1e-6–1e-8 (VaR/ES)
   tolerance.
3. **Rust engine** — the same JSON, generated into `src/golden.rs` (options
   engines) or `tests/golden/mod.rs` (VaR engines) by
   `tools/gen_golden_rs.py`, validated by `cargo test` to the same
   tolerances.

Every generator accepts `--check`, which fails if the committed constants
have drifted from the JSON; CI runs all eight checks before building.

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

These are the loops `.github/workflows/ci.yml` runs on every push (jobs
`python`, `cpp`, `rust`; the `python` job also runs the repo-level
`tests/test_docs_counts.py`, which recounts every number quoted in this
README and in `docs/` from the tree):

```bash
# Python (all 23 projects: 20 dual-asset-class + 3 foundations).
# The 23 projects are 23 separate distributions installed into one
# environment here; use a venv per project (or `uv`/`nox`) if you would
# rather keep them isolated.
for d in python/equity/*/ python/fx/*/ python/foundations/*/; do (cd "$d" && pip install -e . -q && pytest -q); done

# Repo-level docs/count consistency check
python -m pytest -q tests/test_docs_counts.py

# C++ (4 engines; golden provenance check, then build one engine at a time)
for d in cpp/*/; do
  python3 "$d/tools/gen_golden_header.py" --check && \
  cmake -S "$d" -B "/tmp/build-$(basename "$d")" && \
  cmake --build "/tmp/build-$(basename "$d")" -j2 && \
  ctest --test-dir "/tmp/build-$(basename "$d")"
done

# Rust (4 engines)
for d in rust/*/; do python3 "$d/tools/gen_golden_rs.py" --check && (cd "$d" && RUSTFLAGS="-D warnings" cargo test --release); done
```

## Test coverage

**6,446 tests across the portfolio**, all passing, all independently
reproduced from a clean build (not taken from build logs, not taken from
agent self-reports — every count below was reproduced by a fresh
`pip install -e .` + `pytest` / `cmake --build` + `ctest` / `cargo test
--release` run, and `tests/test_docs_counts.py` re-derives every number in
this section from the tree so it cannot go stale silently):

| Area | Projects | Tests |
|---|---|---|
| Python — equity | 10 | 2,350 |
| Python — FX | 10 | 2,830 |
| Python — foundations | 3 | 554 |
| C++ engines | 4 | 326 |
| Rust engines | 4 | 386 |
| **Total** | **31** | **6,446** |

Per-engine, showing the three-language cross-validation:

| Engine | Python ref | C++ | Rust |
|---|---|---|---|
| Equity options/Greeks | `python/equity/01-options-pricing` — 291 | 58 | 88 |
| FX options/Greeks | `python/fx/01-options-pricing` — 434 | 89 | 99 |
| Equity VaR/ES | `python/equity/03-var-es-engine` — 442 | 90 | 99 |
| FX VaR/ES | `python/fx/03-var-es-engine` — 465 | 89 | 100 |

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

Every VaR/ES/backtest and options-pricing entry point in the six golden
engines, the fixed-income bond/YTM/duration stack and the foundations
Black–Scholes rebuild now validate with explicit finiteness checks (NaN
*and* ±inf), pinned by a dedicated `test_input_hardening.py` /
`test_nan_guards.py` suite in each of those projects, and each project's
`docs/VALIDATION.md` documents its validation contract and what the
pre-fix symptom was. Two honest footnotes: the equity Python VaR backtester
— the reference the golden vectors are generated from — was the last
engine to receive the P&L gate (a third review pass found
`exceptions_from_pnl` still counting a NaN day as "no exception" while its
C++/Rust twins already refused it; fixed and pinned in
`python/equity/03-var-es-engine/tests/test_input_hardening.py`), and the
research-lane modules outside those projects (execution, surfaces, regime,
portfolio) still rely on sign guards plus the NaN-free synthetic data
contract, so "portfolio-wide" is a direction of travel there, not a
finished sweep.

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

## Real-world usage notes

What a desk would actually do with this code, and where the honest limits
are (each project's `docs/DESK_GUIDE.md` has the per-area version):

- **Use the Python packages as the reference and the C++/Rust engines as the
  hot path.** The Python code is where semantics are decided and tested; the
  compiled twins exist for the two workloads that are called millions of
  times per revaluation cycle (full-book Greeks, intraday VaR). If you change
  a formula, change it in Python first, regenerate the golden JSON, and let
  the `--check` scripts tell you which engine is now out of date.
- **Treat the golden vectors as the model-validation hand-off.** The
  `tests/golden/golden_vectors.json` files are what an independent
  validation function would ask for first: closed-form inputs, `repr()`
  doubles, committed tolerances. They prove that three implementations agree;
  they do *not* prove the Python reference is right — that burden is carried
  by the analytic-identity, convergence and statistical tests in each
  project.
- **Backtest inputs are gated, not cleaned.** Every VaR/ES/backtest entry
  point refuses NaN/±inf rather than imputing. A production feed will contain
  holes; the intended workflow is to fix or flag them upstream, because a
  silently skipped day is exactly the failure that made a broken model pass
  Kupiec and Basel green before the hardening passes.
- **The FX book backtest is the static-book hypothetical P&L** (today's
  positions revalued under each historical day's factor returns). That is the
  right series for Kupiec/Christoffersen; a desk also needs the *actual* P&L
  with fees and new deals stripped, which you supply to
  `fx_var.evaluate_var_backtest` / `eq_var.exceptions_from_pnl` as plain
  arrays.
- **Kupiec is oversized at 250/99%** (exact finite-sample size 9.5%, pinned in
  `TestKupiecExactSize` in both VaR engines) and the Basel zone function
  derives its zone from the cumulative binomial probability, so it is safe on
  non-250-day windows. Read a Kupiec p-value near 0.05 as "borderline", not as
  a verdict.
- **Cornish–Fisher VaR checks its own validity domain** and raises outside
  it; do not disable `check_domain` to make a number appear.
- **Not implemented, deliberately**: SABR (area 09 discusses it as a rejected
  alternative), Merton/KMV structural PD (area 06 is logistic PD + Vasicek
  IRB), short-rate models (area 04 is curve construction and cash-flow risk).
  If one of those is the first question you would ask, it is a gap, not an
  oversight.
- **Twin APIs differ on purpose.** `eq_var.kupiec_pof(n_obs, n_exceptions,
  alpha=0.01)` takes a tail probability; `fx_var.kupiec_pof(n_exceptions,
  n_obs, alpha=0.99)` takes a confidence level (see `CONVENTIONS.md`). Pass
  by keyword when moving between the equity and FX packages.
- **Dependencies are lower-bounded, not locked.** The golden constants were
  produced with NumPy 2.x / SciPy 1.17; a special-function change in a future
  SciPy could move a `chi2.sf` or `norm.ppf` at the 1e-12 level. CI pins the
  interpreter matrix (3.11, 3.12) and re-checks the goldens on every run, so a
  drift would be caught, but a production deployment should add a lock file
  per project.

## References

The sources each area is built from and validated against. Where the review
that produced this list re-derived a formula, the implementation matched.

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
16. J.P. Morgan / Reuters (1996). *RiskMetrics — Technical Document*, 4th ed. (EWMA λ = 0.94.)
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

56. Hosmer, D. W. & Lemeshow, S. (1980). "Goodness of Fit Tests for the Multiple Logistic Regression Model." *Communications in Statistics — Theory and Methods* A9(10), 1043–1069.
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
90. Hagan, P. S., Kumar, D., Lesniewski, A. S. & Woodward, D. E. (2002). "Managing Smile Risk." *Wilmott*, September, 84–108. (SABR — discussed as an alternative in area 09; not implemented.)

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
99. Merton, R. C. (1974). "On the Pricing of Corporate Debt: The Risk Structure of Interest Rates." *Journal of Finance* 29(2), 449–470. (Structural PD — not implemented in area 06; listed for orientation.)

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

## License

MIT License, Copyright (c) 2026 Ashish Jha — see [LICENSE](LICENSE).
