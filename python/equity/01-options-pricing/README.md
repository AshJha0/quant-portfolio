# Equity Options Pricing and Greeks Engine (`eq_options`)

Reference implementation of the vanilla equity-options stack: analytic Black-Scholes-Merton, Cox-Ross-Rubinstein binomial
(European + American), Black-76 on forwards/futures, exact-scheme Monte
Carlo with variance reduction and MC Greeks, a full analytic Greek set
(incl. vanna/volga), robust implied vol, a discrete delta-hedging
simulator, and a model-comparison harness. The engines are cross-checked
against each other, against golden vectors computed in mpmath, and against
the C++ and Rust ports on seeded random inputs.

```
data (synthetic chain, GBM paths)
      │
      ▼
Black-Scholes ──► Binomial (CRR, American) ──► Black-76 (forwards)
      │                                              │
      ▼                                              ▼
Monte Carlo (antithetic + CV, pathwise/LR Greeks) ──► Greeks (analytic vs FD)
      │                                              │
      ▼                                              ▼
Delta-hedging simulator (P&L, model risk) ──► Model comparison & convergence
```

**Conventions:** rates/dividend yields continuously compounded,
annualised, ACT/365F; `T` in years; vols quoted on log-returns,
annualised. Every stochastic routine takes an explicit seed. Negative
rates supported; `T=0`, `sigma=0`, `K=0` handled as exact limits;
invalid inputs raise `ValueError`.

## Quickstart

```bash
cd python/equity/01-options-pricing
pip install -e .[dev]                # numpy, scipy, pandas; dev adds pytest, hypothesis, mpmath
python -m pytest tests -q            # 2118 tests, offline, about 100 s
python examples/run_pipeline.py      # full report, about 2 s
python examples/validation_tables.py # remaining VALIDATION.md tables
```

```python
from eq_options import bs_price, bs_greeks, crr_price, implied_vol, mc_price

bs_price(100, 100, 1.0, 0.05, 0.20, q=0.01, option_type="call")  # 9.826298
crr_price(100, 100, 1.0, 0.05, 0.20, 0.01, "put", "american", 2000)  # 6.366622
implied_vol(9.826298, 100, 100, 1.0, 0.05, 0.01, "call")         # 0.20000000
mc_price(100, 100, 1.0, 0.05, 0.20, 0.01, "call", n_paths=200_000, seed=42)
# MCResult(value=9.8136, std_error=0.0064, ...)
```

## Result highlights (reference contract S=100, K=100, T=1y, r=5%, q=1%, σ=20%)

**Cross-model agreement** (European call, BS = 9.826298):

| model | price | abs diff vs BS | runtime |
|---|---:|---:|---:|
| Black-Scholes | 9.826298 | n/a | 0.05 ms* |
| CRR tree, 1000 steps | 9.824328 | 1.97e-3 | 3.4 ms |
| Black-76 on `F=S·e^{(r−q)T}` | 9.826298 | <1e-10 | 0.02 ms |
| MC, 200k paths (anti+CV) | 9.813605 | 1.3e-2 (2.0 SE) | 8 ms |

*single call from `examples/run_pipeline.py` (test machine). A timeit loop gives about 4 µs per `bs_price` call; the code is scalar Python.

**Convergence laws, measured:** CRR `error × n ≈ 1.97` constant (O(1/n));
MC `SE × √n ≈ 2.85` constant (O(n^{-1/2})). Full tables in
[docs/VALIDATION.md](docs/VALIDATION.md).

**Greeks** (analytic, call): delta 0.6118, gamma 0.0189, vega 37.76,
theta −5.73/yr, rho 51.35, vanna −0.189, volga 5.66. All match
central finite differences to <1e-4 relative and mpmath to the golden tolerances.

**American exercise:** 1y ATM put early-exercise premium 0.4234 (2000
steps); American call with q=0 equals European to 1e-10 (Merton).

**Implied vol:** solved on the out-of-the-money side with safeguarded
Halley iteration, at most 10 evaluations, about 25 µs per solve in Python.
Accurate to `max(eps price / vega, 1e-12 sigma)` against mpmath over
ln(K/S) in [-5, 5], T in [1e-7, 50], sigma in [1e-4, 5]; worst error on a
22-quote synthetic skewed chain: 1e-14. Refuses prices at or outside the
no-arbitrage bounds with `ValueError`. A deep in-the-money quote carries
less information than its out-of-the-money twin (see VALIDATION.md 6.5).

**Delta hedging** (short 3M ATM call): P&L std 1.61 → 0.22 as rebalances
go 4 → 256 (`std × √N ≈ 3.4` constant, close to the Derman-Kamal
`√(π/4)·vega·σ`); mean P&L +0.001 ± 0.004 at true
vol; selling 25-vol against 15-vol realized earns +2.00 vs the
gamma-weighted vol-spread theory value +1.99; 5 bp transaction costs at
N=128 drag the mean to −0.234.

## Layout

```
src/eq_options/          black_scholes, binomial, black76, monte_carlo,
                         greeks, hedging, comparison, data/{synthetic,live}
tests/                   2039 offline seeded pytest tests (hypothesis, mpmath oracle,
                         cross-language differential test)
tests/golden/            golden_vectors.json (456 mpmath records, per-output tolerances) + generator
examples/                run_pipeline.py and validation_tables.py reproduce every number above
docs/                    METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```

## Documentation contract

- **Why these models, vs alternatives**, [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §1
- **Assumptions register (A1–A8, each with "what breaks")**, [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §2
- **Validation evidence** (benchmarks, convergence, cross-model), [docs/VALIDATION.md](docs/VALIDATION.md) §1–5
- **Failure modes** (smile contradiction, discrete-hedging error, dividend
  modeling error, CRR limitations, numerical limits), [docs/VALIDATION.md](docs/VALIDATION.md) §6
- **Desk usage** (quotes, risk pipeline, EOD reports, limits, governance), [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md) §1–2
- **Real-life scenarios & edge cases** (earnings jump, dividend
  announcement, expiry pinning, 2008/2020 vol spikes; every edge case
  unit-tested), [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md) §3

## Choosing an engine

`eq_options.engine` wraps the main functions of this package with
`engine="python" | "cpp" | "rust"`. The default, `"python"`, is this
package unchanged and stays the reference. `"cpp"` and `"rust"` call the
compiled ports through the `qp_cpp` and `qp_rust` extensions (install
them from the repository root with `pip install ./bindings/cpp-nanobind`
or `pip install ./bindings/rust-pyo3`). An unknown name raises
`ValueError`; a missing extension raises `ImportError` with that install
command. The compiled results agree with the reference to the
cross-language tolerances listed in the module docstring, checked over
10,000 random inputs per engine by `tests/test_engine_differential.py`.

## Newer models (November 2026)

New modules, each with an oracle test in `tests/test_phase5_models.py`:

| Module | What it does | Oracle |
|---|---|---|
| `lr_tree` | Leisen-Reimer tree, European and American | Black-Scholes; error falls 4x per doubling of steps |
| `pde` | Crank-Nicolson with Rannacher start, Brennan-Schwartz American | Black-Scholes; extrapolated LR tree; order-2 convergence |
| `alo` | Andersen-Lake-Offengeim American pricer, including negative rates and the `q < r < 0` double boundary | LR tree at 2001/4001 steps, Richardson-extrapolated, to 2e-5 |
| `dividends` | Escrowed cash dividends, Roll-Geske-Whaley, dividend tree | Haug's published RGW value 4.3860 |
| `term_structure` | Zero curve, vol term structure, Merton averaging | Closed-form averages; flat curves equal `bs_price` |
| `bachelier` | Normal model price, Greeks, implied normal vol | mpmath integral and derivatives at 30 digits |
| `sabr` | Hagan and Obloj SABR vols, stable ATM evaluation, density check | Hagan's ATM formula; beta = 1, nu = 0 gives alpha |
| `sabr_pde` | Arbitrage-free SABR density by the Hagan et al. (2014) PDE on a Le Floc'h-Kennedy grid | Exact mass and forward conservation; non-negative density; Hagan's expansion at short expiry |
| `greeks_higher` | Charm, speed, colour, zomma, veta | `mpmath.diff` of the price at 30 digits |
| `risk_grids` | Spot ladder, vega by expiry bucket, spot-by-vol P&L grid | Direct repricing; bucket totals equal book vega |

Import them from their modules (for example `from eq_options.alo import alo_price`);
the package `__init__` is unchanged this round.
