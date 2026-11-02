# qp-cpp: Python bindings for the C++ engines

`qp_cpp` is a nanobind extension over the four C++ engines in `cpp/`:
equity options (`eqopt::`), FX options (`fxopt::`), equity VaR (`eqvar::`)
and FX VaR (`fxvar::`). It calls the engines' non-throwing `try_*`
functions and batch kernels. The Python reference packages (`eq_options`,
`fx_options`, `eq_var`, `fx_var`) are unchanged and remain the reference.

## Install and test

From the repository root (the package builds the engines from `../../cpp`,
so it installs from the repository tree, not from an sdist):

```
pip install ./bindings/cpp-nanobind
pip install -e python/equity/01-options-pricing -e python/fx/01-options-pricing \
            -e python/equity/03-var-es-engine -e python/fx/03-var-es-engine pytest
pytest bindings/cpp-nanobind/tests
```

Build requirements: a C++23 compiler (GCC 15, Clang 21 or MSVC 19.44),
CMake 3.26 or later and Python 3.11 or later. scikit-build-core and nanobind
are fetched by pip. On Windows run the install from a shell where
`vcvars64.bat` has been called.

## Wheels

The plain idea: CI turns this package into installable wheel files so a
user does not need a C++23 compiler on their machine. The wheels are kept as CI
artefacts only; nothing is uploaded to PyPI.

Formally, the `wheels` job in `.github/workflows/ci.yml` runs cibuildwheel
3.1.4 with the settings under `[tool.cibuildwheel]` in `pyproject.toml`:
CPython 3.11, 3.12 and 3.13 on manylinux_2_34 x86_64 and Windows amd64.
Each wheel is installed into a clean environment and checked by
`bindings/wheel_smoke.py`, which imports the module and prices one
contract. On Linux the build installs gcc-toolset-15 in the container, since the engines need C++23 features the image's default compiler lacks. The full test suite above is not run against the wheel,
because it needs the Python reference packages.

Worked example, the same check by hand on Linux with Python 3.14
(2026-11-02, WSL Ubuntu, GCC 15.2):

```
python -m build --wheel --outdir dist bindings/cpp-nanobind
python -m venv /tmp/clean && /tmp/clean/bin/pip install dist/qp_cpp-0.1.0-cp314-cp314-linux_x86_64.whl
cd /tmp && /tmp/clean/bin/python <repo>/bindings/wheel_smoke.py qp_cpp
# qp_cpp 0.1.0: ok (eq_bs_price = 10.450583572185565)
```

Download a CI wheel from the run's artefacts (`wheels-cpp-nanobind-<os>`) and
`pip install` the file that matches your Python and platform.

## What is exposed

Functions are prefixed by engine. Every engine error raises `ValueError`
with the engine's message. `available()` returns True when the extension
imported.

| Function | Engine call |
|---|---|
| `eq_bs_price`, `eq_bs_greeks` | `eqopt::try_bs_price`, `try_bs_greeks` |
| `eq_bs_price_batch`, `eq_bs_greeks_batch` | `eqopt::bs_price_batch`, `bs_greeks_batch` |
| `eq_implied_vol`, `eq_implied_vol_batch` | `eqopt::try_implied_vol` (batch loops in C++) |
| `eq_crr_price` | `eqopt::try_crr_price` |
| `eq_mc_price` | `eqopt::try_mc_price` with a `ThreadPool` of `threads - 1` workers |
| `fx_gk_price`, `fx_gk_greeks` | `fxopt::try_gk_price`, `try_analytic_greeks` |
| `fx_gk_price_batch`, `fx_gk_greeks_batch` | `fxopt::gk_price_batch`, `gk_greeks_batch` |
| `fx_implied_vol`, `fx_implied_vol_batch` | `fxopt::try_implied_vol` |
| `fx_binomial_price` | `fxopt::try_binomial_price` |
| `fx_mc_price` | `fxopt::try_mc_price` |
| `eq_historical_var`, `eq_expected_shortfall` | `eqvar::try_historical_var`, `try_expected_shortfall` |
| `eq_parametric_var` | `eqvar::try_parametric_var` |
| `eq_monte_carlo_var` | `eqvar::monte_carlo_var` (returns the VaR) |
| `eq_kupiec_pof`, `eq_basel_traffic_light` | `eqvar::kupiec_pof`, `basel_traffic_light` |
| `fx_empirical_var_es` | `fxvar::try_empirical_var_es` |
| `fx_normal_var`, `fx_normal_es`, `fx_t_var`, `fx_t_es` | `fxvar::try_normal_var` and siblings |
| `fx_var_covar` | `fxvar::try_var_covar` |
| `fx_kupiec_pof`, `fx_basel_traffic_light` | `fxvar::kupiec_pof`, `basel_traffic_light` |

Signatures and return types are in `src/qp_cpp/__init__.pyi`.

## Conventions

- Option type is the string `"call"` or `"put"` in scalar calls and a
  boolean array `is_call` in batch calls.
- Batch calls take equal-length 1-D arrays. C-contiguous float64 arrays are
  read in place. Other arrays (strided views, other dtypes) are converted
  first. Outputs are new float64 arrays. The GIL is released while the
  engine runs. A bad element raises `ValueError("element i: ...")` naming
  the first offending index.
- Batch output equals the scalar call bit for bit (tested).
- VaR inputs follow the reference packages: `alpha` is the tail probability
  in (0, 0.5), so `alpha=0.99` is rejected. VaR and ES are positive numbers
  for a loss. `alpha` and the backtest counts are keyword-only.
- `eq_kupiec_pof` returns a dict like `eq_var.kupiec_pof`;
  `fx_kupiec_pof` returns `(lr, p_value)` like `fx_var.kupiec_pof`. The
  Basel functions return dicts with the reference field names.
- Monte Carlo uses the engines' mt19937_64 streams, not NumPy, so it agrees
  with the Python engines statistically, not by value. A result is fixed
  for a given `(seed, threads)`.
- Not bound: the FX VaR Monte Carlo and book-level VaR (they take a
  `Book`/`Market`, which has no array form yet), deltas and strikes from
  delta, Black-76, Christoffersen and Acerbi-Szekely.

## Tests and tolerances

`tests/test_binding.py` (85 tests) compares every function with the
Python reference over seeded random contracts (400 per seed) that include
deep ITM/OTM strikes (K/S from 0.3 to 3), expiries from 1e-3 to 10 years,
vols from 0.01 to 1.5 and negative rates.

| Output | Tolerance | Why |
|---|---|---|
| prices, Greeks | 64 eps (1 + d1^2) (abs(ref) + max(S, K)), times 1/T for Greeks when T < 1 | the differential-test bound: erfc conditioning in the tails, a few roundings in exp/log/sqrt |
| implied vol | 4 max(eps max(S, K) / vega, 1e-12 sigma) | price precision mapped through 1/vega |
| CRR tree | 4 n eps max(S, K) | one rounding per node per step |
| historical VaR | exact | an order statistic is one sample value |
| ES | 1e-13 relative | tail sum, k eps with k <= 250 |
| parametric VaR, closed forms | 1e-12 relative | quadratic form plus the engine's own quantile functions vs scipy |
| Kupiec LR / p-value | 1e-12 rel + 1e-9 abs / 1e-10 rel | log-likelihood difference; chi2 tail from the engine's erfc |
| MC option price | 5 standard errors of the closed form | different RNG stream |
| MC VaR | 1.5 % of the closed-form normal VaR | about 4 quantile standard errors at n = 200,000 |

Largest deviations from Python measured on the test grid (MSVC 19.44,
Windows): price 5.7e-14, Greeks 1.1e-13, FX Greeks 3.6e-15, implied vol 0,
CRR 0.

Error-path tests cover NaN and infinite inputs, negative spot, strike,
expiry and vol, expired implied vol, quotes outside the no-arbitrage
bounds, `alpha` given as a confidence level, exception counts above the
window, bad `dist` and option type strings, length mismatches, and
positional calls to the keyword-only backtests.

Results: 85 passed on Windows (MSVC 19.44, Python 3.12.10), on WSL Ubuntu
26.04 (GCC 15.2, Python 3.12.15 and 3.13.17).

## Overhead from Python

`python bench/overhead.py` prices 100,000 options. Machine: Intel Core 5
120U laptop. These runs shared the machine with other builds, so treat them
as upper bounds. Repeat runs moved by up to a factor of 2. The kernel cost alone is in `cpp/equity-options-engine/bench`.

| Call (ns per option) | Windows, MSVC 19.44, Py 3.12 | WSL, GCC 15.2, Py 3.12 |
|---|---:|---:|
| Python reference `bs_price`, loop | 6,455 | 4,393 |
| `eq_bs_price`, loop | 529 | 652 |
| `eq_bs_price_batch`, one call | 161 | 260 |
| `eq_bs_greeks_batch`, one call | 247 | 299 |
| `eq_bs_price_batch` of length 1, per call | 7,100 | 2,378 |

A batch call has a fixed cost of a few microseconds (argument conversion
for seven arrays), so batches pay off from a few dozen options up.
