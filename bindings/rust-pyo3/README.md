# qp-rust: Python bindings for the Rust engines

`qp_rust` is a PyO3 extension over the four Rust crates in `rust/`:
`eq-options-engine`, `fx-options-engine`, `eq-var-engine` and
`fx-var-engine`. It exposes the same functions, names and conventions as
`qp_cpp` (see `bindings/cpp-nanobind/README.md`), so a caller can switch
engine by switching the import. The Python reference packages are
unchanged and remain the reference.

The package has its own `Cargo.lock` and an empty `[workspace]` table. It
is not a member of `rust/Cargo.toml`, so the engine crates keep zero
third-party dependencies. PyO3 0.30 and numpy 0.30 are the only additions.
The extension is built against the stable ABI (`abi3`, Python 3.10 and
later).

## Install and test

From the repository root:

```
pip install ./bindings/rust-pyo3
pip install -e python/equity/01-options-pricing -e python/fx/01-options-pricing \
            -e python/equity/03-var-es-engine -e python/fx/03-var-es-engine pytest
pytest bindings/rust-pyo3/tests
```

Build requirements: Rust stable (tested with 1.99.1) and Python 3.11 or
later. maturin is fetched by pip as the build backend. Lint with
`cargo clippy --release --locked --all-targets -- -D warnings` and
`cargo fmt --check` from this directory.

## What is exposed

| Function | Engine call |
|---|---|
| `eq_bs_price`, `eq_bs_greeks`, `eq_implied_vol` | `fast::try_bs_price`, `try_bs_greeks`, `try_implied_vol` |
| `eq_bs_price_batch`, `eq_bs_greeks_batch` | `fast::bs_price_batch`, `bs_greeks_batch` |
| `eq_implied_vol_batch` | loop over `fast::try_implied_vol` in Rust |
| `eq_crr_price`, `eq_mc_price` | `fast::try_crr_price`, `fast::try_mc_price` |
| `fx_gk_price`, `fx_gk_greeks`, `fx_implied_vol` | `fast::gk_price_k`, `greeks_k`, `implied_vol_k` |
| `fx_implied_vol_batch` | loop over `fast::implied_vol_k` in Rust |
| `fx_gk_price_batch`, `fx_gk_greeks_batch` | `fast::gk_price_batch`, `greeks_batch` |
| `fx_binomial_price`, `fx_mc_price` | `fast::binomial_price_ws`, `fast::mc_price_par` |
| `eq_historical_var`, `eq_expected_shortfall` | `historical::historical_var`, `expected_shortfall::expected_shortfall` |
| `eq_parametric_var`, `eq_monte_carlo_var` | `parametric::parametric_var_full`, `monte_carlo::monte_carlo_var` |
| `eq_kupiec_pof`, `eq_basel_traffic_light` | `backtest::kupiec_pof`, `basel_traffic_light` |
| `fx_empirical_var_es`, `fx_normal_var`, `fx_normal_es`, `fx_t_var`, `fx_t_es` | `expected_shortfall::*` |
| `fx_var_covar` | `parametric::var_covar` |
| `fx_kupiec_pof`, `fx_basel_traffic_light` | `backtest::kupiec_pof`, `basel_traffic_light` |

Conventions are those of `qp_cpp`: `"call"`/`"put"` strings in scalar
calls, a boolean `is_call` array in batch calls, equal-length 1-D arrays
read in place when C-contiguous float64, new output arrays, the GIL
released while the engine runs, `ValueError` with the engine's message on
any error, `alpha` as the tail probability and keyword-only backtests.
Error messages are the Rust engines' own text, which differs from the C++
text.

`eq_mc_price` rejects `threads=0` with `ValueError`, as `qp_cpp` does. The
engine function it calls, `fast::try_mc_price`, rejects 0 as well
(`ErrorCode::BadThreads`); the binding checks first so the message is the
same in both packages. CONVENTIONS.md lists how each engine entry point
treats a thread count of 0.

The Monte Carlo streams differ from the C++ engines and from NumPy, so MC
results agree with them statistically, not by value.

## Tests and tolerances

`tests/test_binding.py` is the same suite as the C++ package (85 tests),
with the same tolerances and the same reasons (see the table in
`bindings/cpp-nanobind/README.md`). Largest deviations from Python on the
test grid (Windows, Rust 1.99.1): price 5.7e-14, Greeks 9.1e-13, FX Greeks
3.6e-15, CRR 0. The largest implied-vol difference, 2.1e-4, is at a
low-vega contract where the price pins the vol only to that precision; it
is inside the 4 eps M / vega bound.

Results: 85 passed on Windows (Python 3.12.10) and on WSL Ubuntu 26.04
(Python 3.12.15 and 3.13.17).

## Overhead from Python

`python bench/overhead.py` prices 100,000 options. Machine: Intel Core 5
120U laptop, shared with other builds during the runs, so the numbers are
upper bounds and moved by up to a factor of 2 between runs. The kernel cost
alone is in `rust/equity-options-engine/bench`.

| Call (ns per option) | Windows, Py 3.12 | WSL, Py 3.12 |
|---|---:|---:|
| Python reference `bs_price`, loop | 5,555 | 9,380 |
| `eq_bs_price`, loop | 663 | 406 |
| `eq_bs_price_batch`, one call | 158 | 134 |
| `eq_bs_greeks_batch`, one call | 204 | 223 |
| `eq_bs_price_batch` of length 1, per call | 4,900 | 2,207 |
