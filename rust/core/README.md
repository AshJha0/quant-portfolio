# qp-core

Shared numerical core of the four Rust engines in `rust/`. Library name
`qp_core`. It depends on `std` only; `proptest` is a test-only dependency.

The engines re-export what they use from here, so their public paths did
not change. The code was moved, not rewritten: the arithmetic runs in the
same order as before, and every engine golden, bit-equality and
differential test passes unchanged.

## Contents

| Module | Contents | Used by |
|---|---|---|
| `rng` | `splitmix64`, `SplitMix64`, `Xoshiro256PlusPlus` (cached Box-Muller normal), `Xoshiro256StarStar`, the open and half-open uniform maps, `box_muller`, the Marsaglia-Tsang gamma loop | all four engines |
| `special` | Cody `erf`/`erfc` in two coefficient tables, normal CDF and PDF, the Acklam rational inverse normal, one Halley step | all four engines |
| `iv` | Normalised-Black implied-vol core (same algorithm as the Python and C++ engines), generic over the erf table | both options engines |
| `black_scholes` | Black-Scholes in cost-of-carry form (`b = r - q = r_d - r_f`): `carry_d1_d2`, `carry_value`, `carry_price`, `carry_greeks`, generic over the normal pair (`ErfcNormal` for equity, `TruncatedC1Normal` for FX) | both options engines |
| `roots` | Brent's method with `scipy.optimize.brentq` semantics | FX options |
| `linalg` | Row-major Cholesky kernel that rejects NaN pivots | both VaR engines |
| `par` | Fixed-chunk scoped-thread helper; results do not depend on the thread count | FX VaR |
| `pool` | Persistent worker pool for `'static` jobs; results in task-index order, independent of worker count | new, opt-in |
| `philox` | Philox4x32-10 counter-based generator; Random123 known-answer tests; same words as `qp/philox.hpp` | new, opt-in |
| `quantile` | O(n) `select_kth` and type-7 quantile; same bits as a sort | new, opt-in |
| `gemm` | Cache-blocked GEMM and `D^T D`; same bits as the naive loops | eq VaR `sample_covariance` |
| `vmath` | Branch-free batch exp, erfc, normal CDF; exp 2 ulp, erfc 2e-15 relative, CDF 4e-16 absolute (see the module docs) | new, opt-in |
| `stats` | Special functions and sample moments for both VaR engines (one copy since Round 8; each engine's `stats.rs` is a wrapper) | both VaR engines |
| `path_rng` | Per-path and per-resample Philox streams with AS 241 normals (`PathRng`, batched `philox_uniforms`, `philox_indices`); same words as `qp/path_rng.hpp` | both VaR engines |
| `resample` | `RankedLosses`: bootstrap quantiles by counting draws of the largest losses, same bits as buffer-and-select | FX VaR |
| `risk` | Quote-space risk through the bootstrap Jacobian, par DV01, quote gamma, vol grids, P&L explain; twin of `qp_core.risk` | parity tests |
| `crr` | The one Cox-Ross-Rubinstein lattice (`crr_lattice`, `CrrWorkspace`, `CrrError`) | both options engines |
| `holidays` | Rule-based settlement calendars, FX spot dates, spot-date forwards and NDF values | FX, rates |
| `exotics` | Closed-form exotic pricers, the twin of `qp_core.exotics` | parity tests |
| `rates_vol` | SABR implied vol and the Hull-White trinomial Bermudan, twin of `qp_core.vol.sabr` | parity tests |
| `xva` | SA-CCR (BCBS 279 / CRE52), Hull-White bond and swaption closed forms, CVA / DVA / FVA / effective-EPE sums; parity with `qp_core.xva` and `qp/xva.hpp` | parity tests |
| `credit_index` | ISDA CDS Standard Model legs under the CDX and iTraxx indices | parity tests |
| `bond_futures` | CME conversion factor, implied repo, net basis, basic delivery option | parity tests |

`par` stays on `std::thread::scope` because it borrows the caller's
slices, and a persistent pool cannot do that without `unsafe` (this crate
forbids it). `pool` is for work whose inputs can move into an `Arc`.
| `Error`, `require_finite` | Invalid-input and numerical-failure error; each engine converts it with `From` | all four engines |

## Variants kept on purpose

Before this crate existed the engines had drifted apart in small ways.
Each engine pins golden values computed with its own variant, so both
variants are kept under separate names and each engine uses the one it had.

| Variant | Where it differs | Used by |
|---|---|---|
| `erfc` / `erfc_c1_truncated` | Cody's middle-regime `C[1]` correctly rounded, or truncated to 17 digits (one ulp below) | equity options / FX options |
| `norm_pdf` / `norm_pdf_div` | multiply by `1/sqrt(2 pi)`, or divide by `sqrt(2 pi)` | most callers / equity options pricing |
| `Xoshiro256PlusPlus` / `Xoshiro256StarStar` | output scrambler | equity options and both VaR engines / FX options |
| uniform maps | `(m + 0.5) 2^-53` on (0, 1), or `m 2^-53` on [0, 1) | all / FX VaR `Rng::uniform` |

The same seed therefore gives a different stream in the FX options engine
than in the other three. That was already true and is unchanged.

## Not moved

- Each VaR engine's special functions (incomplete gamma and beta, Student
  t, the erfc used by `eq_var_engine::stats`). The two engines use
  different algorithms, so sharing them would change numbers.
- The PSD repair code (`psd.rs`), which is written against each engine's
  own `Matrix` type.
- The engine error enums. Their variant names differ, and downstream code
  (for example the Python bindings) may implement traits for each one
  separately.

## Build and test

From `rust/`:

```sh
cargo test --release --locked -p qp-core
cargo clippy --release --all-targets --locked -p qp-core -- -D warnings
```

The tests are the unit tests in each module (reference vectors for the
generators, erfc against mpmath values), the doctests, and the property
tests in `tests/properties.rs` (erfc reflection, agreement of the two
tables, odd `erf`, CDF symmetry and monotonicity, Acklam plus two Halley
steps inverting the CDF, RNG fill and stream layout, Cholesky
reconstruction, thread-count independence of `par`, Brent convergence).
