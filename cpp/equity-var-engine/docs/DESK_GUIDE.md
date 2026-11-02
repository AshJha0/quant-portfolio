# Desk Guide

How a market-risk team would run this engine: the daily batch, intraday
use, who reads which number, the gates, and reconciliation with the Python
engine. The engine revalues linear positions as a dot product and reprices
European option legs in full in each scenario, with implied vol held at
today's level.

## 1. Where this engine sits

```
Python eq_var   : model development, full revaluation, stress, studies
      |  golden vectors (committed JSON -> generated C++ header)
      v
C++ eqvar       : port for batch and intraday VaR / ES / backtests (linear + options)
```

A methodology change lands in Python first, regenerates the golden vectors,
and is then ported here. A failing cross-language test means a change is not
yet ported.

## 2. Overnight batch

Per book per day: EWMA covariance from the 250-day panel, historical, BRW
and per-factor FHS VaR, parametric normal and t VaR, a 100k-path MC VaR and
ES with a logged seed, and the backtest update (yesterday's VaR vs today's
clean P&L).

Per-call latency for every kernel (p50 to max, GCC and Clang) is in the
README Latency table, produced by `./build/bench_eqvar`.

## 3. Intraday incremental VaR

Parametric VaR on an updated exposure vector costs about 12 microseconds
(Sigma fixed intraday, only w' Sigma w recomputed). Intraday numbers are
indicative (yesterday's Sigma, today's w). The number of record is the
overnight batch.

### Hot-path calls

For per-trade recomputation use `eqvar/kernels.hpp`:

- size the buffers once at start-up: a `std::vector<double>` as long as
  the largest P&L series for `try_historical_var` / `try_expected_shortfall`,
  `L` and `scratch` of n x n for `try_cholesky_lower`, and a
  `McVarWorkspace` reserved for the factor count and scenario count;
- factor the covariance once per covariance update with
  `try_cholesky_lower` (or `cholesky(cov, Repair)` when it may need repair)
  and pass the factor to `try_monte_carlo_var_factored`;
- check each `Result`; `message(err)` gives the same text the throwing API
  would have thrown, and is the only call that allocates.

A kernel that receives a buffer that is too small returns
`Errc::WorkspaceTooSmall` instead of growing it.

## 4. Who reads what

- **Desk heads and traders**: 99 % 1-day VaR from historical and FHS side by
  side (the spread is a regime signal), ES 97.5 %, Euler contributions from
  the parametric decomposition.
- **Risk control**: VaR against limits; the dispersion across methods as a
  model-risk indicator.
- **Capital**: 10-day sqrt-time VaR with its known bias; Basel zone and
  multiplier from the rolling exception count.
- **Model validation**: Kupiec, Christoffersen and Acerbi-Szekely Z2; the
  reconciliation report (section 6).

## 5. Gates (CI)

1. **Build**: GCC and Clang with `-DEQVAR_WERROR=ON`; one ASan/UBSan build.
2. **Unit**: `ctest`, 123 tests, including the fuzz walk of every public
   function.
3. **Golden**: `test_cross_language` against the Python-generated header;
   `tools/gen_golden_header.py --check` must pass.
4. **Repeatability**: `MonteCarlo.BitwiseSeedDeterminism` checks exact
   equality for a fixed seed on the build platform. A switch to
   `std::normal_distribution` would break it. Results on another libm may
   differ in the last bits.
5. **Bench** (nightly): `bench_eqvar --out` latency percentiles per kernel
   against a stored baseline, with compiler, flags and CPU in each record.

Operational controls: log the MC seed per run; log and alert on
`CholeskyResult::jitter_added > 0` and on `repaired`; refresh Kupiec and
Christoffersen on exception days.

## 6. Reconciliation with the Python engine

After a release of either engine:

1. Re-run `tests/golden/generate_golden.py` in the Python project.
2. Run `tools/gen_golden_header.py --check`. A difference means the Python
   engine changed: trace it, port it, regenerate, commit both.
3. MC is not golden-testable across RNGs. Compare both engines at 200k
   paths on the case-B book within 3 combined SE.
4. Record the JSON `versions` block (Python, numpy, scipy) and the git hash.

## 7. Backtesting loop

Daily: append yesterday's exception (`exceptions_from_pnl` on clean P&L).
Rolling window:

- **Traffic light**: `basel_traffic_light({.n_obs, .n_exceptions, .alpha})`
  works on any window. At 250 days and 1 %: green 0-4 (k = 3.0), yellow 5-9
  (k up to 3.85, review note), red 10+ (k = 4.0, escalate).
- **Kupiec p < 0.05**: the exception rate is wrong. Check for a stale
  Sigma or a book whose optionality breaks the linear map.
- **Christoffersen p < 0.05 with Kupiec fine**: exceptions cluster; the
  model lags regimes. Compare with FHS.
- **Acerbi-Szekely Z2 < 0**: ES is understated on exception days.

## 8. Scenarios

- **Volatility spike**: plain historical VaR lags for weeks; FHS and BRW
  respond the next day. Publish the spread.
- **New illiquid name**: a near-constant factor column makes Sigma singular.
  The jitter engages and is logged. Map the name to a liquid proxy.
- **Indefinite Sigma from bad data**: the Cholesky throws and names the most
  negative eigenvalue. Fix the data, or request `Repair::Higham` and log it.
- **Delta-hedged option book**: sigma near 0 with large gross exposure. The
  linear model is blind to gamma. Use the Python engine.
- **Audit replay**: rerun with the logged seed and panel snapshot on the
  same platform for the same MC output.
