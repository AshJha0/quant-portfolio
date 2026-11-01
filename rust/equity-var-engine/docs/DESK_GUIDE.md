# Desk Guide

How a market-risk desk would run this engine: the daily batch, intraday use,
who reads which number, controls, and reconciliation with the Python library
and the C++ engine.

## 1. Where this engine sits

```
research (Python eq_var): model choices, full revaluation, stress
      |  golden vectors (JSON, conv_* keys shared with FX)
      +---------------------------+
      v                           v
C++ eqvar (linear book)     Rust eq_var_engine (linear book)
```

The Python library owns model development and option revaluation. The C++
and Rust engines are linear-book ports written separately against the same
goldens. A methodology change lands in Python first, regenerates the JSON,
and is then ported; a failing golden test is the tripwire for an unported
change.

## 2. Overnight batch

Per book and day: EWMA covariance from the 250-day panel, historical / BRW /
per-factor FHS VaR, parametric normal and t VaR, one 100k-path MC run that
gives VaR, ES and the bootstrap SE (seed derived from the run date and
logged), then the backtest update.

Measured on a 250 x 100 panel, single thread, release + LTO, 2026-11-01
(`src/bin/bench.rs`):

| stage | best ms |
|---|---|
| covariance (sample + EWMA) | 2.0 |
| historical family, FHS per factor, empirical ES | 0.15 |
| parametric family, closed-form ES | 0.03 |
| MC 100k paths normal: VaR + ES + bootstrap SE (200) | 657 |
| MC 100k paths Student-t(6): same | 1744 |

Books are independent, so the batch parallelises across cores. If MC cost
matters, cut paths and report the larger SE; the SE is part of the output.

## 3. Intraday incremental VaR

Parametric VaR on an updated exposure vector recomputes only `w' Sigma w`,
in microseconds, so it can run per trade in the limit check. Intraday numbers
are indicative (yesterday's Sigma, today's w); the number of record is the
overnight batch.

## 4. Who reads what

- **Desk**: 99 % 1-day VaR from historical and FHS side by side (their
  spread is a regime signal), ES 97.5 %.
- **Risk control**: VaR against limits; the spread between historical,
  parametric-t and MC as a model-risk indicator.
- **Capital**: 10-day sqrt-time VaR with its known bias; Basel zone and
  multiplier from the rolling exception count
  (`basel_traffic_light(&BaselInput { n_obs, n_exceptions, alpha })`), which
  now follows the window length.
- **Model validation**: Kupiec, Christoffersen and Acerbi-Szekely Z2.

## 5. Gates for every commit

1. `cargo clippy --all-targets -- -D warnings` and `#![deny(missing_docs)]`.
2. `cargo test --release`: 121 integration tests and 14 doctests, including
   the fuzz walk and its registry of public functions.
3. Goldens: `tools/gen_golden_rs.py --check`, `test_conv_golden.rs`,
   `test_cross_language.rs`. Any drift is an unported methodology change or
   a regression; both block.
4. Determinism: `test_monte_carlo::bitwise_seed_determinism` (same seed,
   same build, exact equality).
5. Nightly bench against a stored baseline.

Operational controls: log the MC seed per run; log `CholeskyFactor::jitter`
whenever it is non-zero (persistent jitter means degenerate factor data
upstream); never pass `PsdRepair::Clip` or `Higham` in the batch without a
recorded reason, since a repaired matrix is a different covariance.

## 6. Reconciliation

After any release of any stack:

1. Regenerate the Python JSON and run `tools/gen_golden_rs.py --check` here
   and the header check in the C++ engine.
2. MC is not golden-testable across RNG streams: run the engines at 200k
   paths on the same book and require pairwise agreement within 3 combined
   bootstrap SEs.
3. Record the JSON `versions` block (Python, NumPy, SciPy), the Rust
   toolchain and both git hashes.

## 7. Backtesting loop

Daily: append yesterday's exception (`exceptions_from_pnl` on clean P&L).
Rolling window:

- **Traffic light**: green `F < 0.95`, yellow `F < 0.9999` (plus factor from
  the equivalent 250-day count), red otherwise. On 250 days at 1 % this is
  the familiar 0-4 / 5-9 / 10+.
- **Kupiec p < 0.05**: coverage is wrong. First suspects: stale Sigma, or
  optionality that the linear map misses.
- **Christoffersen p < 0.05 with Kupiec fine**: exceptions cluster; the
  model lags regimes. Move weight to FHS.
- **Z2 clearly negative**: losses beyond VaR exceed the forecast ES.

## 8. Scenarios

- **Vol spike**: plain HS lags for weeks; FHS and BRW respond the next day.
  Publish the HS-FHS spread.
- **Illiquid name with a near-constant history**: the covariance becomes
  singular and `cholesky_repair` reports a jitter. Map the name to a liquid
  proxy factor.
- **Delta-hedged book**: sigma near zero is right for the linear model and
  wrong if the hedge is only delta-neutral. Use the Python stress replays.
- **Audit replay**: same seed, same panel snapshot and same binary give the
  same MC output. A compiler or `rng.rs` change is a stream change, so pin
  the binary, not only the seed.
