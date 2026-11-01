# Desk Guide, eq-options-engine (Rust)

How a real equity/index options desk would deploy this engine, who
consumes its numbers, and how the three-language setup (Python reference,
C++ incumbent, Rust engine) is governed as one model.

## 1. Where Rust fits on a desk

A typical derivatives technology estate has three tiers, and this
portfolio mirrors them:

| Tier | Stack | This portfolio |
|---|---|---|
| Research / model development | Python + numpy/scipy | `eq_options` (the reference implementation) |
| Incumbent pricing/risk libraries | C++ (20+ years of it) | `cpp/equity-options-engine` |
| **New services being built today** | **Increasingly Rust** | **this crate** |

Rust's beachheads on a desk are the places where C++'s failure modes
hurt most and a GC'd language is disqualified by latency:

- **New risk services.** An intraday reval service, 500k positions x
  full Greek set on every vol-surface tick, is throughput-bound and
  parallel. At ~13M Greek evals/sec/core, a full-book Greek sweep is
  sub-second on a modest box; Rust's compile-time race checking means the
  obvious parallelisation (shard positions across threads, share the
  read-only surface) is safe by construction, not by code review. No GC
  removes one source of latency spikes; no latency distribution has been
  measured for this crate.
- **Safety-critical pipelines.** Anything wired into pre-trade checks,
  margin, or automated hedging, where a memory-corruption bug that
  *silently perturbs a number* is worse than a crash. The borrow checker
  removes that class; `Result`-typed errors mean a bad input is handled
  at every call site or the code does not compile (`#[must_use]` on
  `Result`).
- **FFI cores.** Compiled once, this crate can sit behind a C ABI and be
  called from the Python research stack (PyO3/ctypes) or the C++ estate
 , one audited numeric core, three frontends. No unwinding across the
  boundary because pricing paths never panic.
- **Determinism-sensitive batch reval.** Same seed => identical MC on
  any host. That makes overnight P&L runs reproducible across a
  heterogeneous grid and makes "rerun yesterday's number" a meaningful
  operation, an audit and model-control property, not a nicety.

Where Rust is *not* the tool: exploratory calibration and desk analytics
stay in Python (iteration speed), and nobody rewrites a working
million-line C++ library wholesale, Rust arrives service by service.

## 2. Daily workflow

1. **Open**: risk service loads positions and the marked surface; full
   book of analytic prices + Greeks (delta/gamma/vega/theta/rho, vanna/
   volga for the vol book) revalued continuously intraday.
2. **Quoting off the forward**: index and futures options priced via
   `black76` off the observable forward, so financing/dividend marks are
   absorbed into `F`.
3. **American names**: single stocks with early-exercise risk priced on
   the CRR tree (`n=500-1000`); `early_exercise_premium` (same-tree
   differencing) feeds the borrow/dividend risk view.
4. **Marks in, vols out**: `implied_vol` inverts listed quotes to a vol
   grid; `ArbitrageBound` rejections are surfaced to the data team as
   crossed/stale quotes rather than silently NaN-ing a surface, the
   deep-ITM identifiability limit (VALIDATION.md section 4.1) is why the
   surface is built from OTM wings.
5. **Exotics / structures**: `mc_price` with explicit seeds; the seed is
   logged with the trade so any number can be reproduced on the same
   target and toolchain in a dispute or a model-control review.
6. **Close**: EOD reval + Greek snapshot to P&L attribution
   (theta-carry vs delta/vega explain); results are reproducible from
   the archived inputs + seeds.

Consumers: traders (risk ladders), risk control (limits vs gamma/vega),
product control (P&L explain), model validation (this document plus
VALIDATION.md), and downstream margin systems.

## 3. Golden regression gating across all three stacks

The control that makes a three-language estate governable is the shared
golden-vector contract:

- The **golden vectors come from mpmath** (`generate_golden.py` in the
  Python project), not from any engine: 456 records, each output with a
  stated tolerance.
- The **C++ and Rust engines regenerate their committed copies**
  (`tools/gen_golden_header.py`, `tools/gen_golden_rs.py`, both with
  `--check`) and every record must lie within tolerance. The worst
  deviation is 0.18 of tolerance in all three engines.
- A **differential test** sends the same seeded random inputs through the
  Python, Rust and C++ engines (JSON-lines shims) and compares them.
- **CI runs every suite** (`pytest`, `ctest`, `cargo test` with
  `RUSTFLAGS="-D warnings"`).

This is a small version of multi-implementation model governance: an
independent oracle, machine-checked agreement, and a release gate.

## 4. Controls and limits summary

- Input domain enforced at the API (`PricingError::InvalidInput`); no
  NaN propagation into risk.
- Tree refuses pseudo-probabilities outside (0,1) instead of pricing.
- MC results carry their own error bars (`std_error`, 95% CI) so
  downstream consumers can gate on statistical quality, and every run is
  seed-logged and reproducible on the same target and toolchain.
- Model scope: European vanillas (BSM/B76/MC) and American vanillas
  (CRR) under GBM with continuous dividends, anything outside
  (discrete dividends, smile-dependent exotics, American MC) must go to
  a different model, per the assumptions register in METHODOLOGY.md.
