# Desk Guide

How a market-risk desk actually runs this engine: the daily batch, intraday
use, who consumes which number, controls, and how the C++ production path
stays reconciled with the Python research stack.

## 1. Where this engine sits

```
research (Python eq_var)  ──  model choices, calibration studies, stress
        │  golden vectors (committed, 1e-9)
        ▼
production (C++ eqvar)    ──  overnight batch VaR/ES, intraday incremental VaR,
                              backtest statistics feeding the limit system
```

The Python stack owns *model development*; this engine owns the *numbers of
record*. Any methodology change lands in Python first, regenerates the golden
vectors, and only then is ported here — the cross-language test failing is
the intended tripwire for an unported change.

## 2. Overnight batch and SLAs

Per book, per day: EWMA covariance from the 250-day panel → historical /
BRW / FHS VaR on the book's P&L history → parametric normal + t VaR →
100k-path MC VaR + ES with the **date-derived seed recorded in the run log**
→ backtest update (yesterday's VaR vs today's clean P&L).

Measured cost on a 250×100 panel (single thread, -O2, g++ 13.3):

| stage | best ms |
|---|---|
| covariance (sample + EWMA) | 4.9 |
| historical family + empirical ES | < 0.03 |
| parametric family + closed-form ES | < 0.05 |
| MC 100k paths, normal | 1178 |
| MC 100k paths, Student-t(6) | 1197 |
| **whole batch, one book** | **~1.2 s** |

Practical SLA arithmetic: 500 books ≈ 10 minutes on one core, ~1 minute on
16 (books are embarrassingly parallel). The batch is never the critical path
of the overnight cycle; data readiness is. Budget the SLA around feed
arrival, not compute. If MC cost ever matters, cut paths and *report the
larger SE* rather than dropping the method — the SE is part of the output
contract.

## 3. Intraday incremental VaR

Parametric VaR on an updated exposure vector costs ~12 µs (Σ is fixed
intraday; only `wᵀΣw` is recomputed) — cheap enough to run **per trade** in
the limit check path, not on a snapshot timer. Convention: intraday numbers
are *indicative* (yesterday's Σ, today's w); the number of record is the
overnight batch. Historical VaR on a refreshed P&L vector is equally
sub-ms if the desk maps intraday exposure changes through the stored
scenario panel.

## 4. Who consumes what

- **Desk heads / traders**: 99 % 1d VaR (historical *and* FHS side by side —
  the spread between them is the regime signal), ES 97.5 %, top Euler
  contributors from the parametric decomposition.
- **Risk control**: VaR vs limits; the *method dispersion* (historical vs
  parametric-t vs MC) as a model-risk indicator — if the spread exceeds
  ~25 % investigate before the number moves the limit.
- **Regulatory / capital**: 10-day scaled VaR (sqrt-time, with its
  documented bias), Basel traffic-light zone and multiplier from the
  rolling 250-day exception count.
- **Model validation**: Kupiec / Christoffersen p-values quarterly; the
  cross-language reconciliation report (section 6).

## 5. Controls and regression gates (CI)

Gate every commit on, in order:

1. **Build gate**: `-Wall -Wextra -Werror` clean, C++20.
2. **Unit gate**: `ctest` 100 % (83 tests) — includes all analytic
   identities and edge cases.
3. **Golden gate**: `test_cross_language` — the engine's numbers vs the
   committed Python-generated constants at 1e-9. *Any* drift here is either
   an unported methodology change or a numerical regression; both block.
4. **Determinism gate**: `MonteCarlo.BitwiseSeedDeterminism` — bitwise
   equality on a fixed seed. Catches accidental introduction of
   implementation-defined RNG paths (e.g. someone "simplifying" to
   `std::normal_distribution`).
5. **Bench guard** (nightly, not per-commit): `bench_eqvar` timings vs a
   stored baseline; > 30 % regression on the MC stage or the sub-ms stages
   fails the nightly. The bench prints a checksum line — a changed checksum
   with unchanged code flags a toolchain/libm change worth knowing about.

Operational controls: MC seeds derived from the run date and logged, so any
day's numbers reproduce bitwise on demand (audit, P&L disputes); the
Cholesky `jitter_added` field is logged and alerted on if it exceeds 1e-6 of
mean variance (a persistently jittered Σ means degenerate factor data
upstream); exception days trigger same-day Kupiec/Christoffersen refresh,
not just quarterly.

## 6. Reconciliation vs the Python research stack

Quarterly (and after any release of either stack):

1. Re-run the Python golden generator on the current research code
   (provenance command in `tests/test_cross_language.cpp`).
2. Diff against the committed constants. Any change means research moved —
   trace it to a methodology ticket, port it, regenerate, recommit. The C++
   test suite enforces the rest.
3. For MC (not golden-testable across RNGs): run both stacks at 200k paths
   on the Case-B book and assert |VaR_py − VaR_cpp| < 3·(SE_py ⊕ SE_cpp).
4. File the reconciliation note with model governance: constants version,
   numpy/scipy versions, engine git hash.

This is deliberately the same discipline as any front-office/risk pricer
alignment: one reference, one production twin, committed vectors, a numeric
tolerance with a named owner.

## 7. Backtesting workflow (the monitoring loop)

Daily: append yesterday's exception indicator (`exceptions_from_pnl` on
clean P&L — no fees, no intraday flow). Rolling 250 days:

- **Traffic light**: green 0–4 → k = 3.0; yellow 5–9 → k up to 3.85 and a
  mandatory model-review note; red 10+ → k = 4.0, presumption of a flawed
  model, escalate to model validation with the Christoffersen decomposition
  attached.
- **Kupiec p < 0.05**: coverage is wrong — first suspects are stale Σ (λ too
  slow) or a book whose optionality broke the linear map.
- **Christoffersen independence p < 0.05 with Kupiec fine**: coverage right
  but exceptions cluster — the model lags regimes; move weight from plain
  historical to FHS (this exact signature is what FHS fixed in the Python
  project's 500-day study: plain HS p = 0.02, FHS p = 0.48).

## 8. Real-life scenarios

- **Vol spike (Mar-2020 style)**: plain historical VaR lags for weeks; FHS
  and BRW respond next day (unit-tested behaviours). Expect the
  historical-vs-FHS spread to blow out — that spread is the desk's early
  warning, publish it.
- **New illiquid name enters the book**: sparse history → its factor column
  may be near-constant → singular Σ → jitter path engages silently and
  correctly, but the *logged jitter* is the tell to map the name to a liquid
  proxy factor instead.
- **Perfectly hedged book** (w along a zero-variance direction): sigma ≈ 0,
  VaR ≈ 0 — correct for the linear model, dangerous if the hedge is only
  delta-neutral. Pair with the research stack's stress replays; a zero VaR
  with non-trivial gross exposure should page someone.
- **Quarter-end capital**: 10-day number is sqrt-time-scaled; attach the
  documented understatement caveat under vol clustering rather than
  silently multiplying by √10.
- **Auditor asks to reproduce 2026-03-17's VaR**: replay with that day's
  logged seed and panel snapshot → bitwise identical MC output. This is the
  concrete payoff of the deterministic-RNG design decision.
