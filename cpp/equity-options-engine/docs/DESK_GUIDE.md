# Desk Guide — where a C++ pricing engine sits in the real workflow

The Python reference's
[DESK_GUIDE.md](../../../python/equity/01-options-pricing/docs/DESK_GUIDE.md)
covers the *financial* workflow: who consumes prices and Greeks, hedging
cadence, P&L attribution, limits and scenarios. This guide covers the
*systems* question: which jobs on a desk run through a compiled engine like
this one, and how such an engine is governed.

## 1. The three production seats for a C++ engine

### Intraday risk service
The desk's risk view ("what is my delta/gamma/vega *right now*") is a
long-running service that re-prices the whole position set on every
meaningful underlier/vol tick. Order of magnitude: 100k–1M option positions
x full Greek set, refreshed every few seconds. At ~12M Greek-set
evaluations/sec/core (benchmarked), a full book sweep is tens of
milliseconds on a handful of cores — fast enough that traders see risk
move *with* the market, not 30 seconds behind it. The Python package in the
same role would be the bottleneck; instead it is the *reference* the
service is validated against.

### Quoting / auto-hedging engines
Market-making systems price two-sided quotes off a marked vol surface and
re-hedge on fills. The per-event budget is microseconds, allocation-free,
GC-free, GIL-free. The kernels here fit that shape: a BSM price is ~62 ns,
an implied vol is a few Newton iterations of the same kernel, and nothing
on the hot path allocates. (A real quoting stack adds surface
interpolation and position caching on top; the pricing kernel is the part
that must never be the slow link.)

### Batch reval and scenario grids
End-of-day official P&L, VaR/stress grids (hundreds of scenarios x full
book), and margin replication are throughput problems: billions of
pricings on a schedule. Compiled kernels parallelise trivially across
scenarios (the MC engine's deterministic seed-partitioned threading is the
same pattern), and — critically for sign-off — produce the *same numbers*
as the research library, because both are pinned to the golden vectors.

## 2. Split of responsibilities: Python researches, C++ serves

- Model changes, calibration experiments, new payoffs: prototyped and
  validated in the Python package (fast iteration, rich ecosystem).
- Once accepted, the change is ported here, and the *contract* between the
  two is renewed mechanically: the Python project regenerates
  `golden_vectors.json`, `tools/gen_golden_header.py` regenerates the C++
  header, and both suites must go green with identical numbers to 1e-9
  (measured ~5e-15).

This is the standard two-implementation control quants are asked about in
model governance: an independent implementation agreeing to near machine
precision is strong evidence against implementation error (it deliberately
cannot detect *model* error — both implementations share assumptions
A1–A8; that risk is handled by the model-validation material in the Python
docs).

## 3. Golden-vector regression as the release gate

Every release of this engine (compiler upgrade, flag change, refactor,
new feature) must pass, in order:

1. **Build gate:** `-Wall -Wextra -Werror` — zero warnings.
2. **Cross-language gate:** all 32 golden cases, price + 5 Greeks, to 1e-9.
   Catches: maths edits, accidental convention drift (per-day vs per-year
   theta is the classic), CDF/precision regressions, aggressive-optimisation
   breakage (e.g. someone enabling `-ffast-math`).
3. **Identity gate:** parity, Black-76==BSM, analytic-vs-FD Greeks —
   catches errors that golden vectors alone might miss (a bug symmetric in
   call/put, say).
4. **Statistical gate:** MC within 3 SE, variance reduction effective,
   bit-reproducibility given (seed, threads) — the reproducibility check is
   what makes "rerun yesterday's batch" a meaningful debugging tool and
   satisfies auditors that official numbers can be regenerated exactly.
5. **Convergence gate:** tree within 2e-3 of BSM at n=2000, error decaying.

A red gate blocks the release; there is no "small numeric diff, ship it
anyway" path, because a 1e-6 price drift across a million-position book is
a real P&L restatement.

## 4. Operational notes and controls

- **Determinism is a feature, not a nicety.** Official reval uses pinned
  seeds recorded with the run; any number in a report can be reproduced
  bit-exactly. The in-house inverse-CDF sampler exists precisely so this
  holds across compiler/stdlib upgrades.
- **Input hygiene at the boundary.** The engine throws on negative or
  non-finite (NaN/Inf) inputs — including non-finite rates — and on
  sub-intrinsic implied-vol requests instead of returning
  plausible garbage — upstream data problems (stale quotes, crossed
  markets, bad dividend feeds) surface as exceptions in the service log,
  not as silent risk misstatements.
- **Edge-case policy is part of the interface.** `T=0` intrinsic and
  `sigma=0` discounted-forward-intrinsic limits are documented and tested;
  expiry-day and pinned-strike behaviour therefore matches the reference
  library exactly — no reconciliation break at 3:59pm on expiry Friday.
- **Known limits.** Flat vol, continuous dividends, European MC/BSM:
  the same A1–A8 assumptions as the reference. The engine prices *a* model
  fast; choosing the right vol/dividend inputs per the desk's marking
  process is upstream of it (see the Python DESK_GUIDE for that workflow).
