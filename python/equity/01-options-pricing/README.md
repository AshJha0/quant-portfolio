# Equity Options Pricing & Greeks Engine (`eq_options`)

Flagship-quality reference implementation of the vanilla equity-options
stack: analytic Black-Scholes-Merton, Cox-Ross-Rubinstein binomial
(European + American), Black-76 on forwards/futures, exact-scheme Monte
Carlo with variance reduction and MC Greeks, a full analytic Greek set
(incl. vanna/volga), robust implied vol, a discrete delta-hedging
simulator, and a model-comparison harness — all cross-validated against
each other and against a committed set of golden vectors for the future
C++/Rust engines.

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
pip install -e .                     # numpy, scipy, pandas only
python -m pytest tests -q            # 288 tests, offline, ~5 s
python examples/run_pipeline.py      # full report, ~1.5 s
```

```python
from eq_options import bs_price, bs_greeks, crr_price, implied_vol, mc_price

bs_price(100, 100, 1.0, 0.05, 0.20, q=0.01, option_type="call")  # 9.826298
crr_price(100, 100, 1.0, 0.05, 0.20, 0.01, "put", "american", 2000)  # 6.366622
implied_vol(9.826298, 100, 100, 1.0, 0.05, 0.01, "call")         # 0.20000000
mc_price(100, 100, 1.0, 0.05, 0.20, 0.01, "call", n_paths=200_000, seed=42)
# MCResult(value=9.8411, std_error=0.0178, ci_low=9.8063, ci_high=9.8760, ...)
```

## Result highlights (reference contract S=100, K=100, T=1y, r=5%, q=1%, σ=20%)

**Cross-model agreement** (European call, BS = 9.826298):

| model | price | abs diff vs BS | runtime |
|---|---:|---:|---:|
| Black-Scholes | 9.826298 | — | ~0.3 ms* |
| CRR tree, 1000 steps | 9.824328 | 1.97e-3 | ~4 ms |
| Black-76 on `F=S·e^{(r−q)T}` | 9.826298 | <1e-10 | ~0.3 ms |
| MC, 200k paths (anti+CV) | 9.841109 | 1.5e-2 (0.8 SE) | ~21 ms |

*first-call overhead; steady-state closed-form evaluation is microseconds.

**Convergence laws, measured:** CRR `error × n ≈ 1.97` constant (O(1/n));
MC `SE × √n ≈ 7.9` constant (O(n^{-1/2})). Full tables in
[docs/VALIDATION.md](docs/VALIDATION.md).

**Greeks** (analytic, call): delta 0.6118, gamma 0.0189, vega 37.76,
theta −5.73/yr, rho 51.35, vanna −0.189, volga 5.66 — all matching
central finite differences to <1e-4 relative.

**American exercise:** 1y ATM put early-exercise premium 0.4234 (2000
steps); American call with q=0 equals European to 1e-10 (Merton).

**Implied vol:** round-trips σ→price→σ to 1e-8 across moneyness 0.5–2.0
and expiries 1w–5y; worst error on a 22-quote skewed synthetic chain:
3.1e-13; refuses sub-intrinsic prices with `ValueError`.

**Delta hedging** (short 3M ATM call): P&L std 1.61 → 0.22 as rebalances
go 4 → 256 (`std × √N ≈ 3.4` constant); mean P&L +0.001 ± 0.004 at true
vol; selling 25-vol against 15-vol realized earns +2.00 vs the
gamma-weighted vol-spread theory value +1.99; 5 bp transaction costs at
N=128 drag the mean to −0.234.

## Layout

```
src/eq_options/          black_scholes, binomial, black76, monte_carlo,
                         greeks, hedging, comparison, data/{synthetic,live}
tests/                   288 offline seeded pytest tests
tests/golden/            golden_vectors.json (32 cases, 1e-10) + generator
examples/run_pipeline.py end-to-end report reproducing every number above
docs/                    METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```

## Documentation contract

- **Why these models, vs alternatives** — [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §1
- **Assumptions register (A1–A8, each with "what breaks")** — [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §2
- **Validation evidence** (benchmarks, convergence, cross-model) — [docs/VALIDATION.md](docs/VALIDATION.md) §1–5
- **Failure modes** (smile contradiction, discrete-hedging error, dividend
  modeling error, CRR limitations, numerical limits) — [docs/VALIDATION.md](docs/VALIDATION.md) §6
- **Desk usage** (quotes, risk pipeline, EOD reports, limits, governance) — [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md) §1–2
- **Real-life scenarios & edge cases** (earnings jump, dividend
  announcement, expiry pinning, 2008/2020 vol spikes; every edge case
  unit-tested) — [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md) §3
