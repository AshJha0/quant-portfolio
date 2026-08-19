# Black-Scholes Replication (`eq_bs_replication`)

A from-scratch, **zero-scipy** Black-Scholes-Merton replication: closed-
form call/put pricing and the five analytic Greeks built on nothing but
`math.erf`, an independent Monte Carlo pricer under the same
risk-neutral dynamics, a Newton-Raphson (with bisection fallback)
implied-vol solver, and a controlled experiment that makes the
volatility smile *emerge* from fat-tailed pricing read back through the
model — all cross-validated against theory rather than against a
reference library.

This is deliberately a small, pedagogical, single-asset equity project,
not a pricing library — see `docs/DESK_GUIDE.md` for what it is
actually useful for on a desk (model-validation reference, new-quant
teaching artifact), and `python/equity/01-options-pricing` for the
flagship, vectorised, multi-model pricing engine (with C++/Rust
performance twins) this project's discipline is a warm-up for.

```
closed-form pricing (math.erf only)   Monte Carlo (same risk-neutral GBM)
        │  put-call parity                    │  antithetic variates
        │  analytic Greeks                    │  standard error
        ▼                                      ▼
        └──────────► agree to O(1/√n) ◄────────┘
                          │
                          ▼
         implied-vol round trip (Newton + bisection)
                          │
                          ▼
   model-breakdown demo: fat-tailed MC price → BS implied vol → SMILE
```

**Conventions:** `r` continuously compounded, annualised; `T` in years
(ACT/365F); `sigma` annualised vol of log-returns; no dividends
(q=0 — see `docs/METHODOLOGY.md` §1.3 for the one-line extension).
Every stochastic routine (`mc_call_price`) takes an explicit seed.
Negative rates are supported; `T<=0` and `sigma<=0` raise `ValueError`
with an informative message rather than silently returning NaN.

## Quickstart

```bash
cd python/foundations/03-black-scholes-replication
pip install -e ".[dev,plots]"        # numpy (core), matplotlib (plots), pytest (dev)
pytest -q                            # 44 tests, offline, seeded, <1s
python examples/run_pipeline.py      # report + figures under output/
```

```python
from eq_bs_replication import call_price, put_price, call_greeks, put_greeks, implied_volatility, mc_call_price

call_price(100, 105, 0.03, 0.25, 0.75)          # 7.467460
put_greeks(100, 105, 0.03, 0.25, 0.75).delta    # -0.5053 (via put-call parity)
implied_volatility(7.467460, 100, 105, 0.03, 0.75)  # 0.25000000
mc_call_price(100, 105, 0.03, 0.25, 0.75, n_paths=1_000_000, seed=7)  # (7.4517, 0.0135)
```

## Results summary (reference contract `S=100, K=105, r=3%, sigma=25%, T=0.75y`)

**Identities hold exactly:** put-call parity error **0.00e+00** on the
reference contract; < 1e-9 across a 270-point grid including zero and
negative rates.

**Two independent implementations agree, at the theoretical rate:**
closed-form call price **7.467460**; Monte Carlo converges to it with
`SE·√n ≈ 13.2` constant across 1k–1M paths (the **O(n^{-1/2})** law,
measured, not assumed) — full table in `docs/VALIDATION.md` §2.

**Greeks match finite differences:** delta, vega, gamma all agree with
central finite differences to ≤1e-6 (relative); put Greeks are derived
via put-call-parity relations from call Greeks and independently
cross-checked against those same identities to <1e-12 —
`docs/VALIDATION.md` §3.

**Implied vol round-trips cleanly** across sigma 8%–120% (< 1e-11
error) — except in one documented, expected corner (deep ITM + low vol,
where vega is tiny and the round trip degrades to ~1e-2), which is
tested explicitly rather than hidden — `docs/VALIDATION.md` §4.

**The model breaks on purpose, and the break is measured:** pricing a
strike ladder by Monte Carlo under a fat-tailed (Student-t, df=4)
return distribution and reading the prices back through Black-Scholes
produces a **5.2-vol-point smile** (27.6% at K=70, 22.9% at K=105, 28.6%
at K=140) instead of the flat line constant-vol GBM predicts — the same
qualitative shape real option markets have shown persistently since
1987. Full experiment and numbers in `docs/VALIDATION.md` §5; figure in
`output/figures/black_scholes_overview.png`.

## Layout

```
src/eq_bs_replication/   black_scholes.py (math.erf-only closed form + Greeks
                         + implied vol), monte_carlo.py (independent MC cross-check)
tests/                   44 offline, seeded pytest tests across 5 files
examples/run_pipeline.py end-to-end report + figures reproducing every number above
docs/                    METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```

## Documentation contract

- **Why Black-Scholes, vs alternatives (CRR tree, local vol); why
  Monte Carlo cross-check, not closed-form alone; why `math.erf` and
  not scipy** — [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §1
- **Assumptions register (A1–A4, each with "what breaks")** — [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §2
- **Validation evidence** (identities, MC convergence, Greeks vs FD,
  implied-vol round trips, the smile experiment) — [docs/VALIDATION.md](docs/VALIDATION.md) §1–5
- **Failure modes** (vol smile — demonstrated; jumps, discrete-hedging
  frictions, early exercise — precisely specified, pointing to
  `python/equity/01-options-pricing` where each is built) — [docs/VALIDATION.md](docs/VALIDATION.md) §6
- **Numerical limits** (T→0, sigma→0, deep ITM/OTM, negative rates,
  invalid inputs — every one unit-tested) — [docs/VALIDATION.md](docs/VALIDATION.md) §7
- **Desk usage**: not a pricing library — a model-validation reference
  and teaching artifact, with a concrete new-pricer-onboarding scenario
  — [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md)
