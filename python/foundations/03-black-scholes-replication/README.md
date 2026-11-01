# Black-Scholes Replication (`eq_bs_replication`)

A from-scratch, zero-scipy Black-Scholes-Merton replication. It has
closed-form call and put pricing and the five analytic Greeks built on
`math.erfc`, an independent Monte Carlo pricer under the same
risk-neutral dynamics, an implied-vol solver for calls and puts, and a
controlled experiment in which a volatility smile appears when
fat-tailed prices are read back through the model. Everything is
checked against theory, not against a reference library.

This is a small teaching project for a single equity asset, not a
pricing library. `docs/DESK_GUIDE.md` describes its use as a
model-validation reference. The vectorised, multi-model pricing engine
with C++ and Rust twins is `python/equity/01-options-pricing`.

```
closed-form pricing (math.erfc only)  Monte Carlo (same risk-neutral GBM)
        │  put-call parity                    │  antithetic variates
        │  analytic Greeks                    │  standard error
        ▼                                      ▼
        └──────────► agree to O(1/√n) ◄────────┘
                          │
                          ▼
     implied-vol round trip (OTM side, Newton + bisection)
                          │
                          ▼
   model-breakdown demo: fat-tailed MC price → BS implied vol → SMILE
```

**Conventions:** `r` continuously compounded and annualised; `T` in
years (ACT/365F); `sigma` annualised vol of log-returns; no dividends
(q=0, see `docs/METHODOLOGY.md` §1.3 for the one-line extension).
`mc_call_price` takes an explicit seed. Negative rates are supported.
`T<=0`, `sigma<=0` and non-finite inputs raise `ValueError` in both the
closed form and the Monte Carlo pricer, which share one input contract.

## Quickstart

```bash
cd python/foundations/03-black-scholes-replication
pip install -e ".[test]"             # numpy (core), pytest + hypothesis (test)
pytest -q                            # 304 tests, offline, seeded, ~3 s
python examples/run_pipeline.py      # report to stdout, writes nothing
python examples/run_pipeline.py --out /tmp/bs --plot   # report + figure; needs .[plots]
```

```python
from eq_bs_replication import call_price, put_price, call_greeks, put_greeks, implied_volatility, mc_call_price

call_price(100, 105, 0.03, 0.25, 0.75)          # 7.467460
put_greeks(100, 105, 0.03, 0.25, 0.75).delta    # -0.5053 (via put-call parity)
implied_volatility(7.467460, 100, 105, 0.03, 0.75)  # 0.25000000
implied_volatility(10.131340, 100, 105, 0.03, 0.75, option_type="put")  # 0.25000000
mc_call_price(100, 105, 0.03, 0.25, 0.75, n_paths=1_000_000, seed=7)  # (7.4517, 0.0112)
```

## Results summary

Reference contract `S=100, K=105, r=3%, sigma=25%, T=0.75y`. All
numbers come from `python examples/run_pipeline.py`.

| Check | Result |
|---|---|
| Put-call parity, reference contract | 0.00e+00 |
| Put-call parity, 270-point grid incl. zero and negative rates | < 1e-9 |
| Closed-form call | 7.467460 |
| Monte Carlo SE·√n, 1k to 1M paths | 10.4 to 11.2 (constant, O(n^-1/2)) |
| Delta / vega / gamma vs central differences | 9.3e-14 / 5.2e-08 / 2.1e-10 |
| Implied-vol round trip, sigma 8% to 120% | ≤ 4.4e-16 |
| Fat-tailed (Student-t df=4) smile, synthetic: K=70 / 105 / 140 | 27.6% / 22.9% / 28.6% |

The antithetic standard error is computed from pair averages. Treating
the mirrored draws as independent overstates it by about 33% on this
contract (`docs/VALIDATION.md` §2.1).

Implied vol is solved on the out-of-the-money side: an in-the-money
quote is converted by put-call parity first, so the solver sees only
time value. When the time value is below what double precision can
resolve, or the price sits on the upper bound, the solver warns and
returns `nan` (`docs/VALIDATION.md` §4.1).

The smile from synthetic fat-tailed prices is 5.2 vol points wide
(wing average minus ATM). Listed option markets have shown a smile or
skew since 1987.

## Layout

```
src/eq_bs_replication/   black_scholes.py (erfc-based closed form, Greeks,
                         implied vol), monte_carlo.py (independent MC cross-check)
tests/                   304 offline, seeded pytest tests across 9 files
examples/run_pipeline.py report reproducing every number above (--out, --plot)
docs/                    METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```

## Documentation

- Why Black-Scholes, against a CRR tree and local vol; why a Monte
  Carlo cross-check; why `math.erfc` and not scipy:
  [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §1
- Assumptions register (A1 to A4, each with what breaks):
  [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §2
- Validation evidence (identities, MC convergence, Greeks vs FD,
  implied-vol round trips, the smile experiment):
  [docs/VALIDATION.md](docs/VALIDATION.md) §1 to §5
- Failure modes (smile demonstrated; jumps, discrete hedging and early
  exercise specified): [docs/VALIDATION.md](docs/VALIDATION.md) §6
- Numerical limits, each unit-tested:
  [docs/VALIDATION.md](docs/VALIDATION.md) §7
- Desk usage as a validation reference: [docs/DESK_GUIDE.md](docs/DESK_GUIDE.md)
