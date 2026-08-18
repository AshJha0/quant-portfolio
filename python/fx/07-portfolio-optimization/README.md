# 07 — FX Portfolio Optimization & Currency Risk Allocation

Style-based currency allocation done properly: currency **total returns**
(spot + carry), the three classic FX styles (**CARRY / MOMENTUM / VALUE**) as
dollar-neutral long-short portfolios, shrinkage covariance, mean-variance and
risk-parity allocation, **CVaR-constrained sizing** (Rockafellar–Uryasev LP)
for carry's crash tail, **optimal currency hedging** for international
portfolios, and a walk-forward backtest with pip costs, carry accrual and
GBP-base reporting.

```
Return Estimation ──> Covariance Modeling ──> Mean-Variance ──> Efficient Frontier
 (spot + carry,        (sample / EWMA /        (closed forms      (target-return
  carry/mom/value       Ledoit-Wolf /           + SLSQP, sum-to-    grid, gross
  signal portfolios)    risk-on/off factor)     zero & gross caps)  budget)
        │                                                              │
        ▼                                                              ▼
 Risk Parity (ERC) ──> CVaR sizing (RU-LP) ──> Hedging (h* closed form) ──> Backtest
  (across styles &      (skew-aware carry       (safe-haven underhedge)     (walk-fwd,
   currencies, vol       cut, historical                                     pips, carry,
   targeting)            simulation)                                         GBP base)
```

## Why this is FX and not renamed equity

- Returns are **spot + rate-differential carry** with previous-close rate
  accrual (exact additive decomposition, unit-tested to 0 error).
- The styles are the FX classics: carry ranks on deposit-rate differentials,
  momentum is 12-1 on spot, value is deviation from a PPP anchor; all are
  **dollar-neutral** (weights sum to zero) with a gross-leverage budget.
- The covariance menu includes a **risk-on/off one-factor model** whose
  estimated loadings recover the classic signs (AUD/NZD/EM +, JPY/CHF −).
- Carry's **negative skew** is structural (crash mechanism tied to the rate
  differential), and the CVaR machinery exists precisely to price it.
- Currency **hedging** is a first-class problem: closed-form optimal hedge
  ratios show full hedging is not variance-optimal when safe havens hedge
  your equity book for free.
- Reporting supports a **base-currency choice** (GBP for a London desk),
  with the exact log-return invariance of dollar-neutral books tested.

## Quickstart

```bash
cd python/fx/07-portfolio-optimization
pip install -e .[dev]     # or just have numpy/scipy/pandas on the path
pytest -q                 # 139 tests, offline, ~12 s
python examples/run_pipeline.py   # full pipeline, ~26 s
```

## Headline numbers (seed 123, 3 024 business days, synthetic G10+EM panel)

Style sleeves (dollar-neutral, gross 2.0, one-day implementation lag):

| Style     | Ann. ret | Vol    | Sharpe | Skew   | MDD    | CVaR95/day |
|-----------|---------:|-------:|-------:|-------:|-------:|-----------:|
| CARRY     |  6.46%   | 13.28% |  0.49  | −1.73  | 18.20% | 1.90%      |
| MOMENTUM  |  4.18%   |  8.47% |  0.49  | −0.14  | 13.90% | 1.07%      |
| VALUE     |  6.56%   |  9.88% |  0.66  | −1.34  | 19.66% | 1.38%      |

- **Skew-aware carry sizing**: a 0.50%/day CVaR95 budget cuts the carry book
  30% below naive 5%-vol-target sizing (0.26x vs 0.37x). The multi-style
  RU-LP cuts tail CVaR 81% (2.11% → 0.40%/day) for an 8.0%/yr expected-return
  give-up against the unconstrained mean-chaser.
- **Optimal hedging** (intl equity portfolio): h\* = {EUR 1.15, JPY −1.40,
  GBP 1.56, AUD 3.71, CHF −0.42}; optimal hedge cuts variance 4.9% vs 1.2%
  for the naive full hedge — safe havens are underhedged (even bought), the
  classic result.
- **Walk-forward race** (est. 504d, monthly, 5 bps, carry accrued):
  EW SR 0.87 | MVO SR 1.07 | ERC SR 0.91 | CVaR-constrained SR 0.89 —
  MVO's win rests on unusually estimable synthetic means; see
  docs/VALIDATION.md before believing it out of sample.
- **GBP-base reporting**: dollar-neutral sleeves are base-invariant to
  1.4e-17 (exact for log returns); the net-long book's Sharpe moves
  0.87 → 0.74 switching USD → GBP — net books are not base-invariant.

## Layout

```
src/fx_port/            returns_est, covariance, mvo, risk_parity,
                        hedging, cvar_opt, backtest, metrics, data/
tests/                  139 offline seeded tests (identities to 1e-12)
examples/run_pipeline.py
docs/                   METHODOLOGY.md, VALIDATION.md, DESK_GUIDE.md
```

See `docs/METHODOLOGY.md` for model choices and the assumptions register
(including why carry returns exist at all — the forward premium puzzle — and
why its vol understates its risk), `docs/VALIDATION.md` for every identity
and failure mode, `docs/DESK_GUIDE.md` for how an FX overlay desk would run
this.
