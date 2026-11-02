# 07: FX Portfolio Optimization and Currency Risk Allocation

Style-based currency allocation: currency **total returns**
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
pip install -e ../../core   # qp-core, the shared package (not on PyPI)
pip install -e .[dev]
pytest -q                 # 213 tests, offline, about 40 s
python examples/run_pipeline.py   # full pipeline, about 60 s
```

## Headline numbers (seed 123, 3 024 business days, synthetic G10+EM panel)

All numbers are from `python examples/run_pipeline.py` on synthetic data.
The value signal uses the exact PPP anchor that the synthetic spots revert
to, and momentum is a planted AR(1) trend. Style performance is therefore a
property of the generator, not evidence that the styles work.

Style sleeves (dollar-neutral, gross 2.0, one-day lag), net of 5 bp per
unit of currency-level turnover:

| Style | Ann. ret | Vol | Sharpe (iid SE) | Gross Sharpe | Turnover/yr | Skew | MDD | CVaR95/day |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CARRY | 6.25% | 13.28% | 0.47 (0.29) | 0.49 | 4.2x | -1.73 | 18.43% | 1.90% |
| MOMENTUM | 1.90% | 8.47% | 0.22 (0.30) | 0.49 | 45.5x | -0.14 | 19.90% | 1.08% |
| VALUE | 5.66% | 9.88% | 0.57 (0.29) | 0.66 | 17.9x | -1.33 | 23.82% | 1.38% |

- **Skew-aware carry sizing**: a 0.50%/day CVaR95 budget cuts the carry book
  30% below 5%-vol-target sizing (0.26x vs 0.37x). The multi-style RU-LP
  cuts tail CVaR 86% (2.87% to 0.40%/day) for an 8.1%/yr expected-return
  give-up against the unconstrained mean-chaser.
- **Hedging** (synthetic international equity portfolio): in-sample
  h* = {EUR 1.15, JPY -1.40, GBP 1.56, AUD 3.71, CHF -0.42}, with OLS
  standard errors 0.18 to 0.44. Split-sample estimates move CHF from -1.42
  to +0.59 and AUD from 2.75 to 4.71. Out of sample, the first-half h*
  cuts second-half variance 4.1% against 2.1% for a full hedge. The AUD
  ratio is an equity cross-hedge, not a currency hedge. Forward carry
  raises the full-hedge mean from 4.33% to 5.01% a year.
- **Walk-forward race** (est. 504d, monthly, 5 bp allocation costs plus
  sleeve costs, Ledoit-Wolf covariance, carry accrued): EW SR 0.73, MVO
  0.78, ERC 0.70, CVaR-constrained 0.75, each with SE about 0.33. The
  differences are well inside one standard error.
- **GBP-base reporting**: dollar-neutral sleeves are base-invariant to
  1.4e-17 for log returns. Every combination of the sleeves is also
  dollar-neutral. Adding an unhedged USD collateral account moves the EW
  Sharpe from 0.73 to 0.63 on a GBP base.

## Layout

```
src/fx_port/            returns_est, covariance, mvo, risk_parity,
                        hedging, cvar_opt, backtest, metrics, data/
tests/                  187 offline seeded tests
examples/run_pipeline.py
docs/                   METHODOLOGY.md, VALIDATION.md, DESK_GUIDE.md
```

See `docs/METHODOLOGY.md` for model choices and the assumptions register
(including why carry returns exist at all, the forward premium puzzle, and
why its vol understates its risk), `docs/VALIDATION.md` for every identity
and failure mode, `docs/DESK_GUIDE.md` for how an FX overlay desk would run
this.

## November 2026 models

`hrp`, `robust_mvo` (resampled and robust MVO), `cost_qp` (cost- and turnover-penalised rebalance), `factor_model`, `tracking_error` and `sharpe_tests` (HAC and block-bootstrap Sharpe difference). See docs/METHODOLOGY.md and docs/VALIDATION.md.
