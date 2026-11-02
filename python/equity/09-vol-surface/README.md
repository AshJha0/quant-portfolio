# 09: Equity volatility surface and Heston stochastic volatility

Implied vol, arbitrage-free SVI smiles, a total-variance surface, the
Heston model, calibration, pricing, Greeks and validation. Pure
numpy/scipy/pandas, offline, seeded. All data is synthetic.

```
option chain --> implied vols --> SVI smiles --> total-variance surface
  (synthetic)    (Brent+Newton,   (constrained:    (linear in w(k,T),
                  NaN when no      Lee bound,       calendar check,
                  information)     Durrleman g>=0)  butterfly re-check)
                                         |
                                         v
               Heston calibration (vega-weighted, OTM inversion,
                 multi-start; cond(J) and the kappa/xi ridge reported)
                                         |
              +--------------------------+--------------------------+
              v                          v                          v
       Fourier pricing             Monte Carlo                 Greeks (FD +
   (little-trap CF; P1/P2,    (full-truncation Euler;       Richardson; BS-
    damped integral, fast      QE with martingale            equivalent; sticky-
    Gauss-Legendre)            correction, antithetics,      strike, sticky-
                               control variate)              moneyness, LV rule)
```

## Quickstart

```bash
cd python/equity/09-vol-surface
pip install -e .[test]
pytest -q                          # see the table below for count and runtime
python examples/run_pipeline.py    # end-to-end demo
python examples/validation_tables.py   # tables in docs/VALIDATION.md
```

```python
import numpy as np, eq_surface as es
from eq_surface.data import generate_chain

chain = generate_chain(mode="heston", seed=42)      # known ground truth inside
sl = chain.slice(0.5)
ivs = es.implied_vol_vector(sl.call_mid.values, chain.spot, sl.strike.values,
                            0.5, chain.rate, chain.div_yield)
k = np.log(sl.strike.values / sl.forward.values)
fit = es.fit_svi(k, ivs**2 * 0.5, T=0.5)            # constrained SVI
print(fit.params, fit.arb_free, fit.active_constraints)
```

## Results on synthetic data

Produced by `python examples/run_pipeline.py` (seed 42) unless stated.

| Item | Result |
|---|---|
| Heston P1/P2 vs damped vs Gauss-Legendre | pairwise below 1.5e-7 (tests assert 1e-6) |
| Fourier vs Black-Scholes at xi -> 0, v0 = theta | about 1e-10 |
| Calibration recovery, chain priced with adaptive P1/P2, calibrated with Gauss-Legendre | (v0, kappa, theta, xi, rho) = (0.035, 1.8, 0.045, 0.45, -0.65) recovered to 6 dp, RMSE 0.0000 vol points, cond(J) 1.2e3 |
| SVI slice fit (constrained) vs quadratic-in-delta | 0.0006 to 0.041 vol points vs 0.50 to 0.75 |
| 1y variance swap | strip 19.94%, Heston analytic 20.09%, flat ATM 18.29% |
| ATM 6m call delta | sticky-strike 0.538, sticky-moneyness 0.615, Heston 0.616 |
| Heston fitted to a non-Heston (SVI) surface | residual RMSE largest at 1w, smallest at 2y: see docs/VALIDATION.md section 5.1 |

## Tests and runtime

| Command | Result | Wall time |
|---|---|---|
| `pytest -q` | 161 passed | 53 s |
| `python examples/run_pipeline.py` | exit 0 | 112 s |

Measured on the development machine.

## Layout

```
src/eq_surface/
  black_scholes.py   BS pricer and vega, implied vol (Brent + Newton, NaN when no information)
  smile.py           raw SVI: constrained fit (Lee, Durrleman), analytic derivatives,
                     quadratic-in-delta baseline
  surface.py         total-variance surface, calendar check and enforcement,
                     butterfly re-check at pillars and interpolated expiries
  heston.py          little-trap CF, P1/P2, damped and Gauss-Legendre pricers, Feller check
  heston_mc.py       full-truncation Euler and Andersen QE (martingale-corrected),
                     antithetics, control variate
  calibration.py     vega-weighted multi-start calibration and identifiability diagnostics
  greeks.py          FD Greeks (Richardson), BS-equivalent Greeks, smile-adjusted deltas
  data/synthetic.py  seeded chain generator (known Heston or SVI ground truth, bid/ask)
tests/               analytic identities, cross-method, recovery, bias, property, edge cases
examples/            run_pipeline.py, validation_tables.py
docs/                METHODOLOGY.md, VALIDATION.md, DESK_GUIDE.md
```

METHODOLOGY covers model choice and assumptions, VALIDATION the evidence
and failure modes, DESK_GUIDE the workflow, controls and scenarios.

## November 2026 additions

- `ssvi.py`: SSVI and eSSVI surfaces with the Gatheral-Jacquier no-arbitrage
  checks, a calendar check and a least-squares fit.
- `dupire.py`: Dupire local vol from total variance or from call prices, and
  a local-vol Monte Carlo for repricing checks.
- `bates.py`: Bates (Heston plus lognormal jumps) by the Lewis integral, with
  the Merton series as a closed-form special case.

Tests: `tests/test_round3_volsurface.py` (15 tests).
