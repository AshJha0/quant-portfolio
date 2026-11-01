# 09: FX volatility surface and Heston stochastic volatility

FX volatility surface construction from delta-space broker quotes (ATM DNS,
risk reversals, butterflies), four delta conventions including
premium-adjusted, vanna-volga and SVI smiles, a delta-space surface, and
Heston under Garman-Kohlhagen with two Fourier pricers, QE and Euler Monte
Carlo, calibration and vega/vanna/volga risk. All data is synthetic.

## Pipeline

```
broker quotes {ATM, RR25, BF25, RR10, BF10} x {1w,1m,3m,6m,1y,2y}
      |  smile BF: linear map; broker one-vol BF: smile_quotes_from_broker
      v
five smile vols (10P, 25P, ATM, 25C, 10C)
      |  delta -> strike in the slice's own convention (spot/forward, pa)
      v
pillar strikes -> smile fits: SVI (log-moneyness) and vanna-volga
      |
      v
surface: total variance interpolated in T at fixed delta, each slice
read in its quote convention; calendar check at fixed forward moneyness
      |
      v
Heston under GK (little-trap CF): Gil-Pelaez quadrature and COS
      |  vega-weighted calibration, multi-start (5 pillars x 6 expiries)
      v
pricing, digitals, FD Greeks (both rhos, vanna, volga)
      |
      v
Monte Carlo (full-truncation Euler, QE with martingale correction)
```

## Quickstart

```bash
cd python/fx/09-vol-surface
pip install -e .[test]
pytest -q                          # see the table below for count and runtime
python examples/run_pipeline.py    # EURUSD + USDJPY demo
```

```python
from fx_surface import calibrate_heston
from fx_surface.data import usdjpy_market, calibration_slices
from fx_surface.surface import build_surface

mkt  = usdjpy_market()                    # pa-quoted, strong JPY-call skew
surf = build_surface(mkt, smile_model="vv")
surf.vol_atm(1.0)                         # returns the 1y pa DNS ATM quote
surf.vol(145.0, 0.75)                     # vol by strike and expiry
surf.vol_delta(0.25, 0.75, cp=-1)         # vol at the 25-delta put
res  = calibrate_heston(mkt.S, calibration_slices(mkt))
print(res.summary())
```

## Results on the synthetic presets

Produced by `python examples/run_pipeline.py` and `pytest`.

| Item | Result |
|---|---|
| Pillar reproduction, native convention | every quoted pillar to 1e-10 in vol where the smile fits the quotes (all presets, both smile models, all four conventions on EURUSD) |
| USDJPY SVI pillar misfit | up to 4.15 bp: raw SVI cannot follow a call wing still falling at 10 delta; reported per slice |
| pa quotes solved as unadjusted (USDJPY 1y) | ATM strike shifts by 1.58 JPY; pillar vols wrong by up to 0.32 vol points at 2y |
| COS (L=24, N=1024) vs Gil-Pelaez | below 5e-9 in price at 1y; the old L=14 left a 1e-6 bias |
| Ground-truth calibration (quotes priced with Gil-Pelaez, calibrated with COS) | parameters recovered to 4 dp, RMSE 0.0000 vol points |
| EURUSD calibration | rho -0.184, RMSE 0.227 vol points |
| USDJPY calibration | rho -0.469, RMSE 0.181 vol points |
| EURUSD 6m digital at 1.15 | flat GK 0.2054, vanna-volga 0.1895, Heston 0.1899 |

## Tests and runtime

Measured on the development machine with `pytest -q` and
`python examples/run_pipeline.py`:

| Command | Result | Wall time |
|---|---|---|
| `pytest -q` | 290 passed | 57 s |
| `python examples/run_pipeline.py` | exit 0 | 33 s |

## Layout

```
src/fx_surface/          garman_kohlhagen, smile_from_quotes, smile,
                         surface, heston, heston_mc, calibration,
                         greeks, data/synthetic (seeded presets:
                         EURUSD, USDJPY-pa, 35%-vol EM, Heston ground truth)
tests/                   offline seeded tests
examples/run_pipeline.py end-to-end demo
docs/                    METHODOLOGY.md, VALIDATION.md, DESK_GUIDE.md
```

METHODOLOGY covers model choice and the assumptions register, VALIDATION
the evidence and failure modes F1 to F7, DESK_GUIDE the workflow, risk
buckets and scenarios.
