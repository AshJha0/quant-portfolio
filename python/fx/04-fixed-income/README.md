# FX-Linked Fixed Income: Multi-Currency Curves, FX Forwards and Cross-Currency Analytics

Two-currency curve bootstrapping, FX forward curves via covered interest
parity, cross-currency basis from float-float basis-swap quotes on the
collateral-currency convention, and pricing and risk for FX forwards, FX
swaps, fixed-fixed cross-currency swaps and basis swaps. Per-currency DV01
and KRD ladders, a basis DV01 that is separate from rate DV01, a joint
scenario engine and a CIP arbitrage detector with transaction costs.

```
Curves (USD, EUR)          Basis (non-collateral ccy)         Instruments and risk
deposits + par swaps  -->  D_EUR bootstrapped from     -->  outrights, FX swaps,
bootstrap per ccy          float-float basis swaps          xccy and basis swaps:
                           F_mkt = S * D_EUR / DF_USD       FX delta, DV01, KRD,
                                                            basis DV01, scenarios
```

Conventions: pairs BASE/QUOTE (EURUSD = USD per EUR); domestic = quote
currency, foreign = base currency; zeros continuously compounded; PVs in
the quote currency. Spot is T+2 business days on partial calendars; ON and
TN settle before spot. See `docs/METHODOLOGY.md`.

## Quickstart

```bash
cd python/fx/04-fixed-income
pip install -e ../../core   # qp-core, the shared package (not on PyPI)
pip install -e .[test]
python -m pytest tests -q
python examples/run_pipeline.py
```

```python
from fx_rates import *
from fx_rates.data import generate_market_quotes, build_market_state

market = build_market_state(generate_market_quotes("normal", seed=42))
market_forward(market, 5.0)        # 1.1867 (CIP 1.1719 plus basis)
fwd = FXForward(25e6, 1.10, 0.5)   # long 25m EUR 6m at 1.10
fwd.value(market)                  # MTM in USD
dv01(fwd, market, "EUR")           # signed EUR-curve DV01 (negative)
basis_dv01(fwd, market)            # basis DV01, a separate factor
```

| Command | Result |
| --- | --- |
| `python -m pytest tests -q` | 456 passed, about 5 s |
| `python examples/run_pipeline.py` | about 0.1 s inside the script, 7 s with interpreter start-up |

## Results

`python examples/run_pipeline.py`, synthetic quotes, regime `normal`,
seed 42:

| Quantity | Value |
| --- | --- |
| Max repricing error, deposits and swaps | 4.2e-17 |
| 1y forward points, CIP / market | +152.2 / +168.7 pips |
| 5y forward, CIP / market | 1.1719 / 1.1867 (+147.9 pips) |
| EUR 100m 5y forward at market, PV ignoring basis | -1,207,560 |
| Sample book PV | -292,229 |
| Sample book FX delta | +62.0m |
| USD / EUR / basis DV01 per bp | +20,544 / -21,316 / -21,178 |
| 100m 5y par basis swap: EUR DV01 / basis DV01 | 0 / -51,247 |
| 2008 funding squeeze / March 2020 / year-end turn | -7.34m / -3.08m / +0.86m |
| Carry, long 10m EUR 1y forward, first month | -14,166 |
| Planted CIP violation, riskless P&L | 27.95bp of notional |

## Layout

```
src/fx_rates/            calendars, daycount, curve, bootstrap, fxforward,
                         xccy, risk, scenarios, arbitrage, data/{synthetic,live}
tests/                   offline seeded tests
examples/run_pipeline.py reproduces every number above, writes nothing
docs/                    METHODOLOGY.md, VALIDATION.md, DESK_GUIDE.md
```
