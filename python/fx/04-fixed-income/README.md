# FX-Linked Fixed Income — Multi-Currency Curves, FX Forwards & Cross-Currency Analytics

Two-currency discount curve bootstrapping, FX forward curves via covered
interest parity, cross-currency basis, and pricing + risk for FX forwards,
FX swaps and fixed-fixed cross-currency swaps — with per-currency DV01/KRD
ladders, basis DV01, a joint scenario engine (2008 basis blowout, March
2020, year-end turn) and a CIP arbitrage detector with transaction costs.

```
Yield curves (USD, EUR)        FX forward curve            Instruments & risk
 deposits + par swaps   ──►  F(T) = S·DF_f/DF_d    ──►  outrights / FX swaps /
   bootstrap per ccy         + x-ccy basis s(T):        xccy swaps ──► FX delta,
   (reprice ~1e-16)          F_mkt = S·DF_f·e^{-sT}/DF_d  DV01 & KRD per ccy,
                                                          basis DV01, scenarios,
                                                          carry, CIP arbitrage
```

Conventions: pairs BASE/QUOTE (EURUSD = USD per EUR); domestic = quote ccy
(USD), foreign = base ccy (EUR); zeros continuously compounded annualised;
PVs in the quote currency. See `docs/METHODOLOGY.md`.

## Quickstart

```bash
cd python/fx/04-fixed-income
python -m pytest tests -q        # 418 tests, offline, ~0.4s
python examples/run_pipeline.py  # end-to-end, < 1s
```

```python
import sys; sys.path.insert(0, "src")   # or: pip install -e .
from fx_rates import *
from fx_rates.data import generate_market_quotes, build_market_state

market = build_market_state(generate_market_quotes("normal", seed=42))
market_forward(market, 5.0)        # 1.1867  (CIP 1.1719 + basis)
fwd = FXForward(25e6, 1.10, 0.5)   # long 25m EUR 6m at 1.10
fwd.value(market)                  # MTM in USD
dv01(fwd, market, "EUR")           # signed EUR-curve DV01 (negative!)
```

## Results (regime `normal`, seed 42 — reproduced by `run_pipeline.py`)

- Bootstrap round trip: deposits and par swaps reprice to **< 1e-16**
  (contract 1e-10), both currencies, all four regimes.
- Forward points, 1y: CIP +152.2 pips, market +168.7 pips — the 16.5 pip
  gap is the -15bp basis.
- **Basis mispricing demo:** 5y CIP forward 1.1719 vs market 1.1867
  (+147.4 pips at -25bp basis). A EUR 100m 5y forward struck at the market
  forward is par on the basis curve but shows a fictitious
  **-$1,203,204** PV in a CIP-only model.
- Sample book (2 outrights, 1 FX swap, 1 xccy swap): PV -$292k; FX delta
  +$62.0m; USD DV01 +$20.5k/bp; EUR DV01 -$21.3k/bp (sign flip vs USD);
  basis DV01 -$21.3k/bp.
- Scenarios: 2008 USD funding squeeze **-$7.33m**; March 2020 -$3.07m;
  EUR year-end turn (basis -40bp) +$0.86m.
- Carry: long 10m EUR 1y forward at +169 pips premium rolls down
  **-$14.1k per month** (negative carry, long the low-yielder).
- CIP arbitrage detector: consistent quotes never flag; a forward planted
  30 pips above the no-arb band is detected (sell-forward direction) with
  riskless 27.95bp of notional, P&L verified to 1e-10.

## Layout

```
src/fx_rates/            daycount, curve, bootstrap, fxforward, xccy,
                         risk, scenarios, arbitrage, data/{synthetic,live}
tests/                   269 offline seeded tests (identities, round trips,
                         economic signs, edge cases)
examples/run_pipeline.py end-to-end reproduction of every number above
docs/                    METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```

Documentation contract: model choice vs alternatives and the assumptions
register (A1–A9) in `docs/METHODOLOGY.md`; validation evidence and failure
modes F1–F6 in `docs/VALIDATION.md`; desk usage, scenarios and limits in
`docs/DESK_GUIDE.md`.
