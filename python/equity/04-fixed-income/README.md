# 04 — Fixed Income Pricing & Risk Analytics (`fi_rates`)

Government/corporate bond pricing and rates risk, built the way a desk
builds it:

```
Yield Curve Construction → Bootstrapping → Bond Pricing → Duration
       → Convexity → Key Rate Duration → Scenario Analysis
```

* **Curves** (`curve.py`): discount curve over pillar times with three
  interpolations — log-linear on discount factors (default; piecewise-constant
  forwards), linear on zeros (kept to *demonstrate* its forward sawtooth),
  monotone cubic (PCHIP) on zeros. Zero/forward/par rates derived; flat
  extrapolation warned.
* **Bootstrap** (`bootstrap.py`): sequential bootstrap from deposits, FRAs
  and par swaps (Brent root-solve per pillar), plus a coupon-bond bootstrap.
  Inputs reprice to **1e-10** (measured ~4e-16), order-independent.
* **Bonds** (`bond.py`): fixed-coupon (clean/dirty/accrued, street-convention
  YTM solver), zero-coupon, FRN (== par at reset, exactly), annuities,
  z-spread solver.
* **Risk** (`risk.py`): Macaulay/modified duration, convexity, DV01 —
  analytic *and* bump-based, reconciled; Taylor-vs-full-repricing error
  table; MV-weighted portfolio aggregation.
* **Key rates** (`keyrates.py`): triangular-bump KRDs at 2/5/10/30y
  (configurable); ladder sums to parallel DV01 to 5e-8 (tolerance
  documented).
* **Scenarios** (`scenarios.py`): parallel/steepener/butterfly + historical
  episodes (2013 taper tantrum, 2022 hiking cycle, 2008 GFC — approximate
  published magnitudes), full revaluation vs duration estimate, carry &
  roll-down with the pull-to-par identity.
* **Data** (`data/synthetic.py`): seeded deposit+swap quote sets (upward /
  inverted / flat / negative-rate variants) and a govt+corp sample
  portfolio; `data/live.py`: optional FRED CMT loader, import-guarded,
  never touched by tests.

## Quickstart

```bash
cd python/equity/04-fixed-income
pip install -e .[dev]
python -m pytest tests -q      # 342 tests, ~1s, offline
python examples/run_pipeline.py
```

```python
import datetime as dt
import fi_rates as fr
from fi_rates.data import market_quotes, sample_portfolio

curve = fr.bootstrap_curve(market_quotes("upward", seed=42))
settle = dt.date(2026, 8, 18)
book = sample_portfolio(settle)
print(fr.portfolio_risk(book, settle, curve))     # DV01/duration/convexity
print(fr.krd_report(book, settle, curve))         # 2s/5s/10s/30s ladder
print(fr.scenario_pnl_table(book, settle, curve,
                            list(fr.HISTORICAL_SCENARIOS.values())))
```

## Headline numbers (seed 42, settlement 2026-08-18)

* Curve: 3.06% (3m) → 4.55% (30y) zeros; all 11 quotes reprice to ≤4e-16.
* Book: $1.74mm MV, aggregate modified duration 5.42, DV01 $980/bp.
* KRD ladder 2/5/10/30y: 154 / 278 / 368 / 180 → sum matches parallel DV01
  to 5.3e-8 relative.
* Scenarios (full revaluation): 2022 hiking −$228k (−13.1%), 2013 taper
  −$94k, 2008 GFC +$168k; duration+convexity estimate off by 12–183% on
  non-parallel moves — the reason the KRD ladder exists.
* Convexity halves nothing: it cuts the 100bp Taylor error ~30x
  (−0.401 → −0.014 per 100 face).
* Carry+roll (1y, static curve): 5y govt earns 4.12 per 100 face; identity
  carry+roll == pull-to-par P&L holds to 1e-12.

## Docs (portfolio documentation contract)

* [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — bootstrap+local-interp vs
  Nelson–Siegel/Svensson (who uses which and why); 10-item assumptions
  register, incl. the **single-curve pre-OIS simplification** and its
  multi-curve production replacement.
* [`docs/VALIDATION.md`](docs/VALIDATION.md) — round-trip and
  analytic-identity evidence, KRD-sum tolerance analysis, failure modes with
  numbers (forward sawtooth, extrapolation, duration under twists, negative
  convexity out-of-scope).
* [`docs/DESK_GUIDE.md`](docs/DESK_GUIDE.md) — EOD workflow, DV01-ladder
  hedging, 2s10s curve trades, ALM, auctions/CTD, and the taper-tantrum /
  gilts-LDI / SVB lessons.
