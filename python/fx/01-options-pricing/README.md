# FX Options Pricing & Greeks Engine (`fx_options`)

Flagship-quality pricing library for European (and American, via tree)
FX options with **first-class FX market conventions**: Garman–Kohlhagen,
Black-76 on the forward, CRR binomial, Monte Carlo, all four FX delta
conventions with strike-from-delta inversion, both rhos, vanna/volga, and
a delta-hedging simulator that accounts for foreign interest on the hedge.

```
Garman-Kohlhagen ──► Binomial Tree ──► Black-76 (on F) ──► Monte Carlo
        │                  │                 │                  │
        └──────────────────┴────── cross-model comparison ──────┘
                                   │
     Greeks & FX delta conventions ┴► Delta hedging ─► Desk report
```

**Conventions** (see `CONVENTIONS.md` at repo root): pairs quoted
BASE/QUOTE (EURUSD = USD per EUR); `r_d` = quote-ccy rate, `r_f` =
base-ccy rate; GK = Black–Scholes with `q = r_f`; rates continuously
compounded, ACT/365F; premiums in quote ccy per unit base notional.

## Quickstart

```bash
pip install -e .            # from this directory (deps: numpy/scipy/pandas)
python -m pytest tests -q   # 411 tests, offline, deterministic, ~3s
python examples/run_pipeline.py   # full EURUSD + USDJPY report, <1s
```

```python
from fx_options import (gk_price, analytic_greeks, delta, strike_from_delta,
                        atm_dns_strike, implied_vol, simulate_delta_hedge)

# EURUSD 6m: S=1.10, r_d(USD)=4.25%, r_f(EUR)=2.90%, vol 8.25%
px = gk_price(1.10, 1.1075, 0.5, 0.0425, 0.0290, 0.0825, "call")   # 0.02522800
g  = analytic_greeks(1.10, 1.1075, 0.5, 0.0425, 0.0290, 0.0825, "call")
g.rho_domestic, g.rho_foreign      # +0.2647, -0.2773  (both rates matter!)
g.vanna, g.volga                   # smile-bucket Greeks FX desks live on

# USDJPY quotes use premium-adjusted deltas (premium paid in USD = base ccy):
k25 = strike_from_delta(0.25, 147.5, 0.5, 0.0050, 0.0525, 0.1075,
                        "call", convention="forward_pa")            # 151.64
atm = atm_dns_strike(147.5, 0.5, 0.0050, 0.0525, 0.1075, "forward_pa")
```

## Highlights (all reproduced by `examples/run_pipeline.py`)

- **Model agreement** (EURUSD 6m ATMF call, GK = 0.02522800): Black-76 on
  the CIP forward matches to 1e-16; 1000-step CRR within 6.6e-7; MC
  (200k paths, antithetic + control variate) within 0.8 SE (SE = 6.1e-5).
- **Both rhos**: rho_d = +0.2647, rho_f = −0.2773 per unit notional —
  an FX option is a two-interest-rate instrument.
- **Four deltas for the same option** (EURUSD ATMF call): spot 0.5043,
  forward 0.5116, spot-PA 0.4813, forward-PA 0.4884. On USDJPY the 25Δ
  call strike ranges 151.39–152.05 across conventions — the convention is
  part of the quote. Strike-from-delta round-trips to 1e-8 in all four,
  including the non-monotone premium-adjusted call branch (Brent on the
  documented larger-strike branch).
- **American FX exercise economics**: USDJPY ATMF call with USD carry
  5.25% vs JPY 0.50% carries a **14.4% early-exercise premium**
  (0.628 JPY per USD); the EURUSD equivalent (r_f < r_d) is worthless
  early — exactly the dividend-yield analogy, tested.
- **Implied vol**: Newton + Brent fallback; round-trip max error
  **1.75e-15** across the pipeline grid; hard `ValueError` outside
  no-arbitrage bounds.
- **Hedging simulator with foreign carry**: at true vol, mean P&L ≈ 0 and
  std falls 0.0101 → 0.0014 as rebalances rise 4 → 250 (1/√N verified);
  selling 2 vols rich earns +0.0062 on a 0.0252 premium; 1-pip costs at
  100 rebalances average 0.00041 USD/EUR.
- **Measure care demo**: cash-or-nothing digitals paying in *either*
  currency — foreign-cash digital = `S·e^{−r_f T}N(d1)`, and the test
  suite proves the naive `e^{−r_f T}N(d2)` shortcut is wrong by > 1e-3.
- **Negative rates** (EUR/CHF era), EM vol (35–150%), JPY quotation
  (pip = 0.01, PA deltas, Tokyo cut noted) all first-class and tested.
- **Golden vectors**: 30 cases at 1e-10 in `tests/golden/golden_vectors.json`
  for C++/Rust cross-validation.

## Layout

```
01-options-pricing/
├── README.md
├── pyproject.toml
├── src/fx_options/
│   ├── __init__.py          # public API
│   ├── _common.py           # validation + conventions
│   ├── garman_kohlhagen.py  # GK price, d1/d2, implied vol (Newton+Brent)
│   ├── forwards.py          # CIP forward, forward points, synthetic forward
│   ├── deltas.py            # 4 delta conventions, strike-from-delta, ATM/DNS
│   ├── binomial.py          # CRR tree, European + American, convergence
│   ├── black76.py           # pricing off the FX forward, ≡ GK
│   ├── monte_carlo.py       # seeded MC, antithetic+CV, digitals (measure demo)
│   ├── greeks.py            # analytic Greeks incl. rho_d, rho_f, vanna, volga
│   ├── hedging.py           # delta-hedge sim with foreign-interest accounting
│   ├── comparison.py        # cross-model harness + convergence tables
│   └── data/                # synthetic.py (seeded), live.py (guarded ECB)
├── tests/                   # 359 offline deterministic tests
│   └── golden/              # generate_golden.py + golden_vectors.json
├── examples/run_pipeline.py # EURUSD + USDJPY end-to-end report
└── docs/                    # METHODOLOGY / VALIDATION / DESK_GUIDE
```

## Documentation contract

- **Why GK, vs alternatives** and the full numbered **assumptions
  register** (flat vol misprices wings → project 9; deterministic rates
  fail long-dated; CIP basis; jumps/pegs): `docs/METHODOLOGY.md`.
- **Validation evidence with real numbers** and **failure modes** (CHF
  2015 depeg case study, EM fat tails, long-dated, negative rates):
  `docs/VALIDATION.md`.
- **Desk usage** (vol/delta quoting, delta-vega-vanna-volga buckets,
  25Δ RR/BF marking, NY/Tokyo cuts, T+2, scenario playbook):
  `docs/DESK_GUIDE.md`.
