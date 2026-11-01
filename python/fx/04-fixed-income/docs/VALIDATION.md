# Validation: FX-Linked Fixed Income

Numbers come from `python examples/run_pipeline.py` (synthetic quotes,
regime `normal`, seed 42) unless a test is named. Test counts are in the
README.

## 1. Round trips and identities

| Check | Contract | Achieved | Test |
| --- | --- | --- | --- |
| Bootstrap reprices deposits and par swaps (both currencies, 4 regimes) | < 1e-10 | 0 to 4e-17 | `test_bootstrap.py::TestBootstrapRoundTrip` |
| Pillars only at quoted maturities | exact | passes | `test_basis_swap.py::TestConventionFixes` |
| Float-float bootstrap reprices every quoted basis swap at par | 1e-14 | passes, frequency 1, 2, 4 | `test_basis_swap.py::TestBootstrap` |
| Zero basis spreads give `D == P` at every time | 1e-14 | passes | same |
| CIP identity `F/S = DF_f/DF_d` | exact | < 1e-14 | `test_fxforward.py` |
| Triangular consistency EURJPY = EURUSD x USDJPY | 1e-12 rel | all tenors | `test_fxforward.py` |
| Forward MTM: cashflow method == forward method | 1e-10 rel | incl. negative rates | `test_fxforward.py` |
| FX swap == two outrights | 1e-8 abs | incl. after shocks | `test_fxforward.py` |
| XCCY value == independent discounting of its cashflows | 1e-8 abs | passes | `test_xccy.py` |
| XCCY par-rate solver zeroes PV | < 1e-10 x notional | passes | `test_xccy.py` |
| Par basis swap PV, EURUSD and USDJPY | < 1e-12 x notional | passes | `test_basis_swap.py` |
| FX delta of an outright == `N * D_base(T)` | 1e-9 rel | passes | `test_risk.py` |
| Parallel DV01 == sum of KRD ladder | 1e-6 rel | both currencies | `test_risk.py` |
| Basis DV01 of a par basis swap == `-N S 1e-4 tau (sum D_up + sum D_dn)/2` | 1e-9 rel | passes | `test_basis_swap.py` |
| Rate DV01 of a par basis swap (EUR and USD) | < 1e-6 x basis DV01 | passes | `test_basis_swap.py` |
| Basis DV01 of an FX forward differs from its EUR DV01 | > 0.5% | passes | `test_risk.py` |
| Scenario engine == manual market rebuild | 1e-8 abs | passes | `test_scenarios.py` |
| CIP arbitrage P&L == hand-computed round trip | 1e-10 abs | both directions | `test_arbitrage.py` |
| CIP band with per-currency accruals (USD ACT/360, GBP ACT/365F) | 1e-15 rel | passes | `test_basis_swap.py` |

Sign tests: a long-base forward has positive domestic DV01 and negative
foreign DV01; basis DV01 is negative for long base; EURUSD points are
positive when `r_f < r_d`, including negative EUR rates; long the
low-yielder carries negatively; a wider (more negative) basis helps the
receive-EUR xccy leg.

## 2. Basis mispricing

Spot 1.0866, USD 5y zero 4.061%, EUR 5y zero 2.550%, 5y basis -25bp
(pipeline step 3):

| Quantity | Value |
| --- | --- |
| 5y CIP forward (no basis) | 1.1719 |
| 5y market forward (basis) | 1.1867 |
| Difference | +147.9 pips |
| EUR 100m 5y forward at the market forward, PV with basis | 0.00 |
| Same forward, PV ignoring basis | -1,207,560 |

## 3. Risk separation (pipeline step 5a)

| Position | PV | USD DV01 | EUR DV01 | Basis DV01 |
| --- | --- | --- | --- | --- |
| 100m 5y par basis swap, receive EUR + spread | 0 | 0 | 0 | -51,247 |

Under the old zero-spread model basis DV01 equalled EUR DV01 on every
position. Now an FX forward's two numbers differ slightly and a basis
swap's differ completely, which is the separation a desk hedges with.

## 4. Arbitrage detector

- No false positives: CIP-consistent quotes never flag inside
  `[F_lower, F_upper]`; the band edges are the detection frontier (tested
  at +/-1e-7).
- Planted violation: a forward 30 pips above the band yields 27.95bp of
  notional at maturity ($279,474 on $100m, pipeline step 8), matching the
  hand-computed round trip to 1e-10.
- Costs: a deviation that flags with 1-pip spreads is absorbed by
  10-pip/50bp spreads (tested).

## 5. Failure modes

- **F1 Basis gaps.** DV01s assume small moves. The 2008 scenario on the
  sample book loses -7.34m, while basis DV01 alone (-21.2k per bp x -150bp)
  suggests +3.2m from the basis. Use full-revaluation scenarios for tail
  basis risk.
- **F2 Year-end turn.** The basis curve is smooth in maturity, so it prices
  through the turn rather than the turn itself. The stylised turn scenario
  (-40bp basis) gives +0.86m on the sample book.
- **F3 Short dates.** Value dates are business-day correct on partial
  calendars, but curve times are year fractions and accruals are unit.
  Do not price ON to 1w points with this library.
- **F4 CIP deviations that are not arbitrage.** Persistent post-2008 CIP
  deviations reflect dealer balance-sheet costs (Du, Tepper and
  Verdelhan). `min_pnl` encodes that charge on top of bid/ask.
- **F5 Long-end extrapolation.** Past the last pillar (10y) forwards are
  flat-extrapolated with no warning.
- **F6 Sparse swap strips.** DFs between quoted maturities come from
  log-linear interpolation (A4).
- **F7 Stale cached curves.** `MarketState` is frozen. In-place assignment
  raises `FrozenInstanceError`; `replace()` builds a new instance with
  fresh cached curves (`test_edge_cases_extra.py::TestMarketStateImmutability`).

## 6. Edge cases

| Edge case | Behaviour | Test |
| --- | --- | --- |
| Negative rates (EUR 2019, CHF/EUR both negative) | DF > 1 end to end; identities hold | `test_edge_cases.py`, `test_edge_cases_extra.py` |
| Same-currency cross | `ValueError` at construction | `test_fxforward.py`, `test_xccy.py` |
| Collateral currency not in the pair | `ValueError` | `test_basis_swap.py` |
| Zero notional, empty book | zero PV and risk, no crash | `test_edge_cases.py` |
| Spot over weekends and holidays (EUR 1 May, USD 4 Jul) | business-day spot | `test_fx_dates.py` |
| ON / TN / SN, month-end tenors | correct value dates | `test_fx_dates.py` |
| Invalid quotes (crossed, negative prices, tau <= 0) | `ValueError` | `test_arbitrage.py` |
| Non-finite input anywhere (quote, spot, strike, notional, bump, shock, basis) | `ValueError` naming the field | `test_nan_guards.py` |
| Curve at a non-finite time | raises | `test_nan_guards.py` |
| Extreme curves (50% and -5% zeros) | valid, CIP monotone | `test_edge_cases.py` |
| Crisis basis (-180bp) | bootstraps and prices | `test_synthetic.py` |
| Network | disabled unless `FX_RATES_ALLOW_NETWORK=1` | `test_edge_cases.py::TestOffline` |

A non-finite quote matters most in the CIP detector: every comparison
with NaN is False, so before the guards it reported a confident "no
arbitrage" on data it could not price. `CIPQuotes` now raises.

## 7. Reproducibility

Every stochastic component takes a seed. The generators perturb curve
parameters, not individual quotes, so the quotes are internally
consistent by construction. Tests and the pipeline are offline and
deterministic and write nothing to disk.
