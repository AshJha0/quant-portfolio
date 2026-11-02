# Desk Guide: FX-Linked Fixed Income

Who would use this library, how, and the controls around it. Figures come
from `python examples/run_pipeline.py` (synthetic quotes, normal regime,
seed 42).

## 1. Users

- **FX swaps and forwards desk.** `forward_points_table` is the points
  run: CIP points, market points and the basis contribution per tenor.
  Net EUR DV01 and "what if the basis gaps 40bp over the turn" are
  `book_risk_report` and `scenario_table` calls.
- **Treasury and funding.** With the 5y EURUSD basis at -25bp, synthetic
  USD funding through FX swaps costs about 25bp a year over direct USD
  funding. `detect_cip_arbitrage` with `min_pnl` set to the desk's
  balance-sheet charge shows when the basis pays to lend USD into the swap
  market.
- **Corporate hedgers rolling forwards.** `carry_table`: a long 10m EUR 1y
  forward at a +169 pip premium loses about 14.2k USD in its first month.
- **Real-money hedgers.** A JPY or EUR investor holding USD bonds hedges
  with rolling 3m FX swaps; the hedge cost is the short-rate differential
  plus the basis. For USDJPY the basis sits on the JPY curve
  (`MarketState(pair=("USD", "JPY"))`); shock `basis_bp` and read the carry.

## 2. Daily workflow

1. Curve build: bootstrap USD and EUR curves from deposits and swap closes;
   repricing must be below 1e-10.
2. Basis: load float-float basis-swap quotes into `basis_spreads`;
   `basis_swap_par_spread` confirms each quote reprices. A front-end move
   over 5bp without news is a data error until shown otherwise.
3. Value the book with `book_risk_report`: PV, FX delta, USD DV01, EUR
   DV01 and basis DV01 per position, in USD.
4. P&L attribution: `delta dS + DV01_usd dz_d + DV01_eur dz_f +
   basisDV01 ds + carry`.
5. Scenarios with `historical_scenarios()` for the risk meeting.

Value dates use `spot_date` and `tenor_dates` with the pair's calendars
(partial: weekends plus fixed-date holidays).

## 3. Scenarios on the sample book

Sample book: long 25m EUR 6m, short 15m EUR 2y, 40m 3m/1y buy-sell FX
swap, 50m 5y receive-EUR fixed-fixed xccy swap.

| Quantity | Value |
| --- | --- |
| Base PV | -292,229 |
| FX delta | +62.0m |
| USD DV01 per bp | +20,544 |
| EUR DV01 per bp | -21,316 |
| Basis DV01 per bp | -21,178 |

| Scenario | P&L |
| --- | --- |
| 2008 USD funding squeeze (spot -12%, USD -150bp, EUR -50bp, basis -150bp) | -7,340,781 |
| 2020 March dash for cash (spot -5%, USD -100bp, EUR -30bp, basis -85bp) | -3,084,833 |
| EUR year-end turn (basis -40bp) | +857,029 |
| Fed +100bp surprise | +3,020,137 |
| ECB +75bp catch-up | +509,577 |

Basis DV01 alone suggests +3.2m from the 2008 basis move; full
revaluation of the joint move loses 7.3m. Use scenarios for tail risk.
The book is net long basis widening, which is the number a desk caps into
December.

## 4. Limits and controls

- Per-currency DV01 limits and per-bucket KRD limits. The 5b ladder shows
  the book is mostly a 5y position (23.2k per bp USD at the 5y pillar)
  with short-dated hedges; a net-DV01 limit would miss that.
- A separate basis DV01 limit and a stressed-basis limit from the 2008
  scenario. Basis DV01 is a distinct risk factor: a par 5y basis swap on
  100m has -51,247 basis DV01 and zero rate DV01 (pipeline step 5a).
- Cashflow ladder limits per currency and bucket: an FX swap book can be
  DV01-flat with large offsetting cashflows a week apart.
- Curve controls: repricing tolerance, stale-quote checks, day-over-day
  jump thresholds.
- Model card: assumptions A1-A9 in METHODOLOGY.md. The main reservations
  are A2 (no OIS tenor basis or CSA optionality), A5 (MtM-resetting
  swaps only in `xccy_float`) and A6 (partial calendars).
- Arbitrage monitor: `detect_cip_arbitrage` with `min_pnl` set to the
  balance-sheet charge, alerts to treasury. A non-finite quote raises a
  named error, so a broken feed reaches the alert queue instead of
  producing a silent "no arbitrage".

## 5. Consumers

| Number | Consumer | Frequency |
| --- | --- | --- |
| Forward points run | Sales, e-trading, corporate clients | Continuous |
| Book PV and risk | Trader, market risk, product control | EOD, intraday for risk |
| KRD ladders per currency | Trader, market risk | EOD |
| Basis DV01 and basis scenarios | Trader, treasury | EOD, weekly |
| Carry table | Forward roll desk | Weekly |
| CIP monitor | Treasury | EOD |

## November 2026: which cross-currency swap

For the major pairs the quoted basis is on the MtM (resetting) form; mark it
with `FloatFloatXccy(mtm_reset=True)`. Use the constant-notional form for
trades booked that way, with `notional_quote` set at trade.
