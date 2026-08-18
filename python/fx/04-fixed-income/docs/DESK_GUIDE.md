# Desk Guide — FX-Linked Fixed Income

How a real desk would use this library, who consumes the numbers, and the
controls around them. Figures quoted below are from
`examples/run_pipeline.py` (normal regime, seed 42).

## 1. Who uses this

- **FX swaps / forwards desk.** Quotes forward points and FX swaps off the
  two discount curves plus the basis curve; `forward_points_table` *is* the
  desk's points run (CIP points, market points, and the basis contribution
  split out per tenor). The trader's intraday questions — "what's my net
  EUR DV01?", "what happens if the basis gaps 40bp over the turn?" — are
  `book_risk_report` and `scenario_table` calls.
- **Treasury / funding desk.** Uses the basis-adjusted curve to compare
  raising USD directly vs synthetically (borrow EUR, FX swap into USD).
  With the EURUSD basis at -25bp (5y), synthetic USD funding via the swap
  market costs ~25bp/yr over direct — that spread *is* the basis, read off
  `implied_basis_from_forwards`. The CIP arbitrage detector, run with
  `min_pnl` set to the desk's balance-sheet charge, tells treasury when the
  basis pays them to lend USD into the swap market.
- **Corporate hedgers rolling forwards.** A EUR-based exporter hedging USD
  revenue (or a USD treasurer hedging EUR costs) rolls 3m–1y forwards.
  `forward_carry`/`carry_table` quantifies the roll cost: long EUR 1y
  forward at +169 pips premium bleeds ~14.1k USD per month on 10m EUR —
  the number the salesperson puts in the hedging cost slide.
- **Real-money hedgers (the Japanese lifer story).** A JPY (or EUR) life
  insurer holding USD bonds hedges FX with rolling 3m FX swaps. Their
  hedging cost = short-end rate differential + the basis. When the basis
  widens (more negative for lending their currency into USD), hedged USD
  yields collapse — the mechanism that repeatedly pushed Japanese lifers
  in and out of hedged Treasuries. Reproduce it: shock `basis_bp` and read
  the carry of the 3m swap roll.

## 2. Daily workflow

1. **Curve build (SOD):** bootstrap USD and EUR curves from deposits +
   swap closes (`bootstrap_curve`); verify repricing < 1e-10 (automated
   control — a fail blocks the batch).
2. **Basis calibration:** back out `s(T)` from the FX forward/basis-swap
   marks (`implied_basis_from_forwards`); eyeball vs yesterday — a move
   > 5bp at the front end without a news event is a data error until
   proven otherwise.
3. **Revalue the book** (`book_risk_report`): PV, FX delta, USD DV01, EUR
   DV01, basis DV01 per position and in total, all in USD.
4. **P&L attribution:** explain day-over-day P&L as
   `delta * dS + DV01_usd * dz_d + DV01_eur * dz_f + basisDV01 * ds +
   carry` (all components available); unexplained residual > threshold
   goes to middle office.
5. **Scenarios** (`scenario_table` with `historical_scenarios()`) for the
   risk meeting; **carry report** for the roll desk.

## 3. Real scenarios (and what the book does)

Sample book: long 25m EUR 6m, short 15m EUR 2y, 40m 3m/1y buy-sell FX
swap, 50m 5y receive-EUR xccy swap. Base PV -$292k; net FX delta +$62m
equivalent, USD DV01 +$20.5k/bp, EUR DV01 -$21.3k/bp, basis DV01
-$21.3k/bp.

- **2008 USD funding squeeze** (spot -12%, USD -150bp, EUR -50bp, basis
  -150bp): **-$7.33m**, dominated by the FX delta and the receive-EUR xccy
  gaining on basis but losing on spot. Lesson: basis DV01 (-$21k/bp) alone
  suggests +$3.2m from the basis leg — only full revaluation gets the
  joint move right.
- **2020 March dash-for-cash** (spot -5%, USD -100bp, basis -85bp):
  **-$3.07m**. The episode where central-bank swap lines are the
  mean-reversion mechanism — a desk that survived to April kept the basis
  P&L.
- **EUR year-end turn** (basis -40bp only): **+$0.86m** — the book is net
  long basis-widening. Desks cap exactly this number going into December.
- **Rates surprises** (Fed +100bp: +$3.02m; ECB +75bp: +$0.51m) — the
  per-currency DV01 ladders tell you in advance which one you are.

## 4. Limits & controls

- **Per-currency DV01 limits** (e.g. |USD DV01| < $50k/bp, |EUR DV01| <
  $50k/bp) *and* per-bucket KRD limits — the 5b ladder shows the book is
  really a 5y trade ($23.2k/bp USD at the 5y pillar) dressed up with
  short-dated hedges; a net-DV01-only limit would miss the tenor gap.
- **Basis limits:** separate limit on basis DV01 (here -$21.3k/bp) plus a
  stressed-basis limit using the 2008 scenario, because of failure mode
  F1 (basis gaps, Greeks lie).
- **Tenor gap / cashflow ladder limits:** cap the net cashflow per bucket
  per currency — an FX swap book can be DV01-flat yet have huge offsetting
  cashflows one week apart (settlement/liquidity risk).
- **Curve quality controls:** bootstrap repricing tolerance, stale-quote
  detection, day-over-day zero/basis jump thresholds.
- **Model governance:** assumptions A1–A8 (METHODOLOGY.md) form the model
  card; A2 (no OIS/CSA discounting) and A6 (no calendars) are the two
  documented reservations, with short-date pricing explicitly out of
  scope. Golden-value pillar DFs, forwards and risk numbers are pinned by
  the seeded test suite — any library change that moves a price fails CI.
- **Arbitrage monitor:** `detect_cip_arbitrage` on closes with `min_pnl` =
  balance-sheet charge; alerts route to treasury, not to an auto-trader —
  see failure mode F4 for why post-2008 'CIP arbitrage' is really a
  balance-sheet rent (Du–Tepper–Verdelhan).

## 5. Consumers of the numbers

| Number | Consumer | Frequency |
| --- | --- | --- |
| Forward points run | Sales/e-trading, corporate clients | Continuous |
| Book PV + risk report | Trader, market risk, product control | EOD (risk: intraday) |
| KRD ladders per ccy | Trader (hedging), market risk (limits) | EOD |
| Basis DV01 + basis scenarios | Trader, ALCO/treasury | EOD / weekly |
| Carry/roll table | Forward roll desk, client advisory | Weekly |
| CIP monitor | Treasury funding | EOD |
| P&L attribution | Product control | EOD |
