# Desk Guide: how a rates desk would use `fi_rates`

All figures come from `python examples/run_pipeline.py` on synthetic,
seeded quotes and a synthetic 7-position book.

## 1. Daily workflow

**Curve build.** Closing deposits and swap par rates (or bond prices for a
government desk) go through `bootstrap_curve` or `bootstrap_bond_curve`.
Every input must reprice to 1e-10 (`reprice_instruments`) or the build
fails. A production build would be multi-curve (OIS discounting plus
projection curves); this one is single-curve (METHODOLOGY assumption 1).

**Risk and PV.** `portfolio_risk` gives market value, YTM, modified
duration, convexity and DV01 per position and in aggregate (pipeline
section 4):

| Position | MV | YTM | Mod. duration | Convexity | DV01 |
|---|---|---|---|---|---|
| GOVT_2Y | 498,252 | 3.44% | 1.92 | 4.7 | 97.3 |
| GOVT_10Y | 295,912 | 4.17% | 8.15 | 78.6 | 245.9 |
| GOVT_30Y | 151,738 | 4.48% | 16.19 | 378.7 | 247.5 |
| CORP_BBB_7Y | 133,460 | 6.33% | 5.73 | 40.1 | 78.8 |
| TOTAL (7 positions) | 1,739,957 | 4.35% | 5.53 | 62.0 | 980.2 |

Government bonds accrue ACT/ACT-ICMA; corporates 30/360US.

**Hedging off the DV01 ladder.** `krd_report` on the same book (pipeline
section 5):

| Key rate | Key-rate DV01 | Key-rate duration |
|---|---|---|
| 2y | 154.3 | 0.89 |
| 5y | 278.0 | 1.60 |
| 10y | 368.4 | 2.12 |
| 30y | 179.5 | 1.03 |
| Sum | 980.2 | 5.63 |

The ladder sums to the parallel DV01 to 5.3e-8 relative. To flatten, sell
about 154 DV01 at 2y, 278 at 5y, 368 at 10y and 180 at 30y. Hedging the
total 980 DV01 with one 10y instrument leaves the book flat to parallel
moves and exposed to twists (VALIDATION §4.3). The ladder is in pillar
zero-rate space. A desk hedges in par-rate space, which needs the Jacobian
to the hedge instruments; that is not implemented.

## 2. Curve trades

A 2s10s steepener (long 2y, short 10y, DV01-neutral) is a target ladder:
+X DV01 at 2y, -X at 10y, about 0 elsewhere. `steepener_scenario` and
`butterfly_scenario` give the P&L; the sign tests are in
`test_scenarios.py`.

`carry_rolldown` prices what a position earns if the curve does not move.
Pipeline section 8, 1y horizon, repo 3.10% ACT/360, per 100 face:

| Position | Coupon income | Funding | Net carry | Pull-to-par | Curve roll-down | Total |
|---|---|---|---|---|---|---|
| GOVT_2Y | 3.25 | 3.13 | 0.12 | 0.17 | 0.20 | 0.49 |
| GOVT_5Y | 3.50 | 3.10 | 0.40 | 0.28 | 0.34 | 1.02 |
| GOVT_10Y | 4.00 | 3.10 | 0.90 | 0.11 | 0.31 | 1.32 |
| GOVT_30Y | 4.50 | 3.18 | 1.32 | -0.01 | 0.10 | 1.41 |

On an upward curve the long end carries more than the front end once
funding is paid.

## 3. Scenario analysis

`scenario_pnl_table` runs full revaluation against the historical library
(pipeline section 6, synthetic book):

| Episode | P&L | % of MV | Reading |
|---|---|---|---|
| 2013 taper tantrum (10y +130bp, front end anchored) | -94,037 | -5.4% | Risk sits at the long key rates when the front end is pinned. Check the ladder, not the total. |
| 2022 hiking (2y +370bp, 10y +235bp) | -228,406 | -13.1% | A bear flattener; the short-end buckets dominate. Duration + convexity misses 11% of the loss. |
| 2008 (UST CMT year-end 2007 to year-end 2008, 30y -176bp) | +205,552 | +11.8% | Government duration pays in a flight to quality. Spreads here are static; real credit spreads widened and offset much of this for a credit book. |

Two episodes are discussed but not encoded as shift vectors:

* 2022 UK gilts and LDI: leveraged pension hedges met collateral calls as
  30y gilts moved about 130bp in days. DV01-matched is not
  liquidity-matched.
* 2023 SVB: long-duration assets funded by overnight deposits. The
  duration gap and the 200bp parallel shocks flag it. On the sample book,
  +/-100bp parallel gives +103,679 / -92,995 (+6.0% / -5.3% of MV).

## 4. Where this library hands off

* Auctions and supply: the 10y bucket and roll-down of the old vs new
  on-the-run set up concession trades. `data/live.py` loads FRED CMT
  yields as par-yield quotes.
* Futures and CTD: ladder DV01s convert to futures through the
  cheapest-to-deliver DV01 and conversion factor. The futures basis is not
  modelled.
* Callables and MBS: negative convexity; do not use these analytics
  (VALIDATION §4.5).

## 5. Controls

* Build control: 1e-10 repricing gate on every local-interpolation build.
* Risk control: KRD sum vs parallel DV01 (tolerances in VALIDATION §3).
* Extrapolation control: `ExtrapolationWarning` on every query past the
  last pillar.
* Model risk: interpolation is an explicit parameter with a measured
  forward impact (VALIDATION §4.1); single-curve discounting is the
  largest known limitation (METHODOLOGY §7).
* Calendars are partial (weekend plus fixed-date holidays).

## Round 3: rates products

Build the OIS curve first, then one projection curve per index tenor. Quote
caps and swaptions in normal vol when rates are near zero. Hull-White is for
callable and path-dependent work once a and sigma are calibrated; the
calibration to a swaption grid is not built yet.
