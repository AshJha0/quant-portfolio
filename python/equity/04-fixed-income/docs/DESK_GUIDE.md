# Desk Guide — how a rates desk would use `fi_rates`

## 1. Daily workflow

**07:00 — curve build.** Overnight closes (deposits, swap par rates; or bond
prices for a govie desk) go through `bootstrap_curve` /
`bootstrap_bond_curve`. The build is validated automatically: every input
must reprice to 1e-10 (`reprice_instruments`) or the build fails loudly —
a desk never wants a curve that silently misses its own hedge instruments.
In production this is the single-currency analogue of the multi-curve build
(OIS discounting + projection curves; see METHODOLOGY assumption 1).

**07:15 — EOD risk & PV.** `portfolio_risk` produces the position-level and
aggregate report: market value, YTM, modified duration, convexity, DV01.
Sample output (7-position govt+corp book, ~$1.74mm MV):

```
              mv        weight   ytm    mod_dur  convexity   dv01
GOVT_2Y       498,270   0.286   3.44%    1.92       4.7       97.3
GOVT_10Y      295,929   0.170   4.17%    8.15      78.5      245.9
GOVT_30Y      151,746   0.087   4.48%   16.19     378.7      247.5
CORP_BBB_7Y   133,460   0.077   6.33%    5.73      40.1       78.8
...
TOTAL       1,740,004   1.000   4.45%    5.42      60.9      980.2
```

Consumers: the trader (position DV01s vs limits), the risk manager
(aggregate DV01/KRD vs desk limits), product control (PV for P&L
attribution: carry + roll + curve + spread), and ALM/treasury (duration gap).

**07:30 — hedging off the DV01 ladder.** The KRD report is the hedging
sheet. `krd_report` on the sample book:

```
       key_rate_dv01   key_rate_duration
 2y        154.3            0.89
 5y        278.0            1.60
10y        368.4            2.12
30y        179.5            1.03
SUM        980.2            5.63   (= parallel DV01 to 5e-8 rel)
```

To flatten: sell ~154 DV01 of 2y futures/swaps, ~278 of 5y, ~368 of 10y,
~180 of 30y (bond futures hedge in CTD-DV01 terms — see §4). Hedging the
*total* 980 DV01 with a single 10y instrument leaves the book flat to
parallel moves but fully exposed to twists — exactly the failure quantified
in VALIDATION §4.3.

## 2. Curve trades (KRD-based)

A **2s10s steepener** (long 2y, short 10y, DV01-neutral) is expressed
directly in the KRD ladder: target +X DV01 at the 2y key, −X at 10y, ~0
elsewhere. `steepener_scenario` / `butterfly_scenario` then quantify the
P&L: the pipeline's ±50bp tilt scenarios show a long-duration book losing
in a steepener and gaining in a flattener with the correct signs
(unit-tested), while the 2y book does the opposite. Butterfly (belly vs
wings) trades are monitored with `butterfly_scenario` and the 5y/10y key
rates. Entry/exit levels come from carry & roll: `carry_rolldown` prices
what the trade earns if *nothing* happens — on the sample upward curve, the
5y point earns 3.50 carry + 0.62 roll ≈ 4.12 per 100 face over 1y, so a
steepener financed at the front end has positive carry, which is why "the
carry trade" and "the curve trade" are the same conversation.

## 3. Scenario analysis and what each episode teaches

`scenario_pnl_table` runs full revaluation (curve rebuilt per scenario, no
Taylor shortcut) against the historical library. Sample book results:

| Episode (approximate encoded magnitudes) | P&L | Lesson |
|---|---|---|
| **2013 taper tantrum** (10y +130bp, front end pinned) | −94,035 (−5.4%) | Duration risk concentrates at the long key rates when the front end is anchored — a "hedged" book with net-zero DV01 but long 10y/30y KRDs still lost heavily. Check the *ladder*, not the total. |
| **2022 hiking cycle** (2y +370bp, 10y +235bp, bear flattener) | −228,401 (−13.1%) | The worst year in bond-index history was a *flattener*: short-end KRDs, usually ignored, dominated. Also: the dur+conv estimate misses 13% (VALIDATION §4.3) — scenario P&L must be full revaluation. |
| **2008 GFC** (2y −250bp, bull steepener) | +167,996 (+9.7%) | Flight-to-quality: govt duration is the hedge that pays out in credit crises — but this engine's z-spreads are *static*; in reality corp spreads blew out and ate much of the govt rally for a credit book. Assumption 6 caveat applies. |

Two further real-world episodes are documented for the risk conversation,
though they are not encoded as shift vectors:

* **2022 UK gilts/LDI spiral**: leveraged pension funds hedged long-dated
  liabilities with repo'd gilts; a 130bp 30y move in days triggered
  collateral calls → forced gilt sales → further yield spikes. Lesson:
  DV01-matched is not liquidity-matched; scenario analysis must include the
  *funding* consequences of the mark, and long-end extrapolation/gap risk
  (VALIDATION §4.2) is not academic.
* **2023 SVB**: a held-to-maturity book of long-duration MBS/Treasuries
  funded by overnight-withdrawable deposits — a textbook ALM duration
  mismatch (asset duration ~6y, liability duration ~0). The `portfolio_risk`
  duration gap and the ±200bp parallel scenarios (the standard regulatory
  IRRBB shocks) are precisely the report that would have flagged it:
  ±100bp full-revaluation P&L on the sample book is +103,677/−92,993
  (+6.0%/−5.3% of MV) — scale to ±200bp and a balance sheet levered 10:1 on
  equity and the mismatch is existential.

## 4. Adjacent desk topics (where this library hands off)

* **Auctions / supply**: ahead of a 10y auction the desk checks the 10y KRD
  bucket and the roll-down of the current vs new on-the-run; concession
  trades are set up as KRD-neutral switches. CMT/auction data would come in
  via `data/live.py` (FRED CMT loader, treated as par-yield `ParSwap`
  quotes — approximation documented there).
* **Futures & CTD**: bond-futures hedges convert ladder DV01s via the
  cheapest-to-deliver's DV01 / conversion factor; the futures basis (net
  basis, delivery optionality) is its own model and is out of scope here —
  the ladder outputs are the *inputs* to that calculation.
* **Callables/MBS**: negative convexity — do **not** use these analytics
  (VALIDATION §4.5); route to an OAS model.

## 5. Controls and model governance

* **Build control**: 1e-10 repricing gate on every curve build (tested).
* **Risk control**: KRD-sum-vs-parallel-DV01 reconciliation (tolerance
  documented in VALIDATION §3) catches bump-engine drift.
* **Extrapolation control**: `ExtrapolationWarning` on any query past the
  last pillar; warnings are errors in the CI test profile except where
  explicitly expected.
* **Model risk**: interpolation choice is an explicit parameter with a
  documented P&L impact (the sawtooth table); the single-curve
  simplification is the largest known model limitation and is flagged with
  its production replacement (multi-curve OIS) in METHODOLOGY §5.1.
* **Reproducibility**: every number in these docs regenerates from
  `examples/run_pipeline.py` with fixed seeds.
