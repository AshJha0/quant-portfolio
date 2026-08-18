# Desk Guide — Running FX Market Risk with this Engine

Contract items **5** (how a real desk uses it) and **6** (real-life
scenarios) of `CONVENTIONS.md`. Written for the market-risk manager covering
an FX trading business and the treasury/corporate hedging desk it serves.

---

## 1. The FX risk day: a 24-hour, three-hub cycle

FX never closes; risk runs on a follow-the-sun handover, and the *official
close* is a convention, not a fact of nature.

| Time (NY) | Event | Engine touchpoint |
|---|---|---|
| 17:00 | **Official close** (5pm NY = the standard FX "end of day"; WM/Refinitiv 4pm London fix is the benchmark for many mandates). Positions snapped, `Market` built from official closing spots/points/vols. | `demo_market()`-style snapshot; `Book` frozen for EOD run. |
| 17:30 | EOD batch: factor returns appended (NaN policy: a missing fixing **fails the run loudly** — Tokyo and São Paulo holidays do not line up; the desk decides fill-vs-drop explicitly, the engine never imputes). | `validate_returns` |
| 18:00 | VaR/ES all methods, 95/99, 1d/10d; stress table incl. peg add-ons; backtest updated with yesterday's *hypothetical* (static-book) P&L. | `historical_var` (FHS headline), `parametric_var`, `monte_carlo_var`, `run_stress`, `rolling_backtest` |
| 19:00 | Risk report to desk heads + CRO: VaR vs limits per currency pair and tenor bucket, traffic-light status, method-disagreement flags (HS vs parametric gap >20% = tail-shape alert). | result objects, `basel_traffic_light` |
| 03:00 | London morning: intraday what-ifs on quotes ("client wants 500 EURJPY 6m forward — marginal VaR?") — parametric exposures, seconds. | `linear_exposures`, `var_covar` |
| 08:00–17:00 | NY session: limit monitoring intraday on the parametric proxy; full reval re-run only on big book moves. | `parametric_var` |

**Handover rule**: Tokyo → London → NY each inherit the same factor
convention (`FX:CCY` vs USD), so a JPY book risk-transferred to London nets
correctly — there is no "Tokyo EURJPY factor" to reconcile.

## 2. Netting across desks

Because every position — spot, cross, forward leg, option delta — maps to
the same USD-pivot factors, firm-level risk is literally
`Book(desk1.positions + desk2.positions + ...)`:

* the G10 spot desk's short EURUSD nets against the forwards desk's long
  EUR deposit leg;
* the options desk's EURUSD delta nets against both;
* what *cannot* net (vega, rate legs, peg jump exposure) shows up in its own
  factor family instead of being hidden inside a pair-level number.

Diversification is reported as `ΣVaR_desk − VaR_firm`; ES (not VaR) is used
for capital allocation between desks because it adds up coherently — the
engine's peg counterexample (VALIDATION §2) is the one-slide explanation of
why the desk-level 99% VaRs of two peg books can both be zero while the firm
is carrying a full devaluation risk.

## 3. Limits: per currency pair, per tenor, per risk type

A realistic limit sheet this engine feeds directly:

| Limit | Measure | Example level | Engine source |
|---|---|---|---|
| Firm FX 99%/1d ES | FHS ES | $2.5m | `historical_var(...).es` |
| Per-pair delta | \|`linear_exposures["FX:CCY"]`\| | $25m EUR, $10m EM ccy | `linear_exposures` |
| Tenor bucket DV01 | `IR:CCY` exposure ×1e-4 per bucket | $25k/bp G10, $10k/bp EM | forward rate legs |
| Vega per pair | `VOL:PAIR` exposure ×0.01 | $150k/vol pt | option positions |
| Peg exposure | notional in flagged ccys | $50m HKD-bloc | `PegBlindnessWarning` + factor scan |
| Stress loss | worst of scenario table | $20m | `run_stress` |
| EM jump VaR | jump-mixture MC 99% | $5m | `monte_carlo_var(dist="jump")` |

Peg exposure gets its **own notional limit** precisely because it does not
consume VaR limit — that asymmetry is the whole lesson of January 2015.

## 4. Backtest exception governance

Daily: yesterday's static-book P&L vs yesterday's 99% VaR. On an exception:

1. **Same day**: risk manager annotates the driver (market move vs new
   trade vs data error). Actual-vs-hypothetical P&L split — a fee or a
   booked-late trade is not a model exception.
2. **T+1**: exception memo if the loss > 1.2× VaR: which factor(s), was it
   in the scenario set, did FHS/parametric disagree beforehand.
3. **Rolling 250d**: Kupiec + Christoffersen + traffic light recomputed.
   **Clustered** exceptions (independence p < 5%) trigger a model review
   even in the green zone — clustering means the model is slow, not unlucky
   (the engine's GARCH demo shows exactly this failure shape).
4. **Yellow zone** (5–9): capital multiplier add-on applied automatically
   (3.40→3.85 by count — `basel_traffic_light().multiplier`); model-risk
   committee item.
5. **Red zone** (≥10): multiplier 4.0, model use suspended for capital,
   remediation plan to the regulator. In the pipeline demo,
   parametric-normal on regime-switching data lands red with 14/500 —
   the desk would have been forced onto FHS months earlier by step 3.

Quarterly: Acerbi–Szekely ES backtest on the headline ES model.

## 5. Capital

Regulatory market-risk capital ≈ `multiplier × max(VaR_t−1, avg60(VaR)) `
(1996 MRA shape; FRTB replaces VaR with 97.5% ES — the engine reports both,
which is why every method returns the pair). The multiplier comes straight
from `basel_traffic_light`; the desk-level story: one red zone costs ~33%
more capital on the whole book — the cost of a bad model is not the
exceptions, it is the multiplier.

## 6. Treasury / corporate hedging use case

A EUR-based corporate treasury holding USD revenue is the mirror image of a
trading book, and the engine handles it with `Book(base="EUR")`:

* USD receivables = `Cash("USD", ...)` — the engine reports pure
  translation risk (tested: base-ccy cash is riskless, foreign cash is not);
* the hedge = short `Forward("USDEUR"...)` — hedged VaR collapses to the
  **rate-leg residual** (forward points), which is exactly what the CIP
  decomposition isolates; the treasurer sees why "fully hedged" still shows
  a small VaR;
* hedge-ratio what-ifs: `sensitivity_ladder` over `FX:USD`-equivalent
  exposure, and 10-day/95% numbers for the CFO instead of 1-day/99%.

## 7. Real-life scenarios the desk should rehearse (all runnable)

* **Brexit night (24 Jun 2016)** — `historical_scenarios()["brexit_2016"]`.
  Cable −8.1% with a vol spike: the desk lesson is the *joint* move — short
  GBP vega hurt more than spot delta for many books. Check: option books
  under spot −8% *and* vol +12pts, not sequentially.
* **SNB floor removal (15 Jan 2015)** — `["chf_depeg_2015"]` plus
  `peg_break_scenario("CHF", +0.15)`. The lesson is F1 (VALIDATION): the
  window contained nothing; only the standing stress add-on saw it. Any
  managed currency on the book today (HKD band, GCC pegs) inherits this
  scenario by policy, not by choice.
* **JPY carry unwind (Oct 1998, echoed Oct 2008 and Jul–Aug 2024)** —
  `["jpy_1998"]`. Correlation flip: JPY shorts and long-EM books lose
  *together* precisely when they were "diversified" in the calm covariance.
  Cross-check parametric VaR under `default_correlation(regime="stress")`.
* **MoF/BoJ intervention (Sep–Oct 2022)** — USDJPY −5.5 big figures in
  minutes, twice. Run `sensitivity_ladder(book, market, "FX:JPY")` both
  directions: intervention risk is *two-sided* around a policy level, unlike
  a peg.
* **EM devaluation wave** — `["em_crisis"]` + `peg_break_scenario` with
  contagion (e.g. TRY −25% dragging ZAR −8%) + jump-mixture MC for the
  quantile view. Governance rule: EM 99% VaR is quoted from t/jump MC, never
  normal MC (the demo shows normal underestimates by 12–140%).
* **USD broad move** — `usd_broad_move(±10%)`: the whole-book dollar beta,
  the single number the CIO asks for first.

## 8. Model governance summary

* **Owner**: market risk; **users**: trading desks, treasury, capital.
* **Champion model**: FHS (headline VaR/ES); **challengers**: parametric
  normal/t (speed, decomposition), MC t/jump (EM and peg overlays). The
  daily report prints all of them; a >20% champion–challenger gap is a
  standing agenda item.
* **Change control**: scenario calibrations (§6 of METHODOLOGY) and peg
  policy (threshold 0.05% daily vol) are versioned constants; the seeded
  test suite is the regression harness — any change that shifts a validated
  number fails a test before it reaches production.
