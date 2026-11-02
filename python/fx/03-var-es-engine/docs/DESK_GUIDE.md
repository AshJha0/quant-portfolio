# Desk Guide: Running FX Market Risk with this Engine

Contract items **5** (how a desk uses it) and **6** (real-life scenarios)
of `CONVENTIONS.md`. Written for the market-risk manager covering an FX
trading business and the treasury/corporate hedging desk it serves.

All calls below use `alpha` as the tail probability: 99 % VaR is
`alpha=0.01`, 95 % VaR is `alpha=0.05`.

---

## 1. The FX risk day: a 24-hour, three-hub cycle

FX never closes. Risk runs on a follow-the-sun handover. The *official
close* is a convention, not a fact of nature.

| Time (NY) | Event | Engine touchpoint |
|---|---|---|
| 17:00 | **Official close.** 5pm NY is the standard FX "end of day". The WM/Refinitiv 4pm London fix is the benchmark for many mandates. Positions are snapped and `Market` is built from official closing spots, points and vols. | `demo_market()`-style snapshot. `Book` frozen for the EOD run. |
| 17:30 | EOD batch appends factor returns. NaN policy: a missing fixing **fails the run loudly**. Tokyo and São Paulo holidays do not line up. The desk decides fill or drop explicitly. The engine never imputes. | `validate_returns` |
| 18:00 | VaR/ES for all methods at alpha 0.05 and 0.01, 1d and 10d. Stress table including peg add-ons. Backtest updated with yesterday's *hypothetical* (static-book) P&L. | `historical_var` (FHS headline), `parametric_var`, `monte_carlo_var`, `run_stress`, `rolling_backtest` |
| 19:00 | Risk report to desk heads and the CRO: VaR vs limits per currency pair and tenor bucket, traffic-light status, method-disagreement flags. An HS vs parametric gap above 20 % is a tail-shape alert. | result objects, `basel_traffic_light` |
| 03:00 | London morning: intraday what-ifs on quotes ("client wants 500 EURJPY 6m forward: marginal VaR?"). Parametric exposures answer in seconds. | `linear_exposures`, `var_covar` |
| 08:00 to 17:00 | NY session: intraday limit monitoring on the parametric proxy. Full revaluation re-runs only on big book moves. | `parametric_var` |

**Handover rule**: Tokyo, London and NY each inherit the same factor
convention (`FX:CCY` vs USD). A JPY book risk-transferred to London nets
correctly. There is no "Tokyo EURJPY factor" to reconcile.

## 2. Netting across desks

Every position (spot, cross, forward leg, option delta) maps to the same
USD-pivot factors. Firm-level risk is therefore
`Book(desk1.positions + desk2.positions + ...)`:

* the G10 spot desk's short EURUSD nets against the forwards desk's long
  EUR deposit leg;
* the options desk's EURUSD delta nets against both;
* what *cannot* net (vega, rate legs, peg jump exposure) shows up in its own
  factor family instead of being hidden inside a pair-level number.

Diversification is reported as `ΣVaR_desk − VaR_firm`. ES, not VaR, is used
for capital allocation between desks because it adds up coherently. The
engine's peg counterexample (VALIDATION §2) is the one-slide explanation:
the desk-level 99 % VaRs of two peg books can both be zero while the firm
carries a full devaluation risk.

A book with no positions raises `ValueError` ("book has no positions"). A
book holding only base-currency cash raises `ValueError` ("no risk
factors"). Neither returns a zero VaR.

## 3. Limits: per currency pair, per tenor, per risk type

A limit sheet this engine feeds directly:

| Limit | Measure | Example level | Engine source |
|---|---|---|---|
| Firm FX 99 %/1d ES | FHS ES at alpha 0.01 | $2.5m | `historical_var(..., alpha=0.01).es` |
| Per-pair delta | \|`linear_exposures["FX:CCY"]`\| | $25m EUR, $10m EM ccy | `linear_exposures` |
| Tenor bucket DV01 | `IR:CCY` exposure ×1e-4 per bucket | $25k/bp G10, $10k/bp EM | forward rate legs |
| Vega per pair | `VOL:PAIR` exposure ×0.01 | $150k/vol pt | option positions |
| Peg exposure | notional in flagged ccys | $50m HKD-bloc | `PegBlindnessWarning` + factor scan |
| Stress loss | worst of scenario table | $20m | `run_stress` |
| EM jump VaR | jump-mixture MC at alpha 0.01 | $5m | `monte_carlo_var(dist="jump", alpha=0.01)` |

Peg exposure gets its **own notional limit** because it does not consume
VaR limit. That asymmetry is the main lesson of January 2015.

## 4. Backtest exception governance

Daily: yesterday's static-book P&L vs yesterday's 99 % VaR (alpha 0.01). On
an exception:

1. **Same day**: the risk manager annotates the driver: market move, new
   trade or data error. Split actual from hypothetical P&L. A fee or a
   late-booked trade is not a model exception.
2. **T+1**: write an exception memo if the loss exceeds 1.2× VaR. It covers
   which factors moved, whether the move was in the scenario set, and
   whether FHS and parametric disagreed beforehand.
3. **Rolling 250d**: Kupiec, Christoffersen and the traffic light are
   recomputed. **Clustered** exceptions (independence p < 5 %) trigger a
   model review even in the green zone. Clustering means the model is slow,
   not unlucky. The engine's GARCH demo shows this failure shape.
4. **Yellow zone** (5 to 9 exceptions in 250 days): the capital multiplier
   add-on applies automatically. Call
   `basel_traffic_light(n_obs=..., n_exceptions=..., alpha=0.01)`. It works
   for any window and alpha. The yellow multipliers 3.40 to 3.85 come from
   the 250-day equivalent count. The result carries `.multiplier` and
   `.plus_factor`. This is a model-risk committee item.
5. **Red zone** (10 or more): multiplier 4.00, model use suspended for
   capital, remediation plan to the regulator. In the synthetic pipeline
   demo (`python examples/run_pipeline.py`), parametric-normal on
   regime-switching data has 14 exceptions in 500 days and lands red at
   4.00. Step 3 would have moved the desk onto FHS months earlier.

Quarterly: Acerbi–Szekely ES backtest on the headline ES model, run as
`acerbi_szekely_z2(pnl, var, es, alpha=0.01)`. It uses the paper's sign: a
negative Z2 means ES is understated.

## 5. Capital

Regulatory market-risk capital ≈ `multiplier × max(VaR_t−1, avg60(VaR))`.
This is the 1996 MRA shape. FRTB replaces VaR with 97.5 % ES (alpha 0.025).
The engine reports both, which is why every method returns the pair. The
multiplier comes from `basel_traffic_light(...).multiplier`. One red zone
costs about 33 % more capital on the whole book. The cost of a bad model is
the multiplier, not the exceptions.

## 6. Treasury / corporate hedging use case

A EUR-based corporate treasury holding USD revenue is the mirror image of a
trading book. The engine handles it with `Book(base="EUR")`:

* USD receivables are `Cash("USD", ...)`. The engine reports pure
  translation risk. Tests confirm foreign cash carries risk. A book of
  base-currency cash alone has no risk factors and raises `ValueError`;
* the hedge is a short `Forward("USDEUR"...)`. Hedged VaR collapses to the
  **rate-leg residual** (forward points), which the CIP decomposition
  isolates. The treasurer sees why "fully hedged" still shows a small VaR;
* hedge-ratio what-ifs use `sensitivity_ladder` over the `FX:USD`-equivalent
  exposure. The CFO gets 10-day numbers at alpha 0.05 instead of 1-day at
  alpha 0.01.

## 7. Real-life scenarios the desk should rehearse (all runnable)

* **Brexit night (24 Jun 2016)**: `historical_scenarios()["brexit_2016"]`.
  Cable fell 8.1 % with a vol spike. The lesson is the *joint* move: for many
  books short GBP vega hurt more than spot delta. Check option books under
  spot −8 % *and* vol +12pts together, not one after the other.
* **SNB floor removal (15 Jan 2015)**: `["chf_depeg_2015"]` plus
  `peg_break_scenario("CHF", +0.15)`. The lesson is F1 (VALIDATION). The
  window contained nothing, and only the standing stress add-on saw it. Any
  managed currency on the book today (HKD band, GCC pegs) inherits this
  scenario by policy, not by choice.
* **JPY carry unwind (Oct 1998, echoed Oct 2008 and Jul–Aug 2024)**:
  `["jpy_1998"]`. Correlation flips. JPY shorts and long-EM books lose
  *together* at the moment the calm covariance called them diversified.
  Cross-check parametric VaR under `default_correlation(regime="stress")`.
* **MoF/BoJ intervention (Sep–Oct 2022)**: USDJPY dropped 5.5 big figures
  in minutes, twice. Run `sensitivity_ladder(book, market, "FX:JPY")` in
  both directions. Intervention risk is *two-sided* around a policy level,
  unlike a peg.
* **EM devaluation wave**: `["em_crisis"]` plus `peg_break_scenario` with
  contagion (e.g. TRY −25 % dragging ZAR −8 %), plus jump-mixture MC for the
  quantile view. Governance rule: EM 99 % VaR (alpha 0.01) is quoted from
  t or jump MC, never normal MC. On the synthetic EM book, normal MC
  underestimates 99 % VaR by 12 % vs t(5) and by 138 % vs the jump mixture.
* **USD broad move**: `usd_broad_move(±10%)` gives the whole-book dollar
  beta. It is the first number the CIO asks for.

## 8. Model governance summary

* **Owner**: market risk. **Users**: trading desks, treasury, capital.
* **Champion model**: FHS (headline VaR/ES). **Challengers**: parametric
  normal/t (speed, decomposition) and MC t/jump (EM and peg overlays). The
  daily report prints all of them. A champion–challenger gap above 20 % is
  a standing agenda item.
* **Change control**: scenario calibrations (§6 of METHODOLOGY) and the peg
  policy are versioned constants. The peg threshold is
  `PEG_VOL_THRESHOLD = 5e-4` daily (0.05 % daily vol). The seeded test
  suite is the regression harness. Any change that shifts a validated
  number fails a test before it reaches production.
* **Cross-language contract**: the C++ (`cpp/fx-var-engine`) and Rust
  (`rust/fx-var-engine`) engines are linear-book ports. They revalue spot,
  forward and cash positions, not options. They assert golden constants
  generated from this package by `tests/golden/generate_golden.py`.
  `tests/test_golden_vectors.py` checks the committed JSON against a fresh
  build to a 16-ulp bound. It also checks the shared `conv_*` convention
  values against independent hand computations. A refactor here fails here,
  not in another language's CI. Regenerating the constants requires
  sign-off.
* **Data-quality gate (why the engine refuses NaN).** Every risk number this
  desk publishes feeds a limit check, a traffic light or a capital
  multiplier. **None of those fire on NaN.** `nan > limit` is False, a NaN
  VaR colours no zone, and a NaN report cell looks like a formatting bug.
  The engine therefore refuses and never imputes at every boundary: market
  snapshots, factor histories, position fields, scenario shocks,
  distribution parameters and horizons (VALIDATION §4.1). A stale fixing or
  a failed calibration stops the batch with a named error. It does not
  produce a clean-looking report that understates risk. If a desk wants a
  run to proceed on partial data, the substitution must be made explicitly
  upstream and recorded.

## Round 3: which tool for which question

* "Which trades drive our VaR?" `risk_attribution.parametric_attribution`
  for a quick read, `simulation_attribution` on the scenario P&L matrix for
  the number you report. Components add to the total.
* "What does this hedge do to our ES?" `parametric_incremental` or
  `simulation_incremental`; the latter recomputes, it does not linearise.
* "Are we still on IMA?" `frtb_ima.desk_backtest` (run on actual and on
  hypothetical P&L) and `frtb_ima.pla_test`.
* "What is our 99.9 % loss?" Do not read it off 500 historical days: fit
  `evt.fit_pot` and use `tail.var(0.001)`. Check the mean-excess plot first.
* "Our options book under the historical scenarios?" `revaluation.full_revaluation_pnl`;
  use `taylor_pnl(order="delta_gamma")` only as a fast check and quote the gap.
* "Why did we make money yesterday?" `revaluation.pnl_explain`. A large
  `unexplained` means a missing risk factor or a big move.
* Limits: build a `limits.Limit` tree once; `check_limits` each day; pass the
  aggregated VaR for parents rather than letting it sum the children.
