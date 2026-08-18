# Desk Guide — Running FX Relative Value with `fx_pairs`

How an FX relative-value pod would actually use this library: trade
expression, workflow, sizing, limits, P&L attribution, and the scenarios that
have hurt real books.

---

## 1. How the trades are actually expressed

**Forwards, not cash.** Nobody on an RV pod holds spot balances in two
currencies. A long AUDUSD / short beta·NZDUSD spread is executed as:

* two **FX forwards** (or spot rolled via tom-next swaps) with the pod's
  prime broker, netted under an ISDA/CSA;
* the deposit-rate differential of the model is realised as **forward
  points** on the daily roll — which is why the engine's carry ledger is
  swap-point-based (`(S−F)/S`), not an interest-accrual fiction;
* Wednesday's tom-next covers the weekend (T+2): the desk sees a **3-day
  swap on Wednesday**. The library books the same 3 days on Monday
  (documented simplification, METHODOLOGY §1); over any week the financing
  is identical.

**Prime brokerage & credit.** Positions consume PB credit lines, not cash:
margin is set on net open exposure with add-ons for EM legs; a pairs book
looks small in net USD delta but gross notional (both legs) is what the PB
margins and what the credit officer sees. Cross-currency basis (CIP
deviations, assumption A6) shows up as the difference between model carry and
the PB's actual swap points — reconcile monthly; feed market forward points
into the accrual for basis-heavy pairs (JPY, quarter-ends).

**Costs.** The engine's pip-spread table is indicative interbank: majors
0.5–1 pip, Scandies ~8–15, EM 30–150. Desk reality: spreads are time-of-day
(Wellington open ≈ 3–5× London), and stress multiplies them 5–20×. Re-run
the cost sensitivity (pipeline stage 5) with your PB's realised spreads
before believing any backtest.

## 2. Daily workflow

| When (London) | What | Library call |
|---|---|---|
| 06:45 | Marks, rate panel refresh, carry ledger posted to P&L system | `carry_ledger`, `BacktestResult.decomposition()` |
| 07:00 | Recompute spreads/z-scores with frozen formation params; RLS beta diagnostics vs frozen beta (>0.1 drift → flag) | `log_spread`, `zscore`, `RLSHedge` |
| 07:15 | Signal sheet: entries passing the carry filter, exits, stops, time-stops | `generate_positions`, `carry_entry_veto` |
| 07:30 | Risk: per-pair vol-target sizes, block exposures, regime monitor (40d rolling cross-block correlation) | `vol_target_scale` |
| Monthly | Re-run formation: EG re-fit, funnel re-selection, walk-forward health check | `walk_forward_backtest` |
| Quarterly | Model governance pack: cross-checks vs statsmodels, no-lookahead detector, identity tests (the CI suite is the evidence) | `pytest -q` |

**Who consumes the numbers.** PM (signal sheet + decomposition), risk
(vol-target sizes, block report, stress P&L), operations (roll amounts,
Wednesday 3-day swaps), management (monthly attribution: spot vs carry vs
costs — the decomposition is the anti-fooling device: a "mean reversion" pod
whose P&L is 80% carry is a carry pod with extra steps).

## 3. Sizing: by vol, never by notional

Positions are sized so each spread contributes equal *risk*:
`units = target_vol / realised_spread_vol` (`vol_target_scale`), capped at a
hard max leverage. The cap is not cosmetic — it is the SNB lesson: a pegged
spread's realised vol collapses, an uncapped vol targeter mechanically
maximises leverage at the moment of maximum event risk. Size EM and
policy-maintained spreads on **event scale** (assume a 10–20% gap), not OU
sigma.

## 4. Limits (suggested structure, per $100 of book vol budget)

| Limit | Level | Rationale |
|---|---|---|
| Per-pair vol contribution | ≤ 20% of book target vol | single-relationship risk |
| Per-block gross (commodity bloc / safe havens / EM) | ≤ 40% of gross | risk-off flips cross-block correlation from ~+0.1 to ~−0.6 (VALIDATION §5.4): a "diversified" book becomes one trade |
| EM gross share | ≤ 15%, event-sized | gap + cost regime |
| Hard z-stop | \|z\| ≥ 4, mandatory, no averaging down past it | discipline; but it does **not** protect against gaps (VALIDATION §5.1) |
| Time stop | 3× fitted half-life | a spread that hasn't reverted in 3 half-lives is telling you the model is wrong |
| Formation gate | trade only pairs whose current formation window passes EG at 10% and is non-degenerate | auto-deselects broken relationships |
| Peg/floor list | any pair whose spread vol < ~⅓ of its 2y median: reduce, don't add | vol collapse = policy maintenance = event risk |

**Carry book vs RV book separation.** Run the carry-driven P&L and the
spot-reversion P&L as separate books (the decomposition gives the split
daily). Reasons: (i) they have different risk factors — carry P&L dies in
risk-off exactly when spread P&L draws down (crowded-carry unwind), so
netting them hides the correlation; (ii) the carry filter creates a
systematic long-high-yielder tilt that must be visible to risk as carry
exposure, not disguised as "statistical arbitrage"; (iii) sign-flip pairs
(VALIDATION §4) are carry trades wearing an RV costume — the split exposes
them within a month.

## 5. Real-life scenarios

* **SNB floor, 2011–2015 → 15 Jan 2015.** EURCHF at the 1.20 floor was the
  highest-scoring 'mean reverter' in any G10 scan for three years — and the
  simulated replay (pipeline stage 6) shows the exact book shape: +0.058
  over 750 days at 0.75 hit rate, then −0.150 in one day, 2.6× cumulative
  gains, long-with-the-crowd, gap through every stop. Desk rule: spreads
  whose stationarity is *policy-maintained* go on the peg/floor list and are
  sized on depeg scale. Several real funds and retail brokers (Alpari,
  Everest) did not survive the day.
* **SNB interventions, Aug–Sep 2011.** The floor's *introduction* was also a
  gap (EURCHF +8% in minutes on 6 Sep 2011). Breaks hurt in both
  directions: a short-CHF-vs-EUR RV position built during the panic was
  gapped against too. Interventions are A1 violations whichever way they
  push.
* **Brexit, 24 Jun 2016.** GBP crosses gapped 8–10% overnight; GBP-block RV
  (EURGBP vs GBPUSD structures) saw hedge ratios estimated on pre-vote data
  fail simultaneously with 10× spreads. Walk-forward windows spanning the
  event de-select GBP pairs in the next formation — by design, *after* the
  loss. Pre-scheduled event risk ⇒ flatten or option-hedge; the daily model
  has no vote-risk input.
* **Risk-off JPY spikes (Oct 1998 USDJPY −15% in two days; Mar 2020; Aug
  2024 carry unwind).** Safe-haven legs gap richer while commodity legs gap
  cheaper — the two-block correlation flip, at gap speed. Any RV book short
  JPY/CHF as the 'rich' leg is implicitly short the risk-off factor; the
  block limit exists precisely for this, and the regime monitor (rolling
  cross-block correlation, tested detectable within ~40 days) is the early
  warning.

## 6. P&L attribution & model governance

Daily P&L is published as **spot + carry + costs** (identity-tested to
1e-18, so attribution always sums). Governance artefacts, all reproducible
offline: cross-validation vs statsmodels (VALIDATION §1), spurious-size
control, no-lookahead detector, walk-forward window-integrity tests, and the
seeded scenario suite (SNB, carry-flip, EM costs). Model changes require the
full suite green plus a re-run of `examples/run_pipeline.py` with the numbers
pasted into VALIDATION.md — the same discipline this repository follows.
