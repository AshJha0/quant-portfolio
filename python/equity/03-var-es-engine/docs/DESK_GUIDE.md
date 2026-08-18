# Desk Guide — How a Market-Risk Desk Uses This Engine

Contract item 5: daily workflow, consumers of the numbers, controls and
limits, governance, and real-life scenarios.

## 1. The overnight batch (T+0 close → T+1 open)

1. **Position feed** (post-close): positions → `Portfolio` with the factor
   mapping (equities → own factor; futures → index; options → underlier +
   implied-vol factor). Reconciliation break = the batch does not run;
   an unmapped position is a hard error (`Portfolio` raises on unknown
   factors), never silently dropped.
2. **Market data**: 500+ days of factor returns; EWMA (λ=0.94) covariance
   and vol forecasts refreshed (`ewma_covariance`, `ewma_volatility`).
3. **Risk run**: FHS VaR/ES 95/99 (primary), parametric normal + t
   (secondary/challenger), MC-t full reval on the option book, at 1d and
   10d. On the demo book this whole stack runs in ~90 s
   (`examples/run_pipeline.py`); a real book scales linearly in positions
   and MC paths.
4. **Backtest update**: yesterday's ex-ante VaR vs realised clean P&L →
   exception flag, rolling Kupiec/Christoffersen, traffic-light counter
   (`rolling_var_backtest`, `BacktestResult.summary`).
5. **Morning pack** (before the open): VaR/ES by method + method-disagreement
   table, limit utilisations, new exceptions, stress table, reverse-stress
   direction. The *disagreement* column matters: FHS ≪ parametric says
   "current vol is quiet relative to the window" (and vice versa) — that is
   information, not a bug (VALIDATION.md §3).

## 2. Limit monitoring

- Limits are set on **99 % 1d VaR and 97.5 % ES** per desk and top-of-house,
  with ES the binding FRTB measure. Utilisation = measure / limit.
- Breach protocol: same-day escalation to desk head + market risk manager;
  the desk cuts risk or obtains a documented temporary limit increase.
- The sensitivity ladders (`sensitivity_ladder`) tell the desk *which*
  factor to cut: on the demo book the SPX ladder is asymmetric
  (−$64k at +20 % vs +$182k at −20 %) because the book is net long equity
  with long index puts — cutting equity, not buying more puts, reduces VaR.
- VaR SEs are quoted with the number (MC: $51.0k ± $0.4k). A limit breach
  inside one SE is treated as a breach (conservatism), but flagged as
  statistically marginal.

## 3. Exception investigation & backtest governance

When realised loss < −VaR (an exception):

1. **Attribute it**: real market move vs data error vs unmapped risk
   (dirty P&L containing fees/new deals must be stripped — backtests run on
   clean P&L).
2. **Check clustering**: `exception_cluster_table` — gaps ≤ 5 days flag a
   regime the model is missing; formally, the Christoffersen independence
   p-value. A cluster with acceptable *count* still fails conditional
   coverage (yellow flag on the model, not the desk).
3. **Quarterly model review**: Kupiec + CC p-values per method; a method
   failing two consecutive quarters is remediated (typical fix: unconditional
   → FHS, exactly the VALIDATION.md §4 story) or replaced. Changes go
   through model-risk governance with the backtest evidence attached.

## 4. Regulatory capital link (Basel traffic light)

Market-risk capital ≈ `k × VaR₉₉,₁₀d` (internal-models approach), where the
multiplier `k = 3.0 + add-on` comes from the 250-day exception count:

| exceptions | zone | k |
|---|---|---|
| 0–4 | green | 3.00 |
| 5 | yellow | 3.40 |
| 6 | yellow | 3.50 |
| 7 | yellow | 3.65 |
| 8 | yellow | 3.75 |
| 9 | yellow | 3.85 |
| 10+ | red | 4.00 + presumption of model rejection |

A red zone is a ~33 % capital increase *and* a supervisory finding — on the
GARCH stress period the unconditional models land red (13/250, 11/250)
while FHS stays green (3/250): the capital cost of a lazy model is
measurable (pipeline §Basel). `basel_zone_probabilities` reproduces the
exact binomial table for supervisors' "how likely was this under a correct
model" question (green ≈ 89.2 %, red ≈ 0.03 %).

## 5. Stress committee usage

Monthly (weekly in volatile regimes) the stress committee reviews:

- **Historical replays** (`HISTORICAL_SCENARIOS` — approximate published
  moves, clearly labelled as approximations): 1987 (−20.5 % day, vol
  explosion), 2008 Lehman fortnight (−25 %, VIX 25→80), 2020 COVID (−34 %,
  VIX 15→83). Demo book: −$134k / −$198k / −$254k full-reval — 2.5–4.7×
  the 99 % VaR, which is the committee's headline ("VaR is not a worst
  case").
- **Hypothetical combos** (rate+equity+vol): the risk-off combo (−15 %
  equity, +25 vol pts) and the **melt-up** (+10 %, −5 pts) — the latter
  catches short-upside books that crash scenarios miss (demo book *makes*
  $153k there; a short-call overlay would show the pain).
- **Full reval vs delta-gamma columns side by side**: the divergence
  ($93k in the COVID row) is the committee's evidence that Greeks-based
  intraday risk is unsafe for gap moves and the overnight batch must fully
  revalue.
- **Reverse stress** (`reverse_stress_delta`/`_delta_gamma`): "what joint
  3σ move hurts most?" Demo book: AAPL −4.7 %, JPM −3.6 %, SPX −1.9 %,
  IV +2.1 pts → −$60.4k. The direction (not just the number) is actionable:
  it names the concentration. The delta-gamma optimiser confirms the long
  puts trim the worst case (−$59.8k) — and would flag the reverse for a
  short-gamma book.

## 6. Realistic scenarios and how the desk reads them

- **2008 replay**: correlations → 1 and vol +55 pts; the book's equity
  diversification (AAPL/JPM corr 0.5) is assumed away — stress ignores the
  covariance matrix deliberately (METHODOLOGY.md A9).
- **2020 COVID**: fastest −34 % ever; the delta-gamma error row shows why
  gap risk must be full-reval; the vol factor +68 pts dominates the option
  P&L.
- **Meme-stock concentration**: a single-name factor with +100 %/−60 %
  two-sided shocks via `shocks_by_name` overrides; a short single-name
  position shows unbounded upside loss — VaR (calibrated on pre-squeeze
  vol) is useless, the ladder is the control. This is assumption A1's
  failure mode: names with squeeze risk get their own factor and scenario.
- **Single-name gap risk**: overnight −30 % gap (M&A break, fraud,
  accounting restatement) on the largest single name via `shocks_by_name`:
  hits the equity position's full delta with zero diversification offset.
  Daily-return covariances cannot see gaps; the stress table can.

## 7. Who consumes what

| Consumer | Numbers | Frequency |
|---|---|---|
| Desk head | VaR/ES vs limits, ladders, exceptions | daily, pre-open |
| CRO / risk committee | top-of-house VaR/ES, traffic light, stress table | daily / weekly |
| Regulator | 99 % VaR + ES₉₇.₅, 250-day exceptions, multiplier | quarterly + ad hoc |
| Model validation | backtest p-values, method disagreement, CF domain flags | quarterly |
| Finance / capital | k × VaR₁₀d capital number | monthly |
