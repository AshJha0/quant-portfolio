# Methodology: FX Statistical Pairs / Cross-Rate Mean Reversion

This document states why this model was chosen, against what alternatives,
and under which assumptions (with what breaks when each assumption fails).
Every number in this project comes from synthetic data. The planted pair is
cointegrated by construction, so the profit it shows is put there by the
generator. The only real-data result is the training-period run of the
signed `pairs-fx-g10` pre-registration in `research/` (not out-of-sample
evidence). Validation evidence lives in `VALIDATION.md`; desk usage in
`DESK_GUIDE.md`.

---

## 1. The trade and the objects being modelled

The strategy trades **pairs of currency pairs**: long 1 unit of pair 1,
short `beta` units of pair 2, e.g. AUDUSD vs NZDUSD, EURUSD vs GBPUSD, or
NOKSEK-style relative value expressed through the USD legs. The tradable
object is the **log-rate spread**

```
s_t = log P1_t − alpha − beta · log P2_t
```

Log rates (not levels) because FX P&L is naturally multiplicative, because the
hedge ratio then has a clean elasticity interpretation, and because cross
rates are exact *ratios* of USD legs, in logs, exact *differences*, so
triangular consistency is linear and testable.

Internally every currency is a **USD leg** (CCYUSD, USD per 1 unit of
currency; USD ≡ 1). Any cross is `BASEQUOTE = BASEUSD / QUOTEUSD`, an exact
no-arbitrage identity, which gives us a free machine-checkable null case
(Section 6).

### Total return, not spot

A held FX position is financed: long BASE/QUOTE owns the base currency
(earns its deposit rate) funded in the quote currency (pays its rate). In
practice the position lives in forwards/swaps and financing appears as
**forward points** rolled daily. Backtest P&L is therefore

```
total = spot P&L + carry accrual − transaction costs
```

with carry from covered interest parity, `F(τ) = S(1 + r_q τ)/(1 + r_b τ)`,
daily roll yield `(S − F)/S = (r_b − r_q)τ/(1 + r_b τ)` (swap-point form; the
linear form `(r_b − r_q)τ` is also implemented and agrees to O(τ²)). A
mean-reversion signal computed on spot alone can be systematically
**wrong-carry**, the pipeline constructs an explicit example where including
carry flips the sign of the strategy's P&L (see `VALIDATION.md` §4).

**The T+2 / Wednesday wrinkle.** Real tom-next rolls settle T+2, so the
3-day weekend financing is charged on **Wednesday** (Wednesday's tom-next
spans Saturday–Monday). We accrue by actual calendar-day gaps ACT/365F, which
books the 3 days on Monday instead. Total accrual over any horizon is
identical; only the intra-week booking date shifts by two business days. This
simplification is unit-tested (`test_carry.py`). The equity and FX regime
projects use the same ACT/365F calendar-gap day count for carry.

---

## 2. Why Engle–Granger + OU (model choice vs alternatives)

### Cointegration test: Engle–Granger, from scratch

Chosen: two-step Engle–Granger, OLS `log P1 = alpha + beta log P2 + u`,
then an ADF test (no deterministic terms) on `u` against **MacKinnon (2010)
N=2** critical values.

| Alternative | Why not (here) |
|---|---|
| **Johansen (VECM, ML)** | Estimates the full cointegration rank and is superior for 3+ assets, but for a **pair** it answers the same question with more machinery, is less transparent about *which* regression produced the spread, and its small-sample size distortions are harder to reason about. EG's single OLS beta is also exactly the hedge ratio the desk executes. Johansen is the right upgrade for basket trades (e.g. EUR-block baskets), noted as future work. |
| **Distance method (sum of squared normalized price gaps, Gatev et al.)** | Model-free and robust, but it has no notion of a hedge ratio other than 1:1, no stationarity test (so it happily selects the correlated-but-not-cointegrated pairs our funnel must reject), and no parameter (half-life) to size holding periods with. |
| **Kalman-filter latent-spread models** | Adaptive, but the fully latent version is unidentified without strong priors and easy to overfit. We take the useful half: a **recursive-least-squares hedge ratio with forgetting factor** (`RLSHedge`), which is the Kalman filter's regression limit, adaptivity where it matters, no latent-state machinery. |

Two details that are easy to get wrong and are therefore tested:

1. **Residual-based critical values.** The ADF is run on *estimated*
   residuals; using plain N=1 ADF critical values over-rejects. We use the
   MacKinnon (2010) response surface for N=2 (asymptotic −3.34 at 5% vs
   −2.86 for plain ADF). The test suite asserts the two tables differ and
   that our EG statistic matches `statsmodels.coint` to ~1e-15.
2. **Spurious regression size control.** On 200 seeded independent random
   walks the 5% EG test rejects 4.5% of the time, the funnel's false-positive
   rate is the nominal size, not the near-certainty a levels-OLS t-stat
   would suggest.

### Why price-level cointegration is *rarer* in FX than in equities

This deserves explicit documentation because a renamed equity pairs project
gets it wrong:

* **Currencies are relative prices.** An equity pair can cointegrate because
  both prices load on the same firm-level cash-flow trend. An FX rate is
  already a *ratio* of two money stocks/price levels; there is no "company"
  anchoring the level. Long-run anchors (PPP, real-rate differentials) act
  over years-to-decades with huge deviations, weak glue at trading horizons.
* **The common-factor USD problem.** Any two USD-quoted pairs share the USD
  leg by construction. High return correlation between AUDUSD and NZDUSD is
  partly *mechanical* (common USD factor), and a correlation screen alone
  will flood the funnel with USD-factor pairs that are not cointegrated. This
  is exactly what the synthetic two-block panel reproduces: factor-correlated
  legs whose EG tests (correctly) fail. Cointegration in FX, where it exists,
  usually reflects a *policy or real linkage*: AUD/NZD (twin commodity
  economies), NOK/SEK (Scandinavian bloc), EUR/CHF (SNB policy, see the
  cautionary tale), EUR-satellite pegs.
* Consequently the funnel is: correlation screen (cheap prefilter), then EG
  with correct critical values, then a degeneracy check, then an
  economic-linkage judgement by a person. `cointegration_funnel` runs the
  first three steps. The pipeline hides the planted pair (labelled
  SEKUSD/NOKUSD) among 8 market pairs. Each candidate is regressed in input
  column order, so the orientation uses neither the data nor the ground
  truth. Over 60 synthetic seeds the funnel finds the planted pair in every
  seed and admits 1.1 false positives per seed on about 8 tests at 5%
  (section 8 of `examples/run_pipeline.py`).

### Spread dynamics: Ornstein–Uhlenbeck

Chosen: OU `ds = kappa(theta − s)dt + sigma dW`, fitted two ways -
exact-discretisation **OLS** (AR(1) regression) and exact Gaussian
transition-density **MLE**, which must agree (cross-check, not two models).

* vs **AR(p)/ARMA**: more parameters, marginal gains at daily frequency, and
  no closed-form half-life. OU's `half-life = ln 2/(kappa·dt)` is the single
  number the desk uses for time stops, holding-period estimates, and the
  carry filter.
* vs **no model (pure z-score)**: works until you need to answer "how long
  will I hold this and does the carry eat me first?", the carry-aware entry
  filter *requires* an estimate of holding time.

Estimator bias: the AR(1) coefficient is biased down in finite samples
(Kendall), so kappa = -ln(phi)/dt is biased up and half-lives come out too
short. Recovery is asserted at ±25% at n=8000. A fitted phi <= 0 (an
oscillating series) has no OU representation, and the fitters raise
`ValueError` instead of reporting a half-life of zero.

### Signals: z-score band state machine

`z = (s − mu)/sigma` with formation-frozen or rolling statistics; enter at
`|z| ≥ 2`, exit at `|z| ≤ 0.5`, hard stop at `|z| ≥ 4`, optional time stop,
one-bar cooldown after exits (no same-bar reversals). Vol-targeted sizing
(`hedged_vol_scale`: target vol over 63-day realised hedged-spread vol,
capped at 10x) is reported as a walk-forward variant in the pipeline. On a
pegged spread it raises leverage as realised vol collapses, which is why the
cap exists.

**Carry-aware entry filter**: expected reversion gain ≈ `(|z| − exit)·sigma`;
expected carry over the hold ≈ `carry/day × k·half-life`. Entries whose
expected carry *drag* exceeds the expected reversion gain are vetoed;
carry-favourable entries are never vetoed. It is coarse: it changes which
trades are taken, never how open ones are managed. The pipeline reports the
carry-flip backtest with and without the veto.

---

## 3. Costs

Costs are quoted in **pips** with pair-specific spreads (pip = 0.0001, JPY
pairs 0.01): majors ~0.5–1 pip, Scandi ~8–15, EM 30–150. The engine charges
half the quoted spread per side on each leg's traded notional:
`cost = |Δn| · ½ · pips · pip_size / S`, with `S` the fill price. Dividing by
the price level makes the conversion correct for JPY (pip 0.01) and for
high-price quotes such as USDMXN, USDZAR and USDBRL. Across 60 synthetic
seeds the same in-sample signal has Sharpe 1.77 at major spreads and -0.44 at
EM spreads (`VALIDATION.md` §4).

## 3a. Execution convention

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost. For daily FX the t+1 price is
the next daily close. `ExecutionConfig(mode="next", fill_fraction=0.5,
auction_cost_bp=10.0)` is the default of `run_backtest` and of
`walk_forward_backtest`. In `"next"` mode the position held before the fill
earns `log(q/p_t)` and the new position earns `log(p_{t+1}/q)`, where
`q = p_t + f (p_{t+1} - p_t)`; carry over the period is split in the same
proportions; the half spread is charged at `q`. `"close_auction"` fills at
`p_t` and adds `auction_cost_bp` per unit of traded notional on each leg.
`tests/test_execution.py` fails if a decision at t fills at the close of t.

---

## 4. Assumptions register

Each assumption states **what breaks if violated**.

| # | Assumption | What breaks when it fails |
|---|---|---|
| A1 | **Stable policy regimes** over formation + trading windows (no floors imposed/abandoned, no capital controls, no target-zone changes). | The cointegrating relation is an artefact of policy, not economics. The SNB floor made EURCHF the best-scoring 'cointegration' in the scan and then gapped −15% in a day; the strategy's P&L profile is steady gains then a catastrophic loss exceeding all of them (simulated: +0.058 over 750 days, −0.150 in one day). Stops do not help, the market gaps through them. |
| A2 | **Carry persistence**: deposit-rate differentials move slowly relative to the trade horizon. | The carry filter and the carry ledger both use current rates as forecasts. A surprise hike/cut (EM central bank defending a currency: TRY 2021-23, RUB 2014) changes the sign of expected carry mid-trade; the backtest's accrual remains correct ex post, but the entry decision was made on stale carry. |
| A3 | **No jumps** in the spread (diffusive OU). | Half-life and stop levels are calibrated to diffusion scale. Devaluations, depegs and risk-off gaps produce losses far beyond any z-based stop (A1's failure mode); position sizing must assume gap risk of the *policy* scale, not the OU sigma. |
| A4 | The **hedge ratio is constant** within a trading window (EG beta frozen; RLS optional). | Slow policy divergence drifts beta; the frozen-beta spread trends and z-scores lose meaning. Mitigation: walk-forward re-estimation each window + RLS diagnostics; tested via the RLS beta-shift tracking test. |
| A5 | **The modelled t+1 price is executable** at mid ± half the quoted pip spread; no market impact; no session gaps. | Understates costs in stressed markets (spreads widen 5–20x in risk-off; EM can be untradable), during Wellington/Sydney open gaps, and at fixes. The EM cost scenario bounds the effect; live use needs time-of-day-aware costs. |
| A6 | **Deposit rates proxy forward points** (CIP holds). | Post-2008, cross-currency basis makes CIP fail by 10–50bp for some pairs; carry computed from deposit rates misses the basis. For basis-heavy pairs (USDJPY at year-end turns) the swap-point accrual should be fed market forward points instead, the API accepts any rate series, so this is a data, not a code, change. |
| A7 | Rates panel and prices are **synchronous, daily, gap-tolerant** (ACT/365F on actual calendar gaps; Wednesday T+2 roll simplified to gap-day accrual). | If fed intraday or misaligned data, the no-lookahead guarantees (positions use info through t, carry uses rates at t−1) no longer correspond to executable timestamps. |
| A8 | **Two-regime correlation world** (risk-on/risk-off) is a sufficient stress model for co-movement. | Real correlation breaks are richer (idiosyncratic politics: Brexit GBP, CHF-specific flows). The two-block generator bounds regime risk but does not enumerate it; per-block limits in `DESK_GUIDE.md` are the operational control. |
| A9 | **Log-return P&L approximation**: leg P&L = notional × Δlog S. | Exact for the log spread traded; differs from arithmetic returns at second order (~r²/2 per day, negligible at daily FX vols ~0.5%, material for >5% gap days, the SNB case's true arithmetic loss is slightly less than the log loss; conservative direction). |
| A10 | **Inputs are finite and prices strictly positive.** Quotes, thresholds, notionals, betas and deposit rates are real numbers; a price is never zero or missing. | The package rejects rather than imputes, but the guards used to be inequalities (`if sigma <= 0`) and `isnan` checks, both of which pass NaN and ±Inf respectively. Violated, the failures are silent, not loud: a NaN `stop` disables the regime-break stop; a NaN `sigma` produces an all-flat "strategy" reported as a zero-P&L result; and an Inf from a logged zero price makes `engle_granger` return **"not cointegrated"** rather than an error. Enforced since via `fx_pairs._validation`. See VALIDATION §6.1. |

---

## 5. Walk-forward protocol

Formation 252 bd → trading 63 bd, tiling, non-overlapping; per window: EG fit
on formation, freeze `(alpha, beta, mu, sigma)`, gate on the formation EG
statistic (10% default), trade the next 63 days with frozen parameters, force
flat at the window edge. Window integrity (formation strictly precedes
trading, windows tile with no gaps) is unit-tested, and a lookahead detector
perturbs post-`k` prices and asserts P&L through `k` is unchanged. The EG
gate has limited power on 252 days: on the planted pair (half-life 8.7 days)
82% of windows pass it on average over 60 synthetic seeds.

## 6. The triangular null case

`log EURUSD + log USDJPY − log EURJPY = 0` identically (crosses are ratios of
USD legs). Fed to the machinery, this 'spread' has variance ~1e-31 and must
be flagged **degenerate**, never "perfectly cointegrated". Economically:
there is no true triangular arbitrage at daily frequency once you pay the
bid-ask on three legs; any measured deviation in real data is inside costs
and dies at the tick level to HFT. The degeneracy detector (spread std below
1e-7 in log units) is the guard, and the funnel test asserts it fires.
