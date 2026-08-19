# Methodology — FX Statistical Pairs / Cross-Rate Mean Reversion

This document answers, for the reviewer: **why this model**, **against what
alternatives**, and **under which assumptions** (with what breaks when each
assumption fails). Validation evidence lives in `VALIDATION.md`; desk usage in
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
rates are exact *ratios* of USD legs — in logs, exact *differences*, so
triangular consistency is linear and testable.

Internally every currency is a **USD leg** (CCYUSD, USD per 1 unit of
currency; USD ≡ 1). Any cross is `BASEQUOTE = BASEUSD / QUOTEUSD` — an exact
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
**wrong-carry** — the pipeline constructs an explicit example where including
carry flips the sign of the strategy's P&L (see `VALIDATION.md` §4).

**The T+2 / Wednesday wrinkle.** Real tom-next rolls settle T+2, so the
3-day weekend financing is charged on **Wednesday** (Wednesday's tom-next
spans Saturday–Monday). We accrue by actual calendar-day gaps ACT/365F, which
books the 3 days on Monday instead. Total accrual over any horizon is
identical; only the intra-week booking date shifts by two business days. This
simplification is deliberate and unit-tested (`test_carry.py`).

---

## 2. Why Engle–Granger + OU (model choice vs alternatives)

### Cointegration test: Engle–Granger, from scratch

Chosen: two-step Engle–Granger — OLS `log P1 = alpha + beta log P2 + u`,
then an ADF test (no deterministic terms) on `u` against **MacKinnon (2010)
N=2** critical values.

| Alternative | Why not (here) |
|---|---|
| **Johansen (VECM, ML)** | Estimates the full cointegration rank and is superior for 3+ assets, but for a **pair** it answers the same question with more machinery, is less transparent about *which* regression produced the spread, and its small-sample size distortions are harder to reason about. EG's single OLS beta is also exactly the hedge ratio the desk executes. Johansen is the right upgrade for basket trades (e.g. EUR-block baskets) — noted as future work. |
| **Distance method (sum of squared normalized price gaps, Gatev et al.)** | Model-free and robust, but it has no notion of a hedge ratio other than 1:1, no stationarity test (so it happily selects the correlated-but-not-cointegrated pairs our funnel must reject), and no parameter (half-life) to size holding periods with. |
| **Kalman-filter latent-spread models** | Adaptive, but the fully latent version is unidentified without strong priors and easy to overfit. We take the useful half: a **recursive-least-squares hedge ratio with forgetting factor** (`RLSHedge`), which is the Kalman filter's regression limit — adaptivity where it matters, no latent-state machinery. |

Two details that are easy to get wrong and are therefore tested:

1. **Residual-based critical values.** The ADF is run on *estimated*
   residuals; using plain N=1 ADF critical values over-rejects. We use the
   MacKinnon (2010) response surface for N=2 (asymptotic −3.34 at 5% vs
   −2.86 for plain ADF). The test suite asserts the two tables differ and
   that our EG statistic matches `statsmodels.coint` to ~1e-15.
2. **Spurious regression size control.** On 200 seeded independent random
   walks the 5% EG test rejects 4.5% of the time — the funnel's false-positive
   rate is the nominal size, not the near-certainty a levels-OLS t-stat
   would suggest.

### Why price-level cointegration is *rarer* in FX than in equities

This deserves explicit documentation because a renamed equity pairs project
gets it wrong:

* **Currencies are relative prices.** An equity pair can cointegrate because
  both prices load on the same firm-level cash-flow trend. An FX rate is
  already a *ratio* of two money stocks/price levels; there is no "company"
  anchoring the level. Long-run anchors (PPP, real-rate differentials) act
  over years-to-decades with huge deviations — weak glue at trading horizons.
* **The common-factor USD problem.** Any two USD-quoted pairs share the USD
  leg by construction. High return correlation between AUDUSD and NZDUSD is
  partly *mechanical* (common USD factor), and a correlation screen alone
  will flood the funnel with USD-factor pairs that are not cointegrated. This
  is exactly what the synthetic two-block panel reproduces: factor-correlated
  legs whose EG tests (correctly) fail. Cointegration in FX, where it exists,
  usually reflects a *policy or real linkage*: AUD/NZD (twin commodity
  economies), NOK/SEK (Scandinavian bloc), EUR/CHF (SNB policy — see the
  cautionary tale), EUR-satellite pegs.
* Consequently the honest funnel is: correlation screen (cheap prefilter) →
  EG with correct critical values → **degeneracy check** → economic-linkage
  judgement (human). Expect few survivors; the pipeline's synthetic funnel
  finds 0 survivors among 7 factor-correlated candidates and 1 among 1
  planted true cointegration.

### Spread dynamics: Ornstein–Uhlenbeck

Chosen: OU `ds = kappa(theta − s)dt + sigma dW`, fitted two ways —
exact-discretisation **OLS** (AR(1) regression) and exact Gaussian
transition-density **MLE** — which must agree (cross-check, not two models).

* vs **AR(p)/ARMA**: more parameters, marginal gains at daily frequency, and
  no closed-form half-life. OU's `half-life = ln 2/(kappa·dt)` is the single
  number the desk uses for time stops, holding-period estimates, and the
  carry filter.
* vs **no model (pure z-score)**: works until you need to answer "how long
  will I hold this and does the carry eat me first?" — the carry-aware entry
  filter *requires* an estimate of holding time.

Estimator bias is documented in `VALIDATION.md` (kappa is biased up in finite
samples — the classic AR(1) bias — recovery is asserted at ±25% at n=8000).

### Signals: z-score band state machine

`z = (s − mu)/sigma` with formation-frozen or rolling statistics; enter at
`|z| ≥ 2`, exit at `|z| ≤ 0.5`, hard stop at `|z| ≥ 4`, optional time stop,
one-bar cooldown after exits (no same-bar reversals). Vol-targeted sizing
(`target vol / realised spread vol`, capped) is available and is deliberately
shown maximising leverage right before the synthetic SNB break — the cap is
the lesson.

**Carry-aware entry filter**: expected reversion gain ≈ `(|z| − exit)·sigma`;
expected carry over the hold ≈ `carry/day × k·half-life`. Entries whose
expected carry *drag* exceeds the expected reversion gain are vetoed;
carry-favourable entries are never vetoed. Coarse by design — it changes
which trades are taken, never how open ones are managed.

---

## 3. Costs

Costs are quoted in **pips** with pair-specific spreads (pip = 0.0001, JPY
pairs 0.01): majors ~0.5–1 pip, Scandi ~8–15, EM 30–150. The engine charges
half the quoted spread per side on each leg's traded notional:
`cost = |Δn| · ½ · pips · pip_size / S`. This makes the EM cost sensitivity
first-class: the pipeline shows the identical signal path earning +0.19 at
major spreads and −0.06 at EM spreads (costs −0.26).

---

## 4. Assumptions register

Each assumption states **what breaks if violated**.

| # | Assumption | What breaks when it fails |
|---|---|---|
| A1 | **Stable policy regimes** over formation + trading windows (no floors imposed/abandoned, no capital controls, no target-zone changes). | The cointegrating relation is an artefact of policy, not economics. The SNB floor made EURCHF the best-scoring 'cointegration' in the scan and then gapped −15% in a day; the strategy's P&L profile is steady gains then a catastrophic loss exceeding all of them (simulated: +0.058 over 750 days, −0.150 in one day). Stops do not help — the market gaps through them. |
| A2 | **Carry persistence**: deposit-rate differentials move slowly relative to the trade horizon. | The carry filter and the carry ledger both use current rates as forecasts. A surprise hike/cut (EM central bank defending a currency: TRY 2021-23, RUB 2014) changes the sign of expected carry mid-trade; the backtest's accrual remains correct ex post, but the entry decision was made on stale carry. |
| A3 | **No jumps** in the spread (diffusive OU). | Half-life and stop levels are calibrated to diffusion scale. Devaluations, depegs and risk-off gaps produce losses far beyond any z-based stop (A1's failure mode); position sizing must assume gap risk of the *policy* scale, not the OU sigma. |
| A4 | The **hedge ratio is constant** within a trading window (EG beta frozen; RLS optional). | Slow policy divergence drifts beta; the frozen-beta spread trends and z-scores lose meaning. Mitigation: walk-forward re-estimation each window + RLS diagnostics; tested via the RLS beta-shift tracking test. |
| A5 | **Daily closes are executable** at mid ± half the quoted pip spread; no market impact; no session gaps. | Understates costs in stressed markets (spreads widen 5–20x in risk-off; EM can be untradable), during Wellington/Sydney open gaps, and at fixes. The EM cost scenario bounds the effect; live use needs time-of-day-aware costs. |
| A6 | **Deposit rates proxy forward points** (CIP holds). | Post-2008, cross-currency basis makes CIP fail by 10–50bp for some pairs; carry computed from deposit rates misses the basis. For basis-heavy pairs (USDJPY at year-end turns) the swap-point accrual should be fed market forward points instead — the API accepts any rate series, so this is a data, not a code, change. |
| A7 | Rates panel and prices are **synchronous, daily, gap-tolerant** (ACT/365F on actual calendar gaps; Wednesday T+2 roll simplified to gap-day accrual). | If fed intraday or misaligned data, the no-lookahead guarantees (positions use info through t, carry uses rates at t−1) no longer correspond to executable timestamps. |
| A8 | **Two-regime correlation world** (risk-on/risk-off) is a sufficient stress model for co-movement. | Real correlation breaks are richer (idiosyncratic politics: Brexit GBP, CHF-specific flows). The two-block generator bounds regime risk but does not enumerate it; per-block limits in `DESK_GUIDE.md` are the operational control. |
| A9 | **Log-return P&L approximation**: leg P&L = notional × Δlog S. | Exact for the log spread traded; differs from arithmetic returns at second order (~r²/2 per day, negligible at daily FX vols ~0.5%, material for >5% gap days — the SNB case's true arithmetic loss is slightly less than the log loss; conservative direction). |
| A10 | **Inputs are finite and prices strictly positive.** Quotes, thresholds, notionals, betas and deposit rates are real numbers; a price is never zero or missing. | The package rejects rather than imputes, but the guards used to be inequalities (`if sigma <= 0`) and `isnan` checks — both of which pass NaN and ±Inf respectively. Violated, the failures are silent, not loud: a NaN `stop` disables the regime-break stop; a NaN `sigma` produces an all-flat "strategy" reported as a zero-P&L result; and an Inf from a logged zero price makes `engle_granger` return **"not cointegrated"** rather than an error. Enforced since via `fx_pairs._validation`. See VALIDATION §6.1. |

---

## 5. Walk-forward protocol

Formation 252 bd → trading 63 bd, tiling, non-overlapping; per window: EG fit
on formation, freeze `(alpha, beta, mu, sigma)`, gate on the formation EG
statistic (10% default), trade the next 63 days with frozen parameters, force
flat at the window edge. Window integrity (formation strictly precedes
trading, windows tile with no gaps) is unit-tested, and a lookahead detector
perturbs post-`k` prices and asserts P&L through `k` is bit-identical.

## 6. The triangular null case

`log EURUSD + log USDJPY − log EURJPY = 0` identically (crosses are ratios of
USD legs). Fed to the machinery, this 'spread' has variance ~1e-31 and must
be flagged **degenerate** — never "perfectly cointegrated". Economically:
there is no true triangular arbitrage at daily frequency once you pay the
bid-ask on three legs; any measured deviation in real data is inside costs
and dies at the tick level to HFT. The degeneracy detector (spread std below
1e-7 in log units) is the guard, and the funnel test asserts it fires.
