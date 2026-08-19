# Desk Guide — How an Equity Vol Desk Uses This Engine

Documentation-contract items 5 and 6: the daily workflow around this code,
who consumes the numbers, controls and limits, and realistic scenarios.
The library is the *pricing kernel*; everything below is the operating
context a reviewer should imagine around it.

---

## 1. Where the engine sits in the desk workflow

```
vol surface marks (per strike/expiry)          trades & positions
              \                                     /
               +--> eq_options pricing kernel <----+
                    (BS / B76 / CRR / MC + Greeks)
                          |
        +-----------------+---------------------+
        |                 |                     |
  market-making      intraday risk         EOD risk & P&L
  quotes (bid/ask    (delta ladders,       (Greeks report,
  around theo)       gamma/vega maps)      P&L attribution)
```

### 1.1 Market-making quotes
- Theo = `bs_price` (single names: `crr_price` American) at the desk's
  marked vol for that strike/expiry — **one vol per quote**, taken from
  the surface, never a single flat vol (see VALIDATION.md §6.1).
- Bid/ask = theo ± max(vega · vol_spread, min_tick_edge). A 1y ATM option
  has vega ≈ 37.8 per unit vol (0.378 per vol point): quoting 0.4 vols
  wide ≈ $0.15 edge on a $9.83 option.
- Index futures options are quoted with `black76_price` off the futures —
  the dividend assumption drops out entirely.
- `implied_vol` runs on every incoming quote/print to map the market back
  to vol space; its hard `ValueError` on sub-intrinsic prices is the
  first-line filter against crossed/stale quotes polluting the surface.

### 1.2 Intraday risk pipeline
- Every position re-priced on the current spot/vol marks; `bs_greeks`
  aggregates to book-level delta, gamma, vega, theta, rho plus vanna/volga
  for the smile book.
- Delta is auto-hedged with futures within a band; the hedging simulator's
  1/√N law sets the band: rebalancing a 3M ATM book 64×/quarter leaves
  ~$0.44 P&L noise per option — tighter bands buy less noise but pay the
  5bp-cost drag quantified in VALIDATION.md §5.
- Scenario ladders: spot ±1/2/5/10% × vol ±1/5/10 points, re-priced with
  the same kernel (pure functions make this trivially parallel).

### 1.3 EOD Greeks / risk report
Nightly batch: re-price the book off closing marks, emit per-underlier and
book-level Greeks; P&L attributed as

    dV ≈ delta·dS + ½·gamma·dS² + vega·dsigma + theta·dt + rho·dr + residual

A persistent residual is the model-risk alarm (wrong q around ex-dates,
smile move not captured by parallel vega — split by vanna/volga buckets).
The `comparison.py` harness runs nightly on a probe set: BS vs tree vs MC
must agree within documented tolerance or the batch flags a regression —
same tables as VALIDATION.md, automated.

### 1.4 Vol-surface marking dependency
Everything above is conditional on the surface marks. This project
deliberately treats the surface as an *input* (synthetic skewed chain in
`data/synthetic.py` for reproducibility). Marking, arbitrage-cleaning and
interpolating the real surface is its own project; until then the engine's
per-quote vol interface is exactly the contract a surface service must
provide. Garbage marks → garbage Greeks, with no warning from the kernel:
that is why implied-vol round-trip checks (worst error 3e-13 on the
synthetic chain) run in CI.

## 2. Limits & model governance

**Limits enforced on kernel outputs (typical structure):**
- |net delta| per underlier and book (in $ and %ADV of the hedge trade);
- gamma limit *scaled up near expiry* (see pinning scenario below);
- vega by expiry bucket + net; vanna/volga limits for the skew book;
- stress limits: worst loss over the spot×vol scenario grid
  (crash scenario −20% spot / +15 vol points must stay inside capital).

**Model governance (how this repo maps to it):**
- *Model inventory*: BS/B76 (closed form), CRR (American), MC (exotics) —
  each with documented purpose and validity domain (METHODOLOGY.md §3).
- *Independent validation*: cross-model agreement tests + golden vectors
  (`tests/golden/golden_vectors.json`) that the C++/Rust production
  engines must reproduce to 1e-10 — the classic
  research-implementation vs production-implementation control.
- *Change control*: 288 offline deterministic tests are the regression
  gate; convergence tables are re-generated, not hand-maintained.
- *Known-limitations register*: METHODOLOGY.md §2 (assumptions A1–A8) and
  VALIDATION.md §6 — reviewed when a position pushes into a weak spot
  (e.g. short-dated single names over ex-dates ⇒ A4 escalation).

## 3. Realistic scenarios

### 3.1 Earnings jump (single name)
Overnight earnings gap ±8% with vol crush 35→25. BS has no jump: the desk
runs it as a *scenario*, re-pricing the book at (S±8%, σ−10pts) with the
kernel. Key numbers (from `bs_greeks` at σ=35%, T=1/52): a short 1w ATM
straddle collects $3.87 premium but carries gamma 0.164, so the jump's
gamma term alone is ½·Γ·(8)² ≈ **$5.26 loss** — the premium does not
cover the move, while vega is small — earnings risk is gamma risk, not
vega risk. The A1 (no-jump) assumption is why the desk charges the event
in the vol mark rather than trusting theta.

### 3.2 Dividend announcement
Company unexpectedly raises the dividend. Calls reprice down / puts up
through the forward; the continuous-q engine gets the *direction* right
but the *timing* wrong (VALIDATION.md §6.3: up to +1.15 mispricing on a
2.75-value call straddling the ex-date). Desk action: switch affected
expiries to escrowed-dividend spot inputs, re-check American call
early-exercise flags on deep-ITM short-dated calls (exercise just before
ex-date when dividend > remaining time value — the CRR premium
quantifies the boundary).

### 3.3 Gamma near expiry / pinning
Friday expiry, spot sitting on the 100 strike. T→0 with S≈K sends gamma
→ ∞ (kernel: gamma at T=1/365, ATM, 20 vol is 0.38 — 20× the 1y value)
while delta flips 0↔1 through the strike. Discrete hedging error scales
with Γ·S²: the 1/√N budget that was fine at 3M is inadequate on expiry
day. Desk practice mirrored by the engine's tests: expiry-day gamma
limits, hedge bands shrunk, and the T→0 intrinsic limit handled exactly
(no NaN at the boundary — tested) so the risk system doesn't blow up
precisely when it matters most.

### 3.4 2008/2020-style vol spike
Spot −30%, short-dated vol 20→80+, skew inverts steeper, rates cut
(possibly below zero). Engine behaviours that matter, all tested: σ=1.5+
and r<0 are fully supported; implied vol still round-trips at 80+ vol;
Black-76 keeps index books consistent when dividend forecasts become
meaningless (futures embed the market's guess); the misspecified-vol
hedging result quantifies the P&L of having sold vol too cheap — hedging
a short option marked at 20 when 80 realizes bleeds
½(σ_r²−σ_h²)S²Γ per dt, which the simulator reproduces path-by-path.
Governance hook: the crash cell of the scenario grid is a *limit*, so the
book arrives at the crisis pre-sized for it.

### 3.5 Deep ITM/OTM and other edge cases (daily, not exotic)
Deep-ITM options trade at parity with zero vega — `implied_vol` refuses
to invent a vol there (raises; the quote is information-free); T=0 and
σ=0 return exact limits; K=0 (stock-settled structures) and huge-vol
corners are finite and tested. Every edge case in the contract's item 6
list is both documented (VALIDATION.md §6.5) and unit-tested
(`tests/test_edge_cases.py`).
