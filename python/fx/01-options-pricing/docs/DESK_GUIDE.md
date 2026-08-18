# Desk Guide — How an FX Options Desk Uses This Engine

## 1. The quoting workflow: vol/delta space, not price/strike space

Interbank FX options are quoted as **GK implied vols at deltas**, not
premiums at strikes. A broker run looks like:

```
EURUSD 6M: ATM 8.25  25RR -1.25/-1.05  25BF 0.30/0.40  (vol %, DNS ATM)
```

The chain from quote to trade, and where this engine sits:

1. **Convention resolution** — which delta? EURUSD ≤ 1y: spot delta,
   premium in USD (quote ccy). **USDJPY: premium paid in USD = base ccy →
   premium-adjusted deltas are standard**; most EM and gold likewise.
   Long-dated (> 1y) and EM: forward deltas. `deltas.py` implements all
   four and the conversions — convention mix-ups move the strike
   materially: on the pipeline's USDJPY example the 25Δ call strike is
   151.81 (spot), 152.05 (forward), 151.39 (spot PA), 151.64 (forward
   PA) — up to 66 pips apart for the *same* quoted delta.
2. **Strike-from-delta** (`strike_from_delta`) — turn "25Δ call" into K.
   For PA conventions this is the non-monotone numerical branch — the
   engine picks the market-standard larger strike and *refuses*
   (ValueError) deltas beyond the fold rather than silently returning the
   wrong branch.
3. **ATM resolution** (`atm_dns_strike`) — "ATM" means the delta-neutral
   straddle almost everywhere (`F·e^{±σ²T/2}`); ATM-forward for some
   long-dated/EM markets.
4. **Pricing** (`gk_price` / `black76_price`) — premium in quote ccy per
   unit base notional; multiply by notional; convert premium currency at
   spot if paid in base ccy.
5. **Counter-quote** — reprice at your own vol, quote a two-way around it.

**Cut times and settlement**: an "expiry" is a date *and a cut* — 10am
New York (the default for G10) or 3pm Tokyo (standard for USDJPY and JPY
crosses; a USDJPY option "NY cut" vs "Tokyo cut" differs by ~5 hours of
gamma). Delivery is spot settlement **T+2** after expiry (T+1 for
USDCAD). Time-to-expiry inputs to this engine should be computed to the
cut, ACT/365F.

## 2. Risk: the delta–vega–vanna–volga bucket report

The book is risk-managed in exactly the Greeks `greeks.py` produces:

| Bucket | Engine output | Desk usage |
|---|---|---|
| Delta (per convention) | `delta(..., convention=...)` | Spot hedge; net to the pair's convention so the hedge matches the quote |
| Gamma | `gamma` | Intraday rebalance sizing; pin risk near big strikes at the cut |
| Vega | `vega` | ATM vol hedge (straddles/DNS) |
| **Vanna** | `vanna` | dΔ/dσ ≙ dVega/dS — hedged with **25Δ risk reversals**; the skew position |
| **Volga** | `volga` | dVega/dσ — hedged with **25Δ butterflies**; the wing/kurtosis position |
| Rho_d / Rho_f | `rho_domestic`, `rho_foreign` | Two separate rate buckets feeding the desk's FRA/OIS hedges in *each* currency — an FX option is a two-rate instrument, never net the rhos |
| Theta | `theta` | Daily decay budget vs expected gamma P&L |

A vanilla book marked at flat vol shows zero smile risk; in reality the
desk marks **ATM/25RR/25BF (and 10Δ wings)** per tenor and maps
RR-equivalent risk = vanna bucket, BF-equivalent risk = volga bucket.
This engine supplies the vanna/volga numbers; the RR/BF *marking* (smile
construction) is project 9.

**P&L attribution** (daily): explain P&L = delta·dS + ½gamma·dS² +
vega·dσ + vanna·dS·dσ + ½volga·dσ² + theta·dt + rho_d·dr_d + rho_f·dr_f;
unexplained residual above tolerance triggers investigation (stale marks,
convention mismatch, smile move not captured by flat vol). The Greeks
here are the attribution inputs; `finite_difference_greeks` is the
independent checker used in model validation sign-off.

## 3. Hedging workflow

- New trade → hedge delta immediately in spot (or forward for
  forward-delta books). `hedging.py` mirrors the true mechanics: the
  hedge is a **foreign-currency position earning r_f**, financed in
  domestic at r_d — foreign carry is P&L, not noise (on USDJPY at 5.25%
  USD rates, ignoring it misstates hedge P&L by ~2.6% of spot per year
  per unit delta).
- Rebalance discipline: variance ∝ 1/N vs costs ∝ √N·pips
  (`hedge_frequency_study` quantifies the trade-off — see VALIDATION.md
  §3 for the table). Typical G10 practice: daily + gamma-triggered
  intraday bands.
- Vol marking risk: hedging at the wrong vol biases P&L by roughly
  vega × (σ_mark − σ_realised) — reproduced in the simulator
  (+0.0062 USD/EUR selling 2 vols rich on a 0.0252 premium).
- American OTC positions (rare, but traded): early-exercise monitoring via
  `binomial.py`; exercise a deep-ITM call on a high-carry base ccy when
  the tree's continuation value drops to intrinsic (USDJPY example: 14%
  of European value at stake — see VALIDATION.md §2).

## 4. Controls, limits, governance

- **Model governance**: GK is the booking/quoting model; independent
  implementations (Black-76, tree, MC) cross-check it continuously —
  `comparison.py` is that harness in miniature; golden vectors freeze the
  behaviour for the C++/Rust production rewrites. Any change must
  reproduce `tests/golden/golden_vectors.json` to 1e-10.
- **Input validation as a control**: negative *rates* are legal (EUR/CHF
  era), negative vols/times/spots are hard errors — `ValueError` with the
  offending input named, never a NaN price that flows into risk.
- **Limits**: vega and gamma limits per pair/tenor; notional caps in
  managed/pegged pairs (peg-break risk is unmodelled — METHODOLOGY.md
  assumption 4); stress limits from the scenario set below.

## 5. Realistic scenarios (run these against the engine)

1. **Central-bank surprise (rate shock)**: ECB unexpectedly hikes 50bp →
   EURUSD: `r_f` +50bp. Book impact = rho_f bucket: a long 6m ATMF EUR
   call loses `rho_f × 0.005 ≈ −0.2773 × 0.005 ≈ −0.0014` USD per EUR —
   *and* the forward, all delta-convention strikes and the DNS ATM move.
   Rerun the delta table (`compare_models` + `delta`) at shifted rates;
   the two-rho separation is what makes this scenario computable at all.
2. **Depeg / floor break (CHF Jan-2015 style)**: overnight gap −18%, vol
   3% → 30%+. Flat-vol GK cannot *price* the pre-event risk (see
   VALIDATION.md §4), but the engine quantifies the damage: revalue the
   book at (S×0.82, σ×10), P&L = full jump on net delta + vega × 27 vols;
   the hedging simulator with `sigma_true ≫ sigma_hedge` shows the
   hedged-book loss distribution. Control: notional caps + long-wing
   (10Δ) positions, marked via project-9 smiles.
3. **Carry unwind**: high-carry pair (USDJPY, r_f ≫ r_d) spot drops fast
   as carry trades unwind: correlated spot↓/vol↑ move. Short JPY-call
   (USD-put) books lose on delta, vega *and* vanna simultaneously — the
   vanna bucket is why desks track it as a first-class Greek. Scenario:
   revalue at (S −5%, σ +4 vols) and attribute via the Greek ladder;
   forward points swing (`forward_points`) as rate expectations reprice,
   moving every forward-delta hedge.

## 6. Who consumes the numbers

- **Traders**: prices, per-convention deltas, DNS strikes, vanna/volga
  (intraday).
- **Risk**: Greek ladders in both rate buckets, scenario P&L, limit usage
  (EOD + intraday snaps).
- **Product control**: independent revaluation via `comparison.py`-style
  cross-checks; P&L attribution inputs.
- **Model validation / quant dev**: `finite_difference_greeks`,
  convergence tables, golden vectors (release-gated).
