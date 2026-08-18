# Desk Guide — How an FX Options Desk Uses This

## 1. Daily workflow: broker quotes → marked surface

1. **Collect** — per pair, per tenor (o/n, 1w, 1m, 3m, 6m, 1y, 2y):
   ATM DNS, 25Δ RR, 25Δ fly, 10Δ RR, 10Δ fly from brokers (BGC, TP
   ICAP screens) and the interdealer market. Quotes arrive in the
   pair's native convention — this package's `MarketSlice` carries the
   convention with the quote, which is the correct data model.
2. **Cutoffs** — quotes reference an expiry *cut*: **NY cut** (10:00
   New York) for most pairs, **Tokyo cut** (15:00 Tokyo) for JPY
   crosses and Asian pairs. A 1w option to the Tokyo cut is a
   different instrument (≈ half a trading day of variance) than to the
   NY cut; the marker must normalise before comparing sources
   (assumption A5). Mixing cuts shows up as phantom calendar
   arbitrage at the short end — the `calendar_arbitrage_report` at
   fixed delta is the alarm that catches it.
3. **Validate & fit** — quotes → five vols (`vols_from_quotes`; a
   negative fly warns — usually a one-vol-strangle mixup, failure mode
   F3) → pillar strikes (`solve_pillar_strikes`, native convention) →
   SVI per expiry with the Durrleman check → surface with the
   fixed-delta calendar check. A violation is investigated, not
   auto-smoothed: it is either a stale quote, a cut mismatch, or free
   money.
4. **Calibrate Heston** overnight and on demand (`calibrate_heston`,
   ~0.2 s per pair) for exotics pricing and scenario generation.
   Track (ρ, v0, θ, ξ²/κ) day over day — *not* κ and ξ separately
   (the ridge, VALIDATION §3).
5. **Publish** — vol(K,T) / vol(Δ,T) queries feed the pricers, the
   risk system's scenario engine, and the e-trading auto-quoter.
   Consumers: vanilla market-makers (pillar vols + interpolation),
   exotics desk (Heston + VV adjustments), risk (buckets below),
   product control (independent price verification against broker
   marks).

## 2. Risk in vega / vanna / volga buckets

FX desks do not hedge abstract sensitivities — they hedge with the
three liquid packages, which is exactly why the market quotes them:

| bucket | measured by | hedged with | this package |
|---|---|---|---|
| vega | ∂V/∂σ (parallel) | ATM straddles | `gk_vega`, `heston_greeks_fd["vega"]` |
| vanna | ∂²V/∂S∂σ | **risk reversals** (long call / short put) | `gk_vanna`, FD vanna |
| volga | ∂²V/∂σ² | **butterflies / strangles** | `gk_volga`, FD volga |

A book's smile risk report is (vega, vanna, volga) per pair per tenor
bucket; the trader neutralises vanna with RR trades and volga with
flies, then delta-hedges the residual. The pipeline's Greeks table
(§7) shows Heston FD vs BS-world buckets at the 25Δ pillar: signs and
magnitudes agree OTM (vanna ≈ 3.9 vs 3.2, volga ≈ 2.4 vs 1.6 in the
demo) — the *difference* is the model's smile-dynamics content.
Exactly at ATM both buckets pass through zero and their sign is
model-dependent; desks bucket at 25Δ/10Δ, not at ATM. Both interest-
rate rhos are carried (an FX option is a two-curve position: rho_d > 0
> rho_f for calls) and feed the STIR hedges.

**Sticky delta** is the FX-natural convention: when spot moves, the
desk re-marks the same *deltas* at (approximately) the same vols, so
the strike-space surface slides with spot. A trader's "smile P&L" is
computed against this convention; Heston's own dynamics are close to
sticky-delta (METHODOLOGY §5), which is why its FD delta (which lets
the model smile float) differs from the BS delta at the same vol —
that difference is the vanna-driven hedge correction.

## 3. Exotics: barriers and touches priced with VV adjustments

The market standard for first-generation exotics (RKO/KO barriers,
one-touch / no-touch) is **flat-BS price + vanna–volga adjustment**:
price the exotic at ATM vol, compute its vega/vanna/volga, and charge
the market cost of the pillar portfolio that hedges those exposures
(often weighted by touch/survival probability). The pipeline's digital
demo is the miniature version: flat GK 0.2054 vs VV 0.1895 vs Heston
0.1899 — the VV adjustment moves the price 1.6 vol-points-worth and
lands within ~4 bp of the full stochastic-vol price. Desk practice:
mark touches VV-adjusted, check against Heston, reserve the
difference. When the two *disagree materially* (very long-dated, very
wide barriers, high ξ pairs) the trade goes to the model desk, not
the auto-quoter.

## 4. Realistic scenarios

* **CB surprise (e.g. unexpected 50 bp)** — short-dated ATM gaps
  (7.8 → 12+ overnight), RR flips toward the surprise direction,
  calendar at the front inverts in *vol* but must stay arbitrage-free
  in *total variance* — rerun the fixed-delta calendar check before
  publishing. Pure-diffusion Heston underprices the pre-event 1w
  smile (failure mode F1): the desk overlays event weights.
* **2015 CHF depeg (RR explosion)** — the canonical F6 case: EURCHF
  RRs went from ~0 to double digits and the smile ceased to be a
  diffusion object. Lesson encoded here: pegged-pair surfaces are
  jump-risk prices; Heston calibration chasing ξ to its bound is the
  tell. The 35%-vol EM preset (`em_high_vol_market`, +6% RR topside)
  is the *floating* analogue and calibrates fine — the difference
  between an inverted-smile floating market and a peg is exactly the
  difference between F-mode "works with big ρ > 0" and "do not use".
* **JPY intervention (MoF/BoJ)** — spot drops 3 big figures, USDJPY
  RR (already −2.5) deepens, front flies richen. The pa-convention
  machinery matters most here: post-intervention re-marks at new spot
  move every pa strike (K_ATM = F·e^{−σ²T/2} with both F and σ
  changed); a desk on the wrong convention (F2: 0.2–0.3 vol pts,
  measured) bleeds silently on every re-mark.
* **EM devaluation (smile inversion)** — a "normal" EM smile already
  prices topside (RR +6% in the preset); in a devaluation scare the
  10Δ call wing goes vertical and BF10 explodes. The surface still
  builds (tested at 35% ATM, 30% carry — note the 1y ATM strike sits
  ~30% above spot purely from carry), but 10Δ marks are extrapolation
  (F5): widen spreads, cap auto-quoting deltas.

## 5. Controls, limits, governance

* **Convention registry** — per pair, per tenor: delta type (spot/
  forward, pa or not), ATM definition, cut, BF type (smile vs one-vol),
  premium currency — agreed with each counterparty and versioned.
  The F2 demo (0.2–0.3 vol pts of silent error on USDJPY) is the
  standing justification; confirm-matching on delta-exchange trades
  (where counterparties exchange the hedge at the quoted delta) is
  where mismatches surface first.
* **Arbitrage gates** — Durrleman g(k) ≥ 0 per slice and fixed-delta
  calendar monotonicity are publish-blocking checks (both implemented,
  both tested with planted violations).
* **Model governance** — VV is approved as *marking/adjustment* tool,
  Heston as *pricing/scenario* model; each has a validated domain
  (this repo's VALIDATION.md is the template: identities, cross-model
  agreement, failure modes F1–F7). Changes to smile interpolation are
  model changes and require revalidation — the SVI-vs-VV wing gap is
  the quantified impact.
* **Limits & P&L** — vega/vanna/volga limits per bucket; daily P&L
  explained into delta/vega carry, smile (vanna/volga), rates (both
  rhos) and unexplained; persistent unexplained on JPY books is the
  classic symptom of a convention or cut mismatch.
* **Ridge discipline** — κ/ξ movements along ξ²/κ ≈ const are not a
  market signal (VALIDATION §3); parameter-change alerts key off ρ,
  v0, θ and ξ²/κ.
