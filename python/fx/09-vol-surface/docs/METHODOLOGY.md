# Methodology — FX Volatility Surface & Stochastic Volatility

This project builds the FX-native pipeline: broker quotes in **delta
space** ({ATM, RR25, BF25, RR10, BF10} per expiry) → five-point smiles →
strike solving under the four market delta conventions → smooth smiles
(SVI **and** vanna–volga) → a delta-space surface → Heston on top for
dynamics, calibration, pricing and Greeks. It is deliberately *not* an
equity surface with renamed variables: the quoting, the ATM definition,
the strike solving, the interpolation axis and the risk buckets are all
FX-specific.

## 1. Why delta-space quoting exists

The OTC FX options market is a **sticky-delta** market. Spot moves ~1%
a day; a strike-based quote sheet would be stale within hours and would
not be comparable across the 30+ actively quoted pairs at wildly
different spot levels (EURUSD ≈ 1.10, USDJPY ≈ 150, USDKRW ≈ 1350).
Instead, brokers quote *moneyness in probability units*:

* **ATM** — the delta-neutral straddle (DNS): the strike at which a
  straddle has zero net delta. Unadjusted conventions ⇒ d₁ = 0 ⇒
  `K_ATM = F·exp(+σ²T/2)`; premium-adjusted ⇒ d₂ = 0 ⇒
  `K_ATM = F·exp(−σ²T/2)` (below the forward). Both are implemented
  and tested exactly.
* **RR25** = σ(25Δcall) − σ(25Δput) — the skew, tradable as a package.
* **BF25** = ½(σ(25Δcall) + σ(25Δput)) − σ_ATM — the convexity,
  tradable as a strangle-vs-straddle package. Same at 10Δ.

These five numbers per expiry are self-normalising across pairs, spot
levels and vol regimes — which is precisely why the market settled on
them. The exact linear map to the five smile vols
(σ_C = ATM + BF + RR/2, σ_P = ATM + BF − RR/2) is implemented in
`smile_from_quotes.py` and round-trip tested to 1e-14.

### Broker (one-vol) strangle vs smile strangle — honesty note

We use the **simplified/smile butterfly** throughout: BF is defined so
that the {quotes → vols} map is exactly linear. The *true* broker
market strangle is a **one-vol** quote: the single vol that reprices the
two-leg strangle whose legs are struck at their own smile vols. The two
BF definitions differ at order O(RR²/σ_ATM): negligible for G10 25Δ
(< 0.05 vol pts), but up to 0.3–1.0 vol pts at 10Δ for heavily skewed
pairs (USDJPY, USDBRL). Consuming a genuine broker one-vol strangle as
a smile BF **silently corrupts the wings**; the correct treatment is a
nonlinear solve (find the smile such that the one-vol strangle
reprices). This project generates and consumes smile BFs only, and the
mismatch is listed as failure mode F3 in `VALIDATION.md`.

## 2. Delta conventions and strike solving (`smile_from_quotes.py`)

Four conventions are first-class (`spot`, `forward`, `spot_pa`,
`forward_pa`):

| convention | call delta | used by |
|---|---|---|
| spot | e^{−r_f T} N(d₁) | G10 vs USD (quote-ccy premium), T ≤ 1y |
| forward | N(d₁) | same pairs, T > 1y |
| spot pa | e^{−r_f T} (K/F) N(d₂) | JPY-style pairs (base-ccy premium), T ≤ 1y |
| forward pa | (K/F) N(d₂) | same, T > 1y |

Premium-adjusted deltas arise when the premium is paid in the *base*
currency: the premium is itself a position in the underlying, so the
hedge is `Δ_pa = Δ − V/S` (identity unit-tested). **USDJPY and most
USD-base pairs quote pa deltas** — the presets encode this.

The strike behind "the 25Δ call" solves `Δ(K, σ(K)) = 0.25`: delta
depends on vol which depends on strike. For pillar construction the
pillar vol is known, so unadjusted conventions invert in closed form;
pa puts are monotone in K (Brent). The **pa call delta is non-monotone
in K** — it vanishes at K→0 and K→∞ with an interior maximum where
`N(d₂)σ√T = φ(d₂)` — so a given delta has *two* strikes. The market
standard is the high-strike (OTM, falling-branch) candidate;
`strike_from_delta_pa_candidates` exposes both, and tests verify the
market branch is selected and the low candidate rejected. Smile-
consistent strikes (vol not known a priori) are solved by fixed
point/Brent on the composed map (surface queries, ground-truth
generator).

## 3. Why two smile models (`smile.py`)

**Vanna–volga** is the FX market's workhorse. The price of any strike
is the flat-ATM Black–Scholes price plus the *cost of the hedge*: the
portfolio of the three traded pillars (25P/ATM/25C) that matches the
target's vega, vanna and volga at the reference vol. We implement the
**exact three-instrument replication weights** (a 3×3 linear solve per
strike, residual < 1e-12 tested), not the first-order shortcut. VV is
exact at the pillars by construction, cheap, and is what desks use to
smile-adjust barriers and touches. Its theoretical limits: it is an
interpolation device with no dynamics, its wings extrapolate the
quadratic vol cost and can violate no-arbitrage beyond ~5Δ, and it says
nothing about forward smiles.

**SVI** (raw, in log-moneyness, total variance
`w(k) = a + b[ρ(k−m) + √((k−m)²+s²)]`) provides disciplined wings
(linear total variance, Lee-moment compatible) and an *analytic*
Durrleman butterfly-arbitrage check
`g(k) = (1 − k w′/2w)² − w′²/4 (1/w + ¼) + w″/2 ≥ 0`. Five pillars,
five parameters: exact interpolation for G10 smiles (EURUSD pillar
residual < 1e-7); for the extreme USDJPY skew the exact fit would need
ρ < −1, so the best fit sits on the ρ bound with ≤ 0.05 vol pts pillar
residual — measured and documented in `VALIDATION.md`.

The pipeline compares VV vs SVI at 15Δ (agree within ~5 bp) and in the
wings (diverge ~10+ bp at 5Δ) — the divergence *is* the interpolation
model risk, and quoting both brackets it.

## 4. The surface (`surface.py`): delta-space, not moneyness-space

Equity surfaces interpolate total variance at fixed (log-)moneyness.
FX surfaces interpolate **total variance at fixed delta**: the quoted
objects float with spot and vol (sticky delta), and a fixed log-
moneyness is a very different number of deltas OTM at 1W than at 1Y —
moneyness interpolation badly distorts short-dated wings. `vol(Δ,T)`
interpolates σ²T linearly in T at fixed delta between pillar smiles;
`vol(K,T)` runs the sticky-delta fixed point (strike → delta → vol →
delta …, converges in a few iterations; consistency
`vol(K,T) = vol(Δ(K),T)` tested to 1e-9). Calendar arbitrage is
monitored at fixed delta: w non-decreasing in T along each pillar
coordinate. Outside the pillar range vols extrapolate flat at fixed
delta (documented choice).

## 5. Why Heston on top — and vs the alternatives

The surface answers "what is today's vanilla price". A *model* is
needed for anything path- or dynamics-dependent: barriers/touches,
forward smiles, vega/vanna/volga aggregation under scenarios. We use
**Heston under Garman–Kohlhagen** (q = r_f):

`dS/S = (r_d − r_f)dt + √v dW_S`, `dv = κ(θ−v)dt + ξ√v dW_v`,
`corr = ρ`.

Compared against at least two alternatives:

* **Local vol (Dupire)**: fits vanillas *exactly*, but its smile
  dynamics are sticky-strike-like — the smile moves the wrong way when
  spot moves, mispricing forward-skew products; and FX pillar data
  (5 strikes × 6 expiries) is far too sparse for a stable local-vol
  extraction. Heston's smile floats with spot, much closer to observed
  FX sticky-delta behaviour.
* **SABR (per-expiry)**: excellent single-expiry smile parameterisation
  (and popular for rates), but it is a *slice* model — no consistent
  term structure or calendar dynamics, and β is unidentifiable from FX
  pillars. Heston is one global model across all expiries.
* **Vanna–volga alone**: static, no dynamics, wing arbitrage risk (see
  above) — it is retained as the market benchmark, not the model.
* (Jump additions — Bates — would fix the short-dated skew failure mode
  F1 at the cost of 3 more weakly identified parameters; documented as
  the known extension.)

Implementation choices: the **little-trap** characteristic function
(Albrecher et al. 2007) avoids the complex-log branch instability of
the original Heston form (tested stable to T = 15y); **two independent
Fourier methods** — Gil-Pelaez P1/P2 adaptive quadrature and
Fang–Oosterlee COS with analytic cumulant truncation (per-strike
interval, shared CF grid) — cross-validated to < 1e-6; **Feller
condition** reported, not enforced (calibrated FX smiles routinely
violate it; QE Monte Carlo handles v = 0 correctly). Monte Carlo:
full-truncation Euler (robust baseline, fine steps) and Andersen QE
(moment-matched CIR transition, accurate on coarse steps), both within
3 SE of Fourier in tests.

**Calibration** (`calibration.py`): vega-weighted implied-vol residuals
over 5 pillars × 6 expiries, bounded trust-region least squares, COS
pricing (N = 256) with robust implied-vol inversion. The **κ–ξ ridge**:
vanillas identify ρ and v0 tightly and the *combination* ξ²/κ (long-run
convexity) far better than κ, ξ separately — recovery tolerances are
set accordingly and the ridge is demonstrated numerically in
`VALIDATION.md`.

## 6. Assumptions register

| # | Assumption | What breaks if violated |
|---|---|---|
| A1 | BF quotes are *smile* butterflies (simplified), not one-vol broker strangles. | Wing vols off by up to ~1 vol pt at 10Δ for skewed pairs; surface marks and exotic adjustments wrong (failure mode F3). |
| A2 | Delta convention per pair/tenor is known and correct (pa for JPY-style, spot→forward switch at 1y). | Pillar strikes shift by up to 1.6 JPY (USDJPY 1y ATM); surface wrong by 0.2–0.3 vol pts at pillars (measured, F2). |
| A3 | Rates r_d, r_f are deterministic, continuously compounded, ACT/365F; forward from covered interest parity. | Long-dated (>2y) prices need stochastic rates / cross-currency basis; forwards off by the basis (F6). |
| A4 | Pure diffusion (no jumps). | Short-dated (≤1m) skew/kurtosis around CB events underpriced; calibration pushes ξ up or leaves 1w residuals (F1). |
| A5 | Frictionless single vol per (K,T): no bid/ask, one cutoff (no NY vs Tokyo cut distinction). | Cutoff mismatches move short-dated vols by ~0.1–0.3 vol pts; governance issue, not model (DESK_GUIDE). |
| A6 | Five pillars per expiry suffice to identify the smile. | Beyond 10Δ everything is extrapolation — SVI and VV disagree by design (F5); 10Δ quotes themselves are illiquid. |
| A7 | Total variance interpolated linearly in T at fixed delta; flat extrapolation outside pillars. | Non-monotone forward variance between pillars is smoothed over; event weights (CB dates) need a forward-vol-aware interpolator. |
| A8 | Heston parameters constant over the surface's life; recalibrated daily. | P&L attribution mixes parameter jumps with market moves; ridge drift in (κ, ξ) between days is normal and must not be traded on. |
| A9 | Vol level bump for Heston vega/vanna/volga = parallel shift of √v0 and √θ. | Bucketed (v0-only vs θ-only) vega term-structure risk needs the two bumps separated — both are available in `greeks.py`. |

Each edge case implied above is unit-tested (`tests/test_edge_cases.py`):
T→0, flat smiles (RR = BF = 0 ⇒ degenerate SVI), negative rates both
legs, 35% EM vol with 30% carry, ρ = ±1, ξ→0 (Heston → GK), deep
ITM/OTM, unattainable pa deltas.
