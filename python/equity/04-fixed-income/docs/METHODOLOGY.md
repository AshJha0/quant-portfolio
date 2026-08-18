# Methodology — Fixed Income Pricing & Risk Analytics (`fi_rates`)

Pipeline: **yield curve construction → bootstrapping → bond pricing →
duration → convexity → key-rate duration → scenario analysis.**

---

## 1. Why sequential bootstrap + local interpolation (and not a global fit)?

The curve engine is a **sequential bootstrap** over deposit / FRA / par-swap
(or coupon-bond) quotes, with **log-linear interpolation on discount factors**
as the default. The main alternatives considered:

| Approach | Exact repricing of inputs? | Smooth forwards? | Local? (quote bump moves only its segment) | Typical user |
|---|---|---|---|---|
| **Sequential bootstrap + log-linear DF (chosen)** | Yes, to 1e-10 (tested to ~4e-16) | No — piecewise-constant forwards, steps at pillars | Yes | Swap/rates trading desks (EOD risk, hedging) |
| Bootstrap + monotone cubic on zeros (offered as `pchip_zero`) | Yes at pillars | C1 forwards, no spline overshoot | Mostly (neighbouring segments move slightly) | Desks needing presentable forwards |
| **Nelson–Siegel / Svensson global fit** | No — least-squares residuals of 1–10bp | Yes, parametric and very smooth | No — every quote moves the whole curve | Central banks, economists, relative-value screens |
| Cubic spline on zeros (natural) | Yes at pillars | C2 but oscillates/overshoots between pillars | No | Rarely used raw in production |

Trade-offs, stated:

* A trading desk must **reprice its hedge instruments exactly** — a curve
  that misses the 5y swap by 2bp creates phantom P&L and phantom DV01 on
  every mark. That rules out Nelson–Siegel/Svensson for desk risk. NS/S wins
  where the goal is a *parsimonious description* (term-premium research,
  cross-country comparison, fitting hundreds of noisy bond prices where exact
  repricing is meaningless) — that is why the Fed (Gürkaynak–Sack–Wright) and
  the ECB publish Svensson curves while swap desks bootstrap.
* **Locality matters for hedging**: with log-linear DF, bumping the 10y quote
  changes discount factors only between the 7y and 10y pillars (and beyond via
  the sequential chain), so bucketed DV01s are clean. Global fits smear a 10y
  bump into the 2y bucket.
* The price of locality is **ugly forwards**: log-linear DF ⇒
  piecewise-constant instantaneous forwards with jumps at pillars (this exact
  identity is unit-tested). Linear-on-zeros is worse — a sawtooth
  (demonstrated with numbers in `VALIDATION.md`). The `pchip_zero` option is
  a "monotone-convex-lite": monotone cubic (PCHIP) on zeros gives continuous,
  non-overshooting forwards at a small cost in locality and in bootstrap
  exactness away from pillars. Full Hagan–West monotone-convex is the
  production refinement of the same idea and is out of scope.

All three interpolations share one code path and one test suite; the choice
is a constructor argument, which is itself the point — interpolation is a
**model choice with risk consequences**, not a plumbing detail.

## 2. Instrument treatment

* **Deposits**: simple interest, `P(T) = 1/(1 + r·T)` — closed form.
* **FRAs**: simple forward, `P(T2) = P(T1)/(1 + r·(T2−T1))`. (Futures would
  need a convexity adjustment — out of scope, noted in the assumptions.)
* **Par swaps**: annual (or semi/quarterly) fixed leg with equal accruals
  `1/frequency`; the floating leg is worth par on a single self-discounting
  curve, so the pillar condition is `r·Σ αᵢP(tᵢ) + P(T) = 1`. Each new
  pillar's discount factor is solved with Brent's method because intermediate
  coupons are interpolated off the partially built curve.
* **Coupon bonds** (`bootstrap_bond_curve`): same sequential scheme with the
  pillar DF solved so the quoted dirty price is matched exactly.

Failure behaviour is explicit: any quote that no admissible discount factor
in `(1e-8, 20)` can reprice (e.g. a deposit rate below −1/T, or a bond price
above the sum of its cashflows' maximum PV) raises `ValueError` naming the
instrument, the pillar and the quote.

## 3. Bond pricing conventions

* **Street-convention YTM**: cashflow `k` periods after settlement (with
  fractional current period `w` measured in the bond's own day count) is
  discounted by `(1 + y/m)^−(w+k−1)`. Consequences that are unit-tested: a
  par bond has `YTM == coupon` exactly at any coupon date; price→YTM→price
  round-trips to 1e-10; negative yields work for `1 + y/m > 0`.
* **Clean/dirty/accrued**: `dirty = clean + accrued`, accrued in the bond's
  day count over the current (unadjusted) coupon period.
* **Curve pricing** maps payment dates to times with ACT/365F from settlement
  and discounts with `P(t)·e^(−z·t)` where `z` is a **continuously
  compounded z-spread** (solver provided; round-trips to 1e-10).
* **FRN**: priced off the same curve that projects its forwards; at a reset
  date with zero margin the telescoping sum gives price = par *exactly* —
  used as an identity test rather than an approximation.

## 4. Risk measures

* Macaulay / modified duration and convexity are **analytic** in YTM space
  and are verified against central finite differences of the pricer.
* DV01 is produced two ways on purpose: analytic from YTM
  (`D_mod · P · 1e-4`) and **effective** from a ±1bp parallel bump of the
  pillar zeros with full repricing. They differ by the Jacobian
  `dy/dz ≈ (1 + y/m)` between periodic-yield and continuous-zero spaces —
  the test asserts exactly this relationship rather than pretending the two
  numbers are identical.
* Key-rate durations use **triangular bumps** at 2/5/10/30y (configurable)
  that form a partition of unity across pillars, so the KRD ladder sums to
  the parallel DV01 up to the cross-gamma of the finite bumps (measured:
  ~5e-8 relative under log-linear; ~5e-7 under PCHIP — see `VALIDATION.md`
  for why non-local interpolation loosens the match).

## 5. Assumptions register

Each assumption states *what breaks if violated*.

1. **Single-curve (pre-OIS) framework — the big one.** One curve both
   projects forwards and discounts; the swap floating leg is worth par.
   **This has not been the production standard since 2008.** Post-crisis,
   desks use **multi-curve / OIS discounting**: cash flows are discounted on
   the OIS (SOFR/ESTR) curve while forwards are projected on separate tenor
   curves (and, since the 2020 SOFR transition, on RFR compounding curves);
   collateral (CSA) terms determine the discount curve per counterparty.
   *What breaks*: with a wide OIS–IBOR-style basis, single-curve swap PVs are
   wrong by roughly (basis × annuity) — tens of bp of upfront in 2008–2012;
   the FRN price-equals-par identity fails by the discounting/projection
   basis; DV01 splits across curves. The architecture here (curve objects +
   root-solved bootstrap) extends to multi-curve by adding a second curve and
   re-deriving the par condition; it is deliberately not done to keep the
   project reviewable.
2. **Times, not dates, on the curve.** Curve pillars live in ACT/365F years
   from the valuation date; instruments are quoted directly in year
   fractions. *What breaks*: real quote sheets have spot lag (T+2), IMM
   dates, and per-instrument day counts (ACT/360 money market); ignoring
   them shifts pillars by days — a ~0.5bp effect on short rates, immaterial
   for the risk analytics demonstrated, fatal for production P&L.
3. **No settlement lag / no ex-dividend periods.** Settlement = trade date.
   *What breaks*: accrued interest and short-stub discounting are off by the
   lag; Treasury auctions and ex-div gilts need special handling.
4. **Simplified calendars.** Modified-following over a weekend-only calendar,
   off by default; accrual on unadjusted dates. *What breaks*: payment dates
   can land on holidays; adjusted-vs-unadjusted accrual differences of a few
   days of coupon.
5. **30/360US without the end-of-February rule**; ACT/ACT ISDA (not ICMA).
   *What breaks*: bonds issued/maturing at Feb month-end accrue a day or two
   differently; US Treasuries actually use ACT/ACT **ICMA** (period-based),
   which differs from ISDA in stub periods.
6. **Z-spread as the only credit treatment.** Corporate credit is a constant
   continuous spread over the government/swap curve. *What breaks*: no
   default/recovery dynamics (a CDS-consistent hazard-rate model would price
   distressed bonds very differently), no term structure of spread, no
   spread-vs-rate correlation in scenarios.
7. **Deterministic rates — no optionality.** *What breaks*: callable bonds
   and MBS have **negative convexity** near the call/refi region; everything
   in `risk.py` would overstate their upside. Out of scope and flagged in
   `VALIDATION.md`.
8. **FRAs treated as forward-starting deposits (no futures convexity
   adjustment).** *What breaks*: using futures quotes directly overstates
   forward rates by the convexity adjustment (~1–5bp at 2–3y for realistic
   vol); a HJM/Hull–White adjustment would be needed.
9. **Flat zero extrapolation beyond the last pillar (warned).** *What
   breaks*: 40y+ liabilities discounted off a 30y-last-pillar curve are
   mispriced if the true long end slopes (cf. Solvency II UFR debates); the
   package warns on every such query.
10. **Historical scenarios are stylised approximations** — published
    magnitudes compressed into instantaneous pillar-shift vectors (sources
    and caveats in `scenarios.py` docstrings and `DESK_GUIDE.md`). *What
    breaks*: no timing/carry over the episode, no spread or vol moves, no
    convexity hedging flows.

## 6. Numerical choices

* Brent root-solving with `xtol=1e-16, rtol≈9e-16` for bootstrap pillars,
  YTM and z-spread — round-trip tolerances of 1e-10 are then comfortable
  (measured residuals ~1e-16, see `VALIDATION.md`).
* Central differences for all numerical Greeks (O(h²) error), step 1bp for
  curve bumps, 1e-6..1e-5 for YTM-space checks.
* All randomness (quote noise, portfolio composition) flows through
  `numpy.random.default_rng(seed)` — the entire test suite and pipeline are
  deterministic and offline.
