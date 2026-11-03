# Methodology: Fixed Income Pricing and Risk Analytics (`fi_rates`)

Pipeline: yield curve construction, bootstrapping, bond pricing, duration,
convexity, key-rate duration, scenario analysis.

---

## 1. Sequential bootstrap with local interpolation

The curve engine is a sequential bootstrap over deposit, FRA and par-swap
(or coupon-bond) quotes. The default interpolation is log-linear on discount
factors. The alternatives:

| Approach | Exact repricing of inputs | Forward smoothness | Locality (a quote bump moves only its segment) | Typical user |
|---|---|---|---|---|
| Sequential bootstrap + log-linear DF (default) | Yes, to 1e-10 (measured ~4e-16) | Piecewise-constant forwards, steps at pillars | Yes | Swap and rates trading desks |
| Bootstrap + PCHIP on zeros (`pchip_zero`) | Yes at pillars | C1 zeros, C0 forwards (kinks at pillars), no overshoot | Mostly: a bump moves the two neighbouring segments | Desks that want presentable forwards |
| Nelson-Siegel / Svensson global fit | No, residuals of 1-10bp | Parametric and smooth | No: every quote moves the whole curve | Central banks, relative-value screens |
| Natural cubic spline on zeros | Yes at pillars | C2 zeros but overshoots between pillars | No | Rarely used raw |

A desk must reprice its hedge instruments exactly. A curve that misses the
5y swap by 2bp creates P&L and DV01 that do not exist. That rules out
Nelson-Siegel/Svensson for desk risk. It suits parsimonious description:
the Fed (Gurkaynak, Sack and Wright) and the ECB publish Svensson curves.

Locality matters for hedging. With log-linear DF, bumping the 10y quote
changes discount factors between the 7y and 10y pillars (and later pillars
through the sequential chain), so bucketed DV01s are clean.

The cost of locality is a rough forward curve. Log-linear DF gives
piecewise-constant instantaneous forwards with jumps at pillars (unit
tested). Linear on zeros gives a sawtooth (numbers in `VALIDATION.md`).
PCHIP on zeros gives C1 zero rates; the instantaneous forward
`f = z + t z'` is then continuous but kinked at the pillars
(`tests/test_carry_curve_claims.py`).

Hagan-West monotone convex (`monotone_convex`; Hagan and West 2006,
Applied Mathematical Finance 13(2), Section 5) interpolates the
instantaneous forward itself. On each segment the forward is the discrete
forward plus a quadratic correction that integrates to zero, so pillars
reprice exactly. The correction is picked from four shapes so the forward
never leaves the range of its two node values. In plain terms: the
forward curve is continuous, it rises wherever the inputs rise, and a
pillar bump moves at most two segments on each side. It does not by
itself guarantee positive forwards, and its first segment `(0, t_1]` is
interpolated too, so the short end differs a little from the other
schemes. Tested for exact repricing, continuity at pillars, monotone
inputs giving a monotone forward, locality, and every one of the four
correction shapes (`tests/test_round3_fix_rates.py`).

The paper's positivity collar (their Section 5) is available as an
option: `DiscountCurve(..., "monotone_convex", positive_forwards=True)`.
It clamps each node forward into `[0, 2 min(f^d_i, f^d_{i+1})]` (end nodes
into `[0, 2 f^d]`), and with node values in that band none of the four
correction shapes can dip below zero. So if every discrete forward is
positive, so is the whole forward curve; pillars still reprice exactly.
It is off by default because it changes the curve wherever it binds, and
the constructor refuses it when a discrete forward is zero or negative.
Example: discrete forwards 0.1 %, 5 %, 4.5 %, 0.2 %, 4 % give a plain
forward that goes below zero near the short end and before 5y; with the
collar the minimum is zero or above (`tests/test_round4_collar.py`).
One edge case needed a choice of our own. When a node forward equals the
discrete forward of a neighbouring segment exactly (two equal discrete
forwards side by side), one end of the correction `g` is zero and the
other is not. Each of the paper's four shapes, taken to that limit,
squeezes its transition into a point, so the forward used to jump at the
pillar. We use the plain quadratic of region (i) there instead: it is
continuous, still integrates to zero (pillars reprice), and moves by at
most a third of the non-zero end inside the segment. "Equal" means equal
to within 1e-9 relative, because the discrete forwards come from
differences of `ln P` and rarely cancel to the last bit. Forwards that
are close but not equal still use the paper's shapes, which stay
continuous but turn steeply over a width of about `|g0/g1|` of the
segment; the paper's amelioration step (not implemented) is the cure. With the collar on,
that keeps the forward positive. Worked example: forwards 3 %, 3 %, 5 %
at 1y, 2y, 3y; the forward is now continuous at 2y
(`tests/test_round4_collar.py::test_equal_adjacent_forwards_no_jump`).

All four interpolations share one code path and one test suite; the
choice is a constructor argument.

## 2. Instrument treatment

* Deposits: simple interest, `P(T) = 1/(1 + r T)` on year fractions.
* FRAs: simple forward, `P(T2) = P(T1)/(1 + r (T2 - T1))`. Futures would
  need a convexity adjustment and are not supported.
* Par swaps: fixed leg with equal accruals `1/frequency`; the floating leg
  is worth par on a single self-discounting curve, so the pillar condition
  is `r sum(alpha_i P(t_i)) + P(T) = 1`. Each pillar discount factor is
  solved with Brent's method because intermediate coupons are interpolated
  off the partially built curve.
* Coupon bonds (`bootstrap_bond_curve`): the same sequential scheme, with
  the pillar DF solved to match the quoted dirty price.

Any quote that no discount factor in `(1e-8, 20)` can reprice raises
`ValueError` naming the instrument, the pillar and the quote.

## 3. Dates, day counts and calendars

* Day counts: ACT/365F, ACT/360, 30/360US (without the end-of-February
  rule), ACT/ACT-ISDA and ACT/ACT-ICMA.
* ACT/ACT-ICMA is the government-bond convention (US Treasuries, gilts,
  most sovereigns). A regular coupon is exactly `face * c / f`. Accrued
  interest is that coupon times (days accrued / days in the period). A
  short front stub accrues against its notional regular reference period.
  ACT/ACT-ISDA differs from ICMA in every period: a semiannual 4% bond
  under ISDA pays 1.9836, 1.9891, 2.0123 instead of 2.0000.
* Schedules roll back from maturity. With the end-of-month rule (default
  on) a month-end maturity keeps every coupon on a month end: a 28-Feb
  maturity pays on 31 Aug.
* `fi_rates.calendars` gives weekend plus fixed-date holiday calendars for
  USD, EUR, GBP and JPY, joint calendars (`"EUR+USD"`), and following,
  modified following and preceding rolls. The holiday lists are partial:
  they omit moveable feasts, floating holidays and weekend observance.

## 4. Bond pricing conventions

* Street-convention YTM: cashflow `k` periods after settlement is
  discounted by `(1 + y/m)^-(w+k-1)`, with `w` the remaining fraction of
  the current period (under ICMA, remaining days over reference-period
  days). Unit tested: a par bond has `YTM == coupon` at any coupon date
  (a hypothesis property for ICMA bonds), price to YTM to price round
  trips to 1e-10, and negative yields work for `1 + y/m > 0`.
* `dirty = clean + accrued`; accrued uses the bond's day count over the
  current unadjusted coupon period.
* Curve pricing maps payment dates to times with ACT/365F from settlement
  and discounts with `P(t) exp(-z t)`, `z` a continuously compounded
  z-spread.
* FRN: priced off the curve that projects its forwards. At a reset with
  zero margin the price is par. Mid-period, `current_fixing` gives the
  known coupon (ACT/360) to the next payment date.

## 5. Risk measures

* Macaulay and modified duration and convexity are analytic in YTM space
  and checked against central finite differences of the pricer.
* DV01 is computed two ways: analytic from YTM (`D_mod P 1e-4`) and
  effective from a 1bp parallel bump of the pillar zeros with full
  repricing. They differ by the Jacobian `dy/dz` between periodic-yield
  and continuous-zero spaces; the test asserts that relationship.
* Key-rate DV01s use triangular bumps at 2/5/10/30y (configurable) that
  form a partition of unity across pillars. The ladder sums to the
  parallel DV01 up to the cross-gamma of the finite bumps, plus, under
  PCHIP, the non-locality of the interpolator. Residuals for the sample
  book are in `VALIDATION.md` §3.
* KRDs are sensitivities to pillar zero rates, not to hedge-instrument
  par rates. There is no Jacobian to the quoted instruments.

## 6. Carry and roll-down

`carry_rolldown` works on a static curve over a horizon:

* coupon income = coupons received + change in accrued interest
* funding = dirty price x repo rate x ACT/360 days
* carry = coupon income - funding
* roll-down = change in clean price, split into pull-to-par (clean price
  change at the bond's own constant continuous yield) and curve roll-down
  (the rest)

`carry + rolldown` is the static-curve horizon P&L net of financing.
Coupons are not reinvested.

## 7. Assumptions register

Each assumption states what breaks if it is violated.

1. **Single curve.** One curve projects forwards and discounts, so the
   swap floating leg is worth par. Desks have used OIS discounting with
   separate projection curves since 2008, and RFR compounding since the
   SOFR transition. What breaks: with a wide OIS-IBOR basis, swap PVs are
   wrong by roughly basis x annuity, the FRN par identity fails, and DV01
   splits across curves.
2. **Curve inputs are year fractions.** Quotes are given as times, not
   dated instruments. What breaks: spot lag, IMM dates and per-instrument
   day counts shift pillars by days, about 0.5bp on short rates.
3. **No settlement lag and no ex-dividend periods** in bond pricing.
   Settlement is the date passed in; `calendars.add_business_days` gives
   T+1 or T+2 if the caller wants it.
4. **Calendars cover seven centres.** USD, EUR (TARGET), GBP, JPY, CHF,
   CAD and AUD from `qp_core.holidays`, 2000-2099. Until milestone v2.1
   these were weekend plus fixed-date holidays only, so a schedule could
   roll onto Good Friday or Thanksgiving; other centres are not covered.
5. **30/360US without the end-of-February rule.**
6. **Z-spread is the only credit treatment.** No default or recovery
   dynamics, no spread term structure.
7. **Deterministic rates, no optionality.** Callables and MBS have
   negative convexity that `risk.py` does not capture.
8. **FRAs as forward-starting deposits.** Futures quotes would overstate
   forwards by the convexity adjustment (about 1-5bp at 2-3y).
9. **Flat zero extrapolation beyond the last pillar**, with a warning on
   every query. The fx_rates twin extrapolates the last forward flat.
10. **Historical scenarios are instantaneous pillar shifts.** No timing,
    carry, spread or vol moves over the episode. Sources are in
    `scenarios.py`.

## 8. Numerical choices

* Brent root solves with `xtol=1e-16, rtol ~ 9e-16` for bootstrap
  pillars, YTM, z-spread and the pull-to-par yield.
* Central differences for numerical Greeks, 1bp curve bumps, 1e-6 to 1e-5
  YTM steps.
* All randomness flows through `numpy.random.default_rng(seed)`. Tests and
  the pipeline are deterministic and offline.

## November 2026 additions: OIS multi-curve, caps and swaptions, Hull-White

**Multi-curve.** Collateralised swaps discount on the OIS curve, and each
index tenor gets its own projection curve (Ametrano and Bianchetti 2013).
The OIS curve comes from OIS par rates (a compounded overnight leg is worth
`1 - D(T)`); the projection curve then comes from par swap rates, with the
float leg projected on it and discounted on OIS. Pillars are solved one at
a time with Brent's method, with log-linear discount factors in between.

**Caps and swaptions.** Market formulas: caplet = accrual x discount x
Black-76 on the forward; swaption = annuity x Black-76 on the forward swap
rate (Hull 2018, ch. 29). Bachelier versions for low or negative rates
(Brigo and Mercurio 2006, Section 6.4).

**Hull-White.** `dr = (theta(t) - a r) dt + sigma dW`, fitted to today's
curve. Closed-form zero bonds, zero-bond options, caplets and Jamshidian
swaptions (Brigo and Mercurio 2006, eqs. 3.36-3.45). Constant a and sigma.

*Two-curve.* The model is fitted to the OIS discount curve. A floating leg
on another index uses Henrard's (2010) deterministic multiplicative
spread: the ratio of projection to discount forward factors for each float
period is held at today's value. The payer swaption is then a put on a
coupon bond with coefficient `K tau` on fixed dates, `1` at the end and
`-(beta - 1)` on float dates, struck at `beta_1`. Jamshidian still applies
because every zero bond falls as `r` rises, so the pieces all change sign
at the same root; the code refuses a bond that does not cross its strike.
Checked: with the forward curve equal to the discount curve the
single-curve price comes back to 1e-14; payer minus receiver equals
`annuity x (F - K)` from `rate_options` (an independent code path); a
200k-path Monte Carlo agrees within 4 standard errors.

*Calibration.* `calibrate_hull_white` turns each swaption vol into a price
with the annuity formula, then fits `(a, sigma)` by least squares on price
errors divided by the annuity (so residuals are in rate units). With `a`
given only `sigma` is fitted, which is the usual choice because one
expiry column barely pins `a` down. Checked by round trip: prices from a
known model, turned into normal or Black vols, give back `a` and `sigma`
to 1e-6 relative.
