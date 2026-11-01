# Methodology: FX-Linked Fixed Income

Multi-currency discount curves, FX forward curves via covered interest
parity (CIP), cross-currency basis from float-float basis swaps, and the
pricing and risk of FX forwards, FX swaps, fixed-fixed cross-currency
swaps and float-float basis swaps.

## 1. The pipeline and the maths

### 1.1 Discount curves per currency

Each currency gets a curve `DF(t)` bootstrapped from:

- deposits (simple interest): `DF(tau) = 1 / (1 + r tau)`;
- annual par swaps (unit accrual): `1 = c_n sum_{i=1..n} DF(i) + DF(n)`.

Each quoted swap adds one pillar at its maturity. Discount factors at
unquoted integer years in between follow log-linear interpolation, and the
new pillar is solved with Brent so the quote prices at par on that curve.
No intermediate par quotes are invented. Interpolation is log-linear in DF
(piecewise-constant forwards) with flat-forward extrapolation past the last
pillar. Zero rates are continuously compounded: `z(t) = -ln DF(t) / t`.

### 1.2 FX forwards via CIP

Pairs are BASE/QUOTE (EURUSD = USD per EUR); domestic = quote currency,
foreign = base currency. Without basis:

```
F(T) = S * DF_f(T) / DF_d(T)
```

An outright has two identical MTM forms (tested to 1e-10):

```
V = N * S * D_base(T) - N * K * D_quote(T)
V = N * D_quote(T) * (F_mkt(T) - K)
```

### 1.3 Cross-currency basis

Market forwards deviate from CIP. The basis quote `s(T)` is the par spread
of a float-float cross-currency basis swap of maturity `T` (constant
notionals, exchanged at both ends, `basis_frequency` resets a year,
default 4). The spread sits on the leg of the non-collateral currency;
the collateral currency (USD by default) pays its floating rate flat.

Discounting follows the collateral. Collateral-currency cashflows are
discounted on that currency's own curve, where its floating leg is worth
par. The other currency's discount curve `D` is bootstrapped so every
quoted basis swap prices at par, with its floating rates projected off its
own curve `P`:

```
D(t_n) = (1 - sum_{i<n} (F_i + s_n) tau D(t_i)) / (1 + (F_n + s_n) tau),
F_i tau = P(t_{i-1}) / P(t_i) - 1
```

Between reset dates `ln(D/P)` is linear in `t`, so zero spreads give
`D == P` at every time (tested to 1e-14). For EURUSD the EUR curve
carries the basis; for USDJPY the JPY (quote) curve does, not the USD one.
Market forwards are `F_mkt(T) = S * D_base(T) / D_quote(T)`.

`basis_adjusted_curve` keeps the older zero-spread shortcut
(`z_adj = z_f + s`) and `basis_model="zero_spread"` selects it with a
DeprecationWarning. Under it a parallel foreign-rate move and a basis move
are the same risk factor. `implied_basis_from_forwards` inverts forwards to
that zero-spread equivalent; it differs from the float-float par spread by
the gap between `T` and the spread annuity (under 0.3bp in the synthetic
normal regime, 2.3bp in the crisis regime).

### 1.4 Instruments

- FX forward (outright): above.
- FX swap: near and far exchange of the same base notional, valued as the
  four-cashflow sum and tested against two outrights.
- Fixed-fixed cross-currency swap: two fixed-rate bonds with final
  (optionally initial) notional exchange, each leg discounted on its
  currency's collateral-consistent curve. Par rates are closed-form.
  `solve_par_basis` solves a parallel shift of the basis quotes that zeroes
  the PV.
- Float-float basis swap (`BasisSwap`): constant notionals; the
  mark-to-market resetting variant is not implemented.

### 1.5 Dates and day counts

- `spot_date` counts business days on the non-USD calendars, then rolls to
  a day good in both currencies plus USD (T+2 default, T+1 via
  `spot_lag_days`).
- `tenor_dates`: ON runs T to T+1, TN runs T+1 to spot, SN runs spot to
  spot+1; W/M/Y tenors run from spot with modified following and the
  end-end rule.
- Money-market accrual per currency (`MONEY_MARKET_DAYCOUNT`): ACT/360 for
  USD, EUR, JPY, CHF; ACT/365F for GBP, AUD, CAD, NZD. `CIPQuotes` takes
  per-currency accruals `tau_dom` and `tau_for`.
- Calendars (`fx_rates.calendars`) are partial: weekends plus fixed-date
  holidays for USD, EUR, GBP, JPY.
- Curve times are year fractions; instruments are specified in years.

### 1.6 Risk

- FX delta `dV/dS` in quote currency per unit of spot. For an outright it
  is `N * D_base(T)`.
- DV01 per currency: central difference of a 1bp parallel bump of that
  currency's projection curve. For the basis currency the discount curve
  is re-bootstrapped with the basis quotes held fixed.
- KRD: one pillar at a time; log-linear DF keeps the bump local (tested).
- Basis DV01: 1bp on every basis quote with the projection curves held
  fixed. A par basis swap has zero rate DV01 and basis DV01 equal to minus
  its spread annuity (exact identity, tested). For an FX forward the basis
  DV01 and the foreign-rate DV01 differ (about 0.5% on the sample book).
- Scenario engine: joint spot, curve and basis shocks by full revaluation.
- Carry: age a forward with the market frozen.

## 2. Model choice

| Alternative | Why not here |
| --- | --- |
| CIP only, no basis | With synthetic normal-regime data the 5y EURUSD forward is off by 148 pips, a -$1.21m PV on a EUR 100m forward struck at the market forward (pipeline step 3). |
| Full OIS/CSA multi-curve with tenor basis and global calibration | The production standard. It needs OIS and tenor-basis quotes and CSA terms per counterparty. This library has one curve per currency for projection and collateral-currency discounting, plus the xccy basis curve (assumption A2). |
| Nelson-Siegel / Svensson per currency | Does not reprice the inputs. Used here only to generate synthetic truth. |
| Cubic spline on zeros | Non-local: a 5y quote move reshapes the 2y DF. |

## 3. Assumptions register

Each assumption states what breaks if violated.

- **A1 Deterministic rates and basis.** No convexity or timing
  adjustments. Long-dated forwards and MtM cross-currency swaps carry a
  small convexity adjustment under stochastic rates.
- **A2 One curve per currency plus the xccy basis.** Each currency's
  projection curve also discounts its collateral-currency cashflows; there
  is no OIS-IBOR tenor basis and no CSA optionality. Market forwards stay
  correct because the basis quotes are calibrated.
- **A3 Unit accrual on swap fixed legs** (annual coupons pay `c`; xccy
  coupons `c / frequency`). Real 30/360 vs ACT/360 schedules move par rates
  by a few bp.
- **A4 Log-linear DFs between quoted swap maturities.** KRDs at unquoted
  years are interpolation results, not market facts.
- **A5 Constant-notional basis swaps.** The MtM-resetting basis swap,
  which is the liquid instrument for some pairs, is not modelled.
- **A6 Partial calendars.** Weekends plus fixed-date holidays only; no
  moveable feasts, no turn-of-year day-count spikes on the curve.
- **A7 One pair per `MarketState`.** Triangular consistency of CIP
  forwards is tested to 1e-12 across EURUSD, USDJPY and EURJPY.
- **A8 Continuous compounding for zeros, simple compounding for deposits
  and arbitrage round trips.**
- **A9 Every numeric input is finite.** NaN or Inf raises `ValueError`
  through `fx_rates._validation.require_finite`.

## 4. Units

- Rates: continuously compounded decimal, except deposit and
  arbitrage-detector rates (simple).
- Basis spreads: decimal; reported in bp.
- Forward points: `(F - S) * 1e4`, or `* 1e2` for JPY-quoted pairs.
- PVs, deltas and DV01s in the quote currency; DV01s are signed per +1bp.
