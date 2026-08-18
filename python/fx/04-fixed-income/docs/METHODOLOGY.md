# Methodology — FX-Linked Fixed Income

Multi-currency discount curves, FX forward curves via covered interest
parity (CIP), cross-currency basis, and the pricing/risk of FX forwards,
FX swaps and fixed-fixed cross-currency swaps.

## 1. The pipeline and the maths

### 1.1 Discount curves per currency

Each currency gets a discount curve `DF(t)` bootstrapped from:

- **deposits** (simple interest): `DF(tau) = 1 / (1 + r * tau)`;
- **annual par swaps** (unit accrual): `1 = c_n * sum_{i=1..n} DF(i) + DF(n)`,
  peeled off recursively as `DF(n) = (1 - c_n * A_{n-1}) / (1 + c_n)`.

Interpolation is **log-linear in DF** (piecewise-constant instantaneous
forwards), flat-forward extrapolated beyond the last pillar. Zero rates are
continuously compounded, annualised: `z(t) = -ln DF(t) / t`.

### 1.2 FX forwards via CIP

Portfolio convention (CONVENTIONS.md): pairs are BASE/QUOTE, EURUSD = USD
per EUR; **domestic = quote currency (USD)**, **foreign = base currency
(EUR)**. Covered interest parity:

```
F(T) = S * DF_f(T) / DF_d(T)
```

Both legs of a forward are just zero-coupon bonds, so an outright's MTM has
two algebraically identical forms (identity-tested to 1e-10):

```
V = N * S * DF_f(T) - N * K * DF_d(T)          (two discounted cashflows)
V = N * DF_d(T) * (F(T) - K)                   (forward vs forward)
```

### 1.3 Cross-currency basis

Since 2008, market FX forwards systematically deviate from single-curve
CIP. We represent the deviation as a maturity-dependent **basis spread**
`s(T)` (decimal, negative for EURUSD) added to the foreign zero curve:

```
DF_f_adj(T) = DF_f(T) * exp(-s(T) * T),    z_adj(T) = z_f(T) + s(T)
F_mkt(T)    = S * DF_f_adj(T) / DF_d(T)
```

`s(T)` can be taken from basis-swap quotes directly or backed out of market
FX forwards (`bootstrap.implied_basis_from_forwards`,
`bootstrap.curve_from_fx_forwards`); the two representations are consistent
by construction and tested against each other. All *pricing* uses the
adjusted foreign curve; the pure foreign curve is retained so that interest
rate risk and basis risk are reported as separate factors.

### 1.4 Instruments

- **FX forward (outright)**: above.
- **FX swap**: near + far exchange of the same base notional; valued as the
  explicit four-cashflow sum and identity-tested against the sum of two
  outrights. Its spot delta nearly cancels (`DF_f(T1) - DF_f(T2)`), leaving
  mostly forward-points (rates + basis) risk — exactly the desk intuition.
- **Fixed-fixed cross-currency swap**: two fixed-rate bonds with final
  (optionally initial) notional exchange; the base-currency bond is
  discounted on the adjusted curve and converted at spot:
  `V = sign * (S * B_f[DF_f_adj] - B_d[DF_d])`. Par rates are closed-form
  (PV is linear in the coupon); the **par basis** is solved with Brent to
  ~1e-14 in the spread.

### 1.5 Risk

- **FX delta** `dV/dS` (cash, quote ccy per unit of spot); for an outright
  it equals `N * DF_f_adj(T)` — the base-currency equivalent position.
- **DV01 per currency**: central-difference parallel bump (+/-1bp) of one
  currency's zero pillars; **KRD** bumps one pillar at a time (log-linear DF
  interpolation makes the bump local — tested).
- **Basis DV01**: +/-1bp on the whole spread curve, curves rebuilt.
- **Scenario engine**: joint spot/curve/basis shocks by full revaluation;
  identical to a manual market rebuild (tested).
- **Carry**: age the forward with the market frozen — pure roll down the
  forward-points curve.

## 2. Why this model — alternatives considered

**Chosen: two bootstrapped single curves per currency + a separate
cross-currency basis spread curve applied to the foreign discount curve.**

| Alternative | Why not (here) |
| --- | --- |
| **1. Single-curve CIP only (no basis)** — textbook pre-2008 approach: discount each currency off its own swap curve and set `F = S*DF_f/DF_d`. | Demonstrably misprices: with the normal-regime data the 5y EURUSD forward is off by **147 pips**, a fictitious **-$1.20m PV on a EUR 100m** forward struck at the true market forward (see `run_pipeline.py` step 3). Post-2008 the basis is a persistent, tradeable market factor, not noise. |
| **2. Full multi-curve OIS/CSA framework** — separate projection and discount curves per currency, collateral-currency (CSA) discounting, tenor-basis curves, simultaneous global calibration. | The production standard, but it needs OIS quotes, tenor-basis quotes and CSA terms per counterparty, and a global solver; it obscures the pedagogical chain deposit -> swap -> CIP -> basis. Our single-curve + basis-spread model reproduces the same *market forwards* (they are inputs) and the same first-order risk factors with far less machinery. The simplification and its consequences are assumption **A2** below — a production desk difference deliberately documented rather than hidden. |
| **3. Parametric curves (Nelson–Siegel / Svensson) fitted per currency** | Smooth and great for econometrics, but does not exactly reprice the input instruments — unacceptable for a pricing/hedging library where the bootstrap round trip is the primary control (we hold it to < 1e-10; achieved ~1e-16). NS is still used here as the *generator* of synthetic truth. |
| Interpolation: cubic splines on zeros | Smoother forwards but non-local: a 5y quote move reshapes the 2y DF, contaminating KRD buckets. Log-linear DF keeps bumps local (tested) at the cost of forward-rate steps at pillars. |

## 3. Assumptions register

Each assumption states what breaks if violated.

- **A1 — Deterministic rates and basis.** No convexity/timing adjustments;
  forwards are priced by pure discounting. *Breaks:* long-dated FX forwards
  and MTM cross-currency swaps carry a (small) convexity adjustment under
  stochastic rates; wrong-way FX/rates correlation is invisible. Immaterial
  vs bid/ask below ~10y in G10.
- **A2 — Single discount curve per currency + separate basis spread**
  (vs full OIS/CSA multi-curve). We discount each currency's domestic
  cashflows off its own bootstrapped swap curve and push the *entire*
  cross-currency effect into one spread curve on the foreign leg.
  *What a production desk does differently:* discounts at the OIS rate of
  the **collateral currency** of each CSA (a USD-CSA EUR swap discounts EUR
  flows on EUR-vs-USD-OIS-adjusted curves), separates projection from
  discounting per index tenor, and calibrates all curves jointly.
  *Breaks:* our "basis DV01" mixes what a multi-curve desk splits into
  OIS-Libor and xccy-OIS bases; collateral optionality and CSA switches are
  unpriceable here. Market *forwards* remain correct because they are
  calibration inputs.
- **A3 — Unit accrual fractions on swap fixed legs** (annual coupons pay
  exactly `c * 1.0`; xccy coupons `c / frequency`). *Breaks:* real 30/360 vs
  ACT/360 schedule effects of a few bp on par rates; the day-count module
  exists and is used for date -> time conversion, but schedules are not
  holiday-adjusted.
- **A4 — Par swap quotes at every integer maturity** (complete 2..10y
  strip in the synthetic data). Real markets quote sparsely and interpolate
  par rates; the bootstrap supports that (it interpolates par rates onto
  the annual grid), but then intermediate-year DFs are interpolation-, not
  market-determined. *Breaks:* nothing reprices wrong, but KRD at
  non-quoted pillars is model-dependent.
- **A5 — Basis quoted as a zero-spread on the foreign discount curve**, not
  as the spread on a floating-floating basis swap. The two are equivalent
  to first order; the mapping from market basis-swap quotes to `s(T)` is a
  one-liner at short maturities and a small bootstrap at long ones.
  *Breaks:* second-order differences (annuity effects) of < 1bp on the
  spread for typical levels.
- **A6 — Calendar-day settlement (T+2 spot as trade date + 2 calendar
  days; tenors as year fractions).** No holiday calendars, no
  end-of-month rolls, no turn-of-year day-count spikes. *Breaks:* short-date
  (O/N, T/N) pricing and the year-end turn, where one settlement day can be
  worth tens of forward pips; documented failure mode in VALIDATION.md.
- **A7 — One currency pair per `MarketState`.** Cross-pair books (triangular
  positions) are handled by valuing each pair's book separately; triangular
  *consistency* of CIP forwards is nonetheless an exact identity and is
  tested to 1e-12 across EURUSD/USDJPY/EURJPY.
- **A8 — Continuous compounding for zeros, simple compounding for
  deposit quotes and arbitrage round trips.** Consistent internally;
  mixing conventions with external quotes requires conversion first.

## 4. Units and conventions summary

- Rates: continuously compounded, annualised, decimal (0.04 = 4%), except
  deposit and arbitrage-detector rates which are simple annualised.
- Basis spreads: decimal; reported in bp (1e-4) in tables.
- Forward points: `(F - S) * 1e4` (pips) for EURUSD-style pairs.
- All PVs, deltas and DV01s in the **quote (domestic) currency**; DV01s are
  *signed* sensitivities per +1bp.
