# Validation — FX-Linked Fixed Income

How the library was validated, with the evidence, and where it fails.
All numbers below are from `python examples/run_pipeline.py` (regime
`normal`, seed 42) and are reproduced by the test suite
(`python -m pytest tests -q`: **418 tests, all offline, deterministic**).

## 1. Analytic round trips and identities

| Check | Contract | Achieved | Test |
| --- | --- | --- | --- |
| Bootstrap reprices deposit quotes (both ccys, 4 regimes, 2 seeds) | < 1e-10 | 0.0 – 1e-16 | `test_bootstrap.py::TestBootstrapRoundTrip` |
| Bootstrap reprices par swap quotes | < 1e-10 | ~6e-17 | same |
| Bootstrapped DFs recover the true generator curve at every pillar | < 1e-12 | ~1e-16 | same |
| CIP identity `F/S = DF_f/DF_d` | exact | < 1e-14 | `test_fxforward.py` |
| Triangular consistency EURJPY = EURUSD x USDJPY forwards | 1e-12 rel | passes at all tenors | `test_fxforward.py` |
| Forward MTM: cashflow method == forward-vs-forward method | 1e-10 rel | passes incl. negative rates | `test_fxforward.py`, `test_edge_cases.py` |
| Forward struck at market forward has zero value at inception | ~0 | < 1e-6 USD on 10m EUR | `test_fxforward.py` |
| FX swap == sum of two outright forwards | 1e-8 abs | passes, incl. after shocks | `test_fxforward.py` |
| XCCY swap value == independent discounting of enumerated cashflows | 1e-8 abs | passes | `test_xccy.py` |
| XCCY par-rate solver zeroes PV (closed form) | exact | < 1e-10 * notional | `test_xccy.py` |
| XCCY par-basis solver zeroes PV (Brent, xtol 1e-16) | ~0 | < 1e-4 USD on 50m | `test_xccy.py` |
| Zero basis recovers pure CIP pricing (forwards and xccy) | 1e-14 | passes | `test_fxforward.py`, `test_xccy.py` |
| FX delta of an outright == `N * DF_f_adj(T)` | 1e-9 rel | passes | `test_risk.py` |
| Parallel DV01 == sum of KRD ladder | 1e-6 rel | passes both currencies | `test_risk.py` |
| KRD locality: a 2y cashflow hits only the 2y bucket; a 2.5y cashflow hits exactly {2y, 3y} | exact | passes | `test_risk.py` |
| Basis DV01 == foreign-curve DV01 for a pure FX forward (economic identity of the spread representation) | 1e-9 rel | passes | `test_risk.py` |
| Scenario engine == manual market rebuild | 1e-8 abs | passes | `test_scenarios.py` |
| CIP arbitrage P&L == independent hand-computed round trip | 1e-10 abs | passes both directions | `test_arbitrage.py` |

Economic sign tests (regressions against sign errors, the classic FX bug):
long-base forward has positive domestic DV01 and **negative foreign DV01**
(sign flip); basis DV01 negative for long base; EURUSD points **positive**
when `r_f < r_d` including the negative-EUR-rates regime (2019-style);
long-the-low-yielder carry negative; widening (more negative) basis helps
the receive-EUR xccy leg and the long-EUR forward.

## 2. Basis mispricing demonstration (real numbers)

Normal regime, seed 42: spot 1.0866, USD 5y zero 4.061%, EUR 5y zero
2.550%, 5y basis -25bp.

```
5y EURUSD CIP forward (no basis) : 1.1719
5y EURUSD market forward (basis) : 1.1867      (+147.4 pips)

EUR 100m 5y forward struck at the market forward:
  PV with basis curve   :        $-0.00        (par, by construction)
  PV ignoring the basis : $-1,203,203.97       (fictitious 'edge')
```

A CIP-only model would report a $1.2m day-one profit for selling this
forward at the market price. In the 2008-style crisis regime (front-end
basis -180bp) the same 1y mispricing is ~9x the normal regime's
(`test_edge_cases.py::TestBasisMispricing`).

## 3. Arbitrage detector calibration

- **No false positives:** CIP-consistent two-sided quotes never flag, for
  any forward inside the no-arb band `[F_lower, F_upper]`; the band edges
  are the exact detection frontier (tested at +/-1e-7).
- **Planted violation:** a forward 30 pips above the band yields a
  riskless 27.95bp of notional at maturity ($279,474 on $100m, pipeline
  step 8), matching the hand-computed round trip to 1e-10.
- **Costs matter:** the same mid-curve deviation that flags with 1-pip
  spreads is absorbed by 10-pip/50bp spreads (tested). This is the
  qualitative Du–Tepper–Verdelhan point (see failure mode F4).

## 4. Failure modes

- **F1 — Basis regime shifts.** The basis spread curve is calibrated to
  today's quotes; DV01s assume small moves. In stress the basis gaps
  (2008: EURUSD 3m basis beyond -150/-200bp; March 2020: ~-85bp in days)
  and short-end basis DV01 computed at -15bp badly understates the P&L of
  a -150bp jump. Mitigation: the scenario engine revalues in full — the
  2008 scenario on the sample book moves P&L **-$7.3m**, ~340x its 1bp
  basis DV01. Use scenarios, not Greeks, for tail basis risk.
- **F2 — Year-end turn spikes.** The turn shows up as a sawtooth in
  *daily* forward points; our spread curve is smooth in maturity and
  calendar-free (A6), so it prices *through* the turn, not the turn
  itself. A desk adds explicit turn dates to the curve. The stylised
  "EUR year-end turn" scenario (-40bp basis) bounds the book-level effect
  (+$0.86m on the sample book) without the daily microstructure.
- **F3 — Settlement/holiday approximations (A3/A6).** Calendar-day T+2 and
  unit accruals misplace cashflows by up to a few business days: irrelevant
  beyond ~1m tenors (< 1 pip), material for O/N–1w points and exactly at
  the turn. Do not use this library to price short dates.
- **F4 — CIP 'arbitrage' that isn't.** Post-2008, *observed* CIP deviations
  persist beyond bid/ask. They are not free money: dealer balance-sheet
  costs (leverage-ratio capital against the repo/FX-swap footprint),
  quarter-end regulatory window dressing and counterparty limits explain
  the persistence — the Du–Tepper–Verdelhan finding, qualitatively. The
  detector therefore takes `min_pnl` to encode residual (balance-sheet)
  costs on top of bid/ask; flagged 'arbitrages' should be read as
  *funding-desk opportunities conditional on balance sheet*, not riskless
  P&L for everyone.
- **F5 — Long-end extrapolation.** Beyond the last pillar (10y) forwards
  are flat-extrapolated; 15y+ pricing is a guess, and the library reports
  no warning. Keep instruments within the quoted range.
- **F6 — Sparse swap strips.** With gaps in the par strip (real 2,3,4,5,7,
  10y quoting), intermediate DFs come from par-rate interpolation:
  repricing of *quoted* instruments still holds to 1e-16, but 6y/8y/9y DFs
  (and hence KRDs there) are model choices, not market facts (A4).
- **F7 — Stale cached basis curve on a mutable `MarketState` (fixed).**
  `MarketState.foreign_curve_adjusted` is a `functools.cached_property`
  derived from *two* fields — `foreign_curve` and `basis_spreads`. The
  dataclass was declared `@dataclass(eq=False)`, i.e. **mutable**, so
  assigning to either field in place left the cache populated with the old
  curve and every subsequent forward, MTM and DV01 silently priced off the
  stale basis. Reproduced before the fix: widening the 5y basis from -25bp
  to -500bp by assignment left the 5y forward unchanged at 1.17831 when it
  should have moved to 1.49418 — a **27% pricing error with no error, no
  warning and no NaN**.

  The class is now `frozen=True`: in-place assignment raises
  `dataclasses.FrozenInstanceError` and `MarketState.replace()` (which
  builds a fresh instance, and therefore a fresh cache) is the only way to
  produce a shocked copy. That is already what `risk.py` and
  `scenarios.py` do, so no call site changed — the invariant they relied
  on is now structural rather than conventional.
  (`test_edge_cases_extra.py::TestMarketStateImmutability`.)

## 5. Edge cases (documented AND unit-tested)

| Edge case | Behaviour | Test |
| --- | --- | --- |
| Negative foreign rates (EUR 2019) | DF > 1 supported end-to-end; points positive when `r_f < r_d`; MTM identities hold | `test_edge_cases.py::TestNegativeRatesEra`, `test_curve.py`, `test_arbitrage.py` |
| Same-currency 'cross' (EUR/EUR) | `ValueError` at construction, everywhere | `TestSameCurrencyCross`, `test_fxforward.py`, `test_xccy.py` |
| Zero notional | Valid: zero PV and zero across the whole risk vector | `TestZeroNotional` |
| Empty book | Zero-total risk report, no crash | `TestEmptyBook`, `test_risk.py` |
| Settlement edges (month-end, year-end T+2, 1-day accrual, O/N forward) | Calendar arithmetic correct; sub-week pricing out of scope (F3) | `TestSettlementEdges` |
| Invalid quotes (crossed bid/ask, negative prices, tau <= 0) | Informative `ValueError` | `test_arbitrage.py::TestValidation` |
| **Non-finite quote in the CIP detector** | Rejected at `CIPQuotes` construction; previously every comparison was False and the detector reported "no arbitrage" | `test_nan_guards.py` (§5.1) |
| **NaN/Inf scalar anywhere** (spot, strike, expiry, notional, maturity, bump, `point_factor`, scenario shock, `horizon`) | Rejected with the argument named; previously produced NaN forwards, NaN MTM and NaN DV01 | `test_nan_guards.py` (§5.1) |
| **Curve queried at a non-finite time** | `df(nan)` / `zero_rate(nan)` raise instead of returning NaN | `test_nan_guards.py` (§5.1) |
| Invalid instruments (negative strike/expiry, non-integral schedules, unbracketed par basis) | Informative `ValueError` | `test_edge_cases.py`, `test_xccy.py` |
| Extreme curves (50% and -5% zeros) | Valid curves, CIP monotonicity preserved | `TestOffline::test_extreme_but_valid_curve_inputs` |
| Crisis-wide basis (-180bp) | Bootstraps, prices, and scales mispricing as expected | `test_synthetic.py`, `TestBasisMispricing` |
| Network access | Disabled unless `FX_RATES_ALLOW_NETWORK=1`; loaders raise `RuntimeError`; tests never enable it | `TestOffline` |

Extended coverage in `tests/test_edge_cases_extra.py`:

| Edge case | Behaviour asserted |
| --- | --- |
| `MarketState` mutation | `FrozenInstanceError` on `spot` / `basis_spreads` / `foreign_curve`; `replace()` rebuilds the cache and leaves the original untouched; repeated DV01 / basis-DV01 bumps leak nothing back into the base market (F7) |
| Curve at the boundaries | `DF(0) = 1` exactly; `zero_rate(0)` and `df(-t)` raise; a one-pillar curve is flat at every tenor; `f(10,20) == f(5,10)` exactly (flat-forward extrapolation, F5) |
| Invalid curve inputs | 9 parametrised cases — duplicate / non-increasing / zero first pillar, zero / negative DF, length mismatch, empty, NaN, Inf |
| Key-rate locality | A 2y pillar bump leaves DF(0.25) and DF(10) bit-identical and does not mutate the source curve |
| Negative rates | `DF > 1` across the whole curve at -75bp; `r_d > r_f` gives a forward premium at every tenor; a CHF/EUR book with **both** curves negative prices to zero at the forward strike and has finite, correctly-signed DV01s |
| Pegged pairs (USDHKD) | Spot pinned at 7.80 but a 20bp rate gap still produces >100 pips of forward points — carry risk survives when spot vol does not |
| CIP band | Forward inside the band → no signal; above → `sell_forward`; below → `buy_forward`; wider bid/ask widens the band on both sides; `min_pnl` suppresses marginal signals (F4); negative deposit rates legal; crossed markets and `tau <= 0` raise |
| Basis round-trips | Zero spread reproduces the input curve to 1e-14 and an empty spread tuple returns the *same object*; negative basis lifts market forwards above pure CIP; `implied_basis_from_forwards` recovers the quoted -15bp/-25bp to 1e-12; `curve_from_fx_forwards` reprices its inputs to 1e-14; basis DV01 is defined with no quoted spreads and flips sign with the position |
| Carry / roll | A long low-yielder rolls **down** the points curve (negative carry) with matching signs on `points_roll` and `carry_pnl`; short is the exact mirror; horizons outside `(0, expiry)` raise; a zero scenario is a bit-exact no-op and a shocked scenario never disturbs the base market |
| Bootstrap degeneracies | Missing deposits, duplicate deposit tenors, fractional / duplicate swap maturities all raise; quoted swaps reprice to 1e-12; deposit DF identity holds and admits negative rates |
| Tenor / day-count parsing | 8 unparseable tenor strings raise; `year_fraction` is 0 on equal dates, raises on reversed dates and on an unknown convention |

### 5.1 Non-finite inputs: the guards that silently accepted NaN
(`tests/test_nan_guards.py`)

Nearly every validation in this package was an inequality — `if spot <= 0.0`,
`if tau <= 0.0`, `if bid > ask`, `if strike <= 0.0`, `if maturity <= 0.0`.
**Every comparison against NaN is False**, so each of those guards accepted
NaN and the NaN then travelled through the discount curve, the CIP forward,
the mark-to-market and the DV01 ladder without a single exception.

The most damaging instance was the **CIP arbitrage detector**. Feed
`CIPQuotes` a NaN forward bid (a stale or dropped quote is the realistic
cause) and: the crossed-market check `bid > ask` was False, the positivity
check `min(...) <= 0.0` was False, both round-trip P&Ls came out NaN, and the
final `best_pnl > min_pnl` was False — so `detect_cip_arbitrage` returned
`is_arbitrage=False, direction="none", pnl=0.0`. A confident "no arbitrage"
on data it could not price. That is strictly worse than crashing: nothing in
the output signals that the answer is meaningless.

All of these now route through `fx_rates._validation.require_finite` before
the inequality guards:

| Path | Was | Now |
|---|---|---|
| `CIPQuotes(...)` with any non-finite quote or `tau` | detector reports "no arbitrage" | raises, naming the field |
| `detect_cip_arbitrage(min_pnl=nan)` | every signal suppressed | raises |
| `DiscountCurve.df(nan)` / `.zero_rate(nan)` | `nan` discount factor | raises |
| `DiscountCurve.parallel_shift(nan)` / `.pillar_shift(i, nan)` | all-NaN curve | raises |
| `MarketState(spot=nan)` or a non-finite basis spread | NaN forwards and MTM | raises |
| `cip_forward(spot=nan)` / `expiry=nan` | NaN forward | raises |
| `forward_points(point_factor=nan or <= 0)` | NaN / sign-flipped points | raises |
| `basis_adjusted_curve` with a non-finite spread | NaN adjusted curve | raises |
| `FXForward` / `FXSwap` / `CrossCurrencySwap` with a NaN notional, strike, expiry, rate or maturity | NaN PV and NaN risk | raises at construction |
| `df_from_deposit` / `deposit_rate_from_df` / `implied_basis_from_forwards` with non-finite rate, tau, df, spot or quote | NaN DF / NaN basis | raises |
| `Scenario(spot_pct=nan, …)` | NaN shocked market | raises (and `spot_pct <= -100%` now raises too) |
| `fx_delta(rel_bump=nan)`, `dv01(bump_bp=nan)`, `forward_carry(horizon=nan)` | NaN risk numbers | raises |

Positive companions confirm nothing over-rejects: `DF(0) = 1` exactly, the
zero-rate round trip at 1y, a +1bp shift lowering every DF, the market
forward sitting above pure CIP by exactly the -25bp basis (`-ln(F_mkt/F_cip)/T`
recovers -0.0025 to 1e-9), the two `FXForward` valuation routes agreeing to
1e-10, the deposit DF identity round-tripping at rates from -75bp to +5.2%,
the long-EUR forward's FX delta equalling `N·DF_f_adj(1y)` to 1e-8, and
domestic vs foreign DV01 carrying opposite signs.

## 6. Reproducibility

Every stochastic component takes an explicit seed; the quote generators
noise the *curve parameters*, never individual quotes, so internal
consistency is exact by construction. `pytest -q` passes offline from the
project root in well under 2 minutes (measured: ~0.4s); the pipeline runs
in < 1s (contract: < 60s).
