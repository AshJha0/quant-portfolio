# Validation — `fi_rates`

All numbers below are reproduced by `python examples/run_pipeline.py`
(seeded, offline, <1s) and enforced by the test suite
(`python -m pytest tests -q`, 235 tests, ~1s).

---

## 1. Round-trip evidence

### 1.1 Bootstrap repricing (tolerance 1e-10)

Every input instrument repriced off the bootstrapped curve (upward variant,
seed 42, log-linear DF):

```
Deposit  T= 0.25y  quote=+3.069820%  model-quote = +4.0e-16
Deposit  T= 0.50y  quote=+3.130994%  model-quote = +6.2e-17
Deposit  T= 1.00y  quote=+3.256561%  model-quote = -1.7e-16
ParSwap  T= 2.00y  quote=+3.462035%  model-quote = +2.8e-17
ParSwap  T= 5.00y  quote=+3.868747%  model-quote = -6.9e-18
ParSwap  T=10.00y  quote=+4.207245%  model-quote = +1.4e-17
ParSwap  T=30.00y  quote=+4.532646%  model-quote = +6.9e-18
max abs repricing error: 4.0e-16   (tolerance 1e-10: PASS)
```

Tested for log-linear DF and linear-zero interpolation, for all four curve
variants (upward, inverted, flat, negative), for FRAs, for semiannual swaps,
and for the coupon-bond bootstrap (round trip through synthetic bond prices
back to the generating curve's discount factors). Bootstrapping is verified
**order-independent**: shuffled instrument input reproduces identical pillar
discount factors to 1e-15.

### 1.2 Pricing round trips

| Round trip | Tolerance | Test |
|---|---|---|
| price → YTM → price (on and between coupon dates) | 1e-10 | `test_bond.py::TestYTM` |
| z-spread → clean price → z-spread (incl. negative spread) | 1e-10 | `test_bond.py::TestZSpread` |
| YTM at 500% and at −1% recovered | 1e-10 rel | `test_edge_cases.py::TestExtremeYields` |

## 2. Analytic-identity evidence

| Identity | Result |
|---|---|
| ZCB price == face × P(T) | exact (float equality) |
| Coupon bond == Σ of ZCBs on its cashflows | ≤1e-10 |
| Log-linear DF ⇒ piecewise-constant forwards (all sub-forwards inside a segment equal) | ≤1e-14 |
| `P(t2) = P(t1)·e^(−f(t1,t2)(t2−t1))` for all interpolations | ≤1e-13 rel |
| 1y par rate == simple 1y rate `1/P(1)−1` | ≤1e-14 |
| Par bond YTM == coupon (annual and semiannual, street convention) | ≤1e-12 |
| dirty = clean + accrued; accrued = coupon·¼ a quarter into a 30/360 semiannual period | ≤1e-12 |
| FRN price == par at reset (telescoping identity) | ≤1e-10 |
| Macaulay(ZCB) == maturity; D_mod == D_mac/(1+y/m) | ≤1e-12 / ≤1e-14 |
| Analytic duration/convexity vs central finite differences | ≤1e-8 / ≤1e-6 rel |
| carry + roll-down == full static-curve horizon P&L (pull-to-par) | ≤1e-12 |

## 3. Key-rate duration: why the sum matches the parallel DV01 only "within tolerance"

The triangular key-rate bumps form a **partition of unity** over the curve
pillars (unit-tested to 1e-14), so the *sum of the bumps* is exactly a 1bp
parallel bump. The *sum of the KRDs* nevertheless differs from the parallel
DV01 for two reasons:

1. **Finite-difference cross terms.** PV is not linear in the pillar zeros;
   central differences of bump-vectors do not add exactly. The residual is
   the cross-gamma between key-rate buckets, O(h²) in the 1bp bump.
2. **Non-local interpolation.** Under PCHIP, bumping the pillars in bucket
   *k* also reshapes neighbouring segments, so bucket sensitivities overlap
   slightly.

Measured on the sample portfolio (DV01 ≈ 980.3 per 1bp):

| Interpolation | \|Σ KRDV01 − parallel DV01\| / DV01 | Test tolerance |
|---|---|---|
| log-linear DF | 5.3e-08 | 1e-6 rel |
| PCHIP on zeros | 5.5e-07 | 1e-4 rel |

Locality is also tested directly: for a 2y bond the 30y key-rate DV01 is
below 1e-6 of the 2y bucket; a 5y ZCB carries >99.9% of its KRD at the 5y
key rate.

## 4. Known failure modes

### 4.1 Interpolation artifacts in forwards (the sawtooth) — actual numbers

6-month forward rates (%) off the same bootstrapped quotes, around the
10y/15y/20y pillars:

```
   t     loglinear_df  linear_zero  pchip_zero
  9.0y      4.5855       4.6779       4.6211
  9.5y      4.5855       4.7372       4.6206
 10.0y      4.6862       4.5369  ←    4.6274
 14.5y      4.6862       4.8479       4.7177
 15.0y      4.7336       4.6495  ←    4.7214
 19.5y      4.7336       4.8253       4.7317
 20.0y      4.7656       4.6697  ←    4.7348
```

Under **linear-on-zeros** the forward *rises through each segment and then
drops ~20bp at every pillar* (max adjacent-3m-forward jump measured:
22.4bp) — the classic sawtooth: zero-rate slope is constant per segment, so
the forward `z + t·dz/dt` grows with `t` and resets at each pillar. A trader
quoting forward-starting trades off this curve would systematically misprice
around pillars. **Log-linear DF** jumps only *at* pillars (piecewise-constant
forwards — up to 29bp per step here, but flat and unbiased in between), and
**PCHIP** is continuous (max jump ≈ 2bp from discrete differencing). This is
the concrete reason the interpolation choice is exposed rather than
hard-coded.

### 4.2 Extrapolation risk

Beyond the last pillar the zero rate is held flat and **every query emits an
`ExtrapolationWarning`**. Discounting a 40y cashflow off a 30y-last-pillar
curve embeds the arbitrary flatness assumption directly in PV; long-dated
pension/insurance liabilities are exquisitely sensitive to it (a 10bp long-end
slope error at 40y ≈ 0.4% of PV). The warning is deliberately not
suppressible by config — callers must handle it consciously.

### 4.3 Duration fails for non-parallel moves

From the pipeline scenario table (full revaluation vs duration+convexity
with the DV01-weighted average shift as the parallel proxy):

```
scenario              pnl_full     pnl_dur_conv    error
parallel_+100bp        -92,993       -88,980       +4,013   (4.3% — pure convexity/KRD structure)
steepener_50bp          -4,345       -12,296       -7,951   (183% — duration proxy useless)
flattener_50bp          +5,595       +12,479       +6,884   (123%)
hiking_2022           -228,401      -197,873      +30,529   (13% — large move + curve twist)
gfc_2008              +167,996      +148,079      -19,917   (12%)
```

A single duration number compresses the curve into one factor; for twist
scenarios the sign of the error is not even stable. The KRD ladder exists
precisely to fix this — repricing the steepener off the ladder recovers the
full P&L to within cross-gamma. Also tested: for a 2y+30y barbell under a
steepener, the duration estimate misses >25% of the true P&L
(`test_scenarios.py::test_duration_estimate_fails_for_non_parallel`).

### 4.4 Taylor error growth (duration vs duration+convexity)

10y government bond, YTM space (pipeline section 7):

| shock | full repricing | dur-only error | dur+conv error |
|---|---|---|---|
| −200bp | +17.746 | −1.664 | −0.114 |
| −100bp | +8.443 | −0.401 | −0.014 |
| +100bp | −7.667 | −0.374 | +0.013 |
| +200bp | −14.635 | −1.448 | +0.102 |

Convexity cuts the 100bp error by ~30x (unit-tested as an inequality, not a
number). Duration-only *always* understates the price (overstates losses,
understates gains) for a positive-convexity bond — also tested as an
inequality across the whole shock grid.

### 4.5 Negative convexity — out of scope

Callable bonds and MBS exhibit **negative convexity** when rates approach
the call/refinancing region: price appreciation caps out, duration
*shortens* into a rally. Nothing in `risk.py` models this — applying these
analytics to callables would overstate upside and misdirect hedges
(the 30x Taylor improvement above would reverse sign). An OAS model with an
interest-rate lattice/simulation is required; explicitly out of scope
(assumption 7 in `METHODOLOGY.md`).

### 4.6 Other bounded behaviours (all raise `ValueError` with informative messages)

Unsolvable bootstrap quotes (deposit rate < −1/T, bond price above max PV);
settlement after maturity; YTM at `1+y/m ≤ 0`; invalid frequencies; duplicate
pillars; empty instrument lists; fractional swap maturities. Degenerate but
valid inputs — single-pillar curves, empty portfolios, settlement ==
maturity (price 0, no cashflows), negative-rate curves (DFs > 1) — are
tested as *working* paths, not errors.

## 5. Suite summary

* 235 tests, 100% pass, ~1s wall clock, fully offline, deterministic
  (seeded `default_rng`).
* Edge-case contract (portfolio conventions item 6) covered in
  `tests/test_edge_cases.py` and cross-referenced from the assumptions
  register.
