# Validation: `fi_rates`

Every number below comes from `python examples/run_pipeline.py` (synthetic
seeded quotes, offline) unless a test is named. The suite runs with
`python -m pytest tests -q`; counts and runtime are in the README.

---

## 1. Round-trip evidence

### 1.1 Bootstrap repricing (tolerance 1e-10)

Pipeline section 2, synthetic upward curve, seed 42, log-linear DF:

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
variants (upward, inverted, flat, negative), for FRAs, for semiannual
swaps, and for the coupon-bond bootstrap. Shuffled input reproduces the
same pillar discount factors to 1e-15.

Exact repricing is a property of local interpolation. Appending a pillar
cannot move a local curve before it, so solved quotes stay solved. PCHIP
reshapes earlier segments every time a pillar is appended:

| Interpolation | Max repricing error (rate) | Local |
|---|---|---|
| log-linear DF | 4.0e-16 | yes |
| linear zero | 4.0e-16 | yes |
| PCHIP on zeros | 2.8e-06 | no |

A PCHIP build cannot pass a 1e-10 repricing gate. Pinned by
`test_properties.py::test_local_schemes_reprice_every_instrument_exactly`
and `::test_pchip_bootstrap_is_only_approximately_exact`.

### 1.2 Pricing round trips

| Round trip | Tolerance | Test |
|---|---|---|
| price to YTM to price (on and between coupon dates) | 1e-10 | `test_bond.py::TestYTM` |
| z-spread to clean price to z-spread (incl. negative spread) | 1e-10 | `test_bond.py::TestZSpread` |
| YTM at 500% and at -1% recovered | 1e-10 rel | `test_edge_cases.py::TestExtremeYields` |
| ICMA bond at y = c on any coupon date prices at par (hypothesis) | 1e-9 | `test_icma_calendars.py::test_par_property` |

## 2. Analytic identities

| Identity | Result |
|---|---|
| ZCB price == face x P(T) | float equality |
| Coupon bond == sum of ZCBs on its cashflows | <= 1e-10 |
| Log-linear DF gives piecewise-constant forwards | <= 1e-14 |
| `P(t2) = P(t1) exp(-f(t1,t2)(t2-t1))` for all interpolations | <= 1e-13 rel |
| Par bond YTM == coupon (street convention) | <= 1e-12 |
| ACT/ACT-ICMA regular coupon == c/f | float equality |
| ICMA accrued 15-Feb to 15-May on a 4% semiannual: 2 x 89/181 = 0.9834 | <= 1e-15 rel |
| FRN price == par at reset | <= 1e-10 |
| Mid-period FRN, zero margin: P(t1) face (1 + fixing tau) | <= 1e-12 rel |
| Macaulay(ZCB) == maturity; D_mod == D_mac/(1+y/m) | <= 1e-12 / 1e-14 |
| Analytic duration and convexity vs central differences | <= 1e-8 / 1e-6 rel |
| carry + roll-down == static-curve horizon P&L net of funding | <= 1e-12 |
| Funding == dirty x repo x ACT/360 | <= 1e-14 rel |
| On a flat curve, roll-down is all pull-to-par | <= 1e-9 |

## 3. Key-rate DV01 sum vs parallel DV01

The triangular bumps form a partition of unity over the pillars (tested to
1e-14), so the sum of the bumps is a 1bp parallel bump. The sum of the
key-rate DV01s still differs from the parallel DV01 for two reasons:

1. Finite-difference cross terms. PV is not linear in the pillar zeros, so
   central differences of bump vectors do not add exactly. The residual is
   O(h^2) in the 1bp bump.
2. Non-local interpolation. Under PCHIP a bump in one bucket reshapes the
   neighbouring segments.

The residual depends on the position and the curve. Pipeline section 5
(sample book, DV01 980.17 per bp) and the 20y 4% bond used in
`test_properties.py`:

| Interpolation | Sample book | 20y 4% bond | Test tolerance |
|---|---|---|---|
| log-linear DF | 5.3e-08 | 3.1e-07 | 1e-6 rel |
| linear zero | 5.7e-08 | 3.9e-07 | 1e-6 rel |
| PCHIP on zeros | 5.4e-07 | 2.1e-04 | 1e-3 rel |

The 20y bond column comes from `test_krdv01s_sum_to_the_parallel_dv01_for_local_schemes`
and `test_krdv01_additivity_is_looser_under_pchip` (measured with the same
inputs). Locality is also tested: for a 2y bond the 30y key-rate DV01 is
below 1e-6 of the 2y bucket; a 5y ZCB carries more than 99.9% of its KRD
at the 5y key rate.

## 4. Known failure modes

### 4.1 Forward sawtooth under linear-on-zeros

Pipeline section 9: 6-month forwards `f(t, t+0.5)` in percent near the
10y, 15y and 20y pillars.

| t | loglinear_df | linear_zero | pchip_zero |
|---|---|---|---|
| 9.0 | 4.5809 | 4.6725 | 4.6135 |
| 9.5 | 4.5809 | 4.7312 | 4.6169 |
| 10.0 | 4.6886 | 4.5383 | 4.6278 |
| 14.5 | 4.6886 | 4.8513 | 4.7123 |
| 15.0 | 4.7194 | 4.6385 | 4.7107 |
| 19.5 | 4.7194 | 4.8078 | 4.7328 |
| 20.0 | 4.7888 | 4.6850 | 4.7412 |

Under linear zeros the forward rises through each segment and drops about
20bp at the pillar: the zero slope is constant per segment, so
`z + t dz/dt` grows with `t` and resets. Log-linear DF steps only at
pillars. PCHIP forwards are continuous with kinks at the pillars.

### 4.2 Extrapolation

Beyond the last pillar the zero rate is held flat and every query emits an
`ExtrapolationWarning`. A 10bp long-end slope error at 40y is about 0.4%
of PV.

### 4.3 Duration fails for non-parallel moves

Pipeline section 6. The estimate applies the book's YTM-space modified
duration and convexity to one parallel-equivalent zero-rate move per
scenario: the pillar shifts weighted by `P(t) t`, a curve weight, not a
portfolio weight.

| Scenario | Full revaluation | Duration + convexity | Error |
|---|---|---|---|
| parallel +100bp | -92,995 | -90,880 | +2,115 (2.3%) |
| steepener 50bp | -4,345 | -12,556 | -8,211 (189%) |
| flattener 50bp | +5,595 | +12,742 | +7,147 (128%) |
| hiking 2022 | -228,406 | -202,179 | +26,228 (11%) |
| GFC 2008 | +205,552 | +200,578 | -4,974 (2.4%) |

The parallel error is mostly the mismatch between a continuous zero-rate
shift and a periodic-yield duration (`dy/dz` is about 1.02 here), not
convexity. For twists the sign of the error is not stable. The barbell
test `test_scenarios.py::test_duration_estimate_fails_for_non_parallel`
shows the duration estimate missing more than 25% of the P&L.

### 4.4 Taylor error growth

10y government bond, YTM space (pipeline section 7), per 100 face:

| Shock | Full repricing | Duration-only error | Duration + convexity error |
|---|---|---|---|
| -200bp | +17.747 | -1.664 | -0.114 |
| -100bp | +8.443 | -0.401 | -0.014 |
| +100bp | -7.667 | -0.374 | +0.013 |
| +200bp | -14.635 | -1.448 | +0.102 |

Convexity cuts the 100bp error by about 30x (tested as an inequality).
Duration-only understates the price of a positive-convexity bond across
the whole shock grid (tested).

### 4.5 Negative convexity is out of scope

Callable bonds and MBS have negative convexity near the call region. The
analytics here would overstate their upside. An OAS model on a rate
lattice is needed.

### 4.6 Bounded behaviour

Unsolvable bootstrap quotes, settlement after maturity, `1+y/m <= 0`,
invalid frequencies, unknown day counts, duplicate pillars, empty
instrument lists and fractional swap maturities raise `ValueError`.
Single-pillar curves, empty portfolios, settlement on maturity and
negative-rate curves are tested as working paths.

Every public scalar market input (`face`, `coupon`, `ytm`, `clean_price`,
`z_spread`, `quoted_margin`, `payment`, `current_fixing`, `repo_rate`)
is checked for finiteness and raises `ValueError` naming the argument
(`tests/test_input_hardening.py`, `tests/test_carry_curve_claims.py`).

### 4.7 Zero-net-value books

A duration-neutral long/short book has net market value near 0, so every
market-value-weighted aggregate is undefined. `portfolio_risk` detects
`|net MV| < 1e-9 x gross MV`, emits `ZeroNetValueWarning` and reports NaN
for the weights and weighted aggregates. `mv` and `dv01` are plain sums
and stay exact; DV01 and the key-rate ladder are the right aggregates for
such a book. Pinned by
`test_properties.py::test_zero_net_value_book_warns_and_reports_nan_weights`.

## 5. Test layout

* `test_properties.py`: structural invariants (curve identities, bond
  monotonicity and convexity, duration ordering, pull-to-par, FRN par
  identity at every frequency, portfolio DV01 additivity, scenario signs,
  day-count antisymmetry and additivity for every convention).
* `test_icma_calendars.py`: ICMA coupons and accrued, the par property,
  calendars and rolls.
* `test_carry_curve_claims.py`: funded carry, pull-to-par split, the GFC
  2008 shock, PCHIP smoothness, mid-period FRN.
* `test_edge_cases.py`, `test_input_hardening.py`: edge cases and input
  validation.

## Round 3 oracle tests (`tests/test_round3_rates.py`)

| Item | Oracle | Tolerance |
|---|---|---|
| OIS, projection | the bootstrap reprices every input quote | 1e-13 |
| Projection | OIS quotes used as swap quotes give back the OIS curve | 1e-12 |
| Black, Bachelier | put-call parity, ATM closed forms, implied-vol round trip | 1e-14 / 1e-10 |
| Swaption, cap | payer - receiver = annuity x (F - K); cap - floor = swap | 1e-14 |
| Hull-White | zero bonds reprice the initial curve | 1e-7 rel |
| Hull-White | bond-option parity; Jamshidian payer-receiver parity | 1e-13 |
| Hull-White | exact-OU MC (200k paths): bond, caplet, swaption | 4 SE + 2e-5 |
| Hull-White | sigma -> 0 caplet is intrinsic; small a gives normal vol ~ sigma | 1e-6 / 2e-3 rel |

## Round 3 fixes (`tests/test_round3_fix_rates.py`)

| Item | Oracle | Tolerance |
|---|---|---|
| Monotone convex | every pillar reprices | 1e-15 |
| Monotone convex | forward continuous at pillars; equals -d ln P/dt | 1e-8 / 1e-7 |
| Monotone convex | increasing discrete forwards give an increasing, positive forward | exact sign |
| Monotone convex | bumping one pillar leaves forwards two segments away unchanged | 1e-15 |
| Monotone convex | each of the four shapes: end values, zero integral, quadrature | 1e-14 / 1e-8 |
| HW two-curve | forward curve = discount curve gives the single-curve price | 1e-14 |
| HW two-curve | payer - receiver = annuity x (F - K) from `rate_options` | 1e-13 |
| HW two-curve | exact-OU MC (200k paths) of the two-curve payoff | 4 SE + 2e-5 |
| HW calibration | known (a, sigma) recovered from normal and Black vol grids | 1e-6 rel |
| HW calibration | fixed a: sigma recovered; wrong a leaves a visible misfit | 1e-8 rel |
