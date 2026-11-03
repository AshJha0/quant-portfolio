# Validation: evidence, numbers, failure modes

Every number below comes from synthetic data, produced by
`python examples/run_pipeline.py` (seeded, offline, about 25-30 s, writes
nothing to disk). The signal edge is planted by the generator; no real-data
result exists yet (see `research/`). The test suite is `python -m pytest tests`
(185 tests, about 30-45 s). Tolerances: algebraic identities to 1e-10;
Monte Carlo claims are bounded by their stated standard errors.

---

## 1. Analytic and structural anchors

| Check | Result | Test |
|---|---|---|
| Piecewise AC (KKT/QP) vs closed-form AC, constant eta and sigma | max abs diff < 1e-9 | `test_optimal.py::test_piecewise_reduces_to_closed_form_when_liquidity_constant` |
| lambda -> 0 limit equals the liquidity-weighted schedule | exact to 1e-8 | `test_lambda_zero_gives_liquidity_weighted_twap_analog` |
| Schedule sums to the parent quantity | exact (1e-10), including the active-set case | `test_piecewise_ac_schedule_sums_exactly_to_parent` |
| Liquidity-aware AC cost <= naive AC cost under the time-varying objective | holds for lambda in {1e-6, 1e-5, 1e-4} | `test_liquidity_aware_cost_leq_naive_ac...` |
| IS components sum to total | residual < 1e-10 | `test_tca.py::test_is_components_sum_exactly` |
| Venue identity: effective = quoted half-spread + temporary + rejection cost | exact | `test_venue_scorecard_identity_exact` |
| Fill at the modelled t+1 price `p_t + f (p_{t+1} - p_t)` | exact on a hand ledger | `test_execution_convention.py::test_hand_ledger_fill_at_modelled_next_price` |
| Perfect-foresight signal `sign(r_{t+1})` with f = 1 | mean gross within 4 SE of zero; a same-close fill earns the full abs(r) | `test_perfect_foresight_does_not_pay_with_next_close_fill` |
| Close-auction variant charges exactly `auction_cost_bp` per unit traded | exact | `test_close_auction_charges_exact_extra_cost` |
| Gross P&L equals the position path integral for any f in (0, 1] | hypothesis property, 1e-12 | `test_gross_pnl_equals_position_path_integral` |
| Same-close ledger vs hand-computed toy scenario (session spreads, 3 trades) | exact | `test_ledger_exact_on_toy_scenario_with_session_spreads` |
| Carry accrual vs hand value 1.0 x 1.1 x 0.02 / 365 | exact to 1e-15 | `test_carry_accrual_hand_exact` |
| Temporary impact: 4x size gives 2x impact | exact to 1e-12 | `test_sqrt_impact_scaling_with_size` |
| Features at and before a cutoff do not change when future ticks change | exact frame equality | `test_pit_mutation_leaves_past_features_unchanged` |
| Fix print at zero vol equals the mean of the linear interpolation | exact to 1e-12 | `test_fix_print_zero_vol_is_linear_interpolation_mean` |
| Self-impact on the fix print vanishes with `k_perm = 0` | exact | `test_self_impact_on_print_is_visible_and_vanishes_without_permanent_impact` |
| Research statistics (Lo/Newey-West SE vs Monte Carlo, deflated Sharpe calibration, PBO, block bootstrap) | see the test file | `test_research_stats.py` |

## 2. Signal layer (synthetic, planted alpha)

Feature ICs against the next 1h return, seed 0, 60 days of synthetic hourly
bars with planted AR(1) phi = 0.25:

| feature | IC | t-stat |
|---|---:|---:|
| mom_1 | 0.251 | 9.83 |
| mom_4 | 0.187 | 7.22 |
| reversion | -0.145 | -5.57 |
| breakout | -0.033 | -1.26 |
| mom_1 on phi = 0 noise | 0.003 | 0.12 |

Session-filtered vol-targeted mom_1/mom_4 (0.7/0.3) strategy, EURUSD costs
unless stated, 60 days per seed, 30 synthetic seeds, mean [95% CI] across
seeds:

| execution variant | gross pips | net pips | Sharpe (annualised, 6264 bars/yr) |
|---|---:|---:|---:|
| next, f = 0.5 (default) | +2232 [+2047, +2418] | +2079 [+1893, +2266] | +9.42 [+8.68, +10.16] |
| next, f = 1.0 | +964 [+761, +1166] | +803 [+600, +1006] | +3.16 [+2.39, +3.93] |
| close auction, 0.5 bp | +3501 [+3277, +3725] | +2896 [+2666, +3126] | +10.63 [+9.84, +11.41] |
| close auction, 0 bp | +3501 [+3277, +3725] | +3374 [+3149, +3600] | +12.48 [+11.73, +13.23] |
| EM spreads, f = 0.5 | +2232 [+2047, +2418] | -3246 [-3489, -3004] | -13.42 [-14.49, -12.34] |

Reading. The planted hourly autocorrelation decays within one bar, so most
of the synthetic edge sits in the move immediately after the decision. A
fill one full bar later keeps about a quarter of the same-close gross. The
same planted signal loses money under EM-style spreads. None of these
Sharpe ratios is a market claim (assumption A10).

Seed 0, default variant: Sharpe 11.49, circular block-bootstrap 95% CI
[6.46, 15.88]. The pipeline logs 9 trials on seed 0 (5 execution and cost
variants, 4 other weight sets); the deflated Sharpe ratio at 9 trials is
0.606, and the probability of backtest overfitting over the 5-point weight
grid (CSCV, 8 blocks) is 0.06. The deflated Sharpe is low because the
logged trials include the EM-spread variant, whose large negative Sharpe
widens the null spread.

Test-locked: IC > 0.10 with t > 2 on planted alpha; abs(t) < 2.5 on three
noise seeds (`test_synthetic.py`).

## 3. Scheduler comparison (500mm EURUSD buy, 200 paired synthetic replications)

Controllable cost = spread + temporary + permanent. It is deterministic given
the schedule in this model. IS mean is controllable cost plus mean-zero drift
noise, so its CI is wide and the IS mean column does not rank schedules. IS
sd is execution risk against arrival.

Full 24h day (288 x 5-min buckets):

| schedule | ctrl cost (pips) | IS mean [95% CI] | IS sd |
|---|---:|---:|---:|
| TWAP | 0.578 | +2.13 [-2.24, +6.49] | 31.30 |
| liquidity-weighted | 0.517 | +2.10 [-2.57, +6.78] | 33.54 |
| POV 5% (analog) | 0.709 | +1.08 [-0.52, +2.68] | 11.49 |
| piecewise AC (lambda = 1e-5) | 0.733 | +0.88 [-0.20, +1.96] | 7.73 |

London-only window 07:00-16:00 (108 buckets):

| schedule | ctrl cost (pips) | IS mean [95% CI] | IS sd |
|---|---:|---:|---:|
| TWAP | 0.531 | +1.06 [-2.31, +4.44] | 24.23 |
| liquidity-weighted | 0.525 | +1.05 [-2.56, +4.65] | 25.84 |
| POV 5% (analog) | 0.686 | +0.54 [-1.46, +2.54] | 14.35 |
| piecewise AC (lambda = 1e-5) | 0.750 | +0.71 [-0.62, +2.04] | 9.55 |

Readings, each test-locked:

- Liquidity weighting saves about 11% of controllable cost against TWAP over
  the full synthetic day (0.517 vs 0.578 pips) by avoiding the late and Asia
  sessions. Inside the London window the saving is about 1%
  (`test_liquidity_weighted_beats_twap_controllable_cost`).
- The AC frontier: lambda = 1e-5 front-loading pays 0.22 pips more
  controllable cost to cut IS sd from about 31 to about 8 pips. The
  parameters `k_eta = 0.02` and `lambda = 1e-5` are not calibrated.
- A 500mm single-bucket execution violates the 30%-of-depth cap and raises
  (`test_500mm_parent_requires_multi_session_split`).

## 4. Last look (200 paired synthetic replications, common random numbers)

| flow | venue | quoted half-spread | effective cost | reject rate | reject cost |
|---|---|---:|---:|---:|---:|
| uninformed (alpha = 0) | last-look | 0.107 | 0.290 | 9.0% | 0.053 |
| | firm ECN | 0.178 | 0.309 | 0.0% | 0 |
| informed (alpha = 0.5 pips/bucket) | last-look | 0.107 | 0.584 | 40.8% | 0.347 |
| | firm ECN | 0.178 | 0.309 | 0.0% | 0 |

Paired differences (last-look minus firm), mean [95% CI]: -0.018 [-0.020,
-0.016] pips for uninformed flow, +0.275 [+0.271, +0.280] pips for informed
flow. The informed-flow loss follows from assumption A4 (the dealer sees the
client's whole per-bucket alpha inside a 2-second hold); it is an assumption,
not a finding. Tests: `test_the_trap_effective_cost_higher_with_adverse_selection`,
`test_reject_prob_monotone_increasing_in_adverse_move`.

## 5. Fix benchmark (200 synthetic replications, 100mm, 1-min buckets 14:00-17:00)

The print is the mean of 60 market observations per minute of the 5-minute
window, drawn from an RNG independent of the execution. "Unimpacted" strips
the order's own permanent impact from the path.

| schedule | TE vs print, mean [95% CI] | sd | TE vs unimpacted print | sd |
|---|---:|---:|---:|---:|
| fix-targeting (5-min window) | +1.015 [+0.940, +1.090] | 0.537 | +1.050 [+0.975, +1.125] | 0.537 |
| TWAP 14:00-17:00 | +0.341 [-1.042, +1.724] | 9.921 | +0.388 [-0.996, +1.771] | 9.921 |

This study runs on 1-minute buckets, so the W3 fix to the impact horizon
(METHODOLOGY section 4) changed it. Before the fix (v2.0.0) the fix-targeting
row was +0.498 [+0.423, +0.573] and TWAP +0.242: the 1-minute grid had
understated temporary impact by a factor sqrt(5) and overstated permanent
impact by the same factor. The tracking-error sd is unchanged.

The fix schedule's own permanent impact moves the print by 0.035 pips
(0.078 before the fix). That
number has no seed variance in this model, because permanent impact is a
deterministic function of the schedule. The fix schedule pays about one
pip of spread and impact and carries about 0.5 pip of tracking-error sd; a
3h TWAP is cheaper on average and carries about 10 pips of tracking-error sd.
Tests: `test_fix_print_tracking_error_has_real_variance`,
`test_fix_benchmark_from_own_mids_is_a_tautology` (documents why
`fix_benchmark` is not used for tracking error).

## 6. Failure modes and edge cases

Each is unit-tested.

1. Fix gaming. Trading inside the window moves the print through permanent
   impact, which `fix_self_impact_pips` measures. The print model has no
   median filter (assumption A8).
2. Flash events (GBP, 7 Oct 2016). Gaussian session vol cannot produce an 8%
   gap (assumption A5). With `vol_scale = 5` on GBPUSD, controllable cost is
   more than 2x and IS sd more than 3x calm conditions
   (`test_flash_vol_regime_multiplies_execution_cost`).
3. Last-look adverse selection. Choosing a venue by quoted spread alone picks
   the venue that is worse for flow with alpha (section 4).
4. EM cost regimes. The same planted signal and positions net a loss under
   EM-style spreads (`test_em_wide_spread_flips_strategy_profitability`).
   USDMXN execution cost is more than 20x EURUSD for comparable
   participation (`test_em_pair_execution_costs_dwarf_major`).
5. Weekend gap. Schedulers place zero there, the simulator freezes diffusion
   and raises on any attempted fill (`test_weekend_gap_end_to_end`).
6. Single-bucket execution returns `[X]` across every scheduler and the AC
   solver (`test_twap_single_bucket`, `test_single_bucket_and_sell_side`).
7. An order bigger than the depth cap raises `ValueError`; a POV that cannot
   complete raises `ValueError` (`test_pov_infeasible_raises`).
8. Zero-vol path: execution is deterministic and its IS equals the
   quantity-weighted half-spread (`test_zero_vol_path_deterministic_cost`).
9. Numerical limits. The KKT solve is dense O(N^3), fine to N of about 1440
   (a 1-minute day). At extreme lambda the unconstrained QP sells back in
   thin buckets; the active-set clamp handles it
   (`test_active_set_clamps_thin_buckets_at_high_lambda`).
10. Non-finite inputs. `session_of_hour(NaN)` raises `ValueError`;
    `piecewise_ac_schedule(parent_qty=0)` returns a flat zero schedule;
    NaN or Inf in `eta`, `sigma`, `depths` or `parent_qty` raises before the
    solve. `ExecutionConfig` rejects `fill_fraction` outside (0, 1] and
    non-finite costs (`tests/test_edge_cases_review.py`,
    `test_fill_fraction_validation`).
11. Causality guards. Backtest P&L up to a cutoff does not change when later
    prices change; features do not change under future-tick mutation
    (`test_backtest.py`, `test_features.py`).

## 7. Reproducibility

- Every stochastic component takes an explicit seed or
  `numpy.random.Generator`; the same seed gives the same paths and fills
  (`test_seeded_reproducibility`).
- Venue and scheduler comparisons use common random numbers, so paired
  differences isolate the decision variable.
