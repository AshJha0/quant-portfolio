# Validation: evidence, numbers, failure modes

Numbers come from `python examples/run_pipeline.py` (synthetic data, seeds
named in each table) or from named tests in `tests/`. Every alpha number
here is on a synthetic panel whose generator contains the signal, so it
measures the code, not a market edge. No real-data result exists yet.

---

## 1. Execution convention

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost.

| check | test |
|---|---|
| Fill price equals p_t + f (p_{t+1} - p_t) on a hand-computed 3-day book, f = 0.25, 0.5, 1 | `test_execution_convention.py::test_fill_price_is_interpolated_next_price` |
| A perfect-foresight signal s_t = r_{t+1} earns zero in mean (within 4 SE) at f = 1; the same signal filled at t's close earns over 20 SE | `test_perfect_foresight_earns_nothing_when_filled_at_next_close` |
| Close-auction mode charges exactly `auction_cost_bp` per unit turnover | `test_close_auction_charges_exactly_the_auction_cost` |
| Borrow fee charged on the short gross | `test_borrow_fee_charged_on_short_gross` |
| Bad fill fraction, auction cost, mode, non-finite config, non-positive prices raise `ValueError` | `test_execution_config_rejects_bad_values` et al. |

## 2. Point-in-time checks

- Feature mutation test (`test_features.py::test_point_in_time_mutation`):
  every price and volume after row 300 is multiplied by random factors in
  [0.2, 5.0]; all feature rows up to 300 must be identical.
- Full-pipeline mutation test
  (`test_backtest.py::test_no_lookahead_full_pipeline_mutation`): features,
  z-scores, signal freeze, decile weights, trailing ADV and vol, costs and
  ledger are re-run after mutating all data past day 400. Ledger rows up to
  400 are identical.
- Trailing ADV is causal
  (`test_capacity_and_race.py::test_full_sample_adv_is_deprecated_and_trailing_adv_is_causal`).
  The full-sample `DailyPanel.adv_dollars` now warns.

## 3. Planted alpha recovered (synthetic, seed 42)

The generator plants next-day momentum of strength 0.0008 / 0.02 = IC 0.04
and a reversal effect. On 150 synthetic stocks x 1,250 days:

| feature | mean IC | NW t-stat | planted |
|---|---:|---:|---|
| mom_12_1 | 0.0332 | 13.36 | yes |
| mom_6_1 | 0.0279 | 12.37 | overlaps the planted regressor |
| reversal_1m | 0.0161 | 6.73 | yes |
| ma_20_100 | 0.0132 | 5.84 | overlaps the planted regressor |
| rsi_14 | 0.0129 | 5.37 | overlaps the planted reversal |
| vol_63d | 0.0020 | 0.86 | no |
| turnover_z | 0.0004 | 0.17 | no |

Across 20 seeds the mom_12_1 IC is 0.0325 [0.0315, 0.0335] on synthetic
data. Tests: `test_planted_alpha_has_significant_ic`,
`test_noise_feature_has_no_ic`, `test_planted_momentum_ic_in_target_band`.
These tests show the pipeline recovers what the generator planted. They are
not evidence of alpha.

## 4. Headline book with statistics (synthetic)

| quantity | value | source |
|---|---|---|
| net Sharpe, 20 seeds, mean [95% CI] | 0.28 [-0.05, 0.60] | section 2 of the pipeline |
| gross Sharpe, 20 seeds | 5.05 [4.80, 5.31] | same |
| cost and borrow drag, 20 seeds (bp a year) | 4,075 [3,852, 4,298] | same |
| net Sharpe, seed 42 | 0.93, Lo/NW SE 0.48, bootstrap CI [0.02, 1.84] | section 1 |
| deflated Sharpe, seed 42, 10 logged trials | 0.59 (null spread), 0.00 (trial spread) | section 1 |
| PBO, 8 signal configurations | 0.00 | section 1 |

The trial log holds the seven single features, the combined signal and two
execution variants of the combined book. The second deflated Sharpe uses
the variance of those ten trial Sharpes, which is inflated by
high-turnover trials such as `turnover_z` (net Sharpe -52). The first uses
the sampling variance of a zero-Sharpe estimate. The truth is between them.

The statistics module (`research_stats.py`, a re-export of
`qp_core.stats`) is tested in `tests/test_research_stats.py`: the Newey-West Lo SE
matches the Monte Carlo sd of the Sharpe estimate on AR(1) returns within
10%; the expected maximum Sharpe matches simulation within 3%; the deflated
Sharpe is uniform under the null (5.0% above 0.95, bound 2%); PBO averages
near 0.5 on noise and near 0 when one configuration has skill; the block
bootstrap covers the true mean in more than 89% of 200 trials.

## 5. Costs and capacity (synthetic)

- The default impact coefficient is Y = 0.7, inside the 0.5-1 range the
  methodology cites (`test_default_impact_coefficient_is_in_the_cited_range`).
- Capacity is a full backtest per AUM and Y with per-name trailing ADV
  (`capacity_backtest`; `test_capacity_backtest_monotone_in_aum_and_impact`).
  On synthetic seed 42 the net Sharpe at $100m is 3.56 / 2.03 / 0.89 for
  Y = 0.3 / 0.7 / 1.0, and it is negative at $1bn for Y >= 0.7.
- Costs never feed back into positions: gross returns are identical across
  AUM (`test_impact_cost_increases_total_cost_and_scales_with_aum`).
- The signal band cuts turnover (`test_rebalance_band_reduces_turnover_in_backtest`).

## 6. Execution layer

Simulator mechanics (deterministic path tests):

- A burst moves every later mid by exactly the permanent move; the half
  spread and temporary impact appear only in the fill
  (`test_permanent_impact_persists_temporary_reverts`).
- Permanent impact is linear in size, temporary impact is square-root
  (`test_permanent_impact_linear_temporary_sqrt_in_participation`).
- Slicing consistency (W3). A fixed continuous trading-rate profile v(t),
  cut into n = 10, 20, 40 and 80 buckets with bucket volumes scaled to the
  bucket width, gives a temporary cost within O(1/n) of the quadrature value
  of the integral of eta sigma (v/V)^delta v dt, for delta 0.5, 0.6 and 1 and
  for flat and U-shaped volume (`test_bucket_cost_converges_to_continuous_integral`).
  The same TWAP order costs the same on 13, 26, 52 and 104 buckets
  (`test_twap_cost_does_not_fall_with_finer_slicing`). Before the fix the
  temporary cost of a 100,000-share TWAP on a flat day was 5.26, 3.72, 2.63
  and 1.86 bp on those grids; it is now 18.97 bp on all four
  (0.3 x 2% x sqrt(0.1)).
- Power-law optimiser. With exponent 1 `power_law_trajectory` equals the AC
  closed form to 1e-8 of X for four risk aversions
  (`test_power_law_solver_matches_closed_form_at_exponent_one`), and each
  optimiser wins under its own law (`test_power_law_solver_is_optimal_under_its_own_law`).
- Temporary impact uses the bucket's own volume: the same child order costs
  sqrt(v_open / v_mid) times more in the thin midday bucket
  (`test_thin_bucket_costs_more_than_heavy_bucket`).

Schedule ranking. `evaluate_schedules` reports the model-implied IS
(`model_implied_is_bps`; `expected_is_bps` is the same column under its
v2.0.0 name) from the same simulator with zero price and volume noise, and
the Monte Carlo std and the SE of the Monte Carlo mean. On synthetic seed 42
with 500 replications the model-implied IS is VWAP 49.6 < TWAP 53.0 <
power law 55.5 < AC 57.1 < POV 68.0 < Aggressive 124.7 bp, and the std is
Aggressive 21.0 < POV 61.2 < power law 71.6 < AC 74.5 < VWAP 109.5 <
TWAP 113.7 bp. (v2.0.0, with the double-counted bucket width: VWAP 13.7,
TWAP 14.3, AC 15.1, POV 17.2, Aggressive 27.5 bp; the order is unchanged.)
The TWAP hand value is checked to 1e-10
(`test_race_publishes_deterministic_expected_cost`).

Both optimisers under both laws (`compare_impact_laws`, same config,
lambda 5e-6, zero-noise simulator):

| schedule | objective, linear law | IS, linear law (bp) | objective, square-root law | IS, square-root law (bp) |
|---|---:|---:|---:|---:|
| AC, linear law | 12,698.6 | 20.8 | 31,945.4 | 59.3 |
| power law (delta 0.5) | 13,017.2 | 18.5 | 31,512.8 | 55.5 |

Each schedule has the lower objective under the law it was built for. The
coefficient Y is the same in both columns and is not re-fitted, so compare
down a column, not across.

AC against TWAP on the simulator: `test_ac_beats_twap_on_cost_variance_on_simulator`
requires std_AC < 0.85 std_TWAP and a Levene p < 0.01 over 300 paired
replications. The analytic frontier agrees in direction
(`test_ac_variance_reduction_matches_theory_direction`).

## 7. Exact identities and hand-computed checks

| identity | test | tolerance |
|---|---|---|
| Close-auction ledger on a 3-day toy book | `test_ledger_hand_computed_exact_close_auction` | 1e-15 |
| Next-price ledger on the same book | `test_fill_price_is_interpolated_next_price` | 1e-15 |
| Perold IS: delay + trading + opportunity = total | `test_is_components_sum_to_total_exactly` | 1e-10 |
| Toy IS order (60 at 10.2, 20 at 10.3, 20 unfilled) = 280 bp | `test_is_decomposition_hand_checked_toy_order` | 1e-12 |
| Slippage attribution sums per bucket | `test_slippage_attribution_components_sum_exactly` | 1e-10 |
| VWAP, TWAP, arrival of a toy tape | `test_vwap_hand_computed` et al. | 1e-12 |
| AC recursion x_{j-1} + x_{j+1} = 2 cosh(kappa tau) x_j | `test_trajectory_satisfies_discrete_optimality_recursion` | 1e-8 rel |
| AC at lambda = 0 is TWAP | dedicated tests | 1e-10 |
| AC cost moments on a hand trajectory | `test_cost_moments_hand_computed` | 1e-12 |
| NW SE on [1, 2, 3, 4] with 1 lag | `test_newey_west_hand_computed` | 1e-12 |

## 8. Edge cases

| edge case | behaviour | test |
|---|---|---|
| one-day backtest | nothing fills in the default mode; close auction builds the book | `test_one_day_backtest` |
| single-stock universe | flat book | `test_single_stock_universe_stays_flat` |
| all-NaN signal | flat ledger | `test_all_nan_signal_is_flat` |
| NaN prices mid-sample | name leaves the universe, ledger finite | `test_backtest_with_nan_prices_mid_sample` |
| zero-volume bucket with an order | `ValueError` | `test_zero_volume_bucket_raises_informative_error` |
| child larger than bucket volume | `ValueError` | `test_child_exceeding_bucket_volume_raises` |
| extreme risk aversion (kappa T > 710) | finite front-loaded schedule | `test_ac_extreme_risk_aversion_does_not_overflow_to_nan` |
| fully halted day | VWAP raises, TWAP defined | `test_fully_halted_day_zero_schedule_is_legal_but_has_no_vwap` |
| NaN or Inf schedule, tape, fills | `ValueError` | `test_nan_schedule_rejected_not_silently_filled` et al. |
| partial fill | Perold identity holds | `test_is_components_sum_to_total_on_partial_fill` |

## 9. Known failure modes

1. The edge is planted. On real data the momentum IC is a fraction of the
   synthetic one and decays as capital crowds in.
2. The square-root law is fitted on liquid names at modest participation.
   Above about 10% of ADV costs are worse than the model says, so the
   high-AUM rows of the capacity table are optimistic even at Y = 1.
3. Momentum crashes. The long-short book is short recent losers and takes
   a large loss in a sharp reversal. No statistic here predicts that.
4. Cost regimes. Spreads and impact rise in stress. The cost model is
   static.
5. The cost model works at schedule level on buckets: no queue, no
   resilience, no partial fills. `compare_impact_laws` shows how much the
   choice of law matters; a desk should fit the exponent on its own fills
   before trusting either optimiser.
