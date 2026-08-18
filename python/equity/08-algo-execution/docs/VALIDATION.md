# Validation — evidence, numbers, failure modes

All numbers below are produced by `python examples/run_pipeline.py` (seed
42) or by named tests in `tests/` (145 tests, all passing, ~4 s, offline).

---

## 1. Point-in-time / no-lookahead (the tests that matter most)

- **Feature-level mutation test** (`test_features.py::test_point_in_time_mutation`,
  parametrised over all 6 features): compute features, then multiply every
  price/volume strictly after row 300 by random factors in [0.2, 5.0] and
  recompute. All rows ≤ 300 must be **bit-identical**
  (`assert_frame_equal`, exact). Passes for momentum, reversal, realised
  vol, MA crossover, RSI, turnover-z.
- **Full-pipeline mutation test**
  (`test_backtest.py::test_no_lookahead_full_pipeline_mutation`): the whole
  chain — features -> z-scores -> signal freeze -> decile weights -> ADV/vol
  estimates -> costs -> ledger — re-run after mutating all data past day
  400. Ledger rows ≤ 400 bit-identical, including the cost column (the
  impact model's rolling ADV$ and vol estimators are PIT too).

## 2. Planted alpha recovered at the right strength

Generator plants next-day momentum alpha of strength 0.0008/0.02 = IC 0.04
and a reversal effect. Pipeline output (150 stocks x 1250 days):

| feature     | mean IC | NW t-stat |  verdict |
|-------------|--------:|----------:|----------|
| mom_12_1    |  0.0332 |     13.36 | planted, recovered |
| mom_6_1     |  0.0279 |     12.37 | planted, recovered |
| reversal_1m |  0.0161 |      6.73 | planted, recovered |
| vol_63d     |  0.0020 |      0.86 | not planted, correctly null |
| turnover_z  |  0.0004 |      0.17 | not planted, correctly null |

Statistical tests: `test_planted_alpha_has_significant_ic` (t > 2 required),
`test_noise_feature_has_no_ic` (|t| < 2 required on a pure-noise signal),
`test_planted_momentum_ic_in_target_band` (0.02 < IC < 0.07). Decile
portfolios perfectly monotone: rho = 1.000, Q1 -10.9 -> Q10 +15.3 bps/day.
IC decay: cumulative-horizon IC grows 0.036 (1d) -> 0.129 (20d); against
the no-decay benchmark sqrt(20)·IC_1 = 0.162, i.e. ~20% of per-day alpha
decays over the month.

## 3. Exact identities and hand-computed checks

| identity | test | tolerance |
|---|---|---|
| Backtest ledger (weights, turnover, cost, gross/net) on a 3-day toy book | `test_ledger_hand_computed_exact` | 1e-15 |
| Perold IS: delay + trading + opportunity = total | `test_is_components_sum_to_total_exactly` (25 random orders incl. partial fills) | 1e-10 |
| Toy IS order (60@10.2, 20@10.3, 20 unfilled): 10 + 10 + 8 = 28 = 280 bps | `test_is_decomposition_hand_checked_toy_order` | 1e-12 |
| Slippage attribution: drift + spread + temporary = total per bucket and qty-weighted | `test_slippage_attribution_components_sum_exactly` | 1e-10 |
| VWAP of toy tape (10,11,12 x 1,2,1) = 11.0; TWAP = 11.0; arrival = 10.0 | `test_vwap_hand_computed` et al. | 1e-12 |
| AC trajectory satisfies x_{j-1}+x_{j+1} = 2cosh(kappa tau) x_j | `test_trajectory_satisfies_discrete_optimality_recursion` | 1e-8 rel |
| AC lambda=0 = TWAP; sum of AC trades = X | dedicated tests | 1e-10 / 1e-12 |
| AC cost moments on a hand trajectory [10,4,0] | `test_cost_moments_hand_computed` | 1e-12 |
| DSR = PSR at N = 1; E[maxSR] hand-checked at N=10 | `test_dsr_decreases_with_trials_...`, `test_expected_max_sharpe_properties` | 1e-12 |
| NW SE hand-computed on [1,2,3,4] with 1 lag; lags=0 = naive SE | `test_newey_west_hand_computed` | 1e-12 |
| RSI/momentum/reversal/vol/MA/turnover-z on tiny series | `test_*_hand_computed` | 1e-12 |

## 4. Execution-layer statistical evidence

**Simulator mechanics** (deterministic-path tests): permanent impact of a
50k burst moves *all* later mids by exactly perm = 0.5·0.02·0.05·100 =
0.05 while the fill price additionally carries half-spread + sqrt temporary
impact that never appears in subsequent mids (temporary reverts —
`test_permanent_impact_persists_temporary_reverts`). Permanent impact
verified linear (4x size -> 4x move), temporary verified sqrt (4x size ->
2x cost). Zero participation -> zero impact and an untouched mid path.

**AC vs TWAP, 300 paired replications** (5% ADV buy, sigma 2%/day, sqrt
temp coef 1.0, perm coef 0.5, lognormal volume noise):

| schedule | mean IS (bps) | std IS (bps) |
|---|---:|---:|
| TWAP          |  8.3 | 113.0 |
| AC lambda=5e-6 | 12.0 |  74.0 |

Variance reduction 35% for +3.7 bps expected cost; Levene test W = 42.9,
**p = 1.2e-10** (`test_ac_beats_twap_on_cost_variance_on_simulator`, which
requires std_AC < 0.85·std_TWAP *and* p < 0.01). The analytic frontier
agrees in direction: model-implied V_AC < V_TWAP and E_AC > E_TWAP
(`test_ac_variance_reduction_matches_theory_direction`). Frontier
monotonicity (E strictly up, V strictly down in lambda) is tested across
six lambdas and shown in the demo: (15.5, 99.0) -> (21.1, 73.8) ->
(53.9, 36.5) bps for lambda = 1e-6 / 5e-6 / 5e-5.

**200-replication horse race** (demo): TWAP 29.8 ± 114.2, VWAP 29.6 ±
109.6, POV-10% 25.2 ± 63.6, AC(5e-6) 25.7 ± 76.6, Aggressive-2-buckets
37.6 ± 19.3 bps. Mean differences between TWAP/VWAP are within Monte Carlo
noise (SE ≈ 8 bps); the variance ordering is the robust result.

## 5. Cost, capacity, overfitting guards

- Gross vs net (AUM $200m): Sharpe 4.86 -> 4.06, cost drag 709 bps/yr at
  20% daily turnover. Costs never feed back into positions
  (`test_impact_cost_increases_total_cost_and_scales_with_aum` checks gross
  returns are identical across AUM).
- Capacity curve strictly decreasing in AUM (tested): net Sharpe 4.54 at
  $10m -> 3.94 at $5bn on the demo panel (deep synthetic liquidity; real
  small/mid-cap books decay far faster — see failure modes).
- Signal-band rebalancing (0.25-z freeze) cuts turnover from 30% to 20%/day
  and *raises* net Sharpe 3.45 -> 4.06 in the demo; reduction property is
  tested (`test_rebalance_band_reduces_turnover_in_backtest`).
- Deflated Sharpe: demo MA-crossover strategy SR 1.35, PSR0 = 0.9954, DSR =
  0.8912 (N=7) and 0.6521 (N=45). DSR strictly decreasing in N (tested).

## 6. Edge cases (each documented *and* unit-tested)

| edge case | behaviour | test |
|---|---|---|
| one-day backtest | 1-row ledger, no return, book built | `test_one_day_backtest` |
| single-stock universe | flat book, zero P&L, no crash | `test_single_stock_universe_stays_flat` |
| all-NaN signal | flat ledger | `test_all_nan_signal_is_flat` |
| delisting mid-sample (NaN prices) | name leaves universe, ledger finite | `test_backtest_with_nan_prices_mid_sample` |
| zero-volume bucket, order routed there | informative ValueError ("reschedule around the halt") | `test_zero_volume_bucket_raises_informative_error` |
| zero-volume bucket, POV | bucket skipped, order completes | `test_zero_volume_bucket_pov_skips_it` |
| parent > day volume at cap | ValueError naming day capacity, advising multi-day split | `test_parent_order_larger_than_day_volume_informative_error` |
| child > bucket volume | ValueError (participation > 100%) | `test_child_exceeding_bucket_volume_raises` |
| zero risk aversion | AC = TWAP exactly | `test_zero_risk_aversion_ac_is_twap` |
| lambda -> infinity | >99.9% in first slice | `test_higher_risk_aversion_front_loads` |
| T = 1 slice AC | [X, 0] trajectory for any lambda | `test_single_slice_ac` |
| single-bucket market | exact half-spread fill | `test_single_bucket_intraday_market` |
| eta_tilde <= 0 | ValueError (ill-posed AC) | `test_params_validation` |
| zero-vol returns / N=0 trials / short series | ValueError | `test_deflated_sharpe_edge_inputs` et al. |

## 7. Known failure modes

1. **Alpha decay and crowding.** The planted IC is stationary; real
   momentum ICs decay as capital crowds in. The IC-weighted combiner
   (expanding, lagged) would down-weight a dying signal only slowly; a
   rolling window or decay-time-scale monitoring (DESK_GUIDE) is the
   production answer. Symptom to watch: live IC drifting below the
   backtest's NW confidence band.
2. **Impact model misspecification for illiquid names.** The sqrt law is
   calibrated on liquid equities. For names at > ~10% ADV participation,
   costs are convex-worse than sqrt (book depletion, information leakage);
   the capacity curve and POV caps understate the pain. The simulator's
   guard (child > bucket volume raises) is a blunt version of the real
   constraint.
3. **Momentum crashes (2009-style).** Cross-sectional momentum is short
   the highest-beta losers after a crash; in a sharp reversal the L/S book
   takes a multi-sigma left-tail hit that no daily-IC statistic predicts.
   The synthetic panel can be re-run with `mom_strength` flipped for a
   window to rehearse this; the backtest's max-drawdown and the Sortino/
   Sharpe gap are the early indicators. Guard: crash filters (skip signal
   after market drawdowns), vol-scaled deciles — out of scope here,
   documented as the known limitation of raw momentum.
4. **Backtest overfitting.** The guard *is* the deflated Sharpe ratio:
   report DSR at the honest trial count N (every variant, hyper-parameter
   and universe tweak counts). The demo shows a 0.995-PSR strategy falling
   to DSR 0.65 at N=45.
5. **Regime dependence of costs.** Spread and impact coefficients double or
   worse in stress while volumes shift toward the close; a static cost
   model then misprices both the daily drag and the AC schedule. Production
   answer: re-fit weekly from own-fill TCA, apply stress multipliers in
   risk scenarios (DESK_GUIDE kill-switch section).
6. **Simulator-optimiser mismatch (by design).** AC assumes linear
   temporary impact; the simulator charges sqrt. Consequence visible in the
   demo: concentrating flow (Aggressive) pays 37.6 bps mean vs TWAP's 29.8,
   more than linear-model AC would predict. Conclusions about *rankings*
   (variance ordering) are robust; absolute expected-cost numbers are
   model-dependent.
