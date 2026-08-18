# Validation — evidence, numbers, failure modes

All numbers below are produced by `python examples/run_pipeline.py`
(seeded, offline, ~1 s) and locked in by the test suite
(`python -m pytest tests` — 121 tests). Tolerances follow CONVENTIONS.md:
algebraic identities to 1e-10, model-vs-model to stated tolerance,
statistical claims over seeded replications with paired seeds.

---

## 1. Analytic and structural anchors

| Check | Result | Test |
|---|---|---|
| Piecewise-AC (KKT/QP) vs closed-form AC, constant η,σ | max abs diff < 1e-9 | `test_optimal.py::test_piecewise_reduces_to_closed_form_when_liquidity_constant` |
| λ → 0 limit = liquidity-weighted TWAP-analog (η ∝ 1/depth) | exact to 1e-8 | `test_lambda_zero_gives_liquidity_weighted_twap_analog` |
| Schedule sums = parent quantity | exact (1e-10), incl. active-set clamped case | `test_dp_solution_sums_exactly_to_parent` |
| Liquidity-aware AC ≤ naive (day-average) AC under true time-varying objective | holds for λ ∈ {1e-6, 1e-5, 1e-4}; strictly < 0.99× at λ=1e-5 | `test_liquidity_aware_cost_leq_naive_ac...` |
| Optimality vs random constraint-preserving perturbations | cost(n*) minimal in 20 perturbations | `test_optimality_against_perturbations` |
| IS decomposition components sum to total | residual < 1e-10, every venue/seed | `test_tca.py::test_is_components_sum_exactly` |
| Venue scorecard identity: effective = quoted ½-spread + temp + rejection cost | exact | `test_venue_scorecard_identity_exact` |
| Backtest ledger vs hand-computed toy scenario (session spreads, 3 trades) | exact | `test_ledger_exact_on_toy_scenario_with_session_spreads` |
| Carry accrual vs hand value 1.0·1.1·0.02/365 | exact to 1e-15 | `test_carry_accrual_hand_exact` |
| Zero trade ⇒ zero impact; execute(0) path ≡ simulate_mids path | bitwise equal | `test_zero_trade_zero_impact...` |
| Temporary impact sqrt-scaling: 4× size ⇒ 2× impact | exact to 1e-12 | `test_sqrt_impact_scaling_with_size` |
| Permanent persists / temporary decays | post-trade mids shifted by exactly the permanent amount | `test_permanent_impact_persists_temporary_decays` |
| Point-in-time discipline: mutate future ticks ⇒ features ≤ cutoff bit-identical | exact frame equality | `test_pit_mutation_leaves_past_features_unchanged` |
| No-lookahead: engine books lagged position; cheat signal earns only lagged P&L | exact | `test_no_lookahead_pnl_uses_lagged_position` |

Signal-layer statistics (60 days, hourly bars):

```
feature        IC     t-stat
mom_1        0.251     9.83     (planted phi = 0.25 -> recovered)
mom_4        0.187     7.22
reversion   -0.145    -5.57     (momentum world: honest negative IC)
breakout    -0.033    -1.26
mom_1        0.003     0.12     on phi = 0 noise  -> no false alpha
```

Test-locked: IC > 0.10 with t > 2 on planted alpha; |t| < 2.5 on three noise
seeds (`test_synthetic.py`).

## 2. Scheduler comparison (500mm EURUSD buy, 200 paired replications)

Controllable cost = spread + temporary + permanent (deterministic given the
schedule in this model — std < 1e-12 across seeds); IS std = execution risk
vs arrival.

Full 24h day (288 × 5-min buckets):

| schedule | ctrl cost (pips) | IS mean | IS std |
|---|---|---|---|
| TWAP | 0.578 | 2.13 | 31.23 |
| liquidity-weighted | **0.517** | 2.10 | 33.46 |
| POV 5% (analog) | 0.709 | 1.08 | 11.46 |
| piecewise-AC (λ=1e-5) | 0.733 | 0.88 | **7.71** |

London-only window 07:00–16:00 (108 buckets):

| schedule | ctrl cost (pips) | IS mean | IS std |
|---|---|---|---|
| TWAP | 0.531 | 1.06 | 24.17 |
| liquidity-weighted | **0.525** | 1.05 | 25.77 |
| POV 5% (analog) | 0.686 | 0.54 | 14.31 |
| piecewise-AC (λ=1e-5) | 0.750 | 0.71 | **9.52** |

Readings, each test-locked:

- **Liquidity-weighting saves ~11% of controllable cost vs TWAP over the full
  day** (0.517 vs 0.578 pips ≈ $30k on 500mm at 1e-4 pip) by avoiding the
  late/Asia spread and impact; the saving shrinks to ~1% inside the already
  liquid London window — time-of-day awareness matters most when the horizon
  spans bad hours (`test_liquidity_weighted_beats_twap_controllable_cost`).
- **The AC frontier is visible**: λ=1e-5 front-loading pays +0.22 pips of
  controllable cost to cut IS std from ~31 to ~8 pips — the mean–variance
  trade-off, chosen by client risk preference, not by "cheapest".
- On 500mm, a single-bucket execution violates the 30%-of-depth cap and
  raises with a multi-session hint (`test_500mm_parent_requires_multi_session_split`).

## 3. The last-look trap (200 paired replications, common random numbers)

| flow | venue | quoted ½-spread | effective cost | reject rate | reject cost |
|---|---|---|---|---|---|
| uninformed (α=0) | last-look | 0.107 | **0.290** | 9.0% | 0.053 |
| | firm ECN | 0.178 | 0.309 | 0.0% | 0 |
| informed (α=0.5 pips/bucket) | last-look | 0.107 | **0.584** | 40.8% | 0.347 |
| | firm ECN | 0.178 | 0.309 | 0.0% | 0 |

Paired differences (LL − firm): **−0.018 pips (SE 0.0010)** uninformed;
**+0.275 pips (SE 0.0024)** informed. Both sides of the trap are test-locked
(`test_the_trap_effective_cost_higher_with_adverse_selection`): the tighter
quote is real for benign flow and a mirage for informed flow, with the gap
attributable to rejections (rate 9% → 41%, monotone in α). Rejection
probability is strictly monotone in the adverse move
(`test_reject_prob_monotone_increasing_in_adverse_move`), firm venues never
reject, and every rejected child refills strictly worse than its quote.

## 4. Fix benchmark (200 replications, 100mm, 1-min buckets 14:00–17:00)

| schedule | TE mean (pips) | TE std (pips) |
|---|---|---|
| fix-targeting (5-min window) | 0.512 | **0.000** |
| TWAP 14:00–17:00 | 0.240 | 9.844 |

The fix algo pays a visible spread+impact cost (0.51 pips at 20mm/min — the
window is short, participation is high) but tracks the print with essentially
zero variance; a day-TWAP is *cheaper on average* against the fix but carries
~10 pips of benchmark risk — useless for a client measured on the fix.
Test-locked with std ratio > 10× (`test_fix_targeting_tracks_the_fix`).

## 5. Failure modes & edge cases (documentation contract items 4 & 6)

Each is documented here **and** unit-tested.

1. **Fix gaming / regulatory history.** The mechanics that made the 60-second
   fix gameable (trading inside the window moves the print) are reproduced by
   the simulator's impact model; concentrated fix-window flow moves the
   window mids. The 2013 chat-room scandal, 2014–15 fines and the 5-minute
   window reform are covered in METHODOLOGY.md §2. Model failure mode: our
   fix print has no median filter, so single-bucket impact affects it more
   than the real WM/R methodology (assumption A8).
2. **Flash events (GBP, 7 Oct 2016).** Sterling dropped ~8% in Asia-hours
   minutes on thin liquidity and stop cascades. Gaussian session vol cannot
   produce this (assumption A5); we probe the regime with `vol_scale=5` on
   GBPUSD: controllable cost > 2×, IS std > 3× calm conditions
   (`test_flash_vol_regime_multiplies_execution_cost`). Lesson: any schedule
   optimised on calm profiles understates tail cost in Asia hours for GBP.
3. **Liquidity mirage / last-look adverse selection.** §3 numbers: venue
   choice by quoted spread alone selects the venue that is worse for exactly
   the flow that has alpha. Dealer scorecards must be computed on *effective*
   cost with rejection attribution (DESK_GUIDE.md).
4. **EM cost regimes.** Same planted alpha, same positions: net **+3490
   pips** under EURUSD session spreads vs **−974 pips** under EM-style
   spreads (10–40 pips) — costs flip profitability
   (`test_em_wide_spread_flips_strategy_profitability`); USDMXN execution
   cost is > 20× EURUSD for comparable participation
   (`test_em_pair_execution_costs_dwarf_major`).
5. **Weekend gap.** Non-tradeable buckets: schedulers place zero there,
   the simulator freezes diffusion and raises on any attempted fill
   (`test_weekend_gap_end_to_end`, `test_weekend_bucket_trading_raises...`).
6. **Single-bucket execution** returns `[X]` across every scheduler and the
   AC solver (`test_twap_single_bucket`, `test_single_bucket_and_sell_side`).
7. **Order bigger than depth cap** ⇒ `ValueError` with a multi-session hint;
   POV that cannot complete ⇒ `ValueError` (`test_depth_cap_exceeded...`,
   `test_pov_infeasible_raises`).
8. **Zero-vol path** ⇒ fully deterministic, seed-independent execution whose
   IS equals the quantity-weighted half-spread exactly
   (`test_zero_vol_path_deterministic_cost`, `test_zero_vol_execution_fully_deterministic`).
9. **Numerical limits.** The KKT solve is dense O(N³): fine to N ≈ 1440
   (1-min day, <1 s); beyond that use the banded structure. At extreme λ the
   unconstrained QP sells back in thin buckets — handled by the exact
   active-set clamp, verified non-negative with exact parent sum
   (`test_active_set_clamps_thin_buckets_at_high_lambda`).
10. **Honest-result guards.** Backtest P&L is invariant to future-price
    mutation up to the cutoff; feature matrices are bit-identical under
    future-tick mutation; a same-bar "cheat" signal cannot earn its
    contemporaneous return (`test_backtest.py`, `test_features.py`).

## 6. Reproducibility

- Every stochastic component takes an explicit seed / `numpy.random.Generator`;
  identical seeds give bitwise-identical paths and fills
  (`test_seeded_reproducibility`, `test_ticks_seeded_reproducible`).
- Venue and scheduler comparisons use common random numbers (same seed ⇒ same
  exogenous path), so paired differences isolate the decision variable.
- `pytest -q` runs offline in ~1 s; the pipeline example in ~1 s.
