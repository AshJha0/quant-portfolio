# Validation: FX Portfolio Optimization

All numbers below are reproducible offline: `pytest -q` (187 tests, about 40 s)
and `python examples/run_pipeline.py` (seed 123, 3 024 days, synthetic data, about 60 s).

## 1. Analytic identities (unit-tested tolerances)

| Identity | Tolerance | Test |
|---|---|---|
| total = spot + carry (exact decomposition) | 0 (bitwise) | `test_total_equals_spot_plus_carry_exact` |
| Carry accrual uses previous-close rates (lag check on a hand-built rate jump) | 1e-15 | `test_carry_uses_previous_day_rates` |
| Min-var closed form: Σw ∝ 1, sum = 1 | 1e-12 | `test_min_variance_closed_form_identity` |
| Tangency ∝ Σ⁻¹μ; maximises Sharpe on frontier grid | 1e-12 / 1e-10 | `test_tangency_closed_form_identity` |
| Frontier weights hit target mean, sum 1 | 1e-12 | `test_frontier_weights_hit_target` |
| Dollar-neutral KKT: γΣw − μ constant vector, Σw = 0 | 1e-12 | `test_dollar_neutral_closed_form` |
| SLSQP = closed form (min-var, frontier point, dollar-neutral) | 1e-5 / 1e-6 | `test_slsqp_matches_*` |
| Frontier volatility monotone above min-var | 1e-12 | `test_frontier_monotonic_vol` |
| LW intensity δ ∈ [0,1]; Σ_LW = δmI + (1−δ)S exactly; cond(Σ_LW) < cond(S) at T=2N | exact / strict | `test_lw_*` |
| One-factor loadings signs: AUD/NZD/MXN/BRL +, JPY/CHF − | sign | `test_one_factor_recovers_riskonoff_signs` |
| ERC: max deviation of risk contributions | 1e-8 (achieves ~1e-13) | `test_erc_equal_contributions` |
| Euler: Σ w_i(Σw)_i = w'Σw | 1e-12 | `test_euler_identity` |
| ERC 2-asset diagonal = inverse-vol split (1/3, 2/3) | 1e-10 | `test_erc_two_asset_diagonal_analytic` |
| Ex-ante vol after targeting = target | 1e-12 | `test_vol_target_exact_ex_ante` |
| Hedge ratio closed form = Nelder-Mead brute force | 1e-6 | `test_closed_form_matches_brute_force` |
| var(optimal hedge) ≤ var(unhedged), ≤ var(full hedge) | strict | `test_hedged_variance_leq_unhedged` |
| h\* = 1 exactly when r_u = x'r_fx | 1e-10 | `test_full_hedge_optimal_when_no_local_fx_correlation` |
| Empirical CVaR = RU functional minimum | 1e-9 | `test_empirical_cvar_equals_ru_minimisation` |
| RU-LP objective = empirical CVaR of optimal weights | 1e-9 | `test_ru_objective_equals_empirical_cvar_at_optimum` |
| RU-LP = exhaustive 2001-point grid search (toy 2-asset) | 2e-3 in w, 1e-9 in CVaR | `test_min_cvar_matches_exhaustive_grid` |
| CVaR positive homogeneity CVaR(3r) = 3·CVaR(r) | 1e-12 | `test_cvar_positive_homogeneity` |
| Backtest ledger vs hand computation (incl. carry accrual, pip costs on turnover) | 1e-15 | `test_ledger_hand_computed_with_carry` |
| pips→bps: 25 pips at EURUSD 1.25 = 20 bps; JPY pip size 0.01 | 1e-12 | `test_pips_to_bps_hand_computed` |
| Base switch adds base ccy's own total return (cross-rate reconstruction EURGBP = EURUSD/GBPUSD) | 1e-14 | `test_base_switch_adds_base_ccy_own_return` |
| Dollar-neutral log-return series invariant to base ccy | 1e-14 (exact conditions: log returns, Σw=0) | `test_dollar_neutral_log_returns_invariant_to_base` |
| Sharpe SE = iid formula (Lo 2002, no autocorrelation term) | 1e-14 | `test_sharpe_se_lo_formula` |
| Skew/kurtosis = scipy.stats | 1e-10 | `test_skew_kurtosis_match_scipy` |

No lookahead is verified two ways: the allocator's history endpoint strictly
precedes the first P&L day of its weights, and a violent future spot move
leaves all earlier weights bit-identical.

## 2. Style statistics (pipeline, seed 123, synthetic)

Net of 5 bp per unit of currency-level turnover inside each sleeve:

| Style | Ann. ret | Vol | Sharpe (iid SE) | Gross Sharpe | Turnover/yr | Skew | MDD | CVaR95/day |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CARRY | 6.25% | 13.28% | 0.47 (0.29) | 0.49 | 4.2x | -1.73 | 18.43% | 1.90% |
| MOMENTUM | 1.90% | 8.47% | 0.22 (0.30) | 0.49 | 45.5x | -0.14 | 19.90% | 1.08% |
| VALUE | 5.66% | 9.88% | 0.57 (0.29) | 0.66 | 17.9x | -1.33 | 23.82% | 1.38% |

The sleeves are re-formed daily at the currency level, so momentum's
45x/yr turnover costs it more than half its gross Sharpe at 5 bp. The
standard errors are the iid formula. None of the Sharpe ratios is 2 SE
from zero. The value signal uses the PPP anchor that the synthetic spots
revert to, so value's Sharpe is a property of the generator.

## 3. CVaR sizing numbers (pipeline, seed 123, synthetic)

- Single-sleeve carry: vol-target sizing to 5% ann = 0.37x with
  CVaR95 0.72%/day. A 0.50%/day CVaR budget cuts the book to 0.26x
  (30% smaller).
- Multi-style RU-LP (gross ≤ 1.5): unconstrained mean-chaser CVaR95
  2.87%/day, E[ret] 10.3%/yr. With CVaR95 ≤ 0.40%: CVaR 0.40% (binding),
  E[ret] 2.2%/yr. That is an 86% tail cut for an 8.1%/yr expected-return
  give-up.
- Binding is verified: RU objective = limit = empirical CVaR of the
  solution to 1e-8 (`test_cvar_constraint_binds_and_cuts_tail`).
- `empirical_cvar` is sort-based, O(S log S), and equals the brute-force
  RU minimum to 1e-12 (`test_sort_cvar_equals_rockafellar_uryasev_minimum`).
- `carry_sizing` returns size 0 for a sleeve with non-positive mean
  (`test_carry_sizing_negative_mean_gives_zero_size`).

## 4. Hedging numbers (pipeline, seed 123, synthetic)

| Currency | h* in sample | OLS SE | h* first half | h* second half |
|---|---:|---:|---:|---:|
| EUR | 1.15 | 0.18 | 1.26 | 1.05 |
| JPY | -1.40 | 0.31 | -1.66 | -1.18 |
| GBP | 1.56 | 0.36 | 1.57 | 1.59 |
| AUD | 3.71 | 0.44 | 2.75 | 4.71 |
| CHF | -0.42 | 0.36 | -1.42 | 0.59 |

In sample: unhedged vol 14.46%, full hedge 14.37% (-1.2% variance),
optimal 14.10% (-4.9% variance). Out of sample (h* from the first half,
variance on the second half): full hedge -2.1%, optimal -4.1%. CHF changes
sign between halves. The AUD ratio of 3.71 is an equity cross-hedge. The
synthetic equity generator is linear in FX, so OLS is correctly specified
by construction. Including forward carry raises the full-hedge mean from
4.33% to 5.01% a year with unchanged variance
(`test_hedged_returns_include_forward_carry`). The standard errors match a
300-draw Monte Carlo to 15% (`test_hedge_ratio_standard_errors_match_monte_carlo`).

## 5. Walk-forward race (est. 504d, monthly, synthetic)

Costs: 5 bp on style-level reallocation plus 5 bp on currency-level
turnover inside each sleeve. Covariance: Ledoit-Wolf on the trailing 504
days. MVO uses sample style means, since James-Stein shrinkage is 0 for
three styles.

| Allocator | Ann. ret | Vol | SR (iid SE) | Skew | MDD | CVaR95 | Carry P&L share |
|---|---:|---:|---:|---:|---:|---:|---:|
| EW | 5.65% | 7.73% | 0.73 (0.33) | -1.84 | 12.64% | 1.12% | 82% |
| MVO (sample μ, γ=40) | 4.77% | 6.10% | 0.78 (0.33) | -0.95 | 10.69% | 0.88% | 52% |
| ERC | 4.53% | 6.47% | 0.70 (0.33) | -1.83 | 12.08% | 0.95% | 69% |
| CVaR-constr (≤0.40%/day) | 5.79% | 7.69% | 0.75 (0.33) | -1.86 | 12.64% | 1.12% | 79% |

The Sharpe differences are at most 0.08, against standard errors of about
0.33. The race does not rank the allocators. Every optimiser call raises
on solver failure, so no failed iterate reaches the ledger.

## 6. Failure modes (documented AND unit-tested)

1. **2008-style carry unwind.** Forcing `crash_prob=0.3` (a crash every ~3
   days, the synthetic 2008): carry's mean turns negative and the sleeve
   loses money outright while all metrics stay finite
   (`test_all_crash_sample_carry_loses`). In the base calibration a single
   crash day costs the carry book ~4–8x its daily vol; a 504-day estimation
   window that happens to contain no crash will *understate* carry CVaR:
   this is the peso problem inside the risk model itself.
2. **Correlation regime flips.** High-carry correlations rise with crash
   intensity (AUD-TRY correlation is higher in the stressed world than the
   calm one, `test_crisis_regime_correlations_flip_up`): trailing-window
   diversification benefits evaporate exactly in stress. Mitigant: the
   one-factor risk-on/off model as a stress lens, not a point estimate.
3. **PPP value drawdowns.** The value signal's edge has a multi-year
   half-life; misvaluations of 25%+ persist for years in the generator
   (G10 gaps bounded by ~1, EM wider: `test_spots_track_ppp_loosely`).
   A decade-long value drawdown is consistent with the model being *right*.
4. **EM liquidity in crises.** Costs are linear bps on turnover; the 2016
   GBP flash crash and EM crisis spreads violate this. The backtest's cost
   drag (≤0.06%/yr at the allocation level here) is a floor, not an estimate, for crisis periods.
5. **Pegged currency in the universe.** Zero-vol asset makes Σ singular:
   closed forms raise a clear `ValueError`; the documented remedy
   (`psd_repair(min_eig=1e-10)`) restores invertibility, after which
   min-var correctly piles into the peg (~risk-free in this metric): and
   ERC refuses (undefined risk budget) until the peg is dropped
   (`test_pegged_pair_in_universe_zero_vol_handled`). A peg is a policy
   option, not a riskless asset: cap peg weights by policy (2015 SNB).
6. **Degenerate cross-sections.** Zero rate differentials → carry signal
   identically zero → the book stands down (all-zero weights) rather than
   trading noise; single-currency universe likewise; momentum window longer
   than the sample yields an all-NaN signal and a flat book. All tested.
7. **Gross budget 0** returns the zero portfolio everywhere; infeasible
   budget combinations raise before any solver runs. Two families are
   detected up front, each with a reason in the message:
   * gross-leverage budget below the net budget (`gross_limit=0.5` with
     `sum_to=1.0`: you cannot be 100% net invested on 50% of gross), and
   * a per-currency box that cannot reach the net budget: 3 currencies
     capped at 20% each cannot sum to 1.0 (max attainable 0.6).
   Both raise `ValueError`
   (`test_box_cap_unreachable_for_net_budget_is_rejected`), while an
   exactly-attainable box (3 x 1/3 = 1.0) still solves
   (`test_feasible_box_at_the_boundary_still_solves`).

8. **Constant (zero-dispersion) return series: a numerical-stability
   trap.** An exactly-constant series does not produce `std == 0` in
   floating point: `np.full(50, 4e-4).std(ddof=1)` is ~5.5e-20, not zero.
   An exact `== 0` guard therefore let it through and `sharpe_ratio`
   returned **1.16e17**: a headline stat that is pure floating-point
   noise. The degeneracy guards now compare the dispersion against the
   scale of the data (`1e-14 * max(1, max|r|)`), which is many orders of
   magnitude below any real FX return series, so genuine low-vol books are
   unaffected while constants raise
   (`test_constant_return_series_has_zero_vol_and_undefined_sharpe`).
   The same fix applies to `sharpe_se_lo`, `sortino_ratio`, `skewness`
   and `excess_kurtosis`.

9. **NaN/Inf anywhere in the inputs.** A single missing FX day used to
   propagate silently: `optimal_hedge_ratios` returned all-NaN ratios,
   every metric returned NaN, `psd_repair` returned an all-NaN matrix and
   `is_psd` raised a bare `LinAlgError` ("Eigenvalues did not converge").
   Every public entry point: covariance estimators, `psd_repair`/`is_psd`,
   MVO (`mu` and `sigma`), the CVaR LP and its scenario panel, ERC and
   risk contributions, the hedging block and all performance metrics:
   now raises an informative `ValueError`. Rationale: a NaN risk number
   that reaches a limit check reads as "no breach". Tested in the
   NaN/Inf block of `tests/test_edge_cases_review.py`.

## 7. Cross-model consistency

- SLSQP vs closed forms (three problems) to 1e-5–1e-6.
- RU-LP vs exhaustive grid search; RU-LP vs closed-form homogeneous sizing.
- Hedging closed form vs derivative-free numerical optimum.
- ERC coordinate descent vs analytic 2-asset solution; Euler identity ties
  contributions to total variance.
- EWMA recursion vs 4-observation hand computation.

## Round 3 oracle tests (`tests/test_round3_portfolio.py`, 19 tests)

- HRP gives exactly the inverse-variance weights for a diagonal Sigma and matches the two-asset closed form. It stays long-only on a near-singular Sigma.
- The closed-form MV satisfies its first-order condition. Resampled weights converge to the classic weights as T grows. Robust MVO with kappa = 0 equals the classic solution, and with kappa > 0 it equals the classic solution at the effective risk aversion.
- The cost QP matches the closed-form no-trade band with and without impact. With zero cost it equals the classic MV. The turnover cap binds and the box bounds hold.
- The factor model recovers the true loadings and reproduces lstsq exactly. Its Euler sums match the totals to 1e-13.
- TE: the active weights are budget-neutral, hit the target TE and attain the IR bound. The SLSQP solution equals the closed form when the bounds are slack.
- Sharpe: with no lags, the HAC standard error equals Memmel's to within 2% on 200k normal draws. The gradient matches finite differences. HAC widens under AR(1) returns. The QS kernel matches its closed form and its k_2 = 18 pi^2 / 125 curvature at 0; the Andrews bandwidth matches the AR(1) plug-in; the prewhitened QS estimator recovers the AR(1) long-run variance 1 / (1 - rho)^2 and a VAR(1)'s (I - A)^-1 (I - A)^-T within 3-4% on 200k draws, and the sample covariance on iid data; with `kernel="qs_pw"` the standard error equals Memmel's within 3% on iid normals. The studentised bootstrap rejects at most 9 of 60 seeded null samples at 5% (AR(1) returns, T = 300) and rejects a true difference; `studentize=False` reproduces the old centred statistic exactly. A defect found while testing: the QS kernel's closed form loses about 5e-6 to cancellation at x = 1e-6; it now uses its Taylor series below |6 pi x / 5| = 0.05.
