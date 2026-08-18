# Validation — FX Portfolio Optimization

All numbers below are reproducible offline: `pytest -q` (139 tests, ~12 s)
and `python examples/run_pipeline.py` (seed 123, 3 024 days, ~26 s).

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
| Sharpe SE = Lo (2002) iid formula | 1e-14 | `test_sharpe_se_lo_formula` |
| Skew/kurtosis = scipy.stats | 1e-10 | `test_skew_kurtosis_match_scipy` |

No lookahead is verified two ways: the allocator's history endpoint strictly
precedes the first P&L day of its weights, and a violent future spot move
leaves all earlier weights bit-identical.

## 2. Style statistics (pipeline, seed 123)

| Style | Ann. ret | Vol | Sharpe (Lo SE) | Skew | MDD | CVaR95/day |
|---|---:|---:|---:|---:|---:|---:|
| CARRY | 6.46% | 13.28% | 0.49 (0.29) | −1.73 | 18.20% | 1.90% |
| MOMENTUM | 4.18% | 8.47% | 0.49 (0.30) | −0.14 | 13.90% | 1.07% |
| VALUE | 6.56% | 9.88% | 0.66 (0.29) | −1.34 | 19.66% | 1.38% |

The carry sleeve shows the designed signature: positive premium with strong
negative skew; momentum is near-symmetric. Note the Lo standard errors: with
~11 years of daily data, a Sharpe of 0.49 is not even 2 SE from zero — the
estimation-error theme in numbers.

## 3. CVaR sizing numbers (pipeline, seed 123)

- Single-sleeve carry: vol-target sizing to 5% ann = 0.37x with
  CVaR95 0.72%/day; a 0.50%/day CVaR budget cuts the book to 0.26x
  (**30% smaller**) — same vol lens would never have flagged it.
- Multi-style RU-LP (gross ≤ 1.5): unconstrained mean-chaser CVaR95
  2.11%/day, E[ret] 11.0%/yr; with CVaR95 ≤ 0.40%: CVaR 0.40% (binding),
  E[ret] 3.0%/yr — an **81% tail cut for an 8.0%/yr expected-return
  give-up**, the price of insurance stated explicitly.
- Binding is verified: RU objective = limit = empirical CVaR of the
  solution to 1e-8 (`test_cvar_constraint_binds_and_cuts_tail`).

## 4. Hedging numbers (pipeline, seed 123)

Optimal hedge ratios: EUR 1.15, JPY **−1.40**, GBP 1.56, AUD **3.71**,
CHF **−0.42**. Unhedged vol 14.46% → full hedge 14.37% (−1.2% variance)
→ optimal 14.10% (**−4.9% variance**). Negative safe-haven ratios mean the
optimum *buys* JPY/CHF beyond the unhedged exposure; the risk-on AUD is
overhedged into a proxy equity short. Full hedging is dominated in both
tests and pipeline.

## 5. Walk-forward race (est. 504d, monthly, 5 bps, carry accrued)

| Allocator | Ann. ret | Vol | SR | Skew | MDD | CVaR95 | Carry P&L share |
|---|---:|---:|---:|---:|---:|---:|---:|
| EW | 6.73% | 7.73% | 0.87 | −1.83 | 11.32% | 1.11% | 69% |
| MVO (shrunk μ, γ=40) | 6.50% | 6.05% | 1.07 | −0.87 | 8.58% | 0.86% | 35% |
| ERC | 5.86% | 6.43% | 0.91 | −1.83 | 10.49% | 0.93% | 52% |
| CVaR-constr (≤0.40%/day) | 6.87% | 7.70% | 0.89 | −1.86 | 11.32% | 1.11% | 67% |

Read with care: MVO wins here because the synthetic world has stable,
estimable style means — the one thing real FX markets do not offer. ERC's
near-EW Sharpe with lower drawdown is the more transferable result. The
CVaR allocator's trailing-window constraint rarely binds at 0.40%/day in
calm stretches (its stats sit near EW); it exists for the windows where it
does bind.

## 6. Failure modes (documented AND unit-tested)

1. **2008-style carry unwind.** Forcing `crash_prob=0.3` (a crash every ~3
   days — the synthetic 2008): carry's mean turns negative and the sleeve
   loses money outright while all metrics stay finite
   (`test_all_crash_sample_carry_loses`). In the base calibration a single
   crash day costs the carry book ~4–8x its daily vol; a 504-day estimation
   window that happens to contain no crash will *understate* carry CVaR —
   this is the peso problem inside the risk model itself.
2. **Correlation regime flips.** High-carry correlations rise with crash
   intensity (AUD-TRY correlation is higher in the stressed world than the
   calm one — `test_crisis_regime_correlations_flip_up`): trailing-window
   diversification benefits evaporate exactly in stress. Mitigant: the
   one-factor risk-on/off model as a stress lens, not a point estimate.
3. **PPP value drawdowns.** The value signal's edge has a multi-year
   half-life; misvaluations of 25%+ persist for years in the generator
   (G10 gaps bounded by ~1, EM wider — `test_spots_track_ppp_loosely`).
   A decade-long value drawdown is consistent with the model being *right*.
4. **EM liquidity in crises.** Costs are linear bps on turnover; the 2016
   GBP flash crash and EM crisis spreads violate this. The backtest's cost
   drag (≤0.09%/yr here) is a floor, not an estimate, for crisis periods.
5. **Pegged currency in the universe.** Zero-vol asset makes Σ singular:
   closed forms raise a clear `ValueError`; the documented remedy
   (`psd_repair(min_eig=1e-10)`) restores invertibility, after which
   min-var correctly piles into the peg (~risk-free in this metric) — and
   ERC refuses (undefined risk budget) until the peg is dropped
   (`test_pegged_pair_in_universe_zero_vol_handled`). A peg is a policy
   option, not a riskless asset: cap peg weights by policy (2015 SNB).
6. **Degenerate cross-sections.** Zero rate differentials → carry signal
   identically zero → the book stands down (all-zero weights) rather than
   trading noise; single-currency universe likewise; momentum window longer
   than the sample yields an all-NaN signal and a flat book. All tested.
7. **Gross budget 0** returns the zero portfolio everywhere; infeasible
   budget combinations (|net| > gross) raise before any solver runs.

## 7. Cross-model consistency

- SLSQP vs closed forms (three problems) to 1e-5–1e-6.
- RU-LP vs exhaustive grid search; RU-LP vs closed-form homogeneous sizing.
- Hedging closed form vs derivative-free numerical optimum.
- ERC coordinate descent vs analytic 2-asset solution; Euler identity ties
  contributions to total variance.
- EWMA recursion vs 4-observation hand computation.
