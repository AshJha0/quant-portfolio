# Validation — Equity Portfolio Optimization & Risk Allocation

All numbers below are produced by `python examples/run_pipeline.py`
(seeded, offline, ~15 s) and the pytest suite (`pytest -q`, 141 tests,
~3 s, offline). Tolerances follow CONVENTIONS.md: analytic identities to
1e-10–1e-8 (most hold to machine precision), solver-vs-closed-form to
documented tolerance.

## 1. Analytic identities (unit-tested, exact or near machine precision)

| Identity | Tolerance | Test |
|---|---|---|
| Min-variance weights = `Σ⁻¹1/(1'Σ⁻¹1)` | 1e-12 | `test_min_variance_matches_formula` |
| Tangency weights = `Σ⁻¹μₑ/(1'Σ⁻¹μₑ)` | 1e-12 | `test_tangency_matches_formula` |
| Reverse-optimization round trip: `tangency(δΣw_mkt) = w_mkt` | ≤ 3.5e-16 observed | `test_reverse_optimization_round_trip_identity` |
| Black-Litterman, no views ⇒ posterior = prior | exact (0) | `test_bl_no_views_posterior_equals_prior` |
| Black-Litterman, `Ω = 0` ⇒ `Pμ_BL = Q` | 1e-14 | `test_bl_infinitely_confident_view_holds_exactly` |
| BL posterior between prior and view (finite Ω) | strict inequality | `test_bl_posterior_between_prior_and_view` |
| James-Stein: convex combination, `φ ∈ [0,1]` | exact | `test_js_is_convex_combination` |
| EWMA recursion `S_T = λS_{T−1} + (1−λ)r_T r_T'` | 1e-18 | `test_ewma_recursion_identity` |
| LW shrunk = `δF + (1−δ)S`, `δ ∈ [0,1]`, variances preserved | 1e-18 | `test_lw_*` |
| Euler identity: `Σᵢ RCᵢ = w'Σw` | 1e-18 | `test_rc_euler_identity_exact` |
| ERC: all risk contributions equal | < 1e-8 relative spread (observed ~1e-15) | `test_erc_contributions_all_equal` |
| ERC = inverse-vol under constant correlation | 1e-9 | `test_erc_constant_correlation_equals_inverse_vol` |
| Vol targeting hits ex-ante target | 1e-14 | `test_vol_target_hits_target_exactly` |
| Two-fund theorem: frontier portfolio = affine combo of two others | 1e-10 | `test_two_fund_theorem` |
| SLSQP min-variance vs closed form | 1e-6 (observed ~6e-8) | `test_slsqp_min_variance_matches_closed_form` |
| Target-return portfolio is a variance minimum (feasible perturbations) | perturbation test, 30 directions | `test_target_return_variance_is_minimal_under_perturbation` |
| Effective N of equal weight = N; diversification ratio ≥ 1 | exact / 1e-12 | `test_effective_n_*`, `test_diversification_ratio_*` |
| Backtest cost ledger on hand-computed 2-rebalance scenario | 1e-15 | `test_hand_computed_two_rebalance_cost_ledger` |
| No lookahead: estimation window ends strictly before rebalance; spike on rebalance day uncapturable | exact | `test_no_lookahead_*` |

## 2. The estimation-error study (pipeline §2)

20 disjoint 252-day windows from a calm 21-year panel with known true
moments; each window's estimated portfolio is evaluated under the TRUE
moments (no evaluation noise). Achievable (true tangency) Sharpe 0.452;
equal weight 0.326.

| Strategy | mean true Sharpe | min | max |
|---|---|---|---|
| Tangency, raw mean, unconstrained | **0.141** | −0.125 | 0.325 |
| Tangency, raw mean, long-only | 0.300 | 0.159 | 0.416 |
| Tangency, James-Stein mean, long-only | 0.313 | 0.191 | 0.406 |
| Min-variance, Ledoit-Wolf, long-only | **0.339** | 0.317 | 0.371 |
| ERC, Ledoit-Wolf | 0.333 | 0.328 | 0.339 |

Raw-mean tangency fell below equal weight in 11/20 windows; the
unconstrained version averaged **16.2x gross leverage** (max 58.7x) and
the closed form failed outright (`1'Σ⁻¹μ ≤ 0`) in 8/20 windows. Note the
min/max columns: mean-free allocators (min-var, ERC) are not just better
on average, they are *dramatically more stable* (ERC spans 0.328–0.339
across windows; raw tangency spans −0.125–0.325). Average shrinkage
intensities: James-Stein φ = 0.96 (the data beg to ignore sample means),
Ledoit-Wolf δ = 0.15.

## 3. Ledoit-Wolf conditioning numbers

- 252×8 window: cond(sample) 20.8 → cond(shrunk) 16.9, δ = 0.15.
- 12×10 panel (N close to T): conditioning improvement tested
  (`test_lw_improves_conditioning_when_n_close_to_t`).
- 5×8 panel (**T < N, sample singular, cond = ∞**): LW returns a finite-
  condition PSD matrix and long-only min-variance solves on it
  (`test_short_window_singular_sample_but_lw_invertible`).
- Degenerate case: single asset ⇒ target = sample ⇒ δ = 0, shrinkage is
  a no-op (`test_lw_degenerate_single_asset_returns_sample`).

## 4. Out-of-sample walk-forward race (pipeline §6)

2400-day 8-asset panel with a 120-day crisis (market vol ×3, −120%/yr
market drift during crisis), 252-day window, monthly rebalance, 10bp
costs. Main panel (seed 1), net of costs:

| Strategy | AnnRet | AnnVol | Sharpe | MaxDD | AnnTurnover | AvgEffN |
|---|---|---|---|---|---|---|
| EqualWeight | −0.041 | 0.201 | −0.110 | 0.490 | 0.57 | 8.00 |
| MinVar (LW) | −0.031 | 0.181 | −0.083 | 0.384 | 1.47 | 3.68 |
| Tangency raw | **−0.114** | 0.251 | **−0.354** | **0.658** | **8.27** | **1.57** |
| Tangency JS | −0.019 | 0.245 | 0.044 | 0.494 | 8.61 | 2.26 |
| ERC (LW) | −0.040 | 0.196 | −0.109 | 0.468 | 0.61 | 7.82 |
| Static 60/40 | −0.046 | 0.204 | −0.131 | 0.508 | 0.57 | 7.69 |

Because a single panel is one noisy draw, the race is repeated on six
independent panels (seeds 1–6); mean net Sharpe:

| EqualWeight | MinVar (LW) | Tangency raw | Tangency JS | ERC (LW) | Static 60/40 |
|---|---|---|---|---|---|
| −0.049 | **+0.028** | **−0.064** | −0.002 | −0.038 | −0.060 |

The classic result, reproduced: **raw-mean tangency is the worst
strategy in the race — below naive equal weight — while the mean-free
allocators (min-variance, ERC) and the shrunk-mean tangency do fine.**
Raw tangency also pays ~15x the transaction costs of EW (8.3 vs 0.57
annual turnover). Absolute Sharpes are low because each 8.5-year span
contains a crisis and one realized market-mean draw (Lo-adjusted SE of
every Sharpe is ≈ 0.4–0.5 — none of the absolute levels is
statistically distinguishable from zero, which is itself Merton's point).

## 5. Failure modes

1. **Correlation breakdown in crises** (pipeline §7). Realized average
   pairwise correlation 0.470 (calm) → **0.867** (crisis). Subperiod
   numbers, seed-1 panel:

   | Strategy | Crisis AnnVol | Crisis MaxDD | Crisis TotRet |
   |---|---|---|---|
   | EqualWeight | 0.467 | 0.399 | −0.311 |
   | MinVar (LW) | **0.380** | **0.308** | **−0.202** |
   | Tangency raw | **0.568** | **0.470** | **−0.368** |
   | Tangency JS | 0.555 | 0.407 | −0.327 |
   | ERC (LW) | 0.452 | 0.386 | −0.298 |
   | Static 60/40 | 0.472 | 0.407 | −0.319 |

   With correlations → 1, diversification dies for everyone (all books'
   vol is ~2.5–3x calm levels), but broad books (ERC eff. N 7.8; EW 8)
   degrade gracefully while the concentrated raw-mean tangency book
   (eff. N 1.57) carries idiosyncratic risk it can no longer diversify:
   worst vol, worst drawdown, worst crisis return. Risk parity is hurt
   *less* than concentrated MVO — but note it is NOT hurt less than
   min-variance, which holds the low-beta corner by construction.
2. **Mean estimation error dominating** — the central failure mode; §2
   quantifies it (0.141 vs 0.452 true Sharpe). It does not shrink with
   more assets and shrinks only with calendar time (σ/√Y).
3. **Turnover/cost sensitivity.** Cost drag scales linearly in turnover
   (tested exactly): at 10bp, raw tangency loses 71bp/span vs 5bp for
   EW; at 50bp institutional-stress costs the gap is 5x wider. Mean-
   driven strategies are cost-fragile because their signals churn.
4. **Concentration under constraints.** Long-only caps leverage but
   concentrates: frontier max weight reaches 0.74 at the top of the
   long-only frontier (§3 table) and long-only frontier vol ≥
   unconstrained vol at every target return (dominance tested).
5. **Numerical limits.** Perfectly correlated or T ≤ N sample matrices
   are singular: closed forms raise informative `ValueError`s directing
   to `psd_repair`/`ledoit_wolf_cc` (tested); after eigenvalue-floor
   repair all solvers run. Zero-vol assets: ERC/inverse-vol raise (risk
   contributions undefined — tested); repaired min-variance correctly
   allocates ~everything to the (near-)riskless asset.
6. **Smoothed returns overstate Sharpe.** Lo-adjusted annualisation is
   reported next to iid Sharpe; an AR(1)-smoothed test series shows
   `sharpe_lo < sharpe` (`test_lo_sharpe_penalizes_positive_autocorrelation`).

## 6. Test suite

`pytest -q`: **141 tests, all passing, ~3 s, fully offline, seeded.**
Coverage: analytic identities (above), SLSQP-vs-closed-form
cross-checks, property tests (frontier monotonicity, DR ≥ 1, RC Euler),
no-lookahead and exact cost accounting, and the edge-case contract:
single asset, two-asset closed forms, perfectly correlated (singular)
covariances, zero-vol assets, all-negative means under long-only,
T < N windows, NaN inputs, dimension mismatches.
