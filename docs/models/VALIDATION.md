# Validation pack

One section per model in `models.toml`, in the order of
`VALIDATION_TEMPLATE.md`. Facts that live in the inventory (assumptions,
limitations, tests, KPIs, dates) are repeated here so the page reads on
its own; the inventory is the source of truth and
`tests/test_model_inventory.py` checks that every inventory id has a
section here.

Status of this pack (2026-11-03): first edition. Every entry was reviewed
against its code and tests on that date by the maintainer. Owner is
`maintainer` throughout: this repository has one, so "independent" in
this pack means agreement with code or data the model does not share,
not a separate validation team (see `README.md`, "What this is not").

| Id | Tier | Kind | Status |
|---|---|---|---|
| `CORE-BS-CARRY` | 1 | price | independently-checked |
| `CORE-IV-BLACK` | 1 | sensitivity | independently-checked |
| `EQ-OPT-CRR` | 2 | price | self-tested |
| `EQ-OPT-MC` | 2 | price | self-tested |
| `EQ-OPT-CN` | 2 | price | self-tested |
| `EQ-OPT-ALO` | 1 | price | independently-checked |
| `EQ-OPT-SABR` | 2 | price | self-tested |
| `EQ-OPT-BACHELIER` | 2 | price | independently-checked |
| `FX-OPT-GK` | 1 | price | independently-checked |
| `EQ-VAR-PARAM` | 1 | risk | independently-checked |
| `EQ-VAR-HIST` | 1 | risk | independently-checked |
| `EQ-VAR-MC` | 1 | risk | independently-checked |
| `EQ-VAR-ES` | 1 | risk | independently-checked |
| `EQ-VAR-EVT` | 2 | risk | self-tested |
| `EQ-VAR-SVAR` | 2 | risk | self-tested |
| `EQ-VAR-FRTB-IMA` | 2 | risk | self-tested |
| `FX-VAR-PARAM` | 1 | risk | independently-checked |
| `FX-VAR-HIST` | 1 | risk | independently-checked |
| `FX-VAR-MC` | 2 | risk | self-tested |
| `EQ-FI-BOOTSTRAP` | 1 | price | independently-checked |
| `EQ-FI-HW` | 1 | price | independently-checked |
| `EQ-CR-ISDA-CDS` | 1 | price | independently-checked |
| `EQ-CR-PD-LOGIT` | 2 | forecast | independently-checked |
| `EQ-SURF-HESTON` | 2 | price | independently-checked |
| `EQ-SURF-SSVI` | 2 | price | self-tested |
| `EQ-SURF-DUPIRE` | 2 | price | self-tested |
| `EQ-VOL-GARCH` | 2 | forecast | independently-checked |
| `EQ-VOL-EWMA` | 3 | forecast | self-tested |
| `EQ-VOL-HAR` | 3 | forecast | self-tested |
| `FX-CR-SACCR` | 2 | risk | self-tested |
| `FX-CR-CVA` | 2 | risk | self-tested |
| `FX-FI-BOOTSTRAP` | 2 | price | self-tested |
| `FX-FI-XCCY` | 2 | price | self-tested |
| `CORE-FRTB-SBM` | 2 | risk | self-tested |
| `CORE-FRTB-CURV` | 2 | risk | self-tested |
| `CORE-XVA-COLLATERAL` | 2 | risk | self-tested |

### `CORE-BS-CARRY`: Black-Scholes-Merton with continuous carry (shared kernel)

Tier 1, price, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** European price and Greeks for every Python options package; the C++ and Rust kernels port it.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/black_scholes.py::carry_price`, `python/core/src/qp_core/black_scholes.py::carry_greeks`. Assumptions: lognormal spot, constant volatility, rate and carry; frictionless continuous hedging. Reference: Black and Scholes (1973); Merton (1973) for continuous carry; Hull, *Options, Futures and Other Derivatives*, ch. 15 and 17.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_black_scholes.py::test_worked_example`; `python/core/tests/test_black_scholes.py::test_put_call_parity`. Independent checks: `python/equity/01-options-pricing/tests/test_golden.py::test_record_matches_oracle`; `python/equity/01-options-pricing/tests/test_differential.py::test_rust_matches_python`; `python/equity/01-options-pricing/tests/test_differential.py::test_cpp_matches_python`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: put-call parity residual <= 1e-12 x spot; golden-vector worst deviation inside tolerance. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** no smile, no jumps, no discrete dividends; deep-OTM tails below about 1e-300 underflow to zero. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `CORE-IV-BLACK`: Black implied volatility (shared solver)

Tier 1, sensitivity, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** Inverts a Black price to a volatility; used by every smile, surface and IV reporting path.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/iv_core.py::black_implied_vol`. Assumptions: the quote lies strictly inside the no-arbitrage bounds. Reference: Jaeckel, "Let's be rational" (2015) for the normalised-price formulation; Acklam's inverse-normal approximation for the start value.

**Outcomes analysis.** Own-property tests: `python/equity/01-options-pricing/tests/test_implied_vol.py::test_round_trip_sigma_price_sigma`; `python/equity/01-options-pricing/tests/test_iv_accuracy.py::test_k90_otm_put_recovers_to_1e12`. Independent checks: `python/equity/01-options-pricing/tests/test_iv_accuracy.py::test_iv_round_trip_against_mpmath`; `cpp/core/fuzz/fuzz_implied_vol.cpp`; `rust/core/fuzz/src/lib.rs::check_implied_vol`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: round-trip vol error vs mpmath reference; iterations per solve. Review trigger: a round-trip error above the cited tolerance, or a rise in the mean iteration count.

**Limitations and compensating controls.** quotes within rounding of a bound are refused, not solved; where vega is tiny only the price, not the vol, is recoverable. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** W12 fuzzing (2026-11-03): the first libFuzzer run of `cpp/core/fuzz/fuzz_implied_vol.cpp` failed a check that demanded the recovered vol reprice to 1e-9. Diagnosis: deep out of the money (x = -0.41, s = 0.0127) the price elasticity d ln(beta)/d ln(s) is about x^2/s^2 > 1,000, so a vol correct to 5e-13 moves the price by 1.7e-9. The solver was right and the check was wrong; the property now accepts vol to 1e-10 or price to 1e-9, and the input is kept as `cpp/core/fuzz/regressions/implied_vol/elasticity-deep-otm`. No open finding.

### `EQ-OPT-CRR`: Cox-Ross-Rubinstein binomial tree

Tier 2, price, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** American and European equity option prices by lattice.

**Conceptual soundness.** Entrypoints: `python/equity/01-options-pricing/src/eq_options/binomial.py::crr_price`. Assumptions: recombining lattice; discrete-time limit of Black-Scholes. Reference: Cox, Ross and Rubinstein (1979).

**Outcomes analysis.** Own-property tests: `python/equity/01-options-pricing/tests/test_binomial.py::test_european_tree_converges_to_bs_500_steps`; `python/equity/01-options-pricing/tests/test_binomial.py::test_european_tree_tight_at_5000_steps`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: error vs Black-Scholes at 500 and 5000 steps. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** first-order convergence with odd-even oscillation; cost grows with steps squared. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-OPT-MC`: Monte Carlo European pricer and pathwise / likelihood-ratio Greeks

Tier 2, price, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Simulation price and Greeks; the reference for path-dependent extensions.

**Conceptual soundness.** Entrypoints: `python/equity/01-options-pricing/src/eq_options/monte_carlo.py::mc_price`. Assumptions: geometric Brownian motion under the risk-neutral measure. Reference: Glasserman, *Monte Carlo Methods in Financial Engineering* (2003), ch. 7 for pathwise and likelihood-ratio Greeks.

**Outcomes analysis.** Own-property tests: `python/equity/01-options-pricing/tests/test_edge_cases.py::test_t_zero_returns_intrinsic_tree_and_mc`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: price within 3 standard errors of Black-Scholes. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** standard error falls as 1/sqrt(paths); pathwise Greeks need a payoff differentiable almost everywhere. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-OPT-CN`: Crank-Nicolson finite-difference pricer

Tier 2, price, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** European and American prices on a PDE grid.

**Conceptual soundness.** Entrypoints: `python/equity/01-options-pricing/src/eq_options/pde.py::cn_price`. Assumptions: Black-Scholes PDE on a truncated log-spot grid. Reference: Crank and Nicolson (1947); Rannacher (1984) start-up steps; Duffy, *Finite Difference Methods in Financial Engineering* (2006).

**Outcomes analysis.** Own-property tests: `python/equity/01-options-pricing/tests/test_phase5_models.py::test_cn_european_matches_bs`; `python/equity/01-options-pricing/tests/test_phase5_models.py::test_cn_american_second_order_and_matches_tree`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: error vs Black-Scholes; error vs extrapolated tree for Americans. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** Rannacher start needed for kinked payoffs; grid truncation error at the boundaries. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-OPT-ALO`: Andersen-Lake-Offengenden American option pricer

Tier 1, price, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** Fast high-accuracy American prices.

**Conceptual soundness.** Entrypoints: `python/equity/01-options-pricing/src/eq_options/alo.py::alo_price`. Assumptions: Black-Scholes dynamics; exercise boundary found by fixed-point iteration. Reference: Andersen, Lake and Offengenden (2016), "High-performance American option pricing".

**Outcomes analysis.** Own-property tests: `python/equity/01-options-pricing/tests/test_phase5_models.py::test_alo_no_early_exercise_regions_are_european`. Independent checks: `python/equity/01-options-pricing/tests/test_phase5_models.py::test_alo_matches_extrapolated_tree`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: difference vs extrapolated binomial tree. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** accuracy depends on collocation and quadrature settings. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-OPT-SABR`: SABR implied volatility (Hagan 2002 expansion)

Tier 2, price, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Smile interpolation for rates-style and equity quotes.

**Conceptual soundness.** Entrypoints: `python/equity/01-options-pricing/src/eq_options/sabr.py::sabr_vol`. Assumptions: SABR stochastic-volatility dynamics, short-expiry asymptotic expansion. Reference: Hagan, Kumar, Lesniewski and Woodward (2002), "Managing smile risk".

**Outcomes analysis.** Own-property tests: `python/equity/01-options-pricing/tests/test_phase5_models.py::test_sabr_atm_matches_hagan_atm_formula`; `python/equity/01-options-pricing/tests/test_phase5_models.py::test_sabr_beta1_nu0_is_flat_alpha`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: ATM match to the Hagan ATM formula. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** negative densities at low strikes for long expiries; expansion error grows with nu^2 T. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-OPT-BACHELIER`: Bachelier (normal) option model

Tier 2, price, status `independently-checked`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Prices options on quantities that can go negative.

**Conceptual soundness.** Entrypoints: `python/equity/01-options-pricing/src/eq_options/bachelier.py::bachelier_price`. Assumptions: arithmetic Brownian motion of the forward. Reference: Bachelier (1900); Schachermayer and Teichmann (2008) for the comparison with Black-Scholes.

**Outcomes analysis.** Own-property tests: `python/equity/01-options-pricing/tests/test_phase5_models.py::test_bachelier_worked_example_negative_forward_and_parity`. Independent checks: `python/equity/01-options-pricing/tests/test_phase5_models.py::test_bachelier_matches_mpmath_integral`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: difference vs mpmath integral. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** unbounded below; unsuitable for strictly positive assets at long expiry. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `FX-OPT-GK`: Garman-Kohlhagen FX option model

Tier 1, price, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** FX vanilla prices and Greeks in the FX options package.

**Conceptual soundness.** Entrypoints: `python/fx/01-options-pricing/src/fx_options/garman_kohlhagen.py::gk_price`. Assumptions: lognormal spot, constant domestic and foreign rates and volatility. Reference: Garman and Kohlhagen (1983); Clark, *Foreign Exchange Option Pricing* (2011), ch. 1-3.

**Outcomes analysis.** Own-property tests: `python/fx/01-options-pricing/tests/test_arrays.py::test_matches_scalar_on_boundary_grid`. Independent checks: `python/fx/01-options-pricing/tests/test_differential.py::test_rust_matches_python`; `python/fx/01-options-pricing/tests/test_differential.py::test_cpp_matches_python`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: parity residual; three-language agreement. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** no smile; settlement-date and premium-currency conventions handled outside the kernel. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-VAR-PARAM`: Parametric (delta-normal and Cornish-Fisher) VaR

Tier 1, risk, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** One-day equity book VaR from a covariance matrix.

**Conceptual soundness.** Entrypoints: `python/equity/03-var-es-engine/src/eq_var/parametric_var.py::parametric_var`, `python/equity/03-var-es-engine/src/eq_var/parametric_var.py::cornish_fisher_var`. Assumptions: jointly normal returns (Cornish-Fisher: mild skew and kurtosis); linear book over the horizon. Reference: J.P. Morgan/Reuters, *RiskMetrics Technical Document* (1996); Zangari (1996) for Cornish-Fisher VaR.

**Outcomes analysis.** Own-property tests: `python/equity/03-var-es-engine/tests/test_edge_cases.py::TestSingleAsset.test_parametric_equals_mc_for_single_normal_asset`. Independent checks: `python/equity/03-var-es-engine/tests/golden/test_oracle_equity.py::test_parametric_closed_forms`; `python/equity/03-var-es-engine/tests/golden/test_oracle_equity.py::test_cornish_fisher`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: Kupiec POF p-value on backtest; closed-form agreement with the oracle. Review trigger: a Basel traffic-light zone of amber or red on the backtest, or a Kupiec or Christoffersen p-value below 0.05.

**Limitations and compensating controls.** understates fat-tailed losses; Cornish-Fisher invalid outside its monotone domain. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-VAR-HIST`: Historical simulation VaR (plain, age-weighted, filtered)

Tier 1, risk, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** Equity VaR from replayed historical returns.

**Conceptual soundness.** Entrypoints: `python/equity/03-var-es-engine/src/eq_var/historical_var.py::historical_var`, `python/equity/03-var-es-engine/src/eq_var/historical_var.py::filtered_historical_var`. Assumptions: the look-back window is representative of the next day. Reference: Boudoukh, Richardson and Whitelaw (1998) for age weighting; Barone-Adesi, Giannopoulos and Vosper (1999) for filtered historical simulation.

**Outcomes analysis.** Own-property tests: `python/equity/03-var-es-engine/tests/test_edge_cases.py::TestZeroVolAsset.test_zero_vol_pnl_series_var_is_zero`. Independent checks: `python/equity/03-var-es-engine/tests/golden/test_oracle_equity.py::test_historical_order_statistics`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: exceptions vs Basel traffic light. Review trigger: a Basel traffic-light zone of amber or red on the backtest, or a Kupiec or Christoffersen p-value below 0.05.

**Limitations and compensating controls.** slow to react to regime change (plain variant); quantile noise for short windows. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-VAR-MC`: Monte Carlo VaR with full revaluation

Tier 1, risk, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** Equity VaR for option books via simulated factor returns.

**Conceptual soundness.** Entrypoints: `python/equity/03-var-es-engine/src/eq_var/monte_carlo_var.py::monte_carlo_var`. Assumptions: multivariate normal factor returns from a repaired covariance. Reference: Jorion, *Value at Risk* (3rd ed.), ch. 12; Higham (2002) for the nearest-correlation repair.

**Outcomes analysis.** Own-property tests: `python/equity/03-var-es-engine/tests/test_edge_cases.py::TestOptionsOnlyPortfolio.test_var_computes_and_short_option_var_positive`. Independent checks: `python/equity/03-var-es-engine/tests/test_edge_cases.py::TestSingleAsset.test_parametric_equals_mc_for_single_normal_asset`; `rust/equity-var-engine/fuzz/src/lib.rs::check_cholesky_repair`; `cpp/core/fuzz/fuzz_cholesky_repair.cpp`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: VaR standard error; parity with the parametric VaR for a linear normal book. Review trigger: a Basel traffic-light zone of amber or red on the backtest, or a Kupiec or Christoffersen p-value below 0.05.

**Limitations and compensating controls.** simulation noise in the quantile; standard error reported; tails only as heavy as the factor model. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** W12 fuzzing (2026-11-03): the Rust `psd::cholesky_repair` (Clip and Higham) returns a repaired matrix that is symmetric only to rounding (within 1e-12 of the largest variance, the bound the fuzz property enforces), while the C++ `qp::linalg::repair_covariance` is exactly symmetric. Harmless for the factor (only the lower triangle is read), recorded as a parity note. Owner: maintainer; due with the next VaR-engine change.

### `EQ-VAR-ES`: Expected shortfall (historical and parametric)

Tier 1, risk, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** Tail-average loss for the FRTB-style reports.

**Conceptual soundness.** Entrypoints: `python/equity/03-var-es-engine/src/eq_var/expected_shortfall.py::expected_shortfall`, `python/equity/03-var-es-engine/src/eq_var/expected_shortfall.py::parametric_es`. Assumptions: same as the VaR method it is computed with. Reference: Acerbi and Tasche (2002); Acerbi and Szekely (2014) for the Z2 backtest.

**Outcomes analysis.** Own-property tests: `python/equity/03-var-es-engine/tests/test_edge_cases.py::TestZeroVolAsset.test_zero_vol_pnl_series_var_is_zero`. Independent checks: `python/equity/03-var-es-engine/tests/golden/test_oracle_equity.py::test_parametric_closed_forms`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: Acerbi-Szekely Z2 backtest. Review trigger: a Basel traffic-light zone of amber or red on the backtest, or a Kupiec or Christoffersen p-value below 0.05.

**Limitations and compensating controls.** few observations beyond the quantile; high estimation error. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-VAR-EVT`: Peaks-over-threshold EVT tail VaR and ES

Tier 2, risk, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Extreme quantiles beyond the data with a generalised Pareto tail.

**Conceptual soundness.** Entrypoints: `python/equity/03-var-es-engine/src/eq_var/evt.py::pot_var`, `python/equity/03-var-es-engine/src/eq_var/evt.py::pot_es`. Assumptions: exceedances over the threshold are GPD and independent. Reference: McNeil and Frey (2000); McNeil, Frey and Embrechts, *Quantitative Risk Management* (2015), ch. 5.

**Outcomes analysis.** Own-property tests: `python/equity/03-var-es-engine/tests/test_evt.py::test_pot_matches_exact_gpd_tail`; `python/equity/03-var-es-engine/tests/test_evt.py::test_fit_recovers_parameters`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: parameter recovery on simulated GPD data. Review trigger: a Basel traffic-light zone of amber or red on the backtest, or a Kupiec or Christoffersen p-value below 0.05.

**Limitations and compensating controls.** threshold choice drives the answer; ES undefined for xi >= 1. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-VAR-SVAR`: Stressed VaR and Basel 2.5 capital

Tier 2, risk, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** VaR on the worst 12-month window, combined into Basel 2.5 capital.

**Conceptual soundness.** Entrypoints: `python/equity/03-var-es-engine/src/eq_var/stressed_var.py::stressed_var`. Assumptions: the chosen stress window is relevant to the current book. Reference: BCBS, *Revisions to the Basel II market risk framework* (2009).

**Outcomes analysis.** Own-property tests: `python/equity/03-var-es-engine/tests/test_stressed_var.py::test_planted_stress_window`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: planted stress window recovered. Review trigger: a Basel traffic-light zone of amber or red on the backtest, or a Kupiec or Christoffersen p-value below 0.05.

**Limitations and compensating controls.** window search is in-sample. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-VAR-FRTB-IMA`: FRTB internal-models capital (liquidity-adjusted ES)

Tier 2, risk, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Teaching implementation of the FRTB IMA capital stack.

**Conceptual soundness.** Entrypoints: `python/equity/03-var-es-engine/src/eq_var/frtb_ima.py::ima_capital`. Assumptions: liquidity horizons and risk-factor mapping as configured. Reference: BCBS, *Minimum capital requirements for market risk* (d457, 2019), MAR33.

**Outcomes analysis.** Own-property tests: `python/equity/03-var-es-engine/tests/test_frtb_ima.py::test_capital_uses_average_when_larger`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: capital formula cases. Review trigger: a Basel traffic-light zone of amber or red on the backtest, or a Kupiec or Christoffersen p-value below 0.05.

**Limitations and compensating controls.** not a regulatory implementation; no NMRF charge or DRC. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `FX-VAR-PARAM`: FX parametric VaR

Tier 1, risk, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** FX book VaR from a covariance of FX returns.

**Conceptual soundness.** Entrypoints: `python/fx/03-var-es-engine/src/fx_var/parametric_var.py::parametric_var`. Assumptions: jointly normal FX returns; linear book. Reference: RiskMetrics Technical Document (1996).

**Outcomes analysis.** Own-property tests: `python/fx/03-var-es-engine/tests/test_backtesting.py::test_garch_data_parametric_normal_fails_fhs_passes`. Independent checks: `python/fx/03-var-es-engine/tests/golden/test_oracle_fx.py::test_parametric_closed_forms`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: closed-form agreement with the FX oracle. Review trigger: a Basel traffic-light zone of amber or red on the backtest, or a Kupiec or Christoffersen p-value below 0.05.

**Limitations and compensating controls.** understates fat-tailed losses. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `FX-VAR-HIST`: FX historical simulation VaR

Tier 1, risk, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** FX VaR from replayed historical returns.

**Conceptual soundness.** Entrypoints: `python/fx/03-var-es-engine/src/fx_var/historical_var.py::historical_var`. Assumptions: look-back window representative. Reference: Jorion, *Value at Risk* (3rd ed.), ch. 10.

**Outcomes analysis.** Own-property tests: `python/fx/03-var-es-engine/tests/test_backtesting.py::test_rolling_backtest_shapes_and_window`. Independent checks: `python/fx/03-var-es-engine/tests/golden/test_oracle_fx.py::test_parametric_closed_forms`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: exceptions vs Basel traffic light. Review trigger: a Basel traffic-light zone of amber or red on the backtest, or a Kupiec or Christoffersen p-value below 0.05.

**Limitations and compensating controls.** slow to react to regime change. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `FX-VAR-MC`: FX Monte Carlo VaR

Tier 2, risk, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** FX VaR with optional jumps via simulated factor returns.

**Conceptual soundness.** Entrypoints: `python/fx/03-var-es-engine/src/fx_var/monte_carlo_var.py::monte_carlo_var`. Assumptions: normal factor returns plus optional compound-Poisson jumps. Reference: Jorion, *Value at Risk* (3rd ed.), ch. 12; Merton (1976) for jumps.

**Outcomes analysis.** Own-property tests: `python/fx/03-var-es-engine/tests/test_conventions.py::test_higham_and_clip_repair`; `python/fx/03-var-es-engine/tests/test_conventions.py::test_mc_never_raises_on_a_negative_vol_path`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: VaR standard error. Review trigger: a Basel traffic-light zone of amber or red on the backtest, or a Kupiec or Christoffersen p-value below 0.05.

**Limitations and compensating controls.** simulation noise in the quantile. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-FI-BOOTSTRAP`: Single-curve bootstrap (deposits, FRAs, swaps)

Tier 1, price, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** Discount curve that reprices its input instruments.

**Conceptual soundness.** Entrypoints: `python/equity/04-fixed-income/src/fi_rates/bootstrap.py::bootstrap_curve`. Assumptions: instruments quoted consistently on one curve. Reference: Hagan and West (2006), "Interpolation methods for curve construction".

**Outcomes analysis.** Own-property tests: `python/equity/04-fixed-income/tests/test_bootstrap.py::TestRoundTrip.test_deposits_and_swaps_reprice_to_1e10`; `python/equity/04-fixed-income/tests/test_bootstrap.py::TestRoundTrip.test_fra_repricing`. Independent checks: `python/equity/04-fixed-income/tests/test_hull_white_quantlib.py::test_zero_bond_option_matches_quantlib`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: max repricing error of input instruments. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** no multi-curve basis in this entrypoint; interpolation choice affects forwards between pillars. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-FI-HW`: Hull-White one-factor model and swaption calibration

Tier 1, price, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** Rate options and swaptions under a calibrated short-rate model.

**Conceptual soundness.** Entrypoints: `python/equity/04-fixed-income/src/fi_rates/hull_white.py::calibrate_hull_white`. Assumptions: Gaussian short rate with constant mean reversion and volatility. Reference: Hull and White (1990); Jamshidian (1989) for the swaption decomposition; Brigo and Mercurio (2006), ch. 3.

**Outcomes analysis.** Own-property tests: `python/equity/04-fixed-income/tests/test_round3_fix_rates.py::test_calibration_recovers_known_parameters`. Independent checks: `python/equity/04-fixed-income/tests/test_hull_white_quantlib.py::test_jamshidian_swaption_matches_quantlib`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: calibration RMSE; agreement with QuantLib. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** negative rates possible; one factor: perfectly correlated curve moves. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-CR-ISDA-CDS`: ISDA standard CDS model and hazard bootstrap

Tier 1, price, status `independently-checked`, validated 2026-11-03, next review 2027-05-03.

**Purpose.** CDS pricing and par-spread to hazard-curve bootstrap.

**Conceptual soundness.** Entrypoints: `python/equity/06-credit-risk/src/eq_credit/isda_cds.py::bootstrap_isda`. Assumptions: piecewise-constant hazard; ISDA standard model conventions. Reference: ISDA CDS Standard Model (Markit, 2009); O'Kane, *Modelling Single-name and Multi-name Credit Derivatives* (2008), ch. 6.

**Outcomes analysis.** Own-property tests: `python/equity/06-credit-risk/tests/test_isda_cds.py::test_bootstrap_reprices_every_par_spread`. Independent checks: `python/equity/06-credit-risk/tests/test_isda_cds_quantlib.py::test_isda_cds_matches_quantlib`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: par-spread repricing error; agreement with QuantLib. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** recovery is an input, not estimated. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-CR-PD-LOGIT`: Logistic-regression PD scorecard

Tier 2, forecast, status `independently-checked`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Probability of default from obligor features (teaching scorecard).

**Conceptual soundness.** Entrypoints: `python/equity/06-credit-risk/src/eq_credit/model.py::fit_logistic`. Assumptions: log-odds linear in WoE-binned features; independent obligors. Reference: Siddiqi, *Credit Risk Scorecards* (2006); Thomas, Edelman and Crook (2002).

**Outcomes analysis.** Own-property tests: `python/equity/06-credit-risk/tests/test_edge_cases.py::test_zero_defaults_raises_everywhere`. Independent checks: `python/equity/06-credit-risk/tests/test_model.py::test_matches_sklearn_to_1e6`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: AUC, Gini, calibration by decile. Review trigger: a QLIKE loss worse than the EWMA benchmark over a rolling year, or a failed parameter-recovery test.

**Limitations and compensating controls.** quasi-separation inflates coefficients (warned); trained on synthetic or public data only. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-SURF-HESTON`: Heston stochastic-volatility pricer

Tier 2, price, status `independently-checked`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Smile-consistent prices from the Heston characteristic function.

**Conceptual soundness.** Entrypoints: `python/equity/09-vol-surface/src/eq_surface/heston.py::heston_call`. Assumptions: Heston variance dynamics; Feller condition warned, not required. Reference: Heston (1993); Albrecher et al. (2007), "The little Heston trap", for the stable characteristic function.

**Outcomes analysis.** Own-property tests: `python/equity/09-vol-surface/tests/test_heston.py::test_put_call_parity`. Independent checks: `python/equity/09-vol-surface/tests/test_edge_cases.py::test_xi_zero_consistency_fourier_vs_mc`; `python/fx/09-vol-surface/tests/test_heston_quantlib.py::test_heston_matches_quantlib`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: parity residual; Fourier vs Monte Carlo. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** numerical integration error at very short expiries. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-SURF-SSVI`: SSVI / eSSVI surface fit

Tier 2, price, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Arbitrage-aware implied-volatility surface from quotes.

**Conceptual soundness.** Entrypoints: `python/equity/09-vol-surface/src/eq_surface/ssvi.py::fit_ssvi`. Assumptions: SSVI parameterisation of total variance. Reference: Gatheral and Jacquier (2014), "Arbitrage-free SVI volatility surfaces"; Hendriks and Martini (2019) for eSSVI.

**Outcomes analysis.** Own-property tests: `python/equity/09-vol-surface/tests/test_round3_volsurface.py::test_ssvi_fit_recovers_params`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: fit RMSE; calendar arbitrage check. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** fit quality limited by the parametric form at the wings. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-SURF-DUPIRE`: Dupire local volatility

Tier 2, price, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Local volatility surface from total variance or prices.

**Conceptual soundness.** Entrypoints: `python/equity/09-vol-surface/src/eq_surface/dupire.py::local_vol_from_total_variance`. Assumptions: an arbitrage-free, twice-differentiable input surface. Reference: Dupire (1994); Gatheral, *The Volatility Surface* (2006), ch. 1.

**Outcomes analysis.** Own-property tests: `python/equity/09-vol-surface/tests/test_round3_volsurface.py::test_dupire_flat_vol_returns_sigma`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: flat-vol recovery. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** amplifies noise in the input surface. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-VOL-GARCH`: GARCH(1,1) volatility forecast

Tier 2, forecast, status `independently-checked`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Conditional variance forecasts for VaR filtering and research.

**Conceptual soundness.** Entrypoints: `python/equity/02-volatility-modeling/src/eq_vol/garch.py::fit_garch`. Assumptions: GARCH(1,1) with normal or Student-t innovations. Reference: Bollerslev (1986); Bollerslev and Wooldridge (1992) for robust standard errors.

**Outcomes analysis.** Own-property tests: `python/equity/02-volatility-modeling/tests/test_edge_cases.py::TestDegenerateSeries.test_constant_series_garch_raises`. Independent checks: `python/equity/02-volatility-modeling/tests/test_arch_crosscheck.py::TestGARCHCrossCheck.test_parameters_agree`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: QLIKE and MSE vs realised variance. Review trigger: a QLIKE loss worse than the EWMA benchmark over a rolling year, or a failed parameter-recovery test.

**Limitations and compensating controls.** symmetric response to shocks (see GJR, EGARCH). Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-VOL-EWMA`: EWMA (RiskMetrics) volatility forecast

Tier 3, forecast, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Simple exponentially weighted variance forecast.

**Conceptual soundness.** Entrypoints: `python/equity/02-volatility-modeling/src/eq_vol/ewma.py::ewma_forecast`. Assumptions: variance is an exponentially weighted average of squared returns. Reference: RiskMetrics Technical Document (1996), lambda = 0.94 for daily data.

**Outcomes analysis.** Own-property tests: `python/equity/02-volatility-modeling/tests/test_ewma.py::TestForecast.test_flat_term_structure`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: QLIKE vs realised variance. Review trigger: a QLIKE loss worse than the EWMA benchmark over a rolling year, or a failed parameter-recovery test.

**Limitations and compensating controls.** flat term structure: no mean reversion. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `EQ-VOL-HAR`: HAR-RV realised-variance forecast

Tier 3, forecast, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Daily, weekly and monthly realised-variance regression forecast.

**Conceptual soundness.** Entrypoints: `python/equity/02-volatility-modeling/src/eq_vol/har_rv.py::fit_har`. Assumptions: linear cascade of realised-variance averages. Reference: Corsi (2009), "A simple approximate long-memory model of realized volatility".

**Outcomes analysis.** Own-property tests: `python/equity/02-volatility-modeling/tests/test_har_rv.py::test_noiseless_series_recovers_exact_params`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: parameter recovery on simulated HAR. Review trigger: a QLIKE loss worse than the EWMA benchmark over a rolling year, or a failed parameter-recovery test.

**Limitations and compensating controls.** needs intraday data for realised variance. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `FX-CR-SACCR`: SA-CCR exposure at default for FX trades

Tier 2, risk, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Counterparty EAD under the Basel SA-CCR rules (FX asset class).

**Conceptual soundness.** Entrypoints: `python/fx/06-credit-risk/src/fx_credit/saccr.py::saccr_ead`. Assumptions: BCBS 279 supervisory factors for FX. Reference: BCBS 279, *The standardised approach for measuring counterparty credit risk exposures* (2014).

**Outcomes analysis.** Own-property tests: `python/fx/06-credit-risk/tests/test_xva_saccr.py::test_saccr_unmargined_hand_value`; `python/fx/06-credit-risk/tests/test_xva_saccr.py::test_saccr_multiplier_and_margined_rc`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: hand-computed EAD. Review trigger: a Basel traffic-light zone of amber or red on the backtest, or a Kupiec or Christoffersen p-value below 0.05.

**Limitations and compensating controls.** FX asset class only. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `FX-CR-CVA`: Bilateral CVA / DVA

Tier 2, risk, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Credit valuation adjustment from simulated exposures and flat hazards.

**Conceptual soundness.** Entrypoints: `python/fx/06-credit-risk/src/fx_credit/xva.py::bilateral_cva`. Assumptions: independence of exposure and default except in the wrong-way variant. Reference: Gregory, *The xVA Challenge* (4th ed., 2020), ch. 17.

**Outcomes analysis.** Own-property tests: `python/fx/06-credit-risk/tests/test_xva_saccr.py::test_unilateral_cva_constant_exposure_closed_form`; `python/fx/06-credit-risk/tests/test_xva_saccr.py::test_cva_dva_mirror_symmetry`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: closed form for constant exposure. Review trigger: a Basel traffic-light zone of amber or red on the backtest, or a Kupiec or Christoffersen p-value below 0.05.

**Limitations and compensating controls.** flat hazard curves. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `FX-FI-BOOTSTRAP`: FX-implied and basis-adjusted curve bootstrap

Tier 2, price, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Discount curves from deposits, swaps and FX forwards.

**Conceptual soundness.** Entrypoints: `python/fx/04-fixed-income/src/fx_rates/bootstrap.py::bootstrap_curve`. Assumptions: covered interest parity up to the quoted basis. Reference: Hagan and West (2006); Clark (2011), ch. 2 for FX-implied curves.

**Outcomes analysis.** Own-property tests: `python/fx/04-fixed-income/tests/test_bootstrap.py::TestBootstrapRoundTrip.test_reprices_quotes_to_1e10`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: repricing error of input quotes. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** interpolation choice affects forwards between pillars. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `FX-FI-XCCY`: Cross-currency basis swap par basis

Tier 2, price, status `self-tested`, validated 2026-11-03, next review 2027-11-03.

**Purpose.** Par basis and PV for cross-currency swaps.

**Conceptual soundness.** Entrypoints: `python/fx/04-fixed-income/src/fx_rates/xccy.py::solve_par_basis`. Assumptions: two discount curves consistent with FX forwards. Reference: Fujii, Shimada and Takahashi (2010), "A note on construction of multiple swap curves with and without collateral".

**Outcomes analysis.** Own-property tests: `python/fx/04-fixed-income/tests/test_xccy.py::TestParSolvers.test_par_basis_zeroes_pv`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: par basis zeroes PV. Review trigger: a parity, repricing or cross-language difference above the tolerance in the cited test, or a change in any golden vector.

**Limitations and compensating controls.** no mark-to-market resets in this entrypoint. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `CORE-FRTB-SBM`: FRTB SBM delta and vega aggregation (CSR non-sec, equity, commodity)

Tier 2, risk, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Delta and vega capital across buckets and the three MAR21.6 correlation scenarios.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/capital/frtb_classes.py::aggregate`. Assumptions: BCBS d457 (2019) MAR21 parameters, transcribed and checksum-tested. Reference: BCBS d457 (2019) MAR21; Gregory, *The xVA Challenge*, 4th ed. (2020) for collateral.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_capital_frtb_classes.py::test_footnote_examples`; `python/core/tests/test_capital_frtb_classes.py::test_csr_delta_hand_fixture`; `python/core/tests/test_capital_frtb_classes.py::test_equity_delta_hand_fixture`; `python/core/tests/test_capital_frtb_classes.py::test_commodity_delta_hand_fixture`. Independent checks: none cited.

**Ongoing monitoring.** KPIs: published footnote values and hand fixtures. Review trigger: a change to the Basel text or a failed fixture.

**Limitations and compensating controls.** no CSR securitisation, RRAO or DRC; Python only, no second implementation for these classes. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `CORE-FRTB-CURV`: FRTB SBM curvature charge

Tier 2, risk, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Curvature capital for GIRR, CSR non-sec, equity, commodity and FX.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/capital/frtb_curvature.py::curvature_charge`. Assumptions: CVR inputs come from the caller's full revaluation; squared delta correlations (MAR21.100-101). Reference: BCBS d457 (2019) MAR21; Gregory, *The xVA Challenge*, 4th ed. (2020) for collateral.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_capital_frtb_classes.py::test_curvature_within_bucket_psi_and_selection`; `python/core/tests/test_capital_frtb_classes.py::test_curvature_across_buckets_uses_selected_side`; `python/core/tests/test_capital_frtb_classes.py::test_net_cvr_linear_is_zero_and_option_positive`. Independent checks: none cited.

**Ongoing monitoring.** KPIs: hand fixtures for selection, psi and other-sector buckets. Review trigger: a change to the Basel text or a failed fixture.

**Limitations and compensating controls.** Python only, no second implementation. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `CORE-XVA-COLLATERAL`: Collateralised exposure (variation and initial margin)

Tier 2, risk, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Exposure paths under a CSA: thresholds, MTA, margin period of risk and initial margin.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/xva/collateral.py::collateralised_exposure`. Assumptions: no payment during the margin period of risk; the lag sits on the simulation grid. Reference: BCBS d457 (2019) MAR21; Gregory, *The xVA Challenge*, 4th ed. (2020) for collateral.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_xva_collateral.py::test_mpor_closed_form`; `python/core/tests/test_xva_collateral.py::test_mpor_with_initial_margin_closed_form`; `python/core/tests/test_xva_collateral.py::test_perfect_csa_no_lag_is_zero`. Independent checks: none cited.

**Ongoing monitoring.** KPIs: Brownian closed forms within 4 standard errors. Review trigger: a change to the Basel text or a failed fixture.

**Limitations and compensating controls.** Python only; the swap exposure simulator values only on reset dates. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.
