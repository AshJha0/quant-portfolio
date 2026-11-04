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

Second edition (2026-11-04): added the shared stacks in `qp_core`, `qp_exec`
and `qp_bt`. A port that reproduces a Python-generated fixture is not
counted as independent, and neither is a test against another model in this
repository.

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
| `CORE-FRTB-SBM` | 2 | risk | independently-checked |
| `CORE-FRTB-CURV` | 2 | risk | independently-checked |
| `CORE-XVA-COLLATERAL` | 2 | risk | self-tested |
| `CORE-XVA-ADJ` | 2 | risk | self-tested |
| `CORE-XVA-EXPOSURE` | 2 | risk | self-tested |
| `CORE-XVA-HW` | 2 | price | self-tested |
| `CORE-CAP-SACCR` | 1 | risk | independently-checked |
| `CORE-FRTB-SBM-GIRR-FX` | 1 | risk | independently-checked |
| `CORE-EXO-BARRIER` | 2 | price | independently-checked |
| `CORE-EXO-TOUCH` | 2 | price | self-tested |
| `CORE-EXO-BARRIER-MC` | 2 | price | self-tested |
| `CORE-EXO-ASIAN` | 2 | price | self-tested |
| `CORE-EXO-LSM` | 2 | price | independently-checked |
| `CORE-EXO-LSV` | 2 | price | self-tested |
| `CORE-EXO-TARF` | 2 | price | self-tested |
| `CORE-EXO-VV` | 2 | price | self-tested |
| `CORE-AAD-BS` | 2 | sensitivity | self-tested |
| `CORE-AAD-MC` | 2 | sensitivity | self-tested |
| `CORE-AAD-SWAP` | 2 | sensitivity | self-tested |
| `CORE-VOL-SABR` | 2 | price | independently-checked |
| `CORE-VOL-SABR-PDE` | 2 | price | self-tested |
| `CORE-VOL-ESSVI` | 2 | price | self-tested |
| `CORE-VOL-HESTON-CF` | 2 | price | independently-checked |
| `CORE-CR-ISDA` | 1 | price | independently-checked |
| `CORE-CR-INDEX` | 2 | price | self-tested |
| `CORE-RISK-JACOBIAN` | 2 | sensitivity | self-tested |
| `CORE-RISK-VEGA-GRID` | 2 | sensitivity | self-tested |
| `CORE-RISK-SMILE` | 2 | sensitivity | self-tested |
| `CORE-RISK-PNL` | 2 | risk | self-tested |
| `EXEC-IMPACT-FIT` | 2 | forecast | self-tested |
| `EXEC-IMPACT-TRANSIENT` | 2 | forecast | self-tested |
| `EXEC-MM-AS` | 2 | price | self-tested |
| `EXEC-MM-GLFT` | 2 | price | self-tested |
| `EXEC-MM-TOXICITY` | 2 | forecast | self-tested |
| `EXEC-SOR-ALLOC` | 2 | forecast | independently-checked |
| `EXEC-HAWKES` | 2 | forecast | self-tested |
| `BT-FILL-QUEUE` | 2 | forecast | self-tested |
| `BT-FILL-NEXTBAR` | 2 | forecast | self-tested |
| `BT-COST` | 2 | forecast | self-tested |
| `BT-COV-LW` | 2 | risk | independently-checked |
| `BT-COV-PCA` | 2 | risk | self-tested |

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

Tier 2, risk, status `independently-checked`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Delta and vega capital across buckets and the three MAR21.6 correlation scenarios.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/capital/frtb_classes.py::aggregate`. Assumptions: BCBS d457 (2019) MAR21 parameters, transcribed and checksum-tested. Reference: BCBS d457 (2019) MAR21; Gregory, *The xVA Challenge*, 4th ed. (2020) for collateral.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_capital_frtb_classes.py::test_footnote_examples`; `python/core/tests/test_capital_frtb_classes.py::test_csr_delta_hand_fixture`; `python/core/tests/test_capital_frtb_classes.py::test_equity_delta_hand_fixture`; `python/core/tests/test_capital_frtb_classes.py::test_commodity_delta_hand_fixture`. Independent checks: `tests/independent/frtb/test_frtb_classes_independent.py::test_csr_delta`; `tests/independent/frtb/test_frtb_classes_independent.py::test_csr_vega`; `tests/independent/frtb/test_frtb_classes_independent.py::test_eq_delta`; `tests/independent/frtb/test_frtb_classes_independent.py::test_eq_vega`; `tests/independent/frtb/test_frtb_classes_independent.py::test_com_delta`; `tests/independent/frtb/test_frtb_classes_independent.py::test_com_vega`.

**Ongoing monitoring.** KPIs: published footnote values and hand fixtures. Review trigger: a change to the Basel text or a failed fixture.

**Limitations and compensating controls.** no CSR securitisation, RRAO or DRC; the oracle was written in this repository, so a shared misreading of the text would pass. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `CORE-FRTB-CURV`: FRTB SBM curvature charge

Tier 2, risk, status `independently-checked`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Curvature capital for GIRR, CSR non-sec, equity, commodity and FX.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/capital/frtb_curvature.py::curvature_charge`. Assumptions: CVR inputs come from the caller's full revaluation; squared delta correlations (MAR21.100-101). Reference: BCBS d457 (2019) MAR21; Gregory, *The xVA Challenge*, 4th ed. (2020) for collateral.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_capital_frtb_classes.py::test_curvature_within_bucket_psi_and_selection`; `python/core/tests/test_capital_frtb_classes.py::test_curvature_across_buckets_uses_selected_side`; `python/core/tests/test_capital_frtb_classes.py::test_net_cvr_linear_is_zero_and_option_positive`. Independent checks: `tests/independent/frtb/test_frtb_classes_independent.py::test_girr_curvature`; `tests/independent/frtb/test_frtb_classes_independent.py::test_fx_curvature`; `tests/independent/frtb/test_frtb_classes_independent.py::test_csr_curvature`; `tests/independent/frtb/test_frtb_classes_independent.py::test_eq_curvature`; `tests/independent/frtb/test_frtb_classes_independent.py::test_com_curvature`.

**Ongoing monitoring.** KPIs: hand fixtures for selection, psi and other-sector buckets. Review trigger: a change to the Basel text or a failed fixture.

**Limitations and compensating controls.** the oracle was written in this repository, so a shared misreading of the text would pass. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `CORE-XVA-COLLATERAL`: Collateralised exposure (variation and initial margin)

Tier 2, risk, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Exposure paths under a CSA: thresholds, MTA, margin period of risk and initial margin.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/xva/collateral.py::collateralised_exposure`. Assumptions: no payment during the margin period of risk; the lag sits on the simulation grid. Reference: BCBS d457 (2019) MAR21; Gregory, *The xVA Challenge*, 4th ed. (2020) for collateral.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_xva_collateral.py::test_mpor_closed_form`; `python/core/tests/test_xva_collateral.py::test_mpor_with_initial_margin_closed_form`; `python/core/tests/test_xva_collateral.py::test_perfect_csa_no_lag_is_zero`. Independent checks: none cited.

**Ongoing monitoring.** KPIs: Brownian closed forms within 4 standard errors. Review trigger: a change to the Basel text or a failed fixture.

**Limitations and compensating controls.** the swap exposure simulator values only on reset dates. Inputs are validated at the entrypoint and bad input raises rather than returning a number.

**Findings.** None.

### `CORE-XVA-ADJ`: CVA, DVA and FVA from exposure profiles

Tier 2, risk, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Valuation adjustments on an exposure profile for the shared xVA stack.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/xva/adjustments.py::cva`; `python/core/src/qp_core/xva/adjustments.py::dva`; `python/core/src/qp_core/xva/adjustments.py::fva`; `python/core/src/qp_core/xva/adjustments.py::xva`. Assumptions: exposure independent of default (no wrong-way risk); piecewise-flat hazard curves. Reference: Gregory, *The xVA Challenge*, 4th ed. (2020).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_xva.py::test_cva_closed_form_flat`; `python/core/tests/test_xva.py::test_fva_and_total_sign`; `python/core/tests/test_xva.py::test_piecewise_hazard_survival`; `python/core/tests/test_xva_parity_fixture.py::test_fixture_matches_python`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: flat-profile closed form. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** no wrong-way risk; the Rust and C++ ports match a fixture Python generates, which catches port drift, not a wrong formula.

**Findings.** None.

### `CORE-XVA-EXPOSURE`: Exposure profiles, PFE and EEPE

Tier 2, risk, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** EE, PFE and EEPE from simulated values, and swap exposure paths.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/xva/exposure.py::profile`; `python/core/src/qp_core/xva/exposure.py::eepe`; `python/core/src/qp_core/xva/exposure.py::swap_exposure`. Assumptions: paths come from the Hull-White simulator; PFE is an order statistic of the simulated paths. Reference: Gregory, *The xVA Challenge*, 4th ed. (2020); Basel CRE53 for EEPE.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_xva.py::test_profile_and_pfe_definition`; `python/core/tests/test_xva.py::test_eepe_non_decreasing_and_horizon`; `python/core/tests/test_xva.py::test_irs_cva_matches_swaption_strip`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: swaption-strip agreement within the cited tolerance. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** swap values only on the simulation grid; the Rust and C++ ports match a fixture Python generates, which catches port drift, not a wrong formula.

**Findings.** None.

### `CORE-XVA-HW`: Hull-White simulator for exposure

Tier 2, price, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Curve-fitted short-rate paths and bond prices behind the exposure engine.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/xva/hull_white.py::HullWhite`. Assumptions: Gaussian short rate with constant mean reversion and volatility. Reference: Hull and White (1990); Jamshidian (1989).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_xva.py::test_zcb_reprices_curve_and_mc_martingale`; `python/core/tests/test_xva.py::test_jamshidian_matches_mc`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: curve repricing; Jamshidian against Monte Carlo. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** one factor; the QuantLib Hull-White checks exercise fi_rates, a different implementation, so they are not cited here.

**Findings.** None.

### `CORE-CAP-SACCR`: SA-CCR exposure at default, all asset classes

Tier 1, risk, status `independently-checked`, validated 2026-11-04, next review 2027-05-04.

**Purpose.** Counterparty EAD under BCBS 279 for netting sets across asset classes.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/capital/saccr.py::ead`. Assumptions: BCBS 279 supervisory parameters, transcribed and checksum-tested. Reference: BCBS 279 (2014), Annex 4a.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_capital_saccr.py::test_supervisory_table_checksum`; `python/core/tests/test_capital_saccr.py::test_multiplier_limits`; `python/core/tests/test_capital_saccr.py::test_fx_addon_nets_within_pair`; `python/core/tests/test_capital_saccr.py::test_margined_capped_at_unmargined`. Independent checks: `python/core/tests/test_capital_saccr.py::test_example1_interest_rates`; `python/core/tests/test_capital_saccr.py::test_example2_credit`; `python/core/tests/test_capital_saccr.py::test_example3_commodity`; `python/core/tests/test_capital_saccr.py::test_example4_ir_plus_credit`; `python/core/tests/test_capital_saccr.py::test_example5_margined`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: BCBS 279 Annex 4a examples at published rounding. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** the FX add-on is not in Annex 4a, so it has own-property tests only; the published examples are checked to their printed rounding.

**Findings.** None.

### `CORE-FRTB-SBM-GIRR-FX`: FRTB SBM delta and vega for GIRR and FX

Tier 1, risk, status `independently-checked`, validated 2026-11-04, next review 2027-05-04.

**Purpose.** Delta and vega capital for GIRR and FX across the three MAR21.6 correlation scenarios.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/capital/frtb_sbm.py::girr_delta`; `python/core/src/qp_core/capital/frtb_sbm.py::girr_vega`; `python/core/src/qp_core/capital/frtb_sbm.py::fx_delta`; `python/core/src/qp_core/capital/frtb_sbm.py::fx_vega`. Assumptions: BCBS d457 (2019) MAR21 parameters. Reference: BCBS d457 (2019) MAR21.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_capital_frtb_sbm.py::test_footnote_examples`; `python/core/tests/test_capital_frtb_sbm.py::test_girr_delta_hand_fixture`; `python/core/tests/test_capital_frtb_sbm.py::test_fx_delta_hand_fixture`; `python/core/tests/test_capital_frtb_sbm.py::test_girr_vega_hand_fixture`. Independent checks: `tests/independent/frtb/test_frtb_sbm_independent.py::test_girr_delta`; `tests/independent/frtb/test_frtb_sbm_independent.py::test_girr_vega`; `tests/independent/frtb/test_frtb_sbm_independent.py::test_fx_delta`; `tests/independent/frtb/test_frtb_sbm_independent.py::test_fx_vega`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: oracle agreement to 1e-10 relative on 250 seeded portfolios per measure. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** the oracle was written in this repository from the Basel text, so a shared misreading of that text would pass.

**Findings.** None.

### `CORE-EXO-BARRIER`: Closed-form single and double barrier options

Tier 2, price, status `independently-checked`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Reiner-Rubinstein single barriers and Ikeda-Kunitomo double knock-outs.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/exotics/barriers.py::barrier_price`; `python/core/src/qp_core/exotics/barriers.py::double_knockout_price`. Assumptions: Black-Scholes dynamics with continuous monitoring. Reference: Reiner and Rubinstein (1991); Ikeda and Kunitomo (1992); Haug, *Option Pricing Formulas*, tables 4.13 and 4.15.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_exotics_barriers.py::test_in_out_parity_without_rebate`; `python/core/tests/test_exotics_barriers.py::test_barrier_mc_within_three_se`; `python/core/tests/test_exotics_barriers.py::test_double_knockout_series_has_converged_and_limits`. Independent checks: `python/core/tests/test_exotics_barriers.py::test_reiner_rubinstein_matches_haug_table_4_13`; `python/core/tests/test_exotics_barriers.py::test_ikeda_kunitomo_matches_haug_table_4_15`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: Haug tables to 4 decimal places. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** no smile; the 90/110 row of Haug table 4.15 could not be confirmed and is not used.

**Findings.** None.

### `CORE-EXO-TOUCH`: One-touch, no-touch and double no-touch

Tier 2, price, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Closed-form digital barrier prices.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/exotics/barriers.py::touch_price`; `python/core/src/qp_core/exotics/barriers.py::double_no_touch`. Assumptions: Black-Scholes dynamics with continuous monitoring. Reference: Reiner and Rubinstein (1991).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_exotics_barriers.py::test_touch_parity`; `python/core/tests/test_exotics_barriers.py::test_touch_mc_within_three_se`; `python/core/tests/test_exotics_barriers.py::test_double_no_touch_mc_and_limits`; `python/core/tests/test_exotics_barriers.py::test_dnt_series_converged`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: Monte Carlo within three standard errors. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** no smile; no published table cited.

**Findings.** None.

### `CORE-EXO-BARRIER-MC`: Monte Carlo barrier pricers

Tier 2, price, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Simulation reference for the barrier closed forms.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/exotics/mc.py::barrier_mc`; `python/core/src/qp_core/exotics/mc.py::double_barrier_mc`. Assumptions: geometric Brownian motion on a time grid. Reference: Glasserman, *Monte Carlo Methods in Financial Engineering* (2003).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_exotics_barriers.py::test_barrier_mc_within_three_se`; `python/core/tests/test_exotics_barriers.py::test_double_knockout_mc_within_three_se`; `python/core/tests/test_exotics_barriers.py::test_narrow_corridor_double_knockout_mc`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: closed form within three standard errors. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** statistical error; checked only against this repository's closed forms.

**Findings.** None.

### `CORE-EXO-ASIAN`: Asian options

Tier 2, price, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Geometric closed form, Turnbull-Wakeman approximation and control-variate Monte Carlo.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/exotics/asian.py::geometric_asian_price`; `python/core/src/qp_core/exotics/asian.py::turnbull_wakeman_price`; `python/core/src/qp_core/exotics/asian.py::turnbull_wakeman_continuous`; `python/core/src/qp_core/exotics/asian.py::asian_mc`. Assumptions: lognormal spot; Turnbull-Wakeman matches two moments of the average. Reference: Kemna and Vorst (1990); Turnbull and Wakeman (1991).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_exotics_other.py::test_geometric_closed_form_vs_plain_mc`; `python/core/tests/test_exotics_other.py::test_turnbull_wakeman_vs_control_variate_mc`; `python/core/tests/test_exotics_other.py::test_tw_discrete_tends_to_continuous`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: Turnbull-Wakeman against control-variate Monte Carlo. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** the arithmetic closed form is an approximation.

**Findings.** None.

### `CORE-EXO-LSM`: American options by Longstaff-Schwartz and CRR

Tier 2, price, status `independently-checked`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** American exercise by regression Monte Carlo, with a CRR tree as the reference.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/exotics/lsm.py::crr_american`; `python/core/src/qp_core/exotics/lsm.py::lsm_american`. Assumptions: geometric Brownian motion; polynomial regression basis. Reference: Longstaff and Schwartz (2001), table 1.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_exotics_other.py::test_lsm_vs_crr_tree`; `python/core/tests/test_exotics_other.py::test_american_call_no_dividend_equals_european`. Independent checks: `python/core/tests/test_exotics_other.py::test_lsm_longstaff_schwartz_table_1`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: distance to Longstaff-Schwartz table 1. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** LSM is a low-biased estimate of the exercise policy; the published check is one table cell with a tolerance of three standard errors plus 0.02.

**Findings.** None.

### `CORE-EXO-LSV`: Local-stochastic volatility by particle calibration

Tier 2, price, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Leverage function fitted to a surface, then vanilla repricing under LSV.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/exotics/lsv.py::calibrate_lsv_particles`; `python/core/src/qp_core/exotics/lsv.py::lsv_price_vanillas`. Assumptions: Heston variance with a leverage function; an arbitrage-free input surface. Reference: Guyon and Henry-Labordere (2012).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_exotics_lsv.py::test_lsv_reprices_vanillas_within_3se_and_02_vol`; `python/core/tests/test_exotics_lsv.py::test_pure_local_vol_limit_xi_zero`; `python/core/tests/test_exotics_lsv.py::test_dupire_flat_surface_is_flat`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: vanillas within 3 standard errors and 0.2 vol points. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** no outside implementation to compare with.

**Findings.** None.

### `CORE-EXO-TARF`: Target redemption forward

Tier 2, price, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Monte Carlo price of a TARF.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/exotics/tarf.py::tarf_mc`. Assumptions: lognormal spot, flat vol. Reference: Wystup, *FX Options and Structured Products* (2017).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_exotics_other.py::test_tarf_unreachable_target_is_forward_strip`; `python/core/tests/test_exotics_other.py::test_tarf_orderings`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: forward-strip limit and orderings. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** settlement lag ignored; no smile, which matters for this product.

**Findings.** None.

### `CORE-EXO-VV`: Vanna-volga vanilla and barrier adjustment

Tier 2, price, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Market-practice FX smile adjustment from three pillars.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/exotics/vanna_volga.py::vv_weights`; `python/core/src/qp_core/exotics/vanna_volga.py::vanna_volga_vanilla`; `python/core/src/qp_core/exotics/vanna_volga.py::vanna_volga_barrier`. Assumptions: three pillar quotes describe the smile. Reference: Castagna and Mercurio (2007).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_exotics_other.py::test_vv_reproduces_pillars_exactly`; `python/core/tests/test_exotics_other.py::test_vv_flat_smile_is_black_scholes`; `python/core/tests/test_exotics_other.py::test_vv_knock_in_out_parity_and_far_barrier`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: pillars repriced exactly; flat smile gives Black-Scholes. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** a market heuristic, not an arbitrage-free model; the barrier survival weight is a convention.

**Findings.** None.

### `CORE-AAD-BS`: Adjoint Black-Scholes Greeks

Tier 2, sensitivity, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** All first-order Greeks from one reverse sweep.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/aad/black_scholes.py::bs_greeks_aad`; `python/core/src/qp_core/aad/black_scholes.py::bs_greeks_aad_on`. Assumptions: Black-Scholes with continuous carry. Reference: Giles and Glasserman (2006).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_aad_python.py::test_bs_matches_analytic_greeks`; `python/core/tests/test_aad_python.py::test_bs_matches_bumps`; `python/core/tests/test_aad_tape_reuse.py::test_reused_tape_gives_identical_greeks`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: agreement with closed-form Greeks to 1e-12. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** compared with this repository's closed form and bumps, not an outside library.

**Findings.** None.

### `CORE-AAD-MC`: Adjoint Monte Carlo Greeks

Tier 2, sensitivity, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Pathwise adjoint Greeks of a simulated option price.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/aad/monte_carlo.py::mc_greeks_aad`. Assumptions: geometric Brownian motion; a payoff differentiable almost everywhere. Reference: Giles and Glasserman (2006).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_aad_python.py::test_mc_pathwise_matches_bumps_same_normals`; `python/core/tests/test_aad_python.py::test_mc_european_converges_to_black_scholes`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: bumps on the same normals. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** pathwise Greeks do not apply to discontinuous payoffs.

**Findings.** None.

### `CORE-AAD-SWAP`: Adjoint swap curve Greeks

Tier 2, sensitivity, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Swap risk to curve pillars and, through the Jacobian, to market quotes.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/aad/swap.py::pillar_greeks`; `python/core/src/qp_core/aad/swap.py::quote_greeks`. Assumptions: one quote per pillar, so the Jacobian is square. Reference: Henrard, *Algorithmic Differentiation in Finance Explained* (2017).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_aad_python.py::test_swap_pillar_greeks_match_bumps`; `python/core/tests/test_aad_python.py::test_swap_quote_greeks_match_rebootstrap`; `python/core/tests/test_aad_python.py::test_swap_pv_float_and_tape_agree`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: bumped pillars and rebuilt curves to 1e-6. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** the rebootstrap check needs fi_rates importable.

**Findings.** None.

### `CORE-VOL-SABR`: SABR implied volatility and calibration (shared)

Tier 2, price, status `independently-checked`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Shared Hagan and Obloj expansions, with calibration to a smile.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/vol/sabr.py::sabr_vol`; `python/core/src/qp_core/vol/sabr.py::sabr_calibrate`. Assumptions: SABR dynamics, short-expiry asymptotic expansion. Reference: Hagan et al. (2002); Obloj (2008).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_vol_sabr.py::test_docstring_worked_example`; `python/core/tests/test_vol_sabr.py::test_atm_matches_hagan_atm_formula`; `python/core/tests/test_vol_sabr.py::test_calibration_recovers_parameters`. Independent checks: `python/core/tests/test_vol_sabr_quantlib.py::test_hagan_variant_matches_quantlib`; `python/core/tests/test_vol_sabr_quantlib.py::test_shifted_matches_quantlib_shifted`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: agreement with QuantLib to 1e-12 relative. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** QuantLib checks the hagan variant only; the default obloj variant is checked against stored values and the ATM formula; negative densities at low strikes for long expiries.

**Findings.** None.

### `CORE-VOL-SABR-PDE`: Arbitrage-free SABR density PDE

Tier 2, price, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Finite-difference SABR density that stays non-negative where the expansion fails.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/vol/sabr_pde.py::sabr_pde`. Assumptions: SABR collapsed to one dimension. Reference: Hagan et al. (2014), "Arbitrage-free SABR"; Le Floc'h and Kennedy (2017).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_vol_sabr_pde.py::test_mass_and_forward_conserved`; `python/core/tests/test_vol_sabr_pde.py::test_short_expiry_matches_hagan_expansion`; `python/core/tests/test_vol_sabr_pde.py::test_grid_convergence`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: mass and forward conserved; grid convergence. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** no outside implementation to compare with.

**Findings.** None.

### `CORE-VOL-ESSVI`: eSSVI and raw SVI slice fits

Tier 2, price, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Per-slice smile fits with arbitrage checks.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/vol/essvi.py::fit_essvi_slices`; `python/core/src/qp_core/vol/essvi.py::fit_svi_slices`. Assumptions: SVI family describes each slice. Reference: Gatheral and Jacquier (2014); Hendriks and Martini (2019).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_vol_essvi.py::test_svi_per_slice_closes_f10_within_01_vol`; `python/core/tests/test_vol_essvi.py::test_essvi_by_slice_improves_f10_but_shape_limits_usdjpy`; `python/core/tests/test_vol_essvi.py::test_svi_density_factor_flags_arbitrage`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: fit error in vol points; density factor. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** the eSSVI shape limits the fit on USDJPY.

**Findings.** None.

### `CORE-VOL-HESTON-CF`: Heston characteristic function (shared)

Tier 2, price, status `independently-checked`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** The characteristic function behind the equity and FX Heston pricers.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/vol/heston.py::heston_cf`. Assumptions: Heston variance dynamics. Reference: Heston (1993); Albrecher et al. (2007).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_vol_heston.py::test_martingale_and_normalisation`; `python/core/tests/test_vol_heston.py::test_equals_old_equity_copy_bit_for_bit`. Independent checks: `python/core/tests/test_vol_heston.py::test_gil_pelaez_on_shared_cf_matches_quantlib`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: agreement with QuantLib AnalyticHestonEngine. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** the QuantLib check needs Gil-Pelaez inversion in the test; the function itself returns no price.

**Findings.** None.

### `CORE-CR-ISDA`: ISDA standard CDS model (shared)

Tier 1, price, status `independently-checked`, validated 2026-11-04, next review 2027-05-04.

**Purpose.** CDS legs, hazard bootstrap and quote-to-upfront conversion for the index stack.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/credit_index/isda.py::cds_legs`; `python/core/src/qp_core/credit_index/isda.py::bootstrap_hazard`; `python/core/src/qp_core/credit_index/isda.py::upfront_from_quoted_spread`; `python/core/src/qp_core/credit_index/isda.py::quoted_spread_from_upfront`. Assumptions: piecewise-constant hazard; ISDA standard model conventions. Reference: ISDA CDS Standard Model; Markit values as carried in QuantLib `testIsdaEngine`.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_credit_index.py::test_bootstrap_reprices_par_spreads`; `python/core/tests/test_credit_index.py::test_spread_upfront_round_trip`; `python/core/tests/test_credit_index.py::test_without_half_day_bias_the_markit_values_are_missed`. Independent checks: `python/core/tests/test_credit_index.py::test_markit_published_upfronts`; `python/core/tests/test_credit_index.py::test_frozen_curve_matches_quantlib`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: Markit upfronts to 1e-3 per 1e7 notional. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** recovery is an input, not estimated; the QuantLib test checks the frozen discount curve; the prices are checked against Markit's values.

**Findings.** None.

### `CORE-CR-INDEX`: CDS index intrinsic value and basis

Tier 2, price, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Index intrinsic from constituents, the quoted basis and the curve adjustment.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/credit_index/index.py::intrinsic`; `python/core/src/qp_core/credit_index/index.py::index_basis`; `python/core/src/qp_core/credit_index/index.py::adjust_to_index`. Assumptions: one flat curve at the standard recovery for the quote. Reference: O'Kane, *Modelling Single-name and Multi-name Credit Derivatives* (2008).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_credit_index.py::test_homogeneous_intrinsic_equals_index`; `python/core/tests/test_credit_index.py::test_heterogeneous_basis_and_adjustment`; `python/core/tests/test_credit_index.py::test_factor_and_defaults`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: homogeneous intrinsic equals the index. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** no outside index data.

**Findings.** None.

### `CORE-RISK-JACOBIAN`: Node and quote risk through a Jacobian

Tier 2, sensitivity, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Moves risk between curve nodes and market quotes; cross gamma by differences.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/risk/jacobian.py::node_to_quote_risk`; `python/core/src/qp_core/risk/jacobian.py::quote_to_node_risk`; `python/core/src/qp_core/risk/jacobian.py::cross_gamma`. Assumptions: a square, invertible Jacobian; local linearity. Reference: Henrard, *Interest Rate Modelling in the Multi-curve Framework* (2014).

**Outcomes analysis.** Own-property tests: `python/core/tests/test_risk_core.py::test_node_to_quote_worked_example`; `python/core/tests/test_risk_core.py::test_node_to_quote_matches_reparametrised_gradient`; `python/core/tests/test_risk_core.py::test_cross_gamma_of_quadratic_is_exact_and_symmetric`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: reparametrised gradient agreement. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** finite-difference error depends on the step.

**Findings.** None.

### `CORE-RISK-VEGA-GRID`: Bucketed vega grid

Tier 2, sensitivity, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Splits vega across expiry and strike buckets.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/risk/vega_grid.py::vega_grid`. Assumptions: linear bucket weights. Reference: Bergomi, *Stochastic Volatility Modeling* (2016), on smile risk.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_risk_core.py::test_vega_grid_splits_between_expiry_buckets`; `python/core/tests/test_risk_core.py::test_vega_grid_sums_to_parallel_vega_2d_and_3d`; `python/core/tests/test_risk_core.py::test_grid_weights_partition_of_unity_and_validation`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: buckets sum to parallel vega. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** bucket boundaries are a convention.

**Findings.** None.

### `CORE-RISK-SMILE`: Smile dynamics and smile-aware Greeks

Tier 2, sensitivity, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Delta and gamma under sticky-strike, sticky-delta and sticky-moneyness rules.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/risk/smile_dynamics.py::smile_price`; `python/core/src/qp_core/risk/smile_dynamics.py::smile_delta`; `python/core/src/qp_core/risk/smile_dynamics.py::smile_gamma`; `python/core/src/qp_core/risk/smile_dynamics.py::premium_adjusted_delta`; `python/core/src/qp_core/risk/smile_dynamics.py::rank_smile_dynamics`. Assumptions: one of the named smile rules holds over the move. Reference: Derman (1999); Wystup (2017) for premium-adjusted delta.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_risk_core.py::test_worked_example_numbers`; `python/core/tests/test_risk_core.py::test_delta_matches_fd_of_surface_shifted_the_matching_way`; `python/core/tests/test_risk_core.py::test_premium_adjusted_delta_closed_form`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: agreement with differences of the shifted surface. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** the rules are conventions, not a dynamic model.

**Findings.** None.

### `CORE-RISK-PNL`: P&L explain

Tier 2, risk, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Attributes P&L to Greeks and picks the smile rule that fits a realised path.

**Conceptual soundness.** Entrypoints: `python/core/src/qp_core/risk/pnl_explain.py::explain`; `python/core/src/qp_core/risk/pnl_explain.py::taylor_attribution`. Assumptions: a second-order Taylor expansion over the period. Reference: Taylor expansion of the pricing function; no single paper.

**Outcomes analysis.** Own-property tests: `python/core/tests/test_risk_core.py::test_explain_worked_example`; `python/core/tests/test_risk_core.py::test_taylor_attribution_sums_to_quadratic_form`; `python/core/tests/test_risk_core.py::test_explain_identifies_the_dynamic_of_a_realised_path`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: attribution sums to the quadratic form. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** large moves land in the unexplained term.

**Findings.** None.

### `EXEC-IMPACT-FIT`: Impact exponent estimate

Tier 2, forecast, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Fits the power of price impact against order-flow imbalance, with a bootstrap interval.

**Conceptual soundness.** Entrypoints: `python/execution/src/qp_exec/lob/impact.py::fit_impact_exponent`. Assumptions: a power law between binned imbalance and return. Reference: Bouchaud, Bonart, Donier and Gould, *Trades, Quotes and Prices* (2018).

**Outcomes analysis.** Own-property tests: `python/execution/tests/test_flow_fills_impact.py::test_impact_exponent_recovered_with_ci`; `python/execution/tests/test_flow_fills_impact.py::test_window_imbalance_shapes`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: exponent recovered inside its interval. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** tested on synthetic flow; no outside estimate.

**Findings.** None.

### `EXEC-IMPACT-TRANSIENT`: Transient impact optimal schedules

Tier 2, forecast, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Expected cost and the optimal schedule under a decay kernel.

**Conceptual soundness.** Entrypoints: `python/execution/src/qp_exec/schedules/transient.py::expected_cost`; `python/execution/src/qp_exec/schedules/transient.py::optimal_schedule`. Assumptions: linear propagator with a known kernel. Reference: Obizhaeva and Wang (2013); Gatheral (2010).

**Outcomes analysis.** Own-property tests: `python/execution/tests/test_schedules.py::test_obizhaeva_wang_structure_and_cost`; `python/execution/tests/test_schedules.py::test_power_law_kernel_optimum_is_valid_and_cheaper`; `python/execution/tests/test_schedules.py::test_non_convex_kernel_gives_sign_alternating_optimum`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: Obizhaeva-Wang structure of the optimum. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** a non-convex kernel gives a sign-alternating optimum, so kernels should pass the Gatheral condition.

**Findings.** None.

### `EXEC-MM-AS`: Avellaneda-Stoikov quotes

Tier 2, price, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Reservation price and spread for an inventory-averse market maker.

**Conceptual soundness.** Entrypoints: `python/execution/src/qp_exec/mm/quotes.py::avellaneda_stoikov`. Assumptions: Brownian mid, exponential fill intensity. Reference: Avellaneda and Stoikov (2008).

**Outcomes analysis.** Own-property tests: `python/execution/tests/test_mm.py::test_as_paper_numbers`; `python/execution/tests/test_mm.py::test_as_limits`; `python/execution/tests/test_mm.py::test_as_inventory_skew_and_bound`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: spread at the paper's parameters. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** the paper-parameter test is worked by hand, not read from a printed table.

**Findings.** None.

### `EXEC-MM-GLFT`: Gueant-Lehalle-Fernandez-Tapia quotes

Tier 2, price, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Quotes under an inventory bound: exact, stationary and asymptotic forms.

**Conceptual soundness.** Entrypoints: `python/execution/src/qp_exec/mm/quotes.py::GLFT`; `python/execution/src/qp_exec/mm/quotes.py::glft_asymptotic`. Assumptions: Brownian mid, exponential fill intensity, bounded inventory. Reference: Gueant, Lehalle and Fernandez-Tapia (2013).

**Outcomes analysis.** Own-property tests: `python/execution/tests/test_mm.py::test_glft_matches_matrix_exponential_and_ground_state`; `python/execution/tests/test_mm.py::test_glft_asymptotic_close_to_exact_stationary`; `python/execution/tests/test_mm.py::test_glft_small_gamma_matches_as`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: matrix exponential agreement to 1e-10. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** scipy's matrix exponential checks the solver, not the model, so it is not counted as independent.

**Findings.** None.

### `EXEC-MM-TOXICITY`: Markout toxicity and VPIN

Tier 2, forecast, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Flow toxicity signals that widen quotes.

**Conceptual soundness.** Entrypoints: `python/execution/src/qp_exec/mm/toxicity.py::MarkoutToxicity`; `python/execution/src/qp_exec/mm/toxicity.py::VPIN`. Assumptions: the aggressor side is known. Reference: Easley, Lopez de Prado and O'Hara (2012).

**Outcomes analysis.** Own-property tests: `python/execution/tests/test_mm_toxicity.py::test_markout_ewma_by_hand`; `python/execution/tests/test_mm_toxicity.py::test_vpin_buckets_by_hand`; `python/execution/tests/test_mm_toxicity.py::test_vpin_widens_only_under_toxic_flow_and_pays`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: hand fixtures; widening reacts more to toxic flow. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** VPIN's predictive value is disputed (Andersen and Bondarenko, 2014); evidence is from the repository's simulator.

**Findings.** None.

### `EXEC-SOR-ALLOC`: Cont-Kukanov order allocation

Tier 2, forecast, status `independently-checked`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Splits an order between market and limit orders across venues to minimise expected cost.

**Conceptual soundness.** Entrypoints: `python/execution/src/qp_exec/sor/allocation.py::expected_cost`; `python/execution/src/qp_exec/sor/allocation.py::solve_allocation`; `python/execution/src/qp_exec/sor/allocation.py::single_venue_optimum`; `python/execution/src/qp_exec/sor/allocation.py::deterministic_optimum`. Assumptions: sampled venue flow stands for the fill distribution. Reference: Cont and Kukanov (2017).

**Outcomes analysis.** Own-property tests: `python/execution/src/qp_exec/sor/tests/test_lit.py::test_single_venue_matches_quantile_rule`; `python/execution/src/qp_exec/sor/tests/test_lit.py::test_deterministic_flow_is_greedy`; `python/execution/src/qp_exec/sor/tests/test_lit.py::test_lp_cost_equals_direct_evaluation`. Independent checks: `python/execution/src/qp_exec/sor/tests/test_lit.py::test_matches_cvxpy`; `python/execution/src/qp_exec/sor/tests/test_dark.py::test_live_lit_dark_split_matches_cvxpy`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: cost agreement with cvxpy (Clarabel) to 1e-6. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** the cvxpy tests skip where cvxpy is absent (the qp-exec dev extra installs it); convex venues only; a non-convex venue is refused.

**Findings.** None.

### `EXEC-HAWKES`: Hawkes process fits

Tier 2, forecast, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Exponential-kernel Hawkes maximum likelihood for order-arrival clustering.

**Conceptual soundness.** Entrypoints: `python/execution/src/qp_exec/hawkes/__init__.py::fit_univariate`; `python/execution/src/qp_exec/hawkes/__init__.py::fit_multivariate`. Assumptions: exponential kernel, stationary (branching ratio below one). Reference: Hawkes (1971); Ozaki (1979); Bacry, Mastromatteo and Muzy (2015).

**Outcomes analysis.** Own-property tests: `python/execution/tests/test_hawkes.py::test_loglik_matches_brute_force`; `python/execution/tests/test_hawkes.py::test_univariate_mle_recovers_parameters_within_3_se`; `python/execution/tests/test_hawkes.py::test_time_rescaling_accepts_true_model_rejects_poisson`; `python/execution/tests/test_hawkes.py::test_multivariate_mle_recovers_cross_excitation`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: parameters within 3 standard errors; time-rescaling test. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** one shared decay in the multivariate fit; no outside estimator compared.

**Findings.** None.

### `BT-FILL-QUEUE`: Queue-aware fill model

Tier 2, forecast, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Estimates when a simulated resting order would fill, from book and trade events.

**Conceptual soundness.** Entrypoints: `python/backtest/src/qp_bt/fills/queue.py::fill_marketable`; `python/backtest/src/qp_bt/fills/queue.py::prob_ahead_power`; `python/backtest/src/qp_bt/fills/queue.py::QueueAwareFillModel`. Assumptions: cancels are spread through the queue by the chosen probability. Reference: Moallemi and Yuan (2017).

**Outcomes analysis.** Own-property tests: `python/backtest/tests/test_queue_fills.py::test_worked_example_from_docstring`; `python/backtest/tests/test_queue_fills.py::test_matches_fifo_when_cancels_behind`; `python/backtest/tests/test_queue_fills.py::test_matches_fifo_when_cancels_in_front`; `python/backtest/tests/test_queue_fills.py::test_mutation_wrong_cancel_assumption_is_caught_w25`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: agreement with a FIFO book under the matching cancel rule. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** the simulated order never removes liquidity from the real book.

**Findings.** None.

### `BT-FILL-NEXTBAR`: Next-bar fill model

Tier 2, forecast, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Bar-level fills with a signal-to-fill lag and optional partial fills.

**Conceptual soundness.** Entrypoints: `python/backtest/src/qp_bt/fills/nextbar.py::NextBarFill`. Assumptions: the fill price is taken from a later bar than the decision. Reference: Backtest convention; no single paper.

**Outcomes analysis.** Own-property tests: `python/backtest/tests/test_event_engine.py::test_next_bar_fill_never_uses_the_decision_close`; `python/backtest/tests/test_event_engine.py::test_lag_moves_the_fill_and_drops_orders_past_the_end`; `python/backtest/tests/test_event_engine.py::test_partial_fills_respect_participation_and_finish_later`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: no use of the decision close; fast and event engines agree. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** no book, so no queue or spread effects.

**Findings.** None.

### `BT-COST`: Backtest cost models

Tier 2, forecast, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Linear, half-spread and square-root impact costs, each with a named source.

**Conceptual soundness.** Entrypoints: `python/backtest/src/qp_bt/costs.py::LinearCost`; `python/backtest/src/qp_bt/costs.py::HalfSpreadCost`; `python/backtest/src/qp_bt/costs.py::SqrtImpactCost`. Assumptions: inputs are known at the decision close; square-root coefficient in the published 0.5 to 1 range. Reference: Almgren, Thum, Hauptmann and Li (2005); Toth et al. (2011).

**Outcomes analysis.** Own-property tests: `python/backtest/tests/test_hand_fixtures.py::test_three_asset_ten_day_fixture_matches_exact_oracle`; `python/backtest/tests/test_event_engine.py::test_half_spread_cost_per_asset_and_per_bar`; `python/backtest/tests/test_fast_equals_event.py::test_fast_equals_event`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: hand ledger fixtures. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** a non-positive ADV is treated as infinite depth, so impact is zero; SqrtImpactCost is exercised through engine equality, not a hand value.

**Findings.** None.

### `BT-COV-LW`: Ledoit-Wolf covariance shrinkage

Tier 2, risk, status `independently-checked`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Well-conditioned covariance for portfolio construction.

**Conceptual soundness.** Entrypoints: `python/backtest/src/qp_bt/portfolio/covariance.py::ledoit_wolf`. Assumptions: scaled-identity target; i.i.d. rows. Reference: Ledoit and Wolf (2004).

**Outcomes analysis.** Own-property tests: `python/backtest/tests/test_portfolio_layer.py::test_ledoit_wolf_intensity_in_unit_interval_and_pd`. Independent checks: `python/backtest/tests/test_portfolio_layer.py::test_ledoit_wolf_matches_sklearn`. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: agreement with scikit-learn to 1e-12. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** the shrinkage target is the scaled identity only.

**Findings.** None.

### `BT-COV-PCA`: Sample and PCA covariance

Tier 2, risk, status `self-tested`, validated 2026-11-04, next review 2027-11-04.

**Purpose.** Plain and factor-truncated covariance estimates.

**Conceptual soundness.** Entrypoints: `python/backtest/src/qp_bt/portfolio/covariance.py::sample_cov`; `python/backtest/src/qp_bt/portfolio/covariance.py::pca_cov`. Assumptions: k factors capture the common variation; the diagonal is kept. Reference: Standard estimators; Jolliffe, *Principal Component Analysis* (2002).

**Outcomes analysis.** Own-property tests: `python/backtest/tests/test_portfolio_layer.py::test_pca_full_rank_recovers_sample_and_keeps_diagonal`. Independent checks: none cited. Tolerances are stated in each test.

**Ongoing monitoring.** KPIs: full-rank PCA recovers the sample covariance. Review trigger: a failure of any cited test, or a change to the cited reference.

**Limitations and compensating controls.** noisy when assets approach observations.

**Findings.** None.
