# quant-portfolio

[![ci](https://github.com/AshJha0/quant-portfolio/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AshJha0/quant-portfolio/actions/workflows/ci.yml)

quant-portfolio is a set of study implementations of standard
quantitative-finance models: option pricing, volatility forecasting, VaR and
expected shortfall, fixed income, pairs trading, credit scoring, portfolio
optimisation, execution scheduling, volatility surfaces and regime detection.
Each area is written in Python with equity conventions and again with FX
conventions. The option-pricing and VaR libraries also have C++ and Rust
ports, tested against the Python code and, for options, against an mpmath
oracle. All results come from seeded synthetic data unless a page names a
dataset. The code has not been used on a live book.

The plan has three stages. The repository is at stage one.

| Stage | What it means | Status |
|---|---|---|
| 1. Study portfolio | Correct, tested implementations with plain documentation | In progress |
| 2. Research platform | Free public data, pre-registered experiments, reported results | Planned |
| 3. Desk library | Shared core, bindings, option revaluation in the compiled VaR engines | Planned |

**Status.** See the CI badge above. Test counts are not typed into the docs.
The CI `counts` job writes the executed pass counts to `counts.json` and shows
them in the run summary.

Further reading in [`docs/`](docs/README.md):
[ARCHITECTURE](docs/ARCHITECTURE.md) (layout, conventions, test pipeline),
[DIAGRAMS](docs/DIAGRAMS.md), [LEARN](docs/LEARN.md) (study guide and
self-test question bank), [COOKBOOK](docs/COOKBOOK.md) (recipes) and
[MARKET_RISK](docs/MARKET_RISK.md) (the VaR, ES and backtesting workflow).
Changes are listed in [CHANGELOG.md](CHANGELOG.md).

```
quant-portfolio/
├── CONVENTIONS.md     # engineering and documentation rules for every project
├── CONTRIBUTING.md    # setup, test gate, golden regeneration
├── CHANGELOG.md
├── docs/              # architecture, study guide, recipes, market-risk workflow
├── python/
│   ├── equity/01-…10-…    # 10 equity projects
│   ├── fx/01-…10-…        # 10 FX projects (separate code from equity)
│   └── foundations/       # 3 single-asset projects
├── cpp/               # 4 engines, C++20, CMake, tests in GoogleTest
└── rust/              # the same 4 engines, Rust 2021, std only
```

## Areas

31 sub-projects: 23 Python projects, 4 C++ engines and 4 Rust crates.

| # | Area | Equity | FX |
|---|------|--------|-----|
| 1 | Options pricing and Greeks | `python/equity/01-options-pricing` (+ C++/Rust) | `python/fx/01-options-pricing` (+ C++/Rust) |
| 2 | Volatility modelling and forecasting | `python/equity/02-volatility-modeling` | `python/fx/02-volatility-modeling` |
| 3 | Market risk: VaR and expected shortfall | `python/equity/03-var-es-engine` (+ C++/Rust linear-book ports) | `python/fx/03-var-es-engine` (+ C++/Rust linear-book ports) |
| 4 | Fixed income pricing and risk | `python/equity/04-fixed-income` | `python/fx/04-fixed-income` |
| 5 | Statistical pairs trading | `python/equity/05-pairs-trading` | `python/fx/05-pairs-trading` |
| 6 | Credit risk and PD modelling | `python/equity/06-credit-risk` | `python/fx/06-credit-risk` |
| 7 | Portfolio optimisation and risk allocation | `python/equity/07-portfolio-optimization` | `python/fx/07-portfolio-optimization` |
| 8 | Execution scheduling and TCA | `python/equity/08-algo-execution` | `python/fx/08-algo-execution` |
| 9 | Volatility surface and stochastic volatility | `python/equity/09-vol-surface` | `python/fx/09-vol-surface` |
| 10 | Regime switching | `python/equity/10-regime-switching` | `python/fx/10-regime-switching` |

Foundations projects are smaller: one asset, Python only, no FX version.

| # | Project | Path | What it is |
|---|---|---|---|
| F1 | Risk metrics | `python/foundations/01-risk-metrics` | Single-asset volatility, VaR, ES and Sharpe toolkit on bundled synthetic data. Three VaR methods side by side. No portfolio risk; see `python/equity/03-var-es-engine` for that. |
| F2 | Trading signal backtest | `python/foundations/02-trading-signal-backtest` | Moving-average crossover with fills modelled at t+1 as p_t + f(p_{t+1} - p_t), f = 0.5, transaction costs, in-sample and out-of-sample splits and walk-forward evaluation. |
| F3 | Black-Scholes replication | `python/foundations/03-black-scholes-replication` | Black-Scholes-Merton from scratch with `math.erfc` only, N(x) = 0.5 erfc(-x/sqrt 2), checked against a Monte Carlo pricer, analytic identities and finite-difference Greeks. `python/equity/01-options-pricing` is the fuller implementation. |

Every project has `src/`, `tests/`, `examples/run_pipeline.py` and
`docs/{METHODOLOGY,VALIDATION,DESK_GUIDE}.md`. See `CONVENTIONS.md`.

## How to run

Use a virtual environment outside the repository, one per project if you want
to match CI. Full commands for Python, C++ and Rust are in
[CONTRIBUTING.md](CONTRIBUTING.md).

```bash
python -m venv ~/.venvs/qp && . ~/.venvs/qp/bin/activate
cd python/equity/01-options-pricing
python -m pip install -c ../../../requirements/ci-constraints.txt -e ".[dev]"
python -m pytest -p no:cacheprovider -q
python examples/run_pipeline.py
```

A project declares either a `test` or a `dev` extra; install whichever it has.

## How it is tested

CI (`.github/workflows/ci.yml`) runs on every push and pull request. The
single required check is `ci-ok`, which passes only when every job passed.

| Job | What it runs |
|---|---|
| `python` | pytest for each of the 23 projects, each in its own venv, on Python 3.10 to 3.13 |
| `goldens --check` | regenerates each C++ golden header and Rust golden module and fails on any difference from the committed JSON |
| `cpp` | each engine built and tested with GCC and Clang, plus a Clang AddressSanitizer and UBSan build |
| `rust` | `cargo test --release --locked` with warnings denied |
| `hygiene` | banned-word check and a voice report |
| `counts` | sums executed pass counts into `counts.json` and runs `tests/test_docs_counts.py` |

### Cross-language tests

The C++ and Rust engines are ports of the Python design. They are not
independent derivations. Two kinds of test tie them together.

1. **Golden vectors.** For options, `tests/golden/generate_golden.py` computes
   reference values with an mpmath oracle and stores each output with its own
   tolerance in `tests/golden/golden_vectors.json`. The Python consumer is
   `tests/test_golden.py`. For VaR, the generator uses the Python reference
   with fixed inputs, checked from Python by `tests/test_golden_vectors.py`.
   `tools/gen_golden_header.py` (C++) and `tools/gen_golden_rs.py` (Rust)
   turn the JSON into committed source. The CI `goldens --check` job fails
   if the committed source has drifted from the JSON.
2. **Differential tests.** The options projects' `tests/test_differential.py`
   send random inputs to small JSONL shims (Rust `eq_jsonl`/`fx_jsonl`, C++
   `eqopt_jsonl`/`fxopt_jsonl`, found through `QP_RS_EQ_JSONL`,
   `QP_RS_FX_JSONL`, `QP_CPP_EQ_JSONL`, `QP_CPP_FX_JSONL`) and compare the
   answers with Python. The tests skip when the binaries are not built.

Monte Carlo results do not match path by path across languages, because each
language uses its own random number generator. Agreement there is
statistical, within the tolerance stated in each engine's `docs/VALIDATION.md`.

The compiled VaR engines revalue linear books only. Option revaluation in
those engines is planned for stage three. Only the Python FX VaR engine
reprices option positions.

### Engine latency

<!-- LATENCY-PLACEHOLDER: the latency pass fills this table from the benchmark
     binaries, with CPU, compiler, flags, date and command. -->

| Engine | Language | Operation | Latency | Environment |
|---|---|---|---|---|
| equity-options-engine | C++ | pending latency pass | pending latency pass | pending latency pass |
| equity-options-engine | Rust | pending latency pass | pending latency pass | pending latency pass |
| fx-options-engine | C++ | pending latency pass | pending latency pass | pending latency pass |
| fx-options-engine | Rust | pending latency pass | pending latency pass | pending latency pass |
| equity-var-engine | C++ | pending latency pass | pending latency pass | pending latency pass |
| equity-var-engine | Rust | pending latency pass | pending latency pass | pending latency pass |
| fx-var-engine | C++ | pending latency pass | pending latency pass | pending latency pass |
| fx-var-engine | Rust | pending latency pass | pending latency pass | pending latency pass |

## Conventions to know

- `alpha` is the tail probability in (0, 0.5) in every engine: 0.01 means 99%
  VaR. FX functions used to take a confidence level; passing 0.99 now raises.
- Historical VaR is the order statistic L(ceil(n alpha)) of the sorted losses.
  ES is the Acerbi-Tasche estimator with the fractional atom.
- `kupiec_pof(*, n_obs, n_exceptions, alpha)` takes keywords only.
- `basel_traffic_light(*, n_obs, n_exceptions, alpha=0.01)` works for any
  window and tail probability, using the binomial CDF cut-offs 0.95 and 0.9999.
- The Monte Carlo VaR standard error is a bootstrap estimate. The density
  (KDE) estimate is a diagnostic only.
- Backtests in areas 05, 08, 10 and F2 fill at t+1 by default, as in F2 above.

Old entry points whose signature or convention changed stay for one release
with a `DeprecationWarning`. CHANGELOG.md lists them.

## Limitations

- All strategy results are on synthetic data with planted structure. A
  strategy that finds the planted signal is a recovery test, not evidence of
  an edge.
- The Kupiec test uses the chi-squared(1) approximation. At 250 days and 99%
  its actual size is about 9.5%, not 5% (`TestKupiecExactSize`).
- Not implemented: SABR, structural (Merton/KMV) PD, short-rate models, OIS
  multi-curve, FRTB.
- The compiled engines have no Python bindings.
- Python dependencies are pinned for CI only, in
  `requirements/ci-constraints.txt`.

## References

The sources each area is built from and checked against.

**Options pricing & Greeks (areas 01, F3)**

1. Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy* 81(3), 637–654.
2. Merton, R. C. (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics and Management Science* 4(1), 141–183.
3. Garman, M. B. & Kohlhagen, S. W. (1983). "Foreign Currency Option Values." *Journal of International Money and Finance* 2(3), 231–237.
4. Reiswich, D. & Wystup, U. (2010). "A Guide to FX Options Quoting Conventions." *Journal of Derivatives* 18(2), 58–68.
5. Wystup, U. (2006). *FX Options and Structured Products.* Wiley.
6. Clark, I. J. (2011). *Foreign Exchange Option Pricing: A Practitioner's Guide.* Wiley, ch. 3.
7. Black, F. (1976). "The Pricing of Commodity Contracts." *Journal of Financial Economics* 3(1–2), 167–179.
8. Cox, J. C., Ross, S. A. & Rubinstein, M. (1979). "Option Pricing: A Simplified Approach." *Journal of Financial Economics* 7(3), 229–263.
9. Broadie, M. & Glasserman, P. (1996). "Estimating Security Price Derivatives Using Simulation." *Management Science* 42(2), 269–285.
10. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering.* Springer, ch. 7.
11. Brent, R. P. (1973). *Algorithms for Minimization without Derivatives.* Prentice-Hall, ch. 4.
12. Jäckel, P. (2015). "Let's Be Rational." *Wilmott* 2015(75), 40–53.

**Volatility modelling & forecast evaluation (area 02)**

13. Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics* 31(3), 307–327.
14. Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993). "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *Journal of Finance* 48(5), 1779–1801.
15. Nelson, D. B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New Approach." *Econometrica* 59(2), 347–370.
16. J.P. Morgan / Reuters (1996). *RiskMetrics: Technical Document*, 4th ed. (EWMA λ = 0.94.)
17. Hansen, P. R. & Lunde, A. (2005). "A Forecast Comparison of Volatility Models: Does Anything Beat a GARCH(1,1)?" *Journal of Applied Econometrics* 20(7), 873–889.
18. Patton, A. J. (2011). "Volatility Forecast Comparison Using Imperfect Volatility Proxies." *Journal of Econometrics* 160(1), 246–256.
19. Diebold, F. X. & Mariano, R. S. (1995). "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics* 13(3), 253–263.

**Market risk: VaR, ES, backtesting (areas 03, F1)**

20. Boudoukh, J., Richardson, M. & Whitelaw, R. (1998). "The Best of Both Worlds." *Risk* 11(5), 64–67.
21. Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). "VaR without Correlations for Portfolios of Derivative Securities." *Journal of Futures Markets* 19(5), 583–602.
22. Hull, J. & White, A. (1998). "Incorporating Volatility Updating into the Historical Simulation Method for Value-at-Risk." *Journal of Risk* 1(1), 5–19.
23. Acerbi, C. & Tasche, D. (2002). "On the Coherence of Expected Shortfall." *Journal of Banking & Finance* 26(7), 1487–1503.
24. Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). "Coherent Measures of Risk." *Mathematical Finance* 9(3), 203–228.
25. McNeil, A. J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management*, 2nd ed. Princeton University Press, Ex. 2.14–2.15.
26. Cornish, E. A. & Fisher, R. A. (1938). "Moments and Cumulants in the Specification of Distributions." *Revue de l'Institut International de Statistique* 5(4), 307–320.
27. Maillard, D. (2012). "A User's Guide to the Cornish Fisher Expansion." SSRN 1997178.
28. Kupiec, P. H. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models." *Journal of Derivatives* 3(2), 73–84.
29. Christoffersen, P. F. (1998). "Evaluating Interval Forecasts." *International Economic Review* 39(4), 841–862.
30. Basel Committee on Banking Supervision (1996). *Supervisory Framework for the Use of "Backtesting" in Conjunction with the Internal Models Approach to Market Risk Capital Requirements.* BIS.
31. Acerbi, C. & Székely, B. (2014). "Backtesting Expected Shortfall." *Risk*, December, 76–81.
32. Basel Committee on Banking Supervision (2019). *Minimum Capital Requirements for Market Risk* (FRTB). BIS.
33. Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." *Annals of Statistics* 7(1), 1–26.
34. Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis.* Chapman & Hall.
35. Acklam, P. J. (2003). "An Algorithm for Computing the Inverse Normal Cumulative Distribution Function."
36. Cody, W. J. (1969). "Rational Chebyshev Approximations for the Error Function." *Mathematics of Computation* 23(107), 631–637.
37. Moro, B. (1995). "The Full Monte." *Risk* 8(2), 57–58.
38. Blackman, D. & Vigna, S. (2021). "Scrambled Linear Pseudorandom Number Generators." *ACM Transactions on Mathematical Software* 47(4), art. 36. (xoshiro256++.)
39. Matsumoto, M. & Nishimura, T. (1998). "Mersenne Twister: A 623-Dimensionally Equidistributed Uniform Pseudo-Random Number Generator." *ACM TOMACS* 8(1), 3–30.
40. O'Neill, M. E. (2014). "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation." Harvey Mudd College, HMC-CS-2014-0905.

**Fixed income (area 04)**

41. Macaulay, F. R. (1938). *Some Theoretical Problems Suggested by the Movements of Interest Rates, Bond Yields and Stock Prices in the United States since 1856.* NBER.
42. Tuckman, B. & Serrat, A. (2011). *Fixed Income Securities: Tools for Today's Markets*, 3rd ed. Wiley, ch. 4–5.
43. Ho, T. S. Y. (1992). "Key Rate Durations: Measures of Interest Rate Risks." *Journal of Fixed Income* 2(2), 29–44.
44. Hagan, P. S. & West, G. (2006). "Interpolation Methods for Curve Construction." *Applied Mathematical Finance* 13(2), 89–129.
45. Du, W., Tepper, A. & Verdelhan, A. (2018). "Deviations from Covered Interest Rate Parity." *Journal of Finance* 73(3), 915–957.

**Statistical pairs trading (area 05)**

46. Engle, R. F. & Granger, C. W. J. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing." *Econometrica* 55(2), 251–276.
47. Dickey, D. A. & Fuller, W. A. (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74(366), 427–431.
48. Said, S. E. & Dickey, D. A. (1984). "Testing for Unit Roots in Autoregressive-Moving Average Models of Unknown Order." *Biometrika* 71(3), 599–607.
49. MacKinnon, J. G. (1996). "Numerical Distribution Functions for Unit Root and Cointegration Tests." *Journal of Applied Econometrics* 11(6), 601–618.
50. MacKinnon, J. G. (2010). "Critical Values for Cointegration Tests." Queen's Economics Department Working Paper 1227.
51. Schwert, G. W. (1989). "Tests for Unit Roots: A Monte Carlo Investigation." *Journal of Business & Economic Statistics* 7(2), 147–159.
52. Ng, S. & Perron, P. (2001). "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica* 69(6), 1519–1554.
53. Gatev, E., Goetzmann, W. N. & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies* 19(3), 797–827.
54. Uhlenbeck, G. E. & Ornstein, L. S. (1930). "On the Theory of the Brownian Motion." *Physical Review* 36(5), 823–841.
55. Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal* 58(4), 36–52.

**Credit risk (area 06)**

56. Hosmer, D. W. & Lemeshow, S. (1980). "Goodness of Fit Tests for the Multiple Logistic Regression Model." *Communications in Statistics: Theory and Methods* A9(10), 1043–1069.
57. Hanley, J. A. & McNeil, B. J. (1982). "The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve." *Radiology* 143(1), 29–36.
58. Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.* Wiley. (WoE/IV, PSI.)
59. Vasicek, O. (2002). "The Distribution of Loan Portfolio Value." *Risk* 15(12), 160–162.
60. Gordy, M. B. (2003). "A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules." *Journal of Financial Intermediation* 12(3), 199–232.
61. Basel Committee on Banking Supervision (2005). *An Explanatory Note on the Basel II IRB Risk Weight Functions.* BIS.
62. Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital Standards.* BIS, §272–273.
63. Pykhtin, M. & Zhu, S. (2007). "A Guide to Modelling Counterparty Credit Risk." *GARP Risk Review* 37, 16–22.
64. Gregory, J. (2015). *The xVA Challenge: Counterparty Credit Risk, Funding, Collateral and Capital*, 3rd ed. Wiley.
65. Basel Committee on Banking Supervision (2013). *Supervisory Guidance for Managing Risks Associated with the Settlement of Foreign Exchange Transactions.* BIS.

**Portfolio optimisation & risk allocation (area 07)**

66. Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance* 7(1), 77–91.
67. Black, F. & Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal* 48(5), 28–43.
68. He, G. & Litterman, R. (1999). "The Intuition Behind Black-Litterman Model Portfolios." Goldman Sachs Investment Management Research.
69. Idzorek, T. (2005). "A Step-by-Step Guide to the Black-Litterman Model." Ibbotson Associates.
70. Ledoit, O. & Wolf, M. (2003). "Improved Estimation of the Covariance Matrix of Stock Returns with an Application to Portfolio Selection." *Journal of Empirical Finance* 10(5), 603–621.
71. Jorion, P. (1986). "Bayes-Stein Estimation for Portfolio Analysis." *Journal of Financial and Quantitative Analysis* 21(3), 279–292.
72. Higham, N. J. (1988). "Computing a Nearest Symmetric Positive Semidefinite Matrix." *Linear Algebra and its Applications* 103, 103–118.
73. Maillard, S., Roncalli, T. & Teïletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." *Journal of Portfolio Management* 36(4), 60–70.
74. Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights." SSRN 2297383.
75. Griveau-Billion, T., Richard, J.-C. & Roncalli, T. (2013). "A Fast Algorithm for Computing High-dimensional Risk Parity Portfolios." SSRN 2325255.

**Algorithmic execution & TCA (area 08)**

76. Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions." *Journal of Risk* 3(2), 5–39.
77. Perold, A. F. (1988). "The Implementation Shortfall: Paper versus Reality." *Journal of Portfolio Management* 14(3), 4–9.
78. Bertsimas, D. & Lo, A. W. (1998). "Optimal Control of Execution Costs." *Journal of Financial Markets* 1(1), 1–50.
79. Kissell, R. (2013). *The Science of Algorithmic Trading and Portfolio Management.* Academic Press.
80. Oomen, R. (2017). "Last Look." *Quantitative Finance* 17(7), 1057–1070.
81. Bank for International Settlements (2021). *FX Global Code*, Principle 17.

**Volatility surfaces & stochastic volatility (area 09)**

82. Gatheral, J. (2004). "A Parsimonious Arbitrage-Free Implied Volatility Parameterization with Application to the Valuation of Volatility Derivatives." Global Derivatives & Risk Management, Madrid.
83. Gatheral, J. & Jacquier, A. (2014). "Arbitrage-Free SVI Volatility Surfaces." *Quantitative Finance* 14(1), 59–71.
84. Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *Review of Financial Studies* 6(2), 327–343.
85. Albrecher, H., Mayer, P., Schoutens, W. & Tistaert, J. (2007). "The Little Heston Trap." *Wilmott*, January, 83–92.
86. Carr, P. & Madan, D. (1999). "Option Valuation Using the Fast Fourier Transform." *Journal of Computational Finance* 2(4), 61–73.
87. Andersen, L. (2008). "Simple and Efficient Simulation of the Heston Stochastic Volatility Model." *Journal of Computational Finance* 11(3), 1–42.
88. Lord, R., Koekkoek, R. & van Dijk, D. (2010). "A Comparison of Biased Simulation Schemes for Stochastic Volatility Models." *Quantitative Finance* 10(2), 177–194.
89. Castagna, A. & Mercurio, F. (2007). "The Vanna-Volga Method for Implied Volatilities." *Risk*, January, 106–111.
90. Hagan, P. S., Kumar, D., Lesniewski, A. S. & Woodward, D. E. (2002). "Managing Smile Risk." *Wilmott*, September, 84–108. (SABR: discussed as an alternative in area 09; not implemented.)

**Regime switching (area 10)**

91. Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica* 57(2), 357–384.
92. Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE* 77(2), 257–286.
93. Baum, L. E., Petrie, T., Soules, G. & Weiss, N. (1970). "A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains." *Annals of Mathematical Statistics* 41(1), 164–171.
94. Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm." *Journal of the Royal Statistical Society B* 39(1), 1–38.
95. Viterbi, A. J. (1967). "Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm." *IEEE Transactions on Information Theory* 13(2), 260–269.
96. Ang, A. & Bekaert, G. (2002). "International Asset Allocation with Regime Shifts." *Review of Financial Studies* 15(4), 1137–1187.

**Backtest discipline (areas F2, 05, 08, 10)**

97. Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2014). "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance." *Notices of the AMS* 61(5), 458–471.
98. López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley, ch. 7, 11–12.
99. Merton, R. C. (1974). "On the Pricing of Corporate Debt: The Risk Structure of Interest Rates." *Journal of Finance* 29(2), 449–470. (Structural PD: not implemented in area 06; listed for orientation.)

## Data

Every project uses synthetic data from seeded generators in
`src/*/data/synthetic.py` or similar. Some projects have optional loaders for
free public data (Yahoo Finance, FRED, ECB). Tests never need the network.

## License

MIT License, Copyright (c) 2026 Ashish Jha. See [LICENSE](LICENSE).
