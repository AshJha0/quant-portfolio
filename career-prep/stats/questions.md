# Statistics and econometrics: knowledge check (questions only)

Twenty-five problems, written fresh for this repository, on the statistics a
quant researcher or risk analyst uses every week: estimators and their
standard errors, regression, time-series persistence, multiple testing,
goodness of fit and simple Bayesian updating. The answers and worked
solutions are in [`keys/answers.md`](keys/answers.md). Work each one on
paper first; most take under five minutes.

Every answer in the key is recomputed from the question's own data by
[`verify.py`](verify.py), and the cheap ones are also simulated with a fixed
seed. The CI test `career-prep/tests/test_stats_pack.py` fails if a key and
its check disagree.

Tags: `[est]` estimators, `[test]` hypothesis tests, `[reg]` regression,
`[ts]` time series, `[mult]` multiple testing, `[bayes]`, `[fin]` finance
conventions.

Where a numeric answer is not a simple fraction, give it to about four
significant figures; the key carries more digits only so the check can be
strict. Use z = 1.959964 for a two-sided 95% normal quantile.

## Estimators and standard errors

- **S01** `[est]` A sample is 2, 4, 4, 6. What is the unbiased sample variance?
- **S02** `[est]` A sample of 100 daily P&L figures has sample standard deviation 20. Standard error of the sample mean?
- **S12** `[est]` Waiting times 1, 2, 3 and 6 minutes are modelled as iid exponential with rate lambda. Maximum-likelihood estimate of lambda?
- **S16** `[est]` Var(X) = 4, Var(Y) = 9, Corr(X, Y) = 0.5. Var(X + Y)?
- **S17** `[est]` Ten returns each have variance 1 and every pair has correlation 0.2. Variance of their average?
- **S23** `[est]` A bootstrap resample draws 10 times with replacement from 10 observations. Probability a given observation is never drawn?
- **S24** `[est]` A sample correlation of 0.5 is measured on 103 pairs. Using the Fisher z-transform, the lower end of the two-sided 95% confidence interval?

## Hypothesis tests and power

- **S03** `[test]` A coin lands heads 60 times in 100 flips. The z statistic for H0: p = 1/2 (normal approximation, no continuity correction)?
- **S04** `[test]` The two-sided p-value for z = 2?
- **S05** `[test]` Smallest sample size for which a 95% confidence interval for a proportion near 1/2 has half-width at most 0.01?
- **S15** `[test]` A one-sided z-test at the 5% level. The true effect is 2.5 standard errors. Power?
- **S19** `[test]` A die is rolled 60 times; the faces come up 5, 8, 9, 8, 10 and 20 times. Pearson chi-square statistic against a fair die?
- **S20** `[test]` 100 returns have sample skewness 0.5 and excess kurtosis 1. Jarque-Bera statistic?

## Regression

- **S06** `[reg]` OLS with intercept on the points (1, 2), (2, 3), (3, 5), (4, 6). Slope?
- **S07** `[reg]` R-squared of the fit in S06?

## Time series

- **S10** `[ts]` AR(1): x_t = 0.5 x_{t-1} + e_t with Var(e_t) = 1. Unconditional variance of x_t?
- **S11** `[ts]` AR(1) with coefficient 0.8. Half-life of a shock, in periods?
- **S18** `[ts]` 100 observations from a stationary AR(1) with coefficient 0.5. Effective sample size for estimating the mean (large-sample formula)?

## Multiple testing

- **S13** `[mult]` 20 strategies are tested and the family-wise error rate must stay at 5%. Bonferroni per-test level?
- **S14** `[mult]` 20 independent tests of true null hypotheses at the 5% level. Probability at least one rejects?

## Bayesian updating

- **S21** `[bayes]` A uniform prior on a hit rate p; 7 hits in 10 trials. Posterior mean of p?
- **S22** `[bayes]` Prior mu ~ N(0, 1); one observation x = 2 with x | mu ~ N(mu, 1). Posterior mean of mu?

## Finance conventions

- **S08** `[fin]` Daily returns are iid with standard deviation 1%. Annualised volatility over 252 trading days?
- **S09** `[fin]` A strategy shows an annualised Sharpe ratio of 1 over 4 years of iid returns. Standard error of the Sharpe estimate, using Lo's (2002) formula sqrt((1 + SR^2 / 2) / T) with T in years?
- **S25** `[fin]` A price rises 10% and then falls 10%. Total simple return?
