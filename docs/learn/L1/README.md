# L1: foundations

## Plain idea

Everything above L1 is built from four things: returns, volatility, quantiles
and sampling error. A return says how much a price moved relative to where it
was. Volatility says how much returns usually move. A quantile picks the
return that only a given fraction of days are worse than. Sampling error says
how much any of these would change if history had dealt you a different set
of days.

## Formal statement

- Simple return `R_t = P_t / P_{t-1} - 1`; log return `r_t = ln(P_t / P_{t-1})`.
  Log returns add over time; simple returns add across assets in a portfolio.
- Volatility over `h` days scales as `sigma_h = sigma_1 sqrt(h)` when daily
  returns are independent with constant variance. The annualised figure uses
  252 trading days.
- Historical VaR at confidence `c` on `n` losses is the order statistic
  `L(m)` with `m = ceil(n (1 - c))`, the `m`-th largest loss (type 1, no
  interpolation). This is the repository's single convention; see
  [MARKET_RISK.md](../../MARKET_RISK.md) and appendix Primer B.
- A Monte Carlo estimate's standard error falls as `1 / sqrt(N)`.

## Worked example

1. A stock goes up 10 % and then down 10 %. In simple returns the two-day
   result is `1.10 x 0.90 - 1 = -1 %`, not zero. In log returns the two moves
   are `+0.09531` and `-0.10536`, which add to `-0.01005`, and
   `exp(-0.01005) - 1 = -1 %`. Same answer, different bookkeeping.
2. A daily volatility of 1.26 % annualises to `1.26 % x sqrt(252) = 20.0 %`.
3. With 1,000 days of P&L, 99 % historical VaR is the 10th worst loss
   (`ceil(1000 x 0.01) = 10`). On the seed-0 standard-normal sample used in
   [MARKET_RISK.md](../../MARKET_RISK.md) that loss is 2.506.
4. To halve a Monte Carlo standard error you need four times the paths.

Source: items 1, 2 and 4 are arithmetic checked in
`docs/learn/tools/test_worked_examples.py::test_l1_returns_and_scaling`;
item 3 is quoted from `docs/MARKET_RISK.md` and the order-statistic rule is
`var_historical` in
`python/foundations/01-risk-metrics/src/eq_risk_metrics/var_es.py`.

## Checks

1. Why can you not average the simple returns of two days to get the
   two-day return?
2. With 250 days, which loss is the 99 % historical VaR?
3. Square-root-of-time scaling assumes two things about daily returns.
   Which?

Answers: compounding, `(1 + R1)(1 + R2) - 1`; the 3rd worst
(`ceil(2.5) = 3`); independence and constant variance.

Next: [L1 questions](QUESTIONS.md) and the
[VaR notebook](../../notebooks/02-value-at-risk-exercise.ipynb). Then
[L2](../L2/README.md).
