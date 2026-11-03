# Risk: knowledge check (questions only)

Twenty graded problems on market and credit risk, plus four discussion
prompts. Answers and worked solutions are in [`keys/answers.md`](keys/answers.md).
Every graded answer is recomputed by [`verify.py`](verify.py) by a different
route (root-finding, numerical integration, finite differences, enumeration
or a seeded simulation), and `career-prep/tests/test_numeric_packs.py` fails
if the key and the check disagree.

Conventions, unless a problem says otherwise: P&L is normally distributed
with zero mean; VaR and ES are reported as positive loss amounts; "99% VaR"
is the 1% quantile of the loss; days are independent. Give money to the
nearest unit, ratios to four significant figures.

Background for the code in this repository: `python/quant_portfolio/risk/`
(VaR, ES and backtests). Reference texts: Jorion, *Value at Risk* (3rd ed.);
Hull, *Risk Management and Financial Institutions*; McNeil, Frey and
Embrechts, *Quantitative Risk Management*; Basel Committee (1996),
*Supervisory framework for the use of backtesting*.

## VaR and expected shortfall

- **K01** A $10m position has daily P&L volatility of 1% of notional. What is its one-day 99% VaR?
- **K02** Same position: one-day 97.5% expected shortfall?
- **K03** Scale K01 to a ten-day horizon with the square-root-of-time rule.
- **K04** Two positions of $5m each, daily vols 1% and 2%, correlation 0.3. One-day 99% portfolio VaR?
- **K05** In K04, by how much does the sum of the two stand-alone VaRs exceed the portfolio VaR (the diversification benefit)?
- **K06** In K04, what is the Euler (component) VaR of the second position?
- **K07** Twenty daily P&Ls (in $m): -1.2, 0.4, -3.5, 2.1, 0.9, -0.3, -2.2, 1.5, -4.1, 0.2, 0.7, -0.8, 1.1, -1.9, 0.0, 3.0, -0.6, -2.7, 0.5, 1.8. Historical 90% ES, defined as the mean of the worst 10% of losses?
- **K13** A long position of one share at $100, log-return over one day N(0, 0.02^2). One-day 99% VaR in dollars?
- **K17** For normal P&L, what is the ratio of 99% ES to 99% VaR?
- **K20** For normal P&L, what is the ratio of 97.5% VaR to 99% VaR?
- **K18** Two independent bonds each lose 100 with probability 4%, else 0. Each one's 95% VaR is 0. What is the 95% VaR of holding both?

## Backtesting

- **K08** A 99% VaR model sees 6 exceptions in 250 days. Kupiec's proportion-of-failures likelihood-ratio statistic?
- **K09** Basel traffic light with 250 days at 99%: the yellow zone starts at the smallest exception count whose binomial cumulative probability is at least 95%. What count is that?

## Rates and options

- **K10** A 10-year zero-coupon bond, annually compounded yield 5%, face 100. DV01 (price change for a 1bp fall in yield)?
- **K11** Same bond: the duration-plus-convexity estimate of the relative price change for a +100bp move?
- **K12** 100 options, each with delta 0.6 and gamma 0.05 per $1. The stock rises $2. Delta-gamma P&L?

## Volatility and credit

- **K19** EWMA with lambda 0.94. Yesterday's volatility estimate is 1%; today's return is 3%. Updated volatility?
- **K14** Expected loss for EAD $1m, PD 2%, LGD 60%?
- **K15** Two independent loans, each with PD 5%. Probability at least one defaults?
- **K16** Vasicek one-factor model, PD 1%, asset correlation 0.2. Worst-case default rate at 99.9%?

## Discussion (not graded)

- **KD1** Why is VaR not subadditive in general, and why did Basel move the trading book from VaR to ES?
- **KD2** When does the square-root-of-time rule mislead?
- **KD3** A desk's VaR fell after it added a position. Name two ways that can happen without a bug.
- **KD4** What would you check first if a VaR backtest shows zero exceptions in a year?
