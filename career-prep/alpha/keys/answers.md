# Alpha research: answer key

Questions are in [`../questions.md`](../questions.md).

- **A01** `1`. IR = IC sqrt(BR) = 0.05 x 20 (Grinold 1989). Breadth means independent bets; correlated bets count for less.
- **A02** `3/20`. 1.5/sqrt(100).
- **A03** `2`. With iid returns, t = SR_annual x sqrt(years).
- **A04** `0.1319722`. Lo (2002): SE = sqrt((1 + SR^2/2)/T) = sqrt(1.045/60).
- **A05** `2.807034`. z for a one-sided p of 0.05/20 = 0.0025.
- **A06** `2.507594`. E[max] = integral of x n phi(x) Phi(x)^(n-1) dx. About 2.5: the best of 100 noise strategies looks like a t of 2.5.
- **A07** `6.578813`. 0.9^h = 1/2, h = ln 0.5 / ln 0.9.
- **A08** `1/250`. 0.08/20 = 0.004, i.e. 40 bp a trade.
- **A09** `3.872983`. 0.03/0.06 x sqrt(60), assuming the monthly ICs are independent.
- **A10** `33/35`. Ranks (5,2,3,6,1,4) and (5,3,2,6,1,4); sum d^2 = 2; 1 - 6 x 2/(6 x 35).
- **A11** `0.04163226`. SR_hat ~ N(0, 1/3) in annual units, so P = 1 - Phi(sqrt(3)).
- **A12** `272.9071`. Bailey and Lopez de Prado (2012): MinTRL = 1 + (1 - g3 SR + (g4 - 1)/4 SR^2)(z/SR)^2; for normal returns g3 = 0, g4 = 3: 1 + 1.005 x (1.644854/0.1)^2. About 23 years.
- **A13** `0.6415141`. 1 - 0.95^20.
- **A14** `781`. 800 non-test days. Purge the 9 days 391-399 whose 10-day labels reach into the test fold; embargo the 10 days 600-609. 800 - 19 (Lopez de Prado 2018, sections 7.4.1-7.4.2).

## Discussion notes

- **AD1** Using a value before it was published (restated fundamentals, index membership as of today), using the close to trade at the same close, normalising with full-sample statistics, and survivorship in the universe.
- **AD2** The selection is part of the experiment. The deflated Sharpe ratio (Bailey and Lopez de Prado 2014) needs the number of trials; this repository records each one in a trials ledger for that reason.
- **AD3** Turnover and the horizon of the IC versus the holding period, capacity and spread at the size traded, and whether the IC sits in names you cannot trade (small caps, hard to borrow).
- **AD4** A pre-registered forward test with enough power, a structural-break test on the IC series, and whether the decay coincides with publication or crowding (McLean and Pontiff 2016).
- **AD5** When returns or signals have outliers: ranks bound the influence of a few extreme names, and the portfolio is often formed on ranks anyway.
