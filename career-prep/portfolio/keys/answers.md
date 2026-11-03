# Portfolio construction: answer key

Questions are in [`../questions.md`](../questions.md). Covariance of the
base pair: var1 = 0.01, var2 = 0.04, cov = 0.2 x 0.1 x 0.2 = 0.004.

- **O01** `6/7`. w1 = (var2 - cov)/(var1 + var2 - 2 cov) = 0.036/0.042.
- **O02** `0.09561829`. w'Sigma w with w = (6/7, 1/7): (36 x 0.01 + 1 x 0.04 + 12 x 0.004)/49 = 0.448/49; sqrt is 9.56%, below the lower single-asset vol.
- **O03** `50/69`. w is proportional to Sigma^-1 (mu - rf) = Sigma^-1 (0.06, 0.10) = (5.0, 1.9) after scaling; 5/6.9.
- **O04** `0.7144345`. sqrt(e' Sigma^-1 e) with e = (0.06, 0.10): sqrt(0.5104).
- **O05** `0.1216553`. sigma sqrt(rho + (1 - rho)/N) = 0.2 sqrt(0.3 + 0.07).
- **O06** `0.1095445`. 0.2 sqrt(0.3): correlation sets a floor that diversification cannot remove.
- **O07** `3/2`. Maximise f mu - f^2 sigma^2/2: f* = mu/sigma^2 = 0.06/0.04.
- **O08** `1/5`. f* = p - q = 2p - 1 for an even-money bet.
- **O09** `2/3`. For two assets, inverse-vol weights (1/0.1 : 1/0.2) equalise risk contributions for any correlation, because both cross terms are the same w1 w2 cov.
- **O10** `0.5551401`. No closed form; solve w_i (Sigma w)_i = w_j (Sigma w)_j. The key value is from Newton's method on min y'Sigma y/2 - sum ln(y_i)/3, normalised (Spinu 2013); weights (0.5551, 0.2528, 0.1921), each contributing 0.005125 of variance.
- **O11** `0.02049390`. a'Sigma a = 0.01 x 0.01 + 0.01 x 0.04 - 2 x 0.01 x 0.004 = 0.00042.
- **O12** `1/2`. IR = active return / tracking error.
- **O13** `1.195229`. (6/7 x 0.1 + 1/7 x 0.2)/0.0956183 = 0.1142857/0.0956183.
- **O14** `50/19`. 1/(0.25 + 0.09 + 0.04) = 1/0.38, about 2.63.
- **O15** `0.8660254`. 0.01/0.04 x sqrt(12). The rule assumes no autocorrelation (Lo 2002).

## Discussion notes

- **OD1** Weights depend on Sigma^-1 mu; small errors in mu are amplified along low-variance directions (Michaud 1989; Best and Grauer 1991). Remedies: shrink mu and Sigma (Ledoit and Wolf 2004), Black-Litterman, constraints, or drop mu and use minimum variance or risk parity.
- **OD2** With two assets always (O09); with more, only when all pairwise correlations are equal.
- **OD3** Full Kelly assumes the edge is known; with estimation error the true growth-optimal fraction is smaller, and half Kelly gives about three quarters of the growth with half the volatility (Thorp 2006; MacLean, Thorp and Ziemba 2011).
- **OD4** Turnover and its source (noisy inputs), the cost model's realism against fills, whether the backtest leaked future data into the covariance, and whether the edge survives a shrinkage or turnover penalty.
