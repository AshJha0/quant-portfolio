# Portfolio construction: knowledge check (questions only)

Fifteen graded problems and four discussion prompts. Answers are in
[`keys/answers.md`](keys/answers.md); [`verify.py`](verify.py) reaches every
graded answer by numerical optimisation or a matrix computation rather than
the closed form in the key, and `career-prep/tests/test_numeric_packs.py`
compares them.

Unless stated: two assets with vols 10% and 20%, correlation 0.2, expected
returns 8% and 12%, risk-free rate 2%; weights sum to one; shorting allowed.
Code in this repository: `portfolio-construction` style optimisers live in
`python/quant_portfolio/portfolio/`. References: Markowitz (1952); Grinold
and Kahn, *Active Portfolio Management* (2nd ed.); Maillard, Roncalli and
Teiletche (2010) on risk parity; Kelly (1956); Thorp (2006).

## Mean-variance

- **O01** Weight of asset 1 in the minimum-variance portfolio?
- **O02** Volatility of that portfolio?
- **O03** Weight of asset 1 in the tangency (maximum Sharpe) portfolio?
- **O04** Sharpe ratio of the tangency portfolio?
- **O13** Diversification ratio (weighted average vol over portfolio vol) of the O01 portfolio?

## Many assets

- **O05** Ten assets, each 20% vol, all pairwise correlations 0.3, equal weights. Portfolio volatility?
- **O06** Same, as the number of assets grows without bound (give the limit)?
- **O14** Weights 0.5, 0.3, 0.2. Effective number of positions, 1 / sum of squared weights?

## Sizing

- **O07** Continuous-time Kelly fraction for an asset with 6% excess return and 20% vol?
- **O08** A bet pays even money and wins with probability 0.6. Kelly fraction of wealth?

## Risk budgeting

- **O09** Equal risk contribution weights for the two base assets: weight of asset 1?
- **O10** Three assets with vols 10%, 20%, 30% and correlations rho12 = 0.5, rho13 = 0.2, rho23 = 0.4. Equal-risk-contribution weight of asset 1 (long-only, sum to one)?

## Active management

- **O11** Active weights +0.1 in asset 1 and -0.1 in asset 2. Tracking error?
- **O12** Active return 1.5% a year with 3% tracking error. Information ratio?
- **O15** Monthly mean excess return 1%, monthly sd 4%. Annualised Sharpe ratio (iid months)?

## Discussion (not graded)

- **OD1** Why do unconstrained mean-variance weights swing so much with small changes in expected returns, and what do practitioners do about it?
- **OD2** When does inverse-volatility weighting equal risk parity?
- **OD3** Why do most practitioners bet a fraction of Kelly?
- **OD4** Your backtested optimiser beats equal weight before costs and loses after. What do you look at?
