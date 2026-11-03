# Derivatives and products: knowledge check (questions only)

Twenty-two problems on option pricing, Greeks, trees and simple exotics,
written fresh for this repository in the style of Joshi, Denson and Downes
(2008), Crack (*Heard on the Street*) and Hull (*Options, Futures, and Other
Derivatives*, chs. 13-19). Answers are in [`keys/answers.md`](keys/answers.md).

The key uses the Black-Scholes closed forms. [`verify.py`](verify.py) never
calls them: it integrates each payoff against the risk-neutral lognormal
density, takes Greeks as pathwise derivatives or extrapolated finite
differences of that integral, enumerates trees path by path, and adds a
seeded Monte Carlo where cheap. `career-prep/tests/test_quant_packs.py`
fails if the key and a check disagree.

**Base case** unless a problem says otherwise: spot S = 100, strike K = 100,
risk-free rate r = 5% continuously compounded, volatility 20%, one year to
expiry, no dividends, European exercise. Give numbers to at least 10
significant figures when the answer is not a simple fraction.

Tags: `[bs]` Black-Scholes, `[greek]`, `[tree]` binomial trees, `[exotic]`,
`[measure]` probabilities under Q and P.

## Prices

- **D01** `[bs]` Price of the call.
- **D02** `[bs]` Price of the put.
- **D06** `[bs]` Forward price for one year if the stock pays a 2% continuous dividend yield.
- **D07** `[exotic]` Price of a cash-or-nothing digital call paying 1 if S_T > 100.
- **D08** `[exotic]` Price of an asset-or-nothing call paying S_T if S_T > 100.
- **D09** `[bs]` The base-case call trades at 12. What is its implied volatility?
- **D12** `[bs]` Price of the at-the-money straddle.
- **D13** `[bs]` Price of the 95/105 bull call spread.

## Greeks

- **D03** `[greek]` Delta of the call.
- **D04** `[greek]` Gamma of the call (per unit of spot).
- **D22** `[greek]` Gamma of the put. Why is it the same as D04?
- **D05** `[greek]` Vega of the call, per 1.00 (100 points) of volatility.
- **D14** `[greek]` Delta of the straddle.
- **D19** `[greek]` Theta of the call, per year of calendar time (the change in value as time passes, all else fixed).
- **D20** `[greek]` Rho of the call, per 1.00 of rate.

## Trees

- **D10** `[tree]` Two-step tree, S = 100, up 1.1, down 0.9, zero interest. Price of the call struck at 100.
- **D21** `[tree]` Same tree, but each step grows money by 5% (gross 1.05). Price of the European call struck at 105.
- **D11** `[tree]` Same tree as D21. Price of the American put struck at 105. Where is it optimal to exercise?

## Exotics and measures

- **D15** `[exotic]` Exchange option paying (S1 - S2)^+ at one year, both stocks at 100, vols 20% and 30%, correlation 0.5, no dividends.
- **D16** `[exotic]` Simple chooser: at six months the holder picks a call or a put, both struck at 100 and expiring at one year. Price today.
- **D17** `[measure]` Risk-neutral probability that the call expires in the money.
- **D18** `[measure]` Real-world probability of the same event if the stock's expected return is 10%.
