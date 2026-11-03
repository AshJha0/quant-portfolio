# Risk: answer key

Questions are in [`../questions.md`](../questions.md). Each graded line gives
the answer in backticks, then the shortest argument. `../verify.py`
recomputes every value a different way, and
`career-prep/tests/test_numeric_packs.py` compares the two (relative
tolerance 1e-5, so keys carry six or more significant figures). Here
z_0.99 = 2.326348, z_0.975 = 1.959964, phi is the standard normal density.

- **K01** `232634.8`. z_0.99 x sigma x notional = 2.326348 x 0.01 x 10m.
- **K02** `233780.3`. For normal P&L, ES_a = sigma phi(z_a)/(1 - a) = 100k x 0.0584451/0.025.
- **K03** `735655.8`. 232634.8 x sqrt(10). It assumes iid days and no change in the position.
- **K04** `289628.0`. Dollar sds 50k and 100k; sigma_p^2 = 50k^2 + 100k^2 + 2(0.3)(50k)(100k) = 1.55e10, so sigma_p = 124499.0 and VaR = 2.326348 sigma_p.
- **K05** `59324.21`. Stand-alone VaRs 116317.4 + 232634.8 = 348952.2, less 289628.0.
- **K06** `214885.3`. Component_i = x_i (Sigma x)_i / sigma_p x z. (Sigma x)_2 = 1e10 + 0.3 x 50k x 100k = 1.15e10; 1.15e10/124499 x 2.326348. Components add to the portfolio VaR (Euler's theorem for a homogeneous risk measure).
- **K07** `3.8`. Worst 10% of 20 is two losses: 4.1 and 3.5, mean 3.8 ($m).
- **K08** `3.555355`. LR = -2 ln[0.99^244 0.01^6 / ((244/250)^244 (6/250)^6)]. Below the chi-square(1) 95% point of 3.84, so not rejected at 5%.
- **K09** `5`. P(X <= 4) = 0.8922 and P(X <= 5) = 0.9588 for X ~ Bin(250, 0.01). This matches the Basel (1996) table: green 0-4, yellow 5-9, red 10+.
- **K10** `0.05846793`. P = 100/1.05^10 = 61.39133; modified duration 10/1.05 = 9.523810; DV01 = P x D x 0.0001.
- **K11** `-0.09024943`. -D dy + C dy^2/2 with D = 9.523810, C = T(T+1)/(1+y)^2 = 99.77324: -0.0952381 + 0.0049887. The exact change is (1.05/1.06)^10 - 1 = -0.0904640.
- **K12** `130`. 100 x (0.6 x 2 + 0.5 x 0.05 x 2^2) = 100 x 1.3.
- **K13** `4.546117`. 100(1 - exp(-2.326348 x 0.02)). Slightly below the normal answer 4.6527 because the lognormal loss is bounded.
- **K14** `12000`. EL = EAD x PD x LGD.
- **K15** `0.0975`. 1 - 0.95^2.
- **K16** `0.1455253`. WCDR = N((N^-1(PD) + sqrt(rho) N^-1(0.999))/sqrt(1 - rho)) = N((-2.326348 + 1.381977)/0.894427) = N(-1.055818). This is the Basel IRB formula (Vasicek 2002; Gordy 2003).
- **K17** `1.145665`. phi(z)/(0.01 z) = 0.02665214/0.02326348.
- **K18** `100`. P(loss = 0) = 0.96^2 = 0.9216 < 0.95, so the 95% quantile of loss is 100. Stand-alone VaRs sum to 0: VaR is not subadditive here.
- **K19** `0.01216553`. sigma^2 = 0.94 x 0.0001 + 0.06 x 0.0009 = 0.000148 (RiskMetrics 1996).
- **K20** `0.8425068`. 1.959964/2.326348. Basel's FRTB 97.5% ES and the old 99% VaR are close for normal tails (K02 vs K01) and differ for fat tails.

## Discussion notes

- **KD1** A quantile ignores what happens beyond it, so merging two books with rare, separate losses can push the joint quantile into the loss region (K18). ES averages the tail and is coherent (Artzner et al. 1999; Acerbi and Tasche 2002); FRTB (BCBS 2019) adopted 97.5% ES.
- **KD2** With autocorrelated returns, volatility clustering, fat tails that do not aggregate to normal over the horizon, or a position that will be traded during the horizon.
- **KD3** A hedge with negative correlation to the book; or a change in the covariance window that dropped a volatile day on the same run.
- **KD4** Whether the P&L used is the clean (hypothetical) P&L of the VaR positions, whether the model is far too conservative (also a failure under Basel), and whether the exception count is being computed on the right sign.
