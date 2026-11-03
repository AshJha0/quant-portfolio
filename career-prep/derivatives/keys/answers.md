# Derivatives and products: answer key

Questions are in [`../questions.md`](../questions.md). Each line gives the
answer in backticks, then the shortest argument we know. In the base case
d1 = (ln(S/K) + (r + sigma^2/2) T) / (sigma sqrt T) = 0.35 and
d2 = d1 - sigma sqrt T = 0.15. Each value is recomputed without these
formulas by [`../verify.py`](../verify.py) and compared with this file by
`career-prep/tests/test_quant_packs.py`. If you find an error, open an
issue: the test is the arbiter, not this prose.

- **D01** `10.4505835722`. S N(d1) - K e^{-rT} N(d2).
- **D02** `5.57352602226`. K e^{-rT} N(-d2) - S N(-d1); parity check: C - P = S - K e^{-rT} = 4.877.
- **D06** `103.045453395`. S e^{(r - q) T} = 100 e^{0.03}.
- **D07** `0.532324815454`. e^{-rT} N(d2): the discounted risk-neutral probability of finishing in the money.
- **D08** `63.6830651176`. S N(d1). The call is this minus 100 times D07.
- **D09** `0.241116893682`. Solve BS(sigma) = 12; the price rises monotonically in volatility, so a bracketed root-finder converges. Vega of about 38 per 1.00 says a 1.55 price gap is about 4 vol points.
- **D12** `16.0241095944`. Call plus put, D01 + D02.
- **D13** `5.32511271074`. C(95) - C(105).
- **D03** `0.636830651176`. N(d1).
- **D04** `0.0187620173458`. phi(d1) / (S sigma sqrt T).
- **D22** `0.0187620173458`. Put-call parity: P = C - S + K e^{-rT}, and the last two terms are linear in S, so the second derivatives agree.
- **D05** `37.5240346917`. S phi(d1) sqrt T. Per vol point that is 0.375.
- **D14** `0.273661302351`. N(d1) + (N(d1) - 1) = 2 N(d1) - 1.
- **D19** `-6.41402754644`. -S phi(d1) sigma / (2 sqrt T) - r K e^{-rT} N(d2). About -1.76 cents a calendar day.
- **D20** `53.2324815454`. K T e^{-rT} N(d2).
- **D10** `21/4`. Risk-neutral up probability (1 - 0.9)/(1.1 - 0.9) = 1/2. Only up-up pays (121 - 100 = 21), with probability 1/4.
- **D21** `400/49`. p = (1.05 - 0.9)/0.2 = 3/4. Only up-up pays 16, so the price is (9/16)(16)/1.05^2 = 9/1.1025.
- **D11** `5`. Exercise at once: holding is worth 4.59 (the down node is exercised for 15, the up node is held for 1.43), less than the 5 from exercising now. Early exercise of a put pays when the strike's interest outweighs the remaining optionality.
- **D15** `10.5243157811`. Margrabe: price with the second asset as numeraire, effective vol sqrt(0.04 + 0.09 - 2(0.5)(0.2)(0.3)) = sqrt(0.07); value 100 (N(sigma/2) - N(-sigma/2)).
- **D16** `13.8513299817`. max(C, P) at the choice date equals C + (K e^{-r(T - t)} - S_t)^+ by parity, so the chooser is a one-year call plus a six-month put struck at 100 e^{-0.025}.
- **D17** `0.55961769237`. N(d2).
- **D18** `0.65542174161`. N(d2) with r replaced by the real-world drift 10%: d2 = 0.40.
