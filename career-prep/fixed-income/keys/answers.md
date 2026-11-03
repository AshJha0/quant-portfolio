# Fixed-income maths by hand: answer key

Questions are in [`../questions.md`](../questions.md). Each line gives the
answer in backticks, then the shortest argument we know. Rational answers
are exact; decimals follow in the prose. Each value is recomputed by a
different route in [`../verify.py`](../verify.py) and compared with this
file by `career-prep/tests/test_quant_packs.py`. If you find an error, open
an issue: the test is the arbiter, not this prose.

- **F01** `426092120/4084101`. About 104.3295: 6 a(5, 5%) + 100 v^5 with v = 1/1.05 and a(5, 5%) = (1 - v^5)/0.05 = 4.3295. Above par because the coupon beats the yield.
- **F02** `47698363/10652303`. About 4.4778 years: sum of t PV(CF_t) / price.
- **F03** `953967260/223698363`. About 4.2645: Macaulay / (1 + y).
- **F04** `36710832800/1565888541`. About 23.444: sum t(t + 1) CF_t v^{t+2} / price, the annual-compounding convexity.
- **F05** `0.0444915660812`. Modified duration x price x 0.0001 = 4.2645 x 104.33 / 10^4.
- **F22** `17682020/4084101`. About 4.3295: at 6% the bond is at par, so it falls from F01 to 100. Duration alone predicts 4.449; convexity takes back 0.122 of that.
- **F11** `277260880/2825761`. About 98.119: four coupons of 2 and the face, each discounted at 2.5% a period.
- **F12** `0.0574468791248`. Solve 5v + 5v^2 + 105v^3 = 98 for v = 1/(1 + y). Rough check: (5 + 2/3)/99 = 5.72%.
- **F13** `2384185791015625/35291773913344`. About 67.556: 100 / 1.04^10.
- **F14** `125/13`. About 9.615: a zero's Macaulay duration is its maturity, 10, divided by 1.04.
- **F21** `34375/338`. About 101.70: T(T + 1) / (1 + y)^2 = 110 / 1.0816.
- **F15** `6208620926/969359573`. About 6.4049: durations add with value weights, 0.4 x 9.6154 + 0.6 x 4.2645.
- **F06** `0.0540925533895`. 0.9^{-1/2} - 1.
- **F10** `0.0487901641694`. ln 1.05.
- **F07** `25/416`. About 6.0096%: 1.05^2 / 1.04 - 1. Investing for two years must match rolling one-year money at 4% then the forward.
- **F08** `157/175`. About 0.897143: DF1 = 0.96, then 99 = 5 DF1 + 105 DF2, so DF2 = 94.2/105.
- **F09** `18/325`. About 5.538%: c = (1 - DF2) / (DF1 + DF2).
- **F17** `3/94`. About 3.191%: (1 - DF3) / (DF1 + DF2 + DF3) = 0.09 / 2.82; a floating leg is worth par minus the final discounted principal.
- **F18** `57/25`. 2.28: 4 x 2.82 - 100 x 0.09. Receiving above the par rate is worth the rate gap times the annuity, (0.04 - 3/94) x 2.82 x 100.
- **F16** `500000/203`. About 2463.05: 1,000,000 x (0.06 - 0.05) x 0.25, discounted for the accrual period at the fixing rate, / (1 + 0.06 x 0.25).
- **F19** `0.928070164134`. Vasicek: P = exp(A - B r_0) with B = (1 - e^{-aT})/a and A = (b - sigma^2/(2a^2))(B - T) - sigma^2 B^2 / (4a).
- **F20** `935.95417601`. Black: N tau DF (F N(d1) - K N(d2)) with d1 = sigma/2 = 0.1 at the money; 1,000,000 x 0.25 x 0.94 x 0.003983.
