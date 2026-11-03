# Probability and puzzles: answer key

Questions are in [`../questions.md`](../questions.md). Each line gives the
answer in backticks, then the shortest argument we know. Each value is
recomputed by [`../verify.py`](../verify.py) (the "Check" column of
`python verify.py`) and compared with this file by
`career-prep/tests/test_probability_pack.py`. If you find an error, open an
issue: the test is the arbiter, not this prose.

- **P01** `7/2`. (1+...+6)/6.
- **P02** `161/36`. P(max = k) = (2k-1)/36; sum k(2k-1)/36 = 161/36, about 4.47.
- **P03** `6`. Geometric with p = 1/6 has mean 1/p.
- **P04** `147/10`. Coupon collector: 6(1 + 1/2 + ... + 1/6) = 14.7.
- **P05** `6`. States {start, H}: e0 = 1 + (e_H + e0)/2, e_H = 1 + e0/2, so e0 = 6.
- **P06** `4`. After the first H you only wait for a T (mean 2): 2 + 2. HH is slower because a failed attempt (HT) throws away the H.
- **P07** `1/2`. Both need an H first; the next flip decides, each with probability 1/2.
- **P08** `7/8`. HHH wins only if the first three flips are HHH (1/8); any earlier T means THH must come first.
- **P09** `1/3`. Of BB, BG, GB (equally likely), one is BB.
- **P10** `2/3`. Switching wins exactly when the first pick was wrong.
- **P11** `1/6`. 0.0099 / (0.0099 + 0.0495). About 17%: false positives from the 99% healthy swamp true ones.
- **P12** `3/10`. A fair walk is a martingale; optional stopping gives P = start/target.
- **P13** `41553/58025`. With r = q/p = 2/3: (1 - r^3)/(1 - r^10), about 0.716.
- **P14** `21`. Expected duration of a fair game is a(N - a) = 3 x 7.
- **P15** `25`. S_n^2 - n is a martingale; E[T] = 5^2.
- **P16** `7381/252`. 10 H_10, about 29.29.
- **P17** `23`. P(no match) first falls below 1/2 at n = 23.
- **P18** `0.5072972343`. 1 - prod_{i<23} (365 - i)/365.
- **P19** `1`. Linearity: each item is fixed with probability 1/6, six items.
- **P20** `2119/5760`. Derangements D_8 = 14833 over 8! = 40320; close to 1/e.
- **P21** `3/4`. E[max of n uniforms] = n/(n+1).
- **P22** `1/4`. E[min] = 1/(n+1).
- **P23** `1/3`. E[U_(k)] = k/(n+1) = 2/6.
- **P24** `1/4`. Each piece must be under 1/2; the excluded regions are three triangles of area 1/4 each in the simplex.
- **P25** `3/4`. Integral of max(u, 1 - u) over [0,1].
- **P26** `1/3`. Integral of |u - v| over the unit square.
- **P27** `2.718281828`. P(N > n) = P(sum of n uniforms < 1) = 1/n!, so E[N] = sum 1/n! = e.
- **P28** `0.2231301601`. e^{-3 x 0.5} = e^{-1.5}.
- **P29** `1/16`. Given the count, arrival times are iid uniform: (1/2)^4.
- **P30** `2/5`. lambda_1/(lambda_1 + lambda_2).
- **P31** `1/5`. The minimum is exponential with rate 5.
- **P32** `10`. Memoryless.
- **P33** `20`. Length-biased sampling: E[G^2]/E[G] = 2/rate^2 x rate = 20 for an exponential gap.
- **P34** `3/4`. Balance: pi_0 x 0.3 = pi_1 x 0.1.
- **P35** `4`. Kac: 1/pi_0 = 1/(1/4).
- **P36** `16`. Reflecting at 0, absorbing at N, start at 0: N^2.
- **P37** `11/32`. Reflection: P(M >= 3) = P(S >= 3) + P(S > 3) = 2 P(S_10 >= 4) = 2 x 176/1024.
- **P38** `1/5`. Ballot theorem: (a - b)/(a + b) = 2/10.
- **P39** `42`. Catalan number C_5.
- **P40** `2/3`. Optional stopping: p(+1) + (1 - p)(-2) = 0.
- **P41** `21`. Wald: E[N] E[X] = 6 x 3.5.
- **P42** `1/11`. The number of reds added is uniform on {0,...,10}.
- **P43** `1`. Geometric on {0,1,...} with p = 1/2 has mean q/p = 1.
- **P44** `11/2`. 1 + 9 x P(adjacent flips differ) = 1 + 9/2.
- **P45** `9/4`. 9 adjacent pairs, each HH with probability 1/4.
- **P46** `1/221`. C(4,2)/C(52,2) = 6/1326.
- **P47** `33/16660`. 4 C(13,5) / C(52,5) = 5148/2598960.
- **P48** `6/4165`. 13 C(4,3) x 12 C(4,2) / C(52,5) = 3744/2598960.
- **P49** `6/11`. Ignore other totals: 6/(6 + 5).
- **P50** `244/495`. 8/36 plus sum over points of P(point)^2/(P(point) + 1/6); about 0.4929.
- **P51** `31031/7776`. 6(1 - (5/6)^6), about 3.99.
- **P52** `1/5`. f* = p - q = 0.6 - 0.4 for even odds.
- **P53** `17/4`. Reroll on 1-3 (below 3.5): (4 + 5 + 6)/6 + 3/6 x 3.5.
- **P54** `14/3`. With one reroll left the game is worth 4.25, so keep 5 and 6: (5 + 6)/6 + 4/6 x 4.25.
- **P55** `0.3986904762`. Skip the first 3, then take the first record: (3/10)(1/3 + ... + 1/9).
- **P56** `0.5641895835`. E[max(X,Y)] = E|X - Y|/2 = sqrt(2)/sqrt(2 pi) = 1/sqrt(pi).
- **P57** `1/3`. Sheppard: 1/4 + arcsin(rho)/(2 pi) = 1/4 + 1/12.
- **P58** `0.7978845608`. sqrt(2/pi).
- **P59** `0.3173105079`. Reflection: 2 P(W_1 >= 1) = 2(1 - Phi(1)).
- **P60** `14`. Expected flips for n heads in a row is 2^{n+1} - 2.
