# Mock round 1: answer key

Questions are in [`../round-1.md`](../round-1.md). Each graded answer
is recomputed by [`../verify.py`](../verify.py).

## Graded

- **M1.1** `161/36`. P(max <= k) = k^2/36, so P(max = k) = (2k - 1)/36 and E = sum k(2k - 1)/36 = 161/36, about 4.47.
- **M1.2** `6`. States: no progress (E0), one head (E1). E0 = 1 + (E0 + E1)/2, E1 = 1 + E0/2, so E0 = 6.
- **M1.3** `0.707106781187`. Cov(X, X + Y) = Var X; Var(X + Y) = 2 Var X; correlation = 1/sqrt(2).
- **M1.4** `2`. t = mean / (sd / sqrt n) = (SR / sqrt 252) sqrt(4 x 252) = SR sqrt(years) = 2.

## Discussion rubric

- **M1.5** Survivorship bias: firms that failed or were dropped are missing, so past returns look better. A strong answer says how to fix it (point-in-time membership) and that the error is largest for small-cap and value tilts.
- **M1.6** Use the fundamental law, IR roughly IC sqrt(breadth) = 0.02 sqrt(6000) = 1.5 a year before costs; then ask about turnover, costs, capacity and decay. Strong answers question whether the 500 bets are independent.
