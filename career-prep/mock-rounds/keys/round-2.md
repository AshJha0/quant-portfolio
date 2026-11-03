# Mock round 2: answer key

Questions are in [`../round-2.md`](../round-2.md). Each graded answer
is recomputed by [`../verify.py`](../verify.py).

## Graded

- **M2.1** `7.96556745541`. With r = 0, d1 = 0.1 and d2 = -0.1, so C = 100 (N(0.1) - N(-0.1)) = 100 (2 N(0.1) - 1). The rule of thumb 0.4 sigma sqrt(T) S gives 8.0.
- **M2.2** `0.539827837277`. With r = 0 it is N(-d2) = N(0.1). Above one half because the lognormal median sits below the forward.
- **M2.3** `1/18`. DF1 / DF2 - 1 = 0.95/0.90 - 1 = 1/18, about 5.56%.
- **M2.4** `0.797884560803`. sqrt(T / (2 pi)) with T = 4: 2 / sqrt(2 pi). This is the at-the-money call on a Bachelier asset.

## Discussion rubric

- **M2.5** Delta is the density at the strike, which spikes as time runs out. Desks over-hedge by replicating with a tight call spread, priced with the smile slope (the digital is a call-spread limit).
- **M2.6** Closed-form limits, put-call parity, a second independent method (tree, PDE or Monte Carlo), Greeks by bump versus analytic, and comparison with a trusted library on a grid. A strong answer mentions boundary cases (zero vol, deep in or out of the money).
