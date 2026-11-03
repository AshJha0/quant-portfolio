# Mock round 3: answer key

Questions are in [`../round-3.md`](../round-3.md). Each graded answer
is recomputed by [`../verify.py`](../verify.py).

## Graded

- **M3.1** `465269.574808`. z_0.99 x 0.008 x 25m = 2.3263 x 200,000.
- **M3.2** `23378.027922`. ES = sigma phi(z) / (1 - alpha) = 10,000 x phi(1.95996) / 0.025.
- **M3.3** `0.111803398875`. sqrt(0.25 x 0.01 + 0.25 x 0.04) = sqrt(0.0125).
- **M3.4** `0.0411831840698`. Binomial(250, 0.01): 1 - P(X <= 5). Under the Basel traffic light, five to nine exceptions is the yellow zone.

## Discussion rubric

- **M3.5** Two independent loans each defaulting with 4% probability: each 95% VaR is zero, but the pair has a 7.8% chance of at least one default, so its VaR is positive. It matters because limits set on VaR can be gamed by splitting books; ES is subadditive.
- **M3.6** No: the model is too conservative or the P&L is not the one the risk is measured on (hypothetical versus actual P&L). Strong answers mention capital cost and checking exception clustering too.
