# research/vix: VIX against realised variance (W23 part 3, study E)

A descriptive study. The question: how far above the variance that then
happened does the market's quote for 30-day variance sit? It uses no option
prices, only the CBOE's published closing VIX (`cboe:VIX` through `qp_data`,
no key) and the daily US market return from the Kenneth French library. The
French market is the CRSP value-weighted market, broader than the S&P 500
the VIX is built on, so the comparison is an approximation and is labelled
as one. No trading rule is tested.

| File | What it does |
|---|---|
| `vix_rv.py` | Implied variance `(VIX/100)^2`, forward 21-day realised variance, the variance premium with a Newey-West t-statistic at lag 20, and the Mincer-Zarnowitz regression. Inputs pass the hold-out lock. |
| `tests/test_vix_rv.py` | The docstring's worked example (VIX 20 and +/-1 % days give a premium of 0.0148), the no-look-ahead window, the regression against `numpy.polyfit`, the lock. |

Reference: Carr and Wu (2009), "Variance risk premiums", RFS 22(3).
