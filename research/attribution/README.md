# research/attribution: factor attribution (W23a)

Before a return is called alpha, regress it on the premia it may simply be
collecting. The intercept is what those factors leave unexplained; its
t-statistic uses Newey-West standard errors because monthly returns are
autocorrelated and heteroskedastic.

| Function | What it returns |
|---|---|
| `ols_hac(y, factors, lag=None, small_sample=False)` | Coefficients, HAC standard errors and t-statistics, annualised alpha and its t, information ratio, R squared, residuals. Default lag `floor(4 (T/100)^(2/9))` (Newey and West 1994). |
| `rolling_betas(y, factors, window=36)` | Trailing-window OLS coefficients. |
| `fx_factors(ports)` | `DOL` and `HML_FX` from `research/lib/panel.lrv_sort` (Lustig, Roussanov and Verdelhan 2011). |
| `attribute(y, factors, "ff3")` | `ols_hac` on a named set: `fx`, `ff3`, `ff5`, `ff5mom`. |

French factors come from `qp_data.french.parse_french` (Kenneth French data
library, keyless).

Checked against `statsmodels` OLS with `cov_type="HAC"` to 1e-10 on the
standard errors (`tests/test_attribution.py`).

Example: `y = 0.002 + 0.5 DOL` with no noise gives alpha 0.002 a month
(0.024 a year) and beta 0.5 exactly.
