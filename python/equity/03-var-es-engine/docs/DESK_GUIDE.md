# Desk Guide: Using the Equity VaR Engine

Daily workflow, consumers of the numbers, controls, capital link and stress
review. Dollar figures are for the synthetic demo book in
`examples/run_pipeline.py`.

## 1. The overnight batch

1. **Position feed.** Positions become a `Portfolio` with the factor
   mapping: equities to their own factor, futures to the index, options to
   the underlier and an implied-vol factor. An unmapped position or an
   empty book raises. Nothing is dropped.
2. **Market data.** 500 or more days of factor returns. EWMA (lam 0.94)
   covariance and per-factor vol forecasts (`ewma_covariance`,
   `ewma_volatility`, seed window 30 days).
3. **Risk run.** Per-factor FHS VaR and ES at 95 % and 99 % as the primary
   measure (`factor_filtered_historical_var`), parametric normal and t as
   challengers, Monte Carlo t with full revaluation on the option book
   (`monte_carlo_risk` returns VaR, ES and the bootstrap SE from one
   simulation). The demo pipeline runs in about 1.5 s.
4. **Backtest update.** Yesterday's VaR against realised clean P&L:
   exception flag, Kupiec and Christoffersen, traffic-light count
   (`rolling_var_backtest`, `BacktestResult.summary`).
5. **Morning pack.** VaR and ES by method, the method disagreement,
   limit use, new exceptions, stress table and the reverse-stress direction.
   FHS well below parametric means current vol is quiet relative to the
   window. That is information, not a bug.

If a covariance matrix is materially indefinite (for example correlations
assembled from inconsistent sources), `cholesky` raises and names the most
negative eigenvalue. The batch should stop and the data be fixed, or a repair
(`repair="higham"` or `"clip"`) chosen explicitly and logged. The repair
emits a warning that states the largest entry change.

## 2. Limit monitoring

- Limits on 99 % 1-day VaR and 97.5 % ES per desk and for the firm.
  Utilisation is measure / limit.
- Breach: same-day escalation to the desk head and the market risk
  manager. The desk cuts risk or obtains a documented temporary increase.
- Sensitivity ladders (`sensitivity_ladder`) show which factor to cut. On
  the demo book the SPX ladder is asymmetric (-$64k at +20 %, +$182k at
  -20 %) because the book is net long equity with long index puts.
- Quote the VaR with its SE (MC t: $49.9k, bootstrap SE $0.4k). A breach
  inside one SE is still a breach and is flagged as marginal.

## 3. Exception investigation

1. Attribute the exception: market move, data error or unmapped risk.
   Backtests run on clean P&L (no fees or new deals).
2. Check clustering with `exception_cluster_table` (gaps of 5 days or less)
   and the Christoffersen independence p-value.
3. Quarterly review of Kupiec and conditional-coverage p-values per method.
   At 250 days the Kupiec test rejects a correct model about 9.5 % of the
   time at the nominal 5 % (VALIDATION.md §5).

## 4. Regulatory capital link

Under the Basel 2.5 internal-models approach the general market risk charge
is

`max(VaR_{t-1}, k x avg60(VaR)) + max(sVaR_{t-1}, k_s x avg60(sVaR))`

with 99 % 10-day VaR, stressed VaR (sVaR) calibrated to a stress period, and
`k = 3 + plus factor` from the backtest. The engine computes VaR and the
plus factor. It does not compute stressed VaR or the 60-day averages.

`basel_traffic_light(n_obs=, n_exceptions=, alpha=)` zones any window and
alpha by the cumulative binomial probability (spec S4). On 250 days at 99 %:

| exceptions | zone | k |
|---|---|---|
| 0-4 | green | 3.00 |
| 5 | yellow | 3.40 |
| 6 | yellow | 3.50 |
| 7 | yellow | 3.65 |
| 8 | yellow | 3.75 |
| 9 | yellow | 3.85 |
| 10+ | red | 4.00 |

`basel_zone_probabilities` gives the binomial table (green 89.2 %, red
0.03 % for a correct model at 250 days). On the first 250 forecast days of
the synthetic GARCH backtest the unconditional models are red (13 and 10
exceptions) and per-factor FHS is green (2).

FRTB IMA pieces live in `eq_var.frtb_ima`: the
liquidity-horizon ES cascade, stressed ES, IMCC, capital with the plus
factor, the desk backtest and the P&L attribution test. NMRF/SES (taken as
an input), the default risk charge and risk-factor eligibility are still
not implemented, and the BCBS paragraph references have not been re-checked
against the bis.org text. See "Which tool for which question" below.

## 5. Stress review

- **Historical replays** (`HISTORICAL_SCENARIOS`, approximations of
  published moves): 1987 (-20.5 % in a day), 2008 Lehman fortnight
  (-25 %, VIX 25 to 80), 2020 (-34 % peak to trough, VIX 15 to 83).
  Multi-day episodes are applied as one-day shocks. Demo book: -$134k,
  -$198k, -$254k with full revaluation, 2.5 to 4.6 times the 99 % VaR.
- **Hypothetical combinations**: risk-off (-15 % equity, +25 vol points)
  and melt-up (+10 %, -5 points). The demo book makes $153k in the melt-up.
- **Full revaluation against delta-gamma**: $93k apart in the 2020 row.
  Greeks-based risk is unsafe for gap moves.
- **Reverse stress** (`reverse_stress_delta`, `reverse_stress_delta_gamma`):
  the worst joint 3-sigma move on the demo book is AAPL -4.7 %, JPM -3.6 %,
  SPX -1.9 %, IV +2.1 points, a loss of $60.4k. With gamma the long puts
  trim it to $59.8k. The numerical result reports `converged`; if no start
  converges it warns and returns the closed-form delta direction.

## 6. Scenarios and how to read them

- **2008 replay**: the stress ignores the covariance matrix (A9), so
  diversification between AAPL and JPM does not offset.
- **2020 replay**: the delta-gamma error row shows why gap risk needs full
  revaluation.
- **Single-name squeeze**: use `shocks_by_name` for a two-sided shock on one
  name. VaR calibrated on calm vol does not see it. The ladder is the
  control.
- **Single-name gap** (-30 % overnight): hits the full delta of that name.
  Daily covariances cannot see gaps. The stress table can.

## 7. Who uses what

| Consumer | Numbers | Frequency |
|---|---|---|
| Desk head | VaR and ES against limits, ladders, exceptions | daily |
| CRO and risk committee | firm VaR and ES, traffic light, stress table | daily or weekly |
| Regulator | 99 % VaR, 250-day exceptions, multiplier | quarterly |
| Model validation | backtest p-values, method disagreement, CF validity flags | quarterly |
| Finance | capital charge | monthly |

## Which tool for which question

* "Which trades drive our VaR?" `risk_attribution.parametric_attribution`
  for a quick read, `simulation_attribution` on the scenario P&L matrix for
  the number you report. Components add to the total.
* "What does this hedge do to our ES?" `parametric_incremental` or
  `simulation_incremental`; the latter recomputes, it does not linearise.
* "Are we still on IMA?" `frtb_ima.desk_backtest` (run on actual and on
  hypothetical P&L) and `frtb_ima.pla_test`.
* "What is our 99.9 % loss?" Do not read it off 500 historical days: fit
  `evt.fit_pot` and use `tail.var(0.001)`. Check the mean-excess plot first.
* "Our options book under the historical scenarios?" `revaluation.full_revaluation_pnl`;
  use `taylor_pnl(order="delta_gamma")` only as a fast check and quote the gap.
* "Why did we make money yesterday?" `revaluation.pnl_explain`. A large
  `unexplained` means a missing risk factor or a big move.
* Limits: build a `limits.Limit` tree once; `check_limits` each day; pass the
  aggregated VaR for parents rather than letting it sum the children.
