# Desk Guide: Equity Portfolio Optimization and Risk Allocation

How this kind of stack is used, by whom, with what controls, and what
goes wrong in markets. The numbers quoted here come from the synthetic
panels in `examples/run_pipeline.py`.

## 1. Who consumes these numbers

- **Multi-asset allocation desks / OCIO & wealth platforms**: strategic
  and tactical asset allocation. Black-Litterman is the house workflow:
  the CIO's views (`P`, `Q`, confidences) are blended into the
  equilibrium prior each month; the long-only constrained MVO output is
  the model portfolio pushed to mandates.
- **Risk-parity funds** (Bridgewater All Weather-style, AQR, many bank
  QIS desks): the ERC engine plus the vol-target overlay IS the product.
  Weights are mean-free by design; the P&L lives or dies on covariance
  stability and financing.
- **Pension ALM / insurance**: min-variance and target-risk portfolios
  as the return-seeking sleeve; the frontier table is the board-level
  exhibit ("what return can we buy at 12% vol?"); effective N and
  diversification ratio are governance metrics in the IPS.
- **Risk management**: consumes realized risk contributions (are we
  actually running the budgets we claim?), Lo-adjusted Sharpe SEs (is
  that track record distinguishable from zero? usually not: SE ≈ 0.35
  on 8.5 years of daily data), the Jobson-Korkie/Memmel test for
  comparing two strategies, and the crisis subperiod diagnostics.

## 2. Daily / monthly workflow

1. **Estimation** (T-1 close data): Ledoit-Wolf covariance on the
   trailing 252 days; James-Stein or BL means. Sample means are computed
   but *never* fed to the optimizer directly: they are displayed next
   to their standard errors precisely so nobody is tempted.
2. **Optimization**: each mandate's optimizer (constrained MVO / ERC)
   with mandate bounds. Diagnostics gate the trade: shrinkage intensity
   (δ spiking toward 1 means the sample matrix is uninformative:
   investigate), condition number, effective N, ex-ante vol vs target,
   max risk contribution.
3. **Rebalance decision** (see §3), order generation, cost estimate from
   the turnover ledger before release.
4. **Ex-post attribution**: realized risk contributions vs budgets;
   turnover and cost accounting reconciled to the ledger the backtester
   produces (same code path: one source of truth).
5. **Model governance**: the walk-forward race (EW benchmark included)
   is rerun quarterly; a strategy that cannot beat equal weight net of
   costs over rolling windows is a candidate for retirement. Equal
   weight is the null hypothesis of this business (DeMiguel et al. 2009).

## 3. Rebalancing governance: bands vs calendar

- **Calendar** (this project's backtester: monthly): predictable,
  auditable, easy cost budgeting; but it trades on no-information dates
  and can be front-run if the date is known.
- **Bands**: trade only when drifted weights breach ±x% (absolute or
  relative) of target. Cuts turnover materially for slow signals; the
  first-rebalance/drift machinery in `run_backtest` (drifted weights are
  the turnover baseline) is exactly what a band monitor needs.
- Practice: calendar review with band triggers, plus a turnover budget
  (e.g. ≤ 2x/yr for an allocation product). Raw-mean tangency averaged
  6.8x/yr in the synthetic race and James-Stein tangency 2.7x/yr, so
  both would breach such a budget without a turnover constraint, which
  the optimisers here do not have.

## 4. Capacity and liquidity

- Turnover × AUM = dollars traded; the linear bps model in the
  backtester is a lower bound. Real impact is convex (≈ σ·√(trade/ADV)),
  so strategy capacity is set by turnover and concentration. Raw
  tangency averaged an effective N of 1.9 on the seed-1 synthetic panel,
  about two names. ERC and EW books (effective N about 8) scale.
- Liquidity is regime-dependent: the cost assumption fails worst in the
  crisis subperiod, precisely when vol-target overlays and risk limits
  force trading (see §5).
- Constraints for capacity live naturally in the box bounds (e.g. max
  weight ≤ 5×ADV-days/AUM).

## 5. Leverage for risk parity, and the March-2020 deleveraging spiral

Unlevered ERC on our 8-asset panel runs at 16.7% vol; a 10%-vol product
holds 0.60x. On a real stock/bond universe unlevered ERC runs nearer 5%
and the fund levers **1.5–2.5x** through futures and repo. The overlay
(`vol_target_overlay`, leverage `L = σ_target/σ(w)`, optional cap) makes
the mechanics explicit, and the mechanics are the risk:

**Scenario (March 2020).** Vol estimates triple in two weeks (our crisis
regime: market factor vol ×3). A constant-vol-target book must cut
leverage by ~2/3: selling *both* stocks and bonds into a falling,
illiquid market. Simultaneously: repo/futures financing tightens
(assumption 5 in METHODOLOGY breaks), correlations jump toward 1 so the
diversification the leverage was sized against disappears (realized
0.47 → 0.87 in our panel), and Treasuries briefly sold off *with*
equities: the risk-parity crowd deleveraging together became its own
correlation event. Desk controls this project supports directly:
`max_leverage` cap in the overlay, banded (not instantaneous) vol
targeting, crisis-covariance stress (re-run the optimizer on
`panel.crisis_cov`), and a standing liquidity buffer sized off the
turnover ledger under 5x cost stress.

## 6. Realistic scenarios

- **2008 (all correlations → 1).** Reproduced by the regime generator
  (crisis correlations 0.87). Every long-only book draws down together;
  min-variance loses least (-20% vs -31% EW in the seed-1 synthetic
  subperiod); concentrated raw-mean MVO loses most (-35%). VALIDATION §6:
  diversification is a calm-market quantity; size risk off the crisis
  covariance, not the calm one.
- **2020 crash (speed).** 120-day crisis window with −120%/yr drift:
  monthly rebalancing means the book trades at most once inside the
  crash: vol targeting and risk limits, not the optimizer, determine
  the realized drawdown. Weekly monitoring with band triggers is the
  compromise.
- **2022 stock-bond correlation flip.** Static stock-bond books and risk
  parity rely on a negative stock-bond correlation. In 2022 it turned
  positive and both drew down together. This project has no bond assets
  and no code for this scenario.
- **Estimation-regime trap.** A 252-day window ending three months after
  a crisis contains 120 crisis days: covariance is dominated by the
  crisis, the optimizer de-risks exactly when premia are highest. Known,
  accepted, and the reason windows/half-lives are a governance choice,
  not a free parameter.

## 7. Limits and controls summary

| Control | Where in code | Typical setting |
|---|---|---|
| Box bounds / long-only | `mvo.*(bounds=...)` | 0–10% per name |
| Budget constraint | `budget=` | 1.0 (fully invested) |
| Vol target + leverage cap | `vol_target_overlay(max_leverage=)` | 10% vol, ≤ 2.5x |
| Turnover budget | `BacktestResult.turnover` ledger | ≤ 2x/yr |
| Concentration floor | `effective_n` on proposed weights | eff. N ≥ 4 |
| Risk-budget adherence | `realized_risk_contributions` | RCᵢ within ±25% of budget |
| Estimator health | LW `intensity`, `condition_number` | alert on δ > 0.5 or cond > 1e3 |
| Benchmark null | equal weight in every race | must beat net of costs |
