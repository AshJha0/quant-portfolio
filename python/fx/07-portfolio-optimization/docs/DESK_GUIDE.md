# Desk Guide — FX Portfolio Optimization & Currency Risk Allocation

## 1. Who uses this

- **FX overlay managers** (pension/insurance mandates): the hedging module
  is the core product — per-currency hedge ratios for an international
  asset portfolio, reviewed monthly, with policy bands around h\*.
- **Macro funds / RV currency books**: the style sleeves (carry, momentum,
  value) as a systematic core, allocated by ERC or CVaR-constrained MVO,
  with the dollar-neutral + gross-budget constraint set matching how a
  long-short FX book is actually margined.
- **Corporate treasury / pension hedging programs**: the base-currency
  machinery (GBP reporting for a UK plan), the full-vs-optimal hedge
  variance numbers for the hedging-policy paper, and the carry cost of
  hedging (rate differential) as a separate, reportable mean adjustment.

## 2. Daily / monthly workflow

Daily (risk):
1. Update spot + deposit-rate panels; rebuild total returns
   (`total_log_returns` — carry accrues on previous-close rates).
2. Refresh EWMA covariance and the risk-on/off factor loadings; flag any
   loading sign flip (a safe haven turning risk-on is a regime alert, not a
   data point to average in).
3. Recompute book CVaR95 and per-style attribution
   (`summary`, `style_attribution`); compare against limits (below).

Monthly (allocation committee):
1. Rebuild style signals; re-estimate LW covariance on the trailing 504d.
2. Run the allocator set: ERC (baseline), MVO with shrunk means
   (challenger), CVaR-constrained LP (sizing authority). The **smallest**
   of the MVO and CVaR books is what trades: mean-variance proposes,
   CVaR disposes.
3. Re-run the hedge-ratio closed form for the overlay book; move actual
   hedges only if outside the policy band (see governance).
4. Walk-forward diagnostics (`run_backtest`): realised vs ex-ante vol,
   carry P&L share, cost drag vs budget.

## 3. Governance: limits and controls

| Control | Suggested level (this project's calibration) | Rationale |
|---|---|---|
| Gross leverage | Σ|w| ≤ 2.0 per style, ≤ 1.5 at book level | Matches the tested constraint set; margin realism |
| Per-currency cap | |w_i| ≤ 0.35, EM ≤ 0.20 | Rank weights already cap at ~0.31 gross 2.0; EM cap binds first |
| CVaR95 limit | ≤ 0.40%/day at book level (the pipeline's binding case) | Prices the carry tail that vol targeting misses — 30% smaller carry book than 5%-vol sizing at 0.50%/day budget |
| Vol target | 5% ann ex-ante, exact scaling | `vol_target` is exact ex-ante; realised will drift with regime |
| Hedging policy band | h ∈ [max(0, h\*−0.25), h\*+0.25], capped to [−0.5, 1.5] per ccy | Raw h\* (JPY −1.40, AUD 3.71 in the demo) is a *statistical* optimum; bands keep turnover and governance sane |
| Peg exposure | Explicit cap (e.g. 5%) regardless of optimizer | Min-var piles into pegs; a peg is a policy option that can break (SNB 2015) |
| Stand-down rule | Degenerate signal ⇒ zero weights (built in) | Never trade a rank of noise |

Model governance: closed forms are the golden references (tested to 1e-12);
any library upgrade must re-pass the identity suite. The CVaR LP's binding
status is logged each rebalance — a constraint that never binds is either a
generous limit or a tail-blind estimation window (both worth knowing).

## 4. P&L attribution

The ledger separates **spot P&L**, **carry accrual** and **costs** daily;
`style_attribution` splits book P&L by style. In the pipeline run the EW
book's P&L is 69% carry accrual — a book that looks like "FX alpha" and is
mostly collecting rate differentials should be sized by its crash risk, not
its vol. That number goes on page 1 of the risk report.

## 5. Realistic scenarios (and what this system does)

- **2008 carry unwind.** The synthetic stress (`crash_prob=0.3`) reproduces
  it: carry mean flips negative, crash days cost 4–8x daily vol, high-carry
  correlations jump (VALIDATION §6.1–6.2). What saves the book is the CVaR
  budget set *before* the event and the stand-down discipline — not the
  trailing covariance, which is calm-blind. Action: CVaR limit is a hard
  limit; no averaging down into a rising differential (a widening
  differential in stress is a distress signal, not a better carry signal).
- **2015 SNB de-peg.** A pegged/managed currency shows near-zero vol; every
  variance-based tool misreads it (min-var piles in — tested). Action: the
  peg cap above, plus scenario add-ons (±15% overnight gap) outside the
  historical-simulation window. Assumption 3 (differential persistence)
  fails intraday here: monthly rebalance holds the stale book, so the cap
  is the control, not the signal.
- **2022 USD strength cycle & intervention risk.** A persistent dollar
  trend puts momentum long-USD against carry's short-USD tilt; the style
  correlation matrix (carry-momentum −0.23 in the pipeline) is the hedge,
  and ERC keeps both sleeves alive. MoF/BoJ intervention risk means JPY
  shorts carry gap risk that the historical window won't show until it
  happens — another argument for the CVaR budget over vol.
- **2016 GBP flash crash.** Minutes-long 8% move: linear pip costs and the
  one-day lag both fail (Assumptions 4, 8). The backtest's cost drag is a
  floor in such regimes. Action: crisis cost multipliers in EM/GBP stress
  reports; the GBP-base identity work means a London desk's *dollar-neutral*
  sleeves are unaffected by the reporting-currency move itself (exact
  invariance, tested) — only net exposures need the hedge decision.

## 6. What the numbers mean (and don't)

Synthetic Sharpes calibrate the *machinery*, not the market. The transferable
results are relative: CVaR sizing cuts carry ~30% vs vol sizing at matched
budgets; optimal hedging beats full hedging by ~4x in variance reduction with
safe-haven underhedging; ERC ≈ EW Sharpe with smaller drawdowns; MVO's edge
is exactly as large as your faith in your mean estimates (Lo SE ~0.3 on an
11-year daily Sharpe — most style Sharpe differences are not resolvable).
Quote ranges, not points, to the investment committee.
