# Desk Guide — FX Volatility Modeling & Forecasting

Documentation-contract items 5 (how a real desk uses this) and 6 (real-life
scenarios). Numbers referenced here come from `examples/run_pipeline.py` and
docs/VALIDATION.md.

---

## 1. What the numbers feed

**Vol input to Garman–Kohlhagen pricing.** GK (Black–Scholes with q = r_f,
see portfolio conventions) needs an annualized vol per pair and tenor. The
model forecast provides the *realized-vol anchor*: term forecasts
`sqrt(mean of E[s2_{T+1..T+h}] * 252)` give a model-implied term structure
that mean-reverts from today's state to the unconditional level
(GARCH: geometric decay at rate α+β; the pipeline's EURUSD-like fit has
persistence 0.974 → half-life ≈ 26 days). Traders compare this anchor with
the quoted implied surface: implied far above the anchor with no event in
the window = candidate vol sale; below = candidate buy.

**FX VaR / risk.** One-day 99% VaR per pair ≈ `2.33 · σ_{t+1}` only under
Gaussian innovations — with the fitted t (ν ≈ 7 G10, ν ≈ 3.6 EM), use the
t quantile: the Gaussian multiplier understates the 1% tail by >5% on ν = 6
data (tested). The rolling harness (`rolling_one_step`) is exactly the
production loop: filter the state daily, refit on a schedule (125 days in
the pipeline), and feed σ_{t+1} into the VaR engine. Diagnostics to attach
to the daily run: ARCH-LM and Ljung-Box on standardized residuals (must be
clean), sign-bias (if it trips, escalate to the asymmetric model).

**Cross and illiquid pairs.** Mark EM/G10 crosses off liquid legs with the
vol triangle: EURJPY from EURUSD & USDJPY legs plus their correlation
(exact identity, run_pipeline §6). The same algebra prices proxy hedges:
hedging a cross with its two legs is exactly the triangle decomposition.

## 2. Daily workflow

1. **06:30** — load fixes (or `data/live.py` ECB rates for research), append
   returns; NaN policy = reject: a NaN means a data problem to fix, not fill.
2. **06:40** — filter all pairs' variance states forward under current
   parameters (milliseconds); refit weekly + after any >4σ day (see §4
   depeg protocol: refit only once the jump is *interior* to the sample).
3. **06:50** — publish per pair: σ_{t+1} (daily and annualized), 1w/1m/3m
   term forecasts, distance-to-unconditional, fitted ν, asymmetry γ with SE,
   event-adjusted path for the next 5 days from the GARCH-X calendar.
4. **Consumers**: options desk (anchor vs implied), risk (VaR inputs), carry
   book (§3), management pack (vol regime dashboard).
5. **Diagnostics page** (auto): residual ARCH-LM p, Ljung-Box p, sign-bias p,
   persistence, boundary flags (persistence > 0.995 or SE = NaN → model
   review), QLIKE of last 90 days vs EWMA benchmark.

## 3. Strategy applications

**Vol-risk-premium harvesting.** `fx_vol.vol_premium` compares implied with
subsequent realized (or with the model forecast, tradable ex ante). On the
synthetic demo the premium is 1.2 vol points with 100% positive days — real
G10 premia are ~0.5–1.5 points with 70–85% positive months, and the P&L
shape is the important part: short-var P&L per unit vega is capped near
`K/2` on the upside but *convex against you* on the downside
(`variance_swap_pnl`, tested). Controls: size off the t-quantile not the
Gaussian; hard stop when the premium turns negative (realized above
implied = crisis regime — stand down, do not "average in"); no fresh selling
into scheduled-event windows unless the event premium (GARCH-X γ_x) is
demonstrably overpriced.

**Carry-trade risk monitoring.** Carry portfolios are short vol in
disguise: funding-currency rallies (JPY, CHF) come with vol spikes. Monitor
σ_{t+1}/σ_uncond per funding pair; a spike above ~1.5 with rising
correlation across carry pairs is the classic unwind signature — de-lever
before the VaR engine forces it. The EGARCH γ sign tells you which side of
the pair carries the risk (safe-haven asymmetry, METHODOLOGY §3).

**Event-risk pricing.** GARCH-X with the CB calendar prices meetings
explicitly: the pipeline shows a scheduled day-3 event lifting the day-3
forecast from 8.8% to 14.2% annualized (γ_x t-stat 23.9 on simulation).
Compare that model event premium with the overnight implied straddle around
FOMC/ECB: sell the event when implied event variance ≫ γ_x estimate,
buy when below.

## 4. Scenario playbook (real-life, each mapped to tested behaviour)

- **SNB depeg (EURCHF, 15 Jan 2015, ≈ −15%).** Tested end-to-end with a
  −15% injected jump: fit converges, day-after conditional variance > 20×
  pre-jump, forecasts spike > 10× unconditional then decay geometrically.
  *Protocol*: day 0 — do not trust a refit whose sample *ends* on the jump
  (α unidentified; the MLE degenerates to constant variance — tested,
  documented in VALIDATION §5); risk-manage off the filtered state under
  *pre-jump* parameters with the jump fed through the recursion. Day 1+ —
  refit with the jump interior. Expect elevated ω/α for months; VaR
  backtests will show a one-day breach no vol model prevents — that is what
  the peg-concentration limit was for.
- **BoJ intervention (USDJPY, Sep–Oct 2022, ~¥9tn).** Unscheduled → *not* a
  GARCH-X dummy (assumption A7). Interventions arrive as 2–4σ innovations
  against the trend; the t-likelihood keeps them from whipping α around
  (the reason GARCH-t beats Gaussian OOS on jumpy pairs, DM p = 0.017).
  Practical: near suspected intervention levels, widen VaR add-ons manually;
  the model prices elevated vol *after* the first strike, not before.
- **Brexit (GBPUSD, 24 Jun 2016, ≈ −8%).** A *scheduled* event with an
  unscheduled-sized outcome: put the referendum date in the GARCH-X calendar
  (it was known), and let the realized jump propagate through the recursion.
  The event dummy prices the ex-ante premium; the ex-post jump follows the
  depeg playbook. Implied vol did its job here (GBP 1m implied doubled
  pre-vote) — this is the scenario where the implied-vs-model premium
  monitor (§3) correctly reads "do not sell".
- **EM crisis (TRY 2018/2021, MXN 2020).** One-sided depreciation jumps,
  ν < 4, asymmetry on the depreciation side. Fit the pair in the traded
  quote direction and check γ's sign; remember GJR's γ ≥ 0 sees only
  negative-side asymmetry — on USD/EM quotes use EGARCH or fit the inverted
  pair (pipeline: GJR-t γ = 0 on USDMXN, +0.123 on MXNUSD). Gaussian fits
  pinning persistence at 1 are the misspecification alarm, not a market view.
- **Pegged book (HKD, SAR).** Vol models run fine (tested to 2 bp daily vol)
  but are the wrong tool: the risk is the *peg*, a jump-to-depeg process
  with tiny daily variance. Report the GARCH number for completeness,
  risk-manage with a depeg scenario grid instead (this is a limitation
  stated in writing, VALIDATION §5).

## 5. Controls, limits, governance

- **Model inventory**: GARCH-t is the G10 production default; EGARCH-t for
  EM and safe-haven pairs; EWMA runs in parallel as the challenger benchmark
  every model must beat on trailing QLIKE.
- **Recalibration**: weekly, plus triggered refits after >4σ days (jump
  interior rule). Parameters, SEs, persistence and ν archived per run —
  parameter jumps between refits are themselves a monitored signal.
- **Model-risk limits**: boundary flags (persistence > 0.995, SE = NaN,
  ν < 3) force human review before numbers publish. Forecasts are floored
  at the pegged-pair band vol and capped at 3× the trailing 21d realized
  for VaR use (fat-finger guard).
- **Backtesting**: quarterly OOS race (the `rolling_one_step` harness) with
  DM tests against the challenger; Mincer–Zarnowitz calibration on the year;
  VaR breach counting at 95/99 with Kupiec-style monitoring by risk.
- **P&L attribution**: vol-book P&L split into premium carry (implied −
  forecast at inception), realized-vs-forecast surprise, and event-day
  P&L (GARCH-X term) — the split makes it visible whether the desk earns
  the premium or is just short the jumps.
- **Governance artefacts**: METHODOLOGY.md (assumptions register A1–A10 with
  break conditions) and VALIDATION.md (recovery tables, arch cross-check,
  failure modes) are exactly the documents an internal model-validation
  team asks for; the offline deterministic test suite is the regression
  gate for any change.
