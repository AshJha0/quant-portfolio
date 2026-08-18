# Methodology — Equity Statistical Pairs Trading

Pipeline: **pair selection → correlation screen → Engle-Granger cointegration
→ ADF on residuals → OU spread model → z-score signals → event-driven
backtest with costs → walk-forward validation**.

All conventions: daily closes in dollars, trading-day time (252/yr), OU
`kappa` per trading day so half-life `ln 2 / kappa` is in trading days,
transaction costs in basis points of per-leg traded notional, borrow
annualised ACT/252.

---

## 1. Why this model? (contract item 1)

The core modelling decision is **Engle-Granger cointegration + an
Ornstein-Uhlenbeck spread model**, chosen against four alternatives:

### vs the distance method (Gatev-Goetzmann-Rouwenkamp SSD)

The distance method ranks pairs by the sum of squared differences of
normalised price paths and trades divergences from the formation-window
spread. It is assumption-light and cheap — which is why we keep SSD as a
*pre-screen* (`ssd_screen`) — but it has no null hypothesis: it cannot
distinguish "these paths tracked each other because a common factor ties
them" from "they tracked each other by luck". It also gives no model of the
spread, hence no principled half-life, no time-stop calibration, and no
stationary variance for the z-score. EG gives a testable null
(no-cointegration) with correct critical values, and the residual defines
the tradeable hedge ratio directly.

### vs Johansen's ML procedure

Johansen estimates the cointegration *rank* and the cointegrating vectors of
an N-dimensional VAR by maximum likelihood. It is the right tool for baskets
of 3+ series (EG can only test one candidate relation at a time, and its
result depends on which leg is the regressand). For **two-leg pairs** — this
project — the rank can only be 0 or 1, EG answers exactly that question, the
two-step OLS is transparent and fast enough to run thousands of times inside
a walk-forward loop, and its output *is* the hedge ratio a trader needs.
Johansen adds VAR lag-order selection and eigenvalue machinery for no
practical gain at N=2, so it is documented as the scaling path and kept out
of scope. (The EG dependence on regression direction is real; the desk
convention here is to fix the orientation in the pair spec and re-test both
directions in research.)

### vs ML approaches (clustering, autoencoders, RL execution)

Learned pair selection (embedding + clustering) can propose candidates
beyond sector tags, and RL can refine entry/exit. But they need large cross
sections to avoid overfitting, are hard to govern (model-risk sign-off wants
a null distribution, not a validation-set loss), and — decisive here — they
do not *replace* the statistical question: a proposed pair must still pass a
stationarity test before capital is deployed. This project implements the
part that survives any overlay.

### Why OU for the spread (vs rolling z-score only)

A rolling z-score is a non-parametric fallback (implemented as
`zscore_rolling`), but it conflates level and speed: it cannot say *how
fast* the spread reverts. The OU model gives kappa → half-life → a
principled **time stop** (k× half-life: a convergence trade that hasn't
converged in ~3 half-lives is evidence against the model, tested and used in
the walk-forward), and a stationary variance for a warm-up-free z-score.
OU is fitted two ways — OLS on the exact AR(1) discretisation and numerical
MLE — which must agree (cross-check in tests).

### The critical-values decision (the classic mistake, made testable)

The step-2 ADF statistic on EG residuals must be compared against
**MacKinnon's Engle-Granger response surfaces (N=2)**, not plain ADF (N=1)
values. OLS chooses (α, β) to minimise residual variance, biasing the
residuals towards stationarity; the null distribution shifts left. At the
5% level and T=500: plain ADF −2.87 vs EG N=2 −3.35. Using the wrong table
more than doubles the spurious "discovery" rate (measured in
`test_spurious_rejection_rate_close_to_size`: EG values reject ~5% of
independent random-walk pairs; plain ADF values reject >1.5× that). Both
surfaces are implemented from MacKinnon (1996, 2010) and verified against
`statsmodels` to 1e-10.

### Why the correlation screen runs on returns, never prices

Price levels of two independent random walks are spuriously correlated
(Granger-Newbold): the sample correlation of levels does not converge to 0.
In `test_price_corr_spurious_return_corr_not`, independent walks average
|price corr| > 0.45 while |return corr| < 0.10. A price-level screen selects
shared drift; a return screen selects shared shocks. Note the screen is a
*cheap filter*, not evidence of cointegration — the panel's trap pairs have
return correlation ≈ 0.92 and are still not cointegrated (that is what the
ADF stage is for).

### Hedge ratio: intercept, and static vs adaptive

- **Intercept included by default** in the cointegrating regression: share
  price levels are arbitrary (splits), so forcing the line through the
  origin misspecifies the relation. With an intercept in step 1, the step-2
  ADF regression runs *without* a constant (residuals are exactly mean-zero)
  against the N=2 "constant" surface. MacKinnon tabulates no N≥2
  no-constant surface; `engle_granger(intercept=False)` flags the
  approximation (`crit_approx=True`).
- **RLS with forgetting factor ("Kalman-lite")** is the adaptive option:
  exponentially-weighted recursive least squares, algebraically the Kalman
  filter for a random-walk-coefficient model with the state-noise ratio tied
  to the forgetting factor λ instead of separately estimated. One knob
  (memory ≈ 1/(1−λ)) instead of two noise covariances, O(1) per step, no
  smoothing. A full Kalman filter buys explicit uncertainty bands and
  estimated noise ratios at the price of calibrating them — not needed for
  hedge-ratio tracking here. **Identification caveat** (tested): with short
  memory, an intercept is nearly collinear with the price level; track with
  `intercept=False` and let the OU mean absorb the level.

---

## 2. Assumptions register (contract item 2)

| # | Assumption | What breaks if violated |
|---|------------|-------------------------|
| A1 | **The spread's stationarity persists out of sample.** Cointegration found in the formation window holds in the trading window. | The strategy shorts a diverging spread. The regime-break simulation quantifies it: +$81k pre-break becomes −$21k post-break with stops, **−$715k without stops** (examples/run_pipeline.py §6). Mitigants: walk-forward re-testing, stop-loss, time-stop, re-entry arming. |
| A2 | **The hedge ratio β is stable** over formation + trading horizon. | The "hedged" book carries residual directional exposure; spread P&L is contaminated by market moves. Mitigants: RLS tracking (tracks a 1.0→2.0 drifting β with mean error <0.02 in tests), re-fit every walk-forward window. |
| A3 | **The spread is OU** (linear mean reversion, Gaussian shocks). | Half-life and stationary variance are wrong ⇒ mis-set time stops and z-scores. Fat tails make |z|>4 events far more common than Gaussian OU predicts — stops fire more than modelled. Mitigants: stop rules do not depend on distributional tails; rolling z-score fallback. |
| A4 | **Borrow is available and its fee is stable** (50bp/yr default). | Hard-to-borrow names: fee spikes (100bp → 10–100% annualised in squeezes) or recalls force-close the short leg at the worst moment. This is unhedgeable model-externally; see DESK_GUIDE (GME scenario). |
| A5 | **Execution at the close at quoted cost** (5bp commission + 2bp impact per leg), fills always available. | Fixed-bps impact understates cost for large size (impact scales ~√size); closing auctions can gap. Cost sensitivity table in VALIDATION.md quantifies bps → P&L; capacity limits in DESK_GUIDE. |
| A6 | **Daily closes are synchronous and clean.** Gaps ≤5 days may be forward-filled. | Stale-fill over halts fabricates convergence P&L. Longer gaps raise `ValueError` by design (`align_pair`) rather than silently extrapolating. |
| A7 | **Signals computed at t are executable at t+1's close** (one-bar lag). | Any same-bar execution is lookahead: the detector test constructs a spread where same-day execution is profitable and honest execution loses — the engine must lose. |
| A8 | **Pair P&Ls are approximately independent** across the book. | Crowding correlates all stat-arb books through the *positions*, not the prices — August 2007: simultaneous unwinds made "independent" spreads diverge together (DESK_GUIDE). Portfolio Sharpe overstates capacity. |

---

## 3. The mathematics

**Engle-Granger step 1.** OLS: `y_t = α + β x_t + u_t`. Under
cointegration, β̂ is super-consistent (converges at rate T).

**Step 2 — ADF from scratch.** On residuals û:
`Δû_t = ρ û_{t-1} + Σᵢ φᵢ Δû_{t-i} + ε_t` (no constant), statistic =
t-value of ρ̂. Lag order by AIC over p = 0..p_max (Schwert rule), all
candidates fitted on a common sample (Ng-Perron convention) so AICs are
comparable — this reproduces `statsmodels.adfuller(autolag="AIC")` to 1e-8,
including the selected lag. Critical values:
`crit(T) = b₀ + b₁/T + b₂/T² + b₃/T³` with MacKinnon's (2010) coefficients,
N=2 surface.

**OU spread.** `ds = κ(μ − s)dt + σ dW`; exact discretisation
`s_{t+1} = μ(1−b) + b s_t + ε_t`, `b = e^{−κΔt}`,
`Var ε = σ²(1−b²)/(2κ)`. OLS on the AR(1) gives (κ, μ, σ) in closed form;
the conditional-likelihood MLE coincides with OLS in (c, b) and differs only
in variance normalisation — the agreement is a tested implementation check.
Half-life `= ln 2 / κ`; stationary sd `= σ/√(2κ)`. b ≥ 1 ⇒ flagged
non-mean-reverting (κ=0, half-life ∞, **do not trade**), never a fabricated
huge-but-finite half-life.

**Signals.** z from the OU stationary distribution
(`z = (s−μ)/(σ/√(2κ))`, no warm-up) or rolling window. State machine:
enter |z|≥2, exit on reversion through the exit band (default z→0), hard
stop |z|≥4, time stop at ⌈3×half-life⌉ bars, and **re-entry arming**: after
any exit the pair may not re-enter until |z| < entry — otherwise a stop at
|z|>4 is immediately followed by re-entry at |z|>2, re-fighting the trade
the stop just cut (this arming is what caps the regime-break loss).

**Sizing.** Dollar-neutral (gross/2 per leg — zero net dollars at entry,
tested exact) or beta-neutral (shares in ratio 1:β — tracks the spread
exactly, small net dollar exposure).

**Backtest accounting.** Executed position at t = target decided at t−1;
MTM `q·ΔP` on holdings carried in; commission+slippage in bps of per-leg
traded notional charged as explicit cash; borrow accrued daily on short
market value. Identity `net = gross − commission − slippage − borrow` holds
to the penny (tested). Walk-forward: formation [i, i+F) fits everything;
trading [i+F, i+F+T) uses frozen parameters; windows can never overlap
(constructor-enforced + tested); positions force-closed at window ends.

**Metrics.** Sharpe annualised √252; its SE both iid
(`√((1+SR²/2)/T)`, Lo 2002) and **Lo-adjusted** for serial correlation
(Bartlett/Newey-West long-run variance ratio on the mean) — mean-reversion
books have autocorrelated P&L, and the iid SE overstates significance
exactly then (tested: AR(1) φ=0.6 P&L widens the SE by >30%). Sortino, max
drawdown, hit rate, holding period, annualised turnover and cost drag, and
per-pair attribution.

---

## 4. References

- Engle & Granger (1987), "Co-integration and Error Correction". 
- MacKinnon (1996, 2010), critical-value response surfaces (QED WP 1227).
- Gatev, Goetzmann & Rouwenkamp (2006), "Pairs Trading: Performance of a
  Relative-Value Arbitrage Rule".
- Lo (2002), "The Statistics of Sharpe Ratios".
- Avellaneda & Lee (2010), "Statistical Arbitrage in the U.S. Equities
  Market".
- Khandani & Lo (2007), "What Happened to the Quants in August 2007?"
