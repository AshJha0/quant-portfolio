# Methodology: FX algorithmic trading and execution modelling

This project models the execution problem in OTC FX, which differs in
structure from listed equities. This file covers the market-structure
rationale, the model choices against alternatives, the execution and
statistics conventions, and the assumptions register.

Every result in this project is computed on synthetic data. The intraday
signal edge is planted by the generator. No real-data result exists yet; real
data arrives in the research platform of the repository plan.

---

## 1. Benchmarks: OTC FX has no consolidated tape

Spot FX is an over-the-counter dealer market. There is no exchange, no
consolidated tape and no official volume print. Liquidity is fragmented
across dealer streams (single-bank platforms, RFQ), ECNs (the main interdealer
platforms) and internalisation pools. Published volume is partial,
survey-based (BIS triennial) or venue-proprietary.

Consequences for the design:

- VWAP is ill-defined. The equity VWAP benchmark needs a tape. The FX
  benchmarks are arrival price (implementation shortfall), interval TWAP
  and the WM/R 4pm London fix. These three are implemented in
  `execution/tca.py`.
- Volume participation is participation of modelled liquidity.
  `pov_schedule` participates at a capped rate of the modelled session depth
  profile, because realised market volume is not observable in real time.
- The intraday mean-reversion feature reverts to the running time-weighted
  average mid of the day (`features.reversion_to_session_mean`), the FX
  analog of VWAP reversion.

Alternatives considered: (a) implement VWAP as if a tape existed, rejected as
wrong for FX; (b) use dealer-poll volume curves as a pseudo-tape. That curve
is still a model, which is what the session depth profile is, with fewer
parameters.

## 2. The WM/R fix: print model and self-impact

The WM/R London 4pm fix is the reference rate for index
funds, corporate hedging and custody pricing. Large flow is benchmarked to
it, especially at month-end.

- History. Until 2015 the fix was computed over a 60-second window. In 2013
  it emerged that dealers at several banks shared client fix-order
  information and traded ahead of it to move the print. Fines of about $10bn
  followed (2014-2015 FCA, CFTC and DoJ settlements), and the 2015 reform
  widened the window to 5 minutes.
- Print model. `tca.fix_print` computes the print from many market
  observations inside the 5-minute window centred on 16:00 London. Between
  bucket mids the market path is a Brownian bridge with the bucket's vol,
  drawn from an RNG that is independent of the execution and sampled 60
  times per bucket. The child orders fill at bucket starts, so a fix
  schedule does not reproduce the print exactly and its tracking error has
  a real variance.
- Self-impact. `fix_print(..., include_own_impact=False)` removes the
  order's own permanent impact from the path: the print the market would
  have produced without the order. `fix_self_impact_pips` is the
  difference. A benchmark computed from the impacted mids hides this cost,
  because the print moves with the order.
- `tca.fix_benchmark` (the window TWAP of the same simulated mids the
  schedule trades on) is kept for one release as a descriptive statistic.
  It is not a tracking-error benchmark: against it the fix schedule's
  tracking error is half-spread plus temporary impact with zero variance,
  by construction.
- Why gaming matters. A benchmark that trading inside its own window can
  move invites manipulation. The simulator reproduces the mechanics (window
  trading moves window mids through permanent impact), which is why the FX
  Global Code (Principles 9-12) and the 2015 reform constrain fix handling.
  The project models the mechanics, not the manipulation.

## 3. Session liquidity: the 24h FX day

FX trades from the Wellington Monday open to the NY Friday close. Liquidity
depends on time of day, and the London-NY overlap is the deepest and
tightest window. Five stylised sessions on a London clock
(`sessions.SESSION_BOUNDS`):

| session | hours (London) | EURUSD spread (pips) | depth (mm/min) | vol (pips/sqrt(min)) |
|---|---|---|---|---|
| Asia | 00-07 | 0.6 | 20 | 0.9 |
| London | 07-12 | 0.35 | 50 | 1.8 |
| Overlap | 12-17 | 0.2 | 70 | 2.2 |
| NY | 17-21 | 0.4 | 40 | 1.6 |
| Late/rollover | 21-24 | 1.0 | 8 | 0.7 |

These profile values are stylised inputs, not estimates from market data.
Spread, depth and vol are step functions of the bucket midpoint hour. The
liquidity-weighted schedule trades in proportion to depth and has about 11%
lower controllable cost than TWAP over the full day in the synthetic
simulator (VALIDATION.md section 2).

EM pairs (USDMXN profile) invert parts of the pattern: NY liquidity is best,
Asia is thin, and spreads are one to two orders of magnitude wider in pips.

## 4. Impact model

Executing `q` mm base in a bucket with session depth `D` (mm absorbed per
bucket), so participation `rho = q / D` is a dimensionless rate, and session
vol `sigma_H` in pips over a fixed reference horizon `H` =
`impact_horizon_minutes` (default 5 minutes, the default bucket):

- Temporary (fill only, gone next bucket): `k_temp * sigma_H * rho^delta`
  pips, `k_temp = 0.35`, `delta = impact_exponent = 0.5` (1 gives the
  linear law). The square-root law is the most robust empirical
  impact regularity across asset classes. Scaling by session vol and depth
  makes the same 10mm clip about 3x more expensive per unit vol in the late
  session than in the overlap (tested exactly).
- Permanent (shifts all later mids): `k_perm * sigma_H * q / D_H` pips with
  `D_H` the session depth over `H`, `k_perm = 0.05`, linear in flow, so an
  order's total permanent impact does not depend on how it is sliced.

`H` is fixed rather than equal to the bucket width so that the same trading
rate costs the same on any grid (`tests/test_impact_consistency.py`). Up to
v2.0.0 both terms used the per-bucket vol `sigma * sqrt(dt)`: a 200mm TWAP
over two overlap hours paid 0.119 pips of temporary impact on 1-minute
buckets and 0.651 on 30-minute buckets, with permanent impact 0.314 and
0.057 pips. It now pays 0.266 and 0.141 pips on every grid. Results on
5-minute grids are unchanged; the 1-minute fix study below changed. Linear permanent impact is required for
  no dynamic arbitrage (Huberman and Stanzl). It also makes the permanent
  cost of a schedule nearly independent of the trajectory, which is why it
  is left out of the AC objective (section 5).

`k_temp` and `k_perm` are set by hand. They are not calibrated to FX trade
data.

Alternatives considered: (a) linear temporary impact only, convenient
(classic AC) but wrong at size. The simulator keeps the square root
(evaluation model) and the optimiser uses a quadratic stand-in (decision
model); the mismatch is stated. (b) A full limit-order-book simulation, the
wrong granularity for OTC FX where top of book is a dealer quote.

Internalisation is left out. Major dealers internalise 60-90% of EURUSD flow
against opposing client flow, so realised impact for a bank algo can be far
below the public-market estimate. Every child order here consumes external
liquidity, so the absolute cost levels are an upper bound (assumption A7).

## 5. Optimal execution: piecewise Almgren-Chriss

Classic Almgren-Chriss (2001, Journal of Risk 3(2)) assumes constant
temporary impact `eta` and vol `sigma`. Over a 24h FX day both vary about
10x across sessions, so the discrete mean-variance problem is solved with
bucket-specific coefficients:

    min_n  sum_j eta_j n_j^2 / tau  +  lambda sum_j sigma_j^2 tau x_j^2,
    x_j = X - sum_{i<=j} n_i,   sum_j n_j = X

- Solved as a strictly convex equality-constrained QP through its KKT system
  (`scipy.linalg.solve`), with an active-set loop that enforces one-sided
  execution (at high lambda the unconstrained optimum sells back in illiquid
  buckets).
- `eta_j` is proportional to `1/depth_j` (`eta_from_depth`) with
  `k_eta = 0.02`. The risk aversion `lambda` (1e-5 in the pipeline) has no
  utility calibration. Both are uncalibrated choices.
- Anchors (unit-tested): with constant `eta, sigma` the solution equals the
  closed-form discrete AC trajectory
  `x_j = X sinh(kappa (T - t_j)) / sinh(kappa T)`,
  `cosh(kappa tau) = 1 + lambda sigma^2 tau^2 / (2 eta)`, to 1e-9; with
  `lambda -> 0` it reduces to the liquidity-weighted schedule.
- Permanent impact is left out of the objective: with linear permanent
  impact its cost is schedule-independent to leading order
  (`integral of gamma x dx = gamma X^2 / 2`).

Alternatives considered: (a) closed-form AC with day-average parameters
(naive AC), implemented as the baseline and worse under the time-varying
objective (tested); (b) dynamic programming over an inventory grid, needed
when constraints break the QP structure. For the quadratic objective the KKT
solve is the exact answer.

## 6. Last look

On dealer streams the dealer may hold an order briefly before accepting it
(last look). Past practice, criticised and litigated (a $150m NYDFS penalty
in 2015) and now covered by FX Global Code Principle 17, was to reject when
the price moved against the dealer during the hold.

Model (`LastLookVenue`):

- quoted half-spread = 0.6 x firm spread;
- hold window 2 seconds; the mid move seen in the hold is the bucket
  innovation scaled by `sqrt(hold/bucket)` plus the client's per-bucket
  alpha. Informed flow is fully visible to the dealer inside the hold. That
  is an assumption, and it is what makes informed flow lose;
- rejection probability is a monotone logistic in the move against the
  dealer, `p = expit((m - 0.6)/0.2)` with `m` in pips. Moves in the
  dealer's favour are filled;
- a rejected child resubmits at the post-move price and pays the firm spread
  plus a 0.05 pip penalty.

Synthetic result (VALIDATION.md section 3): the last-look stream is cheaper
for uninformed flow by 0.018 pips and dearer for informed flow by 0.275
pips, although it quotes 40% tighter.

Alternatives considered: (a) a fixed rejection probability, which misses the
correlation with adverse moves; (b) a dealer-inventory game, not identifiable
on synthetic data.

## 7. Execution convention, signals, carry and P&L

Signals use data up to the close of period t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost.

- In `IntradayBacktester.run(..., execution=ExecutionConfig(...))` the
  position decided at bar t fills inside bar t+1. The P&L booked at bar t+1
  is `pos_{t-1} * f * d + pos_t * (1 - f) * d` with
  `d = close_{t+1} - close_t`, and the trade pays half the spread of the
  bar-(t+1) session. `fill_fraction = 1` is the next bar's close.
- `mode="close_auction"` fills at the decision bar's close mid and pays
  `auction_cost_bp` (default 0.5 bp, about 0.55 pips on EURUSD at 1.10) on
  top of half the bar-t spread. Spot FX has no closing auction; in this
  project the name marks the same-close fill with an extra immediacy cost,
  so the semantics match the other alpha projects in the repository.
- Pairs are quoted BASE/QUOTE per CONVENTIONS.md (EURUSD = USD per 1 EUR).
  Pip = `pip_size` price units. Costs and slippage are in pips. Ledger P&L
  is per unit base notional in quote currency, pips and base currency
  (`net_base = net_quote / S`).
- Carry: a long BASE/QUOTE position rolled overnight accrues
  `S (r_base - r_quote) dt` with ACT/365F, booked at the 21:00 London
  rollover to the position held across it, time-weighted by `f` when the
  position changes inside that bar. `signals.carry_gate` flattens positions
  whose sign disagrees with the rate differential before rollover.
- Signals: 1h and 4h momentum, reversion to the running day TWAP mid,
  London-open breakout of the Asia range, carry. Features are causal; a test
  mutates future ticks and asserts the features at and before the cutoff do
  not change.

## 8. Statistics

`research_stats.py` re-exports `qp_core.stats`, which the other alpha
projects use too.

- Sharpe ratios are annualised with `BARS_PER_YEAR = 24 x 261 = 6264`
  hourly bars, counting session-filtered zero-position bars.
- `summary["sharpe_se"]` is the Lo (2002) standard error with a Newey-West
  (Bartlett) long-run variance at lag `floor(4 (n/100)^(2/9))`.
- Every headline is a distribution over 30 synthetic seeds (signal layer) or
  200 replications (execution), reported as mean [95% Student-t CI].
- The pipeline logs every configuration it evaluates on seed 0 in a
  `TrialLog` (9 trials: 5 execution and cost variants and 4 other signal
  weight sets) and reports the deflated Sharpe ratio at that count, a
  circular block-bootstrap CI for the seed-0 Sharpe, and the probability of
  backtest overfitting (CSCV, 8 blocks) over the 5-point weight grid.

## 9. Assumptions register

| # | Assumption | What breaks if violated |
|---|---|---|
| A1 | Session profiles are deterministic step functions of hour of day. | Real liquidity shifts with data releases, month-end and holidays; the liquidity-weighted and AC schedules would mis-time. |
| A2 | Temporary impact is square-root in participation, scaled by session vol and depth, and decays fully in one bucket. | Slower decay makes consecutive child orders interact; clustered schedules (fix, high-lambda AC) would look too cheap. |
| A3 | Permanent impact is linear and never decays. | If concave or transient, splitting across sessions is over-rewarded; if super-linear, the depth cap is too lenient. |
| A4 | Last look is a logistic rejection on the hold-window adverse move, with resubmission at the post-move price. | Real dealers vary hold times, use symmetric or pre-hedged last look, and may price-improve; the effective-cost gap is a stylised bound. |
| A5 | Mid dynamics are driftless Gaussian in pips with session vol; no fat tails, no gaps. | Flash events produce discontinuous mids; impact and execution risk are understated. Partly probed with `vol_scale`. |
| A6 | No news or event gaps in the simulator. | Schedules that trade through an FOMC bucket look fine in simulation and are dangerous live; desks impose blackout windows. |
| A7 | No internalisation. | A large dealer's realised impact is lower; absolute cost levels are an upper bound. |
| A8 | The fix print is the mean of Brownian-bridge observations of the mid in the 5-minute window (no median filter, no trade/quote blend). | Tracking-error levels shift; the fix-vs-TWAP comparison is robust to this. |
| A9 | Carry accrues once per calendar day at rollover with dt = 1/365 (no T+2 spot date, no triple-roll Wednesday). | Absolute carry P&L is off by day-count details; sign-based carry gating is unaffected. |
| A10 | The planted alpha (hourly AR(1), phi = 0.25) is far stronger than any real intraday FX autocorrelation, which is within a few hundredths of zero in majors. | The synthetic backtest Sharpe means nothing as a market claim. Only the machinery is the deliverable. |
| A11 | One parent order per run, one pair, no cross-pair netting; children within a bucket fill at one price. | Portfolio execution and clip-level microstructure are out of scope. |
| A12 | Impact coefficients, `k_eta` and the AC risk aversion are set by hand. | Absolute costs and the AC schedule shape are not calibrated to any market. |

## 10. Model choice summary

| Layer | Chosen | Alternatives rejected | Trade-off |
|---|---|---|---|
| Benchmarks | Arrival, TWAP, WM/R fix from independent window observations | VWAP (no tape); dealer-poll volume curve | FX structure over familiarity |
| Impact | square-root temporary and linear permanent, session-scaled | linear only; full LOB simulation | realism vs AC tractability (decision/evaluation split) |
| Optimal execution | piecewise AC via KKT/QP and active set | naive constant-parameter AC; inventory-grid DP | exact for the quadratic objective |
| Venue model | logistic last look and firm ECN, common random numbers | fixed reject probability; dealer game | captures adverse selection with 4 parameters |
| Signals | momentum, reversion, breakout, carry on 1h bars, causal | tick-level alpha; ML feature stacks | auditability and testable causality |
| Execution timing | decide at close t, fill at a modelled t+1 price | same-close fill without extra cost | matches the other alpha projects; same-close kept as a costed variant |
