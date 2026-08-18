# Methodology — FX Algorithmic Trading & Execution Modeling

This project models the *execution* problem as it actually presents itself in
OTC FX, which differs structurally from listed equities. This document covers
the market-structure rationale, the model choices (each against alternatives),
and the full assumptions register.

---

## 1. Why these benchmarks: OTC FX has no consolidated tape

Spot FX is an over-the-counter dealer market. There is no exchange, no
consolidated tape, and no official volume print. Liquidity is fragmented
across dealer streams (single-bank platforms, RFQ), ECNs (EBS, LSEG/Refinitiv
Matching, ParFX) and internalisation pools; published "volume" is partial,
survey-based (BIS triennial) or venue-proprietary.

Consequences that drive the whole design:

- **VWAP is ill-defined.** The equity VWAP benchmark and VWAP schedule need a
  tape. In FX the desk-standard benchmarks are **arrival price**
  (implementation shortfall), **interval TWAP**, and the **WM/R 4pm London
  fix**. Those are the three benchmarks implemented in `execution/tca.py`.
- **Volume participation is participation of *modeled* liquidity.** Our
  `pov_schedule` is explicitly a POV-*analog*: it participates at a capped
  rate of the *modeled* session depth profile, because realised market volume
  is unobservable in real time.
- **The "VWAP-reversion" signal becomes TWAP-mid reversion.** The intraday
  mean-reversion feature reverts to the running time-weighted average mid of
  the day (`features.reversion_to_session_mean`), the honest FX analog.

**Alternatives considered.** (a) Pretend a tape exists and implement VWAP
(common in renamed-equity projects) — rejected as structurally wrong for FX;
(b) use dealer-poll volume curves as a pseudo-tape — closer to practice at
large banks, but the curve is still a *model*, which is exactly what our
session depth profile is, with fewer moving parts.

## 2. The WM/R fix: window model and why benchmark gaming matters

The WM/Reuters (now LSEG) London 4pm fix is the reference rate for index
funds, corporate hedging and custody pricing; enormous flow is benchmarked to
it, particularly at month-end.

- **History.** Until 2015 the fix was computed over a **60-second** window.
  In 2013 it emerged that dealers at several major banks had been sharing
  client fix-order information in chat rooms ("The Cartel") and front-running
  / banging the close to move the print in their favour. The scandal led to
  ~$10bn of fines (2014–2015 FCA/CFTC/DoJ settlements), criminal charges, and
  the **2015 reform** that widened the calculation window to **5 minutes**
  and pushed banks to charge explicit fees rather than trade around the fix.
- **Model.** We model the post-2015 methodology: the fix print is the TWAP of
  mids over a 5-minute window centred on 16:00 London
  (`sessions.fix_window_mask`, `tca.fix_benchmark`; the window is quantised
  to bucket boundaries on coarse grids). The `fix_schedule` executes flat
  across exactly that window, which minimises tracking error to the print
  (measured TE std ≈ 0.00 pips vs ≈ 9.8 pips for a 3h TWAP — see
  VALIDATION.md).
- **Why gaming matters.** A benchmark that can be moved by trading *inside
  its own window* invites manipulation: a dealer holding client fix orders
  knows the sign of the window flow. Our simulator reproduces the mechanics
  (trading in the window moves the window mids via impact), which is exactly
  why the FX Global Code (Principles 9–12) and the 2015 reform constrain fix
  handling. We model the *mechanics*, not the manipulation.

## 3. Session liquidity: the 24h FX day

FX trades continuously from Wellington Monday to NY Friday close. Liquidity
is strongly time-of-day dependent; the London–NY overlap is the deepest and
tightest window. We encode five stylized sessions on a London clock
(`sessions.SESSION_BOUNDS`):

| session | hours (London) | EURUSD spread (pips) | depth (mm/min) | vol (pips/√min) |
|---|---|---|---|---|
| Asia | 00–07 | 0.6 | 20 | 0.9 |
| London | 07–12 | 0.35 | 50 | 1.8 |
| **Overlap** | **12–17** | **0.2** | **70** | 2.2 |
| NY | 17–21 | 0.4 | 40 | 1.6 |
| Late/rollover | 21–24 | 1.0 | 8 | 0.7 |

Spread/depth/vol are step functions of the bucket midpoint hour. The
scheduler layer exploits the profile: the liquidity-weighted schedule trades
proportionally to depth (constant participation rate) and beats naive TWAP on
controllable cost by ~11% over the full day (VALIDATION.md §2).

EM pairs (USDMXN profile) invert parts of the pattern: home-session (NY)
liquidity is best, Asia is a desert, and spreads are 1–2 orders of magnitude
wider in pips — wide enough to flip an intraday strategy from profitable to
unprofitable (tested).

## 4. Impact model

Executing `q` mm base in a bucket with session depth `D` (mm absorbed per
bucket) and vol `σ` (pips per √bucket):

- **Temporary** (fill-only, gone next bucket): `k_temp · σ · sqrt(q / D)`
  pips, `k_temp = 0.35`. The square-root law is the most robust empirical
  impact regularity across asset classes; scaling by *session* σ and depth
  makes the same 10mm clip ~3× more expensive per unit vol in the late
  session than in the overlap (tested exactly).
- **Permanent** (shifts all later mids): `k_perm · σ · (q / D)` pips,
  `k_perm = 0.05`, **linear** in flow. Linearity of permanent impact is
  required for no-dynamic-arbitrage (Huberman–Stanzl); it also makes the
  permanent cost of a schedule almost trajectory-independent, which is why it
  is excluded from the AC optimisation objective (§5).

**Alternatives considered.** (a) Linear temporary impact only — analytically
convenient (classic AC) but empirically wrong at size; we keep sqrt in the
*simulator* (evaluation model) and use a quadratic-cost stand-in in the
*optimiser* (decision model), a standard desk practice, with the mismatch
documented. (b) Full limit-order-book simulation — wrong granularity for OTC
FX where top-of-book is a dealer quote, and needlessly heavy for scheduling.

**Internalisation (deliberately ignored).** Major FX dealers internalise
60–90% of EURUSD flow against opposing client flow, so realised impact for a
bank algo can be far below the public-market estimate. We model *external*
liquidity consumption only; an internalising desk should treat our cost
numbers as an upper bound. This is a documented simplification, not an
accident (assumption A7).

## 5. Optimal execution: piecewise Almgren–Chriss

Classic Almgren–Chriss (2000) assumes constant temporary impact `η` and vol
`σ`. Over a 24h FX day both vary ~10× across sessions, so we solve the
discrete mean–variance problem with bucket-specific coefficients:

    min_n  Σ_j η_j n_j²/τ  +  λ Σ_j σ_j² τ x_j²,   x_j = X − Σ_{i≤j} n_i,  Σ n_j = X

- Solved **numerically** as a strictly convex equality-constrained QP via its
  KKT system (`scipy.linalg.solve`), with an exact active-set loop enforcing
  one-sided execution (at high λ the unconstrained optimum sells back in
  illiquid buckets).
- `η_j ∝ 1/depth_j` (`eta_from_depth`): the quadratic decision-model
  calibration of the simulator's sqrt law, matching marginal cost at a
  reference participation rate.
- **Verification anchors** (both unit-tested): with constant `η, σ` the
  numerical solution equals the closed-form discrete AC trajectory
  `x_j = X sinh(κ(T−t_j))/sinh(κT)`, `cosh(κτ) = 1 + λσ²τ²/(2η)`, to 1e-9;
  with `λ → 0` it reduces exactly to the liquidity-weighted (TWAP-analog)
  schedule.
- Permanent impact is excluded from the objective: with linear permanent
  impact its cost is schedule-independent to leading order (continuous-time
  identity `∫γ x dx = γX²/2`), so including it only shifts the objective by a
  near-constant.

**Alternatives considered.** (a) Closed-form AC with day-average parameters
("naive AC") — implemented as the comparison baseline; measurably worse under
the true time-varying objective (tested: aware ≤ naive, strictly < at
λ = 1e-5). (b) Dynamic programming over a discretised inventory grid — the
natural extension when constraints (participation caps, discrete clip sizes)
break the QP structure; for the quadratic objective the KKT solve *is* the
exact answer, so the grid DP would only add discretisation error.

## 6. Last-look: the dealer's option and the client's trap

On dealer streams (in contrast to firm-liquidity ECNs), the dealer receives
the order and may hold it briefly ("last look") before accepting. Historic
practice — heavily criticised, litigated (a major dealer paid a $150m NYDFS
penalty in 2015 for last-look abuse), and now regulated by **FX Global Code
Principle 17** — was to reject trades when the price moved against the dealer
during the hold, and (worse) to use the order information meanwhile.

Model (`LastLookVenue`):

- quoted half-spread = 0.6 × firm spread (the lure);
- hold window `hold_seconds = 2`; the mid move seen in the hold is the bucket
  innovation scaled by `sqrt(hold/bucket)` plus the client's per-bucket alpha
  (informed flow reveals itself immediately — the markout);
- rejection probability is a **monotone logistic** in the move against the
  dealer: `p = expit((m − 0.6)/0.2)` pips (asymmetric by design: moves in the
  dealer's favour are filled happily — that asymmetry *is* the controversy);
- a rejected child resubmits at the post-move price, paying the firm spread
  plus a 0.05 pip aggression penalty.

This produces the empirically-observed trap, quantified in VALIDATION.md §3:
the last-look stream is genuinely cheaper for uninformed flow (−0.018 pips)
and strictly more expensive for informed flow (+0.275 pips) despite quoting
40% tighter — adverse selection converts the spread saving into rejection
cost.

**Alternatives considered.** (a) Fixed rejection probability — misses the
whole point (rejections must correlate with adverse moves); (b) full
dealer-inventory game — interesting but unidentifiable with synthetic data.

## 7. Signals, carry and P&L conventions

- Pairs quoted BASE/QUOTE per CONVENTIONS.md (EURUSD = USD per 1 EUR); pip =
  `pip_size` price units; costs and slippage in **pips**; ledger P&L per unit
  base notional in quote ccy, pips, and base ccy (`net_base = net_quote/S`).
- Carry: long BASE/QUOTE rolled overnight accrues
  `S·(r_base − r_quote)·Δt`, ACT/365F, booked at the 21:00-London (5pm NY)
  rollover to the position held across it (hand-exact test). The **carry
  filter** (`signals.carry_gate`) flattens positions whose sign disagrees
  with the rate differential before rollover.
- Signals: 1h/4h momentum, reversion to the running day TWAP-mid,
  London-open breakout of the Asia range, carry. All features are strictly
  causal; **point-in-time discipline is test-enforced** by mutating future
  ticks and asserting bit-identical features at/before the cutoff.
- The backtester books `pos_{t−1}·Δclose_t` (structural no-lookahead), costs
  turnover at half the *session* spread of the trading bar, and is verified
  against a hand-computed ledger.

## 8. Assumptions register

Each assumption states what breaks if violated.

| # | Assumption | What breaks if violated |
|---|---|---|
| A1 | Session profiles are deterministic step functions of hour-of-day. | Real liquidity shifts with data releases, month-end, holidays; the liquidity-weighted and AC schedules would mis-time. Mitigate: profile recalibration, event calendars (see A6). |
| A2 | Temporary impact is sqrt in participation, scaled by session σ and depth; fully decays in one bucket. | Slower decay ⇒ consecutive child orders interact and the scheduler under-prices clustering (fix schedule, high-λ AC front-loading would look too cheap). |
| A3 | Permanent impact is linear and never decays. | If concave/transient, splitting across sessions is over-rewarded; if super-linear, the depth cap is too lenient. Linearity is also what lets the AC objective drop the permanent term. |
| A4 | Last-look behaviour = logistic rejection on hold-window adverse move + resubmit at post-move price; hold-window drift is a scaled bucket innovation (common random numbers). | Real dealers vary hold times, apply symmetric or pre-hedged last look, and may price-improve; our effective-cost gap is a stylised bound, not a venue ranking. Rejection→resubmit also ignores the option to *not* re-trade. |
| A5 | Mid dynamics are driftless Gaussian in pips with session vol; no fat tails, no gaps. | Flash events (GBP Oct-2016) produce discontinuous mids: sqrt impact and the logistic last-look model both understate cost; execution risk (IS std) is understated. Partially probed with `vol_scale` (VALIDATION.md §4). |
| A6 | **No news/event gaps in the simulator** — no scheduled-data vol spikes, no CB announcements. | Schedules that trade through (say) an FOMC bucket look fine in-sim and are dangerous in production; desks impose blackout windows (DESK_GUIDE.md). Documented, deliberately out of scope of the generator. |
| A7 | No internalisation: every child order consumes external liquidity. | A large dealer's realised impact is materially lower; our absolute cost levels are an upper bound (relative scheduler rankings are less affected). |
| A8 | The WM/R fix = plain 5-min TWAP of mids (no median filtering, no trade/quote blend), quantised to bucket boundaries. | Levels of fix tracking error shift slightly; the fix-vs-TWAP *comparison* is robust. |
| A9 | Carry accrues once per calendar day at rollover with Δt = 1/365 (no T+2 spot-date, no triple-roll Wednesdays, no weekend day-count). | Absolute carry P&L off by day-count details (~3/365 on Wednesdays); sign-based carry gating unaffected. |
| A10 | Planted alpha (hourly AR(1), φ = 0.25) is far stronger than real intraday FX alpha, by design, for statistical power in tests/demos. | Absolute backtest Sharpe (~13) is meaningless as a market claim; only the *machinery* (IC positive with t > 2 on planted, ≈ 0 on noise, costs flip EM profitability) is the deliverable. |
| A11 | One parent order per run, single pair, no cross-pair netting; children within a bucket fill at one price. | Portfolio execution (EURUSD legs vs crosses) and clip-level microstructure are out of scope. |

## 9. Model choice summary (documentation contract item 1)

| Layer | Chosen | Alternatives rejected | Trade-off |
|---|---|---|---|
| Benchmarks | Arrival / TWAP / WM-R fix | VWAP (no tape); dealer-poll volume curve | honesty vs familiarity |
| Impact | sqrt temporary + linear permanent, session-scaled | linear-only; full LOB sim | realism vs AC tractability (handled by decision/evaluation split) |
| Optimal exec | piecewise AC via KKT/QP + active set | naive constant-parameter AC; inventory-grid DP | exact for quadratic objective; DP reserved for non-QP constraints |
| Venue model | logistic last-look + firm ECN, common random numbers | fixed reject prob; dealer game | captures adverse-selection mechanism with 4 parameters |
| Signals | momentum/reversion/breakout/carry on 1h bars, PIT-enforced | tick-level alpha; ML feature stacks | auditability and test-enforceable causality |
