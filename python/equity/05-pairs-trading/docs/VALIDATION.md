# Validation — Equity Statistical Pairs Trading

How the implementation was validated (contract items 3, 4, 6). All numbers
below are reproduced by `python examples/run_pipeline.py` (seeded, offline,
~5 s) and the test suite (`pytest -q`, 256 tests, ~5 s, offline).

---

## 1. Cross-validation against statsmodels

The ADF machinery is written from scratch and must reproduce
`statsmodels.tsa.stattools.adfuller` **exactly** on the same specification:

| Check | Tolerance | Test |
|---|---|---|
| ADF t-stat, fixed lag ∈ {0,1,4,8}, regression `c` | 1e-8 (measured ~1e-14) | `test_fixed_lag_stat_matches` |
| ADF t-stat + selected lag, `autolag="AIC"`, regressions `n`/`c`/`ct` | 1e-8, lag identical | `test_autolag_aic_matches` |
| MacKinnon critical values (N=1 and N=2 response surfaces, finite-sample) | 1e-10 vs `mackinnoncrit` | `test_finite_sample_surface_matches_statsmodels` |

## 2. Parameter recovery on planted truth (run_pipeline §3)

EG + OU on the seed-7 panel's cointegrated pairs (n=1500 days):

| pair | β true | β est | κ true | κ OLS | κ MLE | HL true (d) | HL est (d) |
|------|-------:|------:|-------:|------:|------:|------------:|-----------:|
| CO1  | 1.353 | 1.356 | 0.084 | 0.074 | 0.074 | 8.2 | 9.4 |
| CO2  | 1.718 | 1.729 | 0.101 | 0.100 | 0.100 | 6.9 | 7.0 |
| CO3  | 1.567 | 1.553 | 0.081 | 0.081 | 0.081 | 8.5 | 8.6 |
| CO0  | 1.425 | 1.429 | 0.140 | 0.128 | 0.128 | 5.0 | 5.4 |

Unit tests additionally verify OU recovery on 20k-step simulations (κ within
15%, σ within 5%), OLS↔MLE agreement (κ rel. 1e-3), the half-life identity,
RLS tracking of a drifting β (mean error < 0.02), and that a random-walk
spread yields a huge/infinite half-life with the `mean_reverting=False` flag
rather than a tradeable number.

## 3. Statistical size: the spurious-regression guard

200 replications of EG on **independent** random walks (T=400):

- With the correct MacKinnon **EG (N=2)** critical values: rejection rate ≈
  nominal 5% (asserted within [0.5%, 12%]).
- With the classic mistake — plain **ADF (N=1)** values on the same
  statistics: rejection rate > 1.5× higher (asserted).

(`test_spurious_rejection_rate_close_to_size`.)

## 4. The selection funnel shows the trap (run_pipeline §2)

Seed-7 panel, 20 names: **40 same-sector candidates → 8 pass the return
correlation screen (ρ ≥ 0.6) → 4 pass Engle-Granger at 5%**. The rejected
four passing the correlation screen are exactly the three
correlated-random-walk traps (return correlations 0.919–0.923, EG stats
−1.45 to −2.13 vs −3.34 critical) and the regime-break pair. Every true
cointegrated pair is accepted (EG stats −7.4 to −9.8); no trap is.

## 5. No-lookahead proof

`test_engine_reports_the_honest_losing_result` constructs a spread that
alternates ±3 daily. Executing today's signal at today's close is a money
machine (mean reversion happens overnight); executing yesterday's signal —
the only causal option — systematically enters after the reversion:

- cheat (same-day info): **net P&L > 0**;
- engine (t−1 signal at t close): **net P&L < 0** — asserted.

The engine physically reads `target[t-1]` at t; the cheat is only reachable
by pre-shifting the input. Companion tests pin the exact one-bar lag between
signal and position and that the first bar can never trade.

## 6. Exact accounting

- `net = gross − commission − slippage − borrow` to 1e-9, with
  commission/slippage equal to the trade-ledger sums (costs reduce P&L by
  exactly the ledger sum, and gross P&L is invariant to the cost model).
- A 3-round-trip scenario is matched against hand-computed closed forms
  (fractions like 200/102 + 200/98 + 100/101) to 1e-9, including per-trade
  P&L, holding periods and exit reasons.
- Borrow: 252bp/yr ⇒ 1e-4/day accrues on the short leg's daily market value
  from the day after entry, matched day-by-day to 1e-12.
- Dollar neutrality at entry: net exposure exactly 0, gross exactly the
  target (1e-9).

## 7. Walk-forward hygiene

- Formation and trading windows **cannot overlap**: enforced in the
  `WalkForwardWindow` constructor and asserted over full schedules.
- **Parameters frozen**: a traded window's P&L is reconstructed from the
  recorded formation-window parameters only and must match the engine
  output to 1e-9 (`test_parameters_frozen_during_trading_window`).
- Positions force-closed at every trading-window end (asserted).

Out-of-sample results, seed-7 panel (capital $6mm = 6 × $1mm gross/pair,
costs 5bp+2bp/leg, borrow 50bp):

| metric | in-sample (§4) | walk-forward (§5) |
|---|---:|---:|
| net P&L | $2.92mm | $1.40mm |
| ann. return | 12.3% | 7.4% |
| Sharpe | 2.42 | 1.59 |
| Sharpe SE (iid / Lo) | 0.41 / 0.57 | 0.46 / 0.50 |
| hit rate | 95.6% | 67.2% |
| max drawdown | $92k | $128k |
| turnover | 5.7x | 6.7x |

The in-sample/out-of-sample gap (Sharpe 2.4 → 1.6, hit rate 96% → 67%) is
itself a validation result: selection and OU parameters fitted on the trading
data flatter every number.

## 8. Cost sensitivity (walk-forward book, slippage 2bp, borrow 50bp fixed)

| commission (bp/leg) | net P&L | Sharpe | cost drag (bp/yr on capital) | hit rate |
|---:|---:|---:|---:|---:|
| 0  | $1,463,482 | 1.66 | 11.5 | 68.8% |
| 2  | $1,438,103 | 1.63 | 20.4 | 67.2% |
| 5  | $1,400,033 | 1.59 | 33.8 | 67.2% |
| 10 | $1,336,584 | 1.51 | 56.0 | 62.5% |
| 20 | $1,209,685 | 1.37 | 100.6 | 56.2% |

At ~6.7x annual turnover the book loses ≈ 4.5bp/yr of capital per bp of
per-leg commission; a strategy this size survives realistic institutional
costs but would not survive retail spreads on less liquid names.

## 9. Failure modes (contract item 4)

**Cointegration breakdown — the regime-break case study (run_pipeline §6).**
Planted pair: OU spread (true half-life 11.6d) for 750 days, then the spread
becomes a drifting random walk. Fitted on pre-break data (EG −5.01,
half-life est. 10.0d — the fit is *good*; nothing in-sample warns you):

- net P&L before break: **+$81,497**
- net P&L after break, with stops + re-entry arming: **−$20,963**
  (1 post-break entry; the stop fires, arming blocks re-entry while |z|
  stays extreme)
- same book **without** stop-loss/time-stop: **−$715,158** — the strategy
  averages into a diverging spread. This is the single most important risk
  control in the project.

**Short squeeze (2021 GME-style).** A short leg that rallies 100%+ produces
exactly the "spread rich, add to short" signal while borrow fees explode and
recalls force closure. Level-based stops help only if honoured with gap
risk; borrow-fee spikes are modelled (borrow bps) but recalls are not — the
mitigation is desk process (DESK_GUIDE §4), not statistics.

**Crowding — August 2007 quant quake.** Khandani-Lo: heavily overlapping
stat-arb books were unwound simultaneously; spreads that "could not" widen
did, together, for three days, then snapped back. Diversification across
pairs is illusory when the *holders* are correlated: A8 in the assumptions
register. Monitoring proxy: pair-level days-to-cover and the correlation of
the book's P&L with public stat-arb factors.

**Low-vol spread ⇒ leverage temptation.** A 5bp-vol spread "supports" 20x
leverage to hit a return target; that is exactly the book that dies in a
quake (drawdown scales with leverage, stationarity does not). The desk
guide caps leverage by stressed spread vol, not recent vol.

**Estimator bias in the OU half-life (measured, and it has a sign).** OLS on
the discretised AR(1) inherits the Dickey-Fuller/Kendall small-sample bias:
`kappa` is under-estimated, so **estimated half-lives are systematically too
LONG**. Measured at n = 12,000 (`test_properties.py::
test_ols_kappa_is_biased_downward_and_worse_for_slow_reversion`):

| true κ | 0.01 | 0.03 | 0.08 | 0.20 |
|---|---|---|---|---|
| true half-life (d) | 69.3 | 23.1 | 8.7 | 3.5 |
| estimated half-life (d) | 87.2 | 25.8 | 9.2 | 3.6 |
| relative bias in κ | **−20.5%** | −10.5% | −5.9% | −3.5% |

The bias worsens as reversion slows — exactly the regime where pairs are
marginal anyway. Desk consequence: a time stop set at *k × estimated
half-life* is systematically **too loose**, letting losers run past the point
where the model says they should have converged. Either shrink the estimate
or set `k` conservatively.

**An OU fit alone cannot reject a random walk (why cointegration is the
gate).** On a *pure random walk* OLS returns `b` slightly below 1 (finite-
sample bias ≈ −1.5/n), so `OUFit.mean_reverting` comes back **True** with a
spurious half-life of 300–950 days across seeds. The `mean_reverting` flag
only trips on `b ≥ 1` exactly, so it does **not** protect you here
(`test_an_ou_fit_alone_cannot_reject_a_random_walk`). Two screens do:

1. the Engle-Granger test with **N=2** MacKinnon values (verified to reject
   random-walk pairs), and
2. a hard cap on the accepted half-life — an estimate in the hundreds of days
   is the tell, and the pipeline's selection funnel filters on it.

This is the concrete reason the funnel gates on cointegration *first* and
never trades off a half-life alone.

**Numerical/degenerate edges (all unit-tested):** zero-variance legs raise
informative errors everywhere (screen excludes them without crashing); >5-day
data gaps refuse to forward-fill; all-cash periods produce NaN Sharpe (not
inf); zero-trade backtests produce empty ledgers and zero turnover; b≥1
AR(1) fits are flagged non-mean-reverting; oscillatory (b<0) spreads are
rejected as non-OU; a monotone-rising equity curve reports max drawdown
`+0.0` rather than `−0.0`.

## 10. Structural invariants (property-based tests)

`tests/test_properties.py` pins the *shape* of the library rather than point
values, so the checks hold for any correct implementation:

| Invariant | Tolerance | Test group |
|---|---|---|
| Sharpe and Sortino **scale-invariant** under leverage; Sharpe flips sign with returns; annualisation scales as √periods | 1e-12 rel | `TestMetricInvariances` |
| Max drawdown homogeneous of degree 1 and **shift-invariant** | 1e-12 rel | `TestMetricInvariances` |
| Lo-adjusted Sharpe SE **exceeds** the iid SE under positive autocorrelation and falls below it under negative | directional | `TestMetricInvariances` |
| OU fit **equivariant** under `s → s + d` (μ shifts by d, κ/σ unchanged) and `s → c·s` (μ, σ scale by c, κ unchanged) | 1e-8 – 1e-9 | `TestSpreadAlgebra` |
| OLS and MLE agree on a 4,000-point sample | 5% on κ/σ | `test_ols_and_mle_agree_on_a_long_sample` |
| Half-life recovered for κ ∈ {0.03, 0.08, 0.20}; faster κ ⇒ strictly shorter half-life | 15% | `TestSpreadAlgebra` |
| Stationary std matches both the simulated dispersion and σ/√(2κ) | 10% / 15% | `test_stationary_std_matches_the_simulated_dispersion` |
| MacKinnon N=2 values strictly below N=1 at every level; gap at 5% > 0.4 t-units | exact | `test_mackinnon_values_are_ordered_and_n2_is_stricter` |
| EG **power** (rejects a planted cointegrated pair) and **size** (rejects ≤25% of 40 correlated-random-walk samples at 5%) | — | `TestCointegrationProperties` |
| Signal machine: positions ⊂ {−1,0,1}; **exact sign antisymmetry** under z → −z; entries non-increasing in `entry_z`; stop disarms until z re-enters the band | exact | `TestSignalProperties` |
| Sizing: dollar mode dollar-neutral and hits the gross target; beta mode uses the cointegrating share ratio; both linear in gross and antisymmetric in direction | 1e-12 rel | `TestSizingProperties` |
| Backtest: `net = gross − commission − slippage − borrow` daily; zero costs ⇒ net == gross; net monotone in cost bps; gross P&L homogeneous in the gross target; **antisymmetric under target → −target**; trade P&Ls reconcile to daily net | 1e-9 – 1e-6 | `TestBacktestIdentities` |
| **No look-ahead**: truncating the sample leaves every earlier day's P&L bit-identical; the first bar never carries a position | 1e-9 | `test_no_lookahead_shifting_the_target_later_changes_nothing_before` |
