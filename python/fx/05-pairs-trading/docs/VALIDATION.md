# Validation: FX statistical pairs

How the implementation was validated, with the numbers. Every performance
number here comes from synthetic data in which the planted pair is
cointegrated by construction, so the profit is a property of the generator,
not evidence of an edge in FX markets. No real-data result exists yet (Phase
4 of the remediation plan).

Reproduce: `python examples/run_pipeline.py` (seeded, offline, 196 s with the
default 60 seeds; it writes no files) and `pytest -q` (286 tests, 106 s).

Execution convention for every backtest below: signals use data up to the
close of day t. Orders fill at a modelled t+1 price, p_t + f (p_{t+1} - p_t)
with f = 0.5 by default; a close-auction variant fills at p_t and pays an
auction cost.

---

## 1. Cross-validation against statsmodels (from-scratch econometrics)

| Check | Result |
|---|---|
| ADF t-stat, fixed lags (0 and 4), `regression="c"` | matches `statsmodels.adfuller` to **< 1e-10** (observed ~8e-16) |
| ADF, no-constant (`"n"`) on residual-type series | matches to < 1e-10 |
| ADF auto-lag (AIC) | selected lag identical; stat matches to < 1e-8 |
| MacKinnon critical values (N=1, finite-sample) | match statsmodels-reported values to < 1e-10 |
| Engle–Granger statistic | matches `statsmodels.coint(trend='c', autolag='aic')` to **~2e-15**; 5% critical value matches to < 1e-8 |
| EG vs plain ADF critical values | N=2 table more negative by > 0.3 at every level (asymptotic 5%: −3.336 vs −2.862): using the N=1 table on residuals would over-reject |

## 2. Statistical size, power and the funnel

* **Spurious-regression size control**: 200 seeded independent random-walk
  pairs, EG at 5%: rejection rate **4.5%** (nominal 5%; asserted < 10%).
* **Recovery on one synthetic seed** (n=1500, true beta 1.0): EG stat
  **-7.76**, recovered beta **0.999**, alpha -0.0004.
* **Funnel** (`cointegration_funnel`): the planted pair is added to the
  8-pair two-block universe under the labels SEKUSD/NOKUSD. Candidates are
  regressed in input column order, so the orientation uses neither the data
  nor the ground truth. On seed 9 the screen keeps 8 candidates and EG at 5%
  keeps only the planted pair. Over 60 synthetic seeds the funnel finds the
  planted pair in 60 of 60 and admits 1.12 [0.80, 1.43] false positives per
  seed out of about 8 tests (`tests/test_wiring.py`).
* **EG gate power inside the walk-forward**: on 252-day formation windows the
  10% gate passes 82% [79%, 85%] of windows on a pair that is cointegrated by
  construction (60 synthetic seeds). The gate costs trading time even when
  the relation holds.

## 3. Model recovery and cross-model consistency

* **OU recovery** (true kappa=20/yr, sigma=0.05, half-life 8.7 bd; n=1500):
  OLS kappa **20.1**, sigma **0.051**, half-life **8.7 bd**. MLE matches OLS
  (kappa within 2%, as it must, conditional MLE equals OLS for AR(1)); this is a
  consistency check, not an independent estimate. Kappa carries the upward
  small-sample bias of -ln(phi) (phi is biased down); recovery tolerance ±25%
  at n=8000. A fitted phi <= 0 raises `ValueError`
  (`test_spread.py::TestOscillatingSpreadRejected`).
* **Half-life identity**: `ln 2/(kappa·dt)` asserted to 1e-12 against the
  fitted kappa; white noise → half-life < 1 bar; random walk → > 250 bars.
* **RLS hedge**: lam=1 converges to batch OLS (|Δbeta| < 1e-5); with lam=0.98
  it tracks a mid-sample beta break 1.0 → 1.6 to within 0.15 on both sides.
* **Accounting identities** (exact, atol 1e-18): `total = spot + carry +
  cost` elementwise; carry-inclusive backtest = spot-only backtest + an
  independently computed carry ledger; costs scale linearly in notional, pip
  spread, and hedge-leg beta; a hand-built 3-trade ledger (weekend 3-day
  accrual included) matches the engine number-for-number.
* **No lookahead**: perturbing all prices after day k (×1.5 / ×0.7 shocks)
  leaves P&L and positions through k unchanged (detector test). The same
  check covers the RLS hedge path and the vol-target scale
  (`tests/test_wiring.py`).
* **Execution**: a hand ledger checks the fill price `p_t + f (p_{t+1} - p_t)`
  for f = 0.25, 0.5 and 1; a perfect-foresight signal earns zero (within 4
  SE) at f = 1, while the same signal filled at the decision close earns the
  full |r|; the close-auction variant charges exactly `auction_cost_bp` per
  unit traded (`tests/test_execution.py`).

## 4. Costs, carry and the carry-flip pair (synthetic)

One synthetic seed (seed 9), same signal path, full-sample hedge ratio (so
these runs are in-sample):

| run | total P&L | costs | Sharpe (Lo/NW SE) |
|---|---|---|---|
| frictionless | +0.1995 | 0 | 1.58 (0.32) |
| major costs (0.7 / 1.0 pips) | +0.1946 | -0.0049 | 1.54 (0.32) |
| major costs, RLS hedge (lam 0.995, causal) | +0.1920 | -0.0050 | 1.52 (0.33) |
| major costs, close-auction fill (+10 bp) | +0.1096 | -0.0849 | 0.87 (0.35) |
| EM costs (60 / 30 pips) | -0.0601 | -0.2596 | -0.37 (0.29) |

Across 60 synthetic seeds (mean [95% CI] (sd)):

| quantity | value |
|---|---|
| in-sample Sharpe, frictionless | 1.82 [1.75, 1.89] (0.28) |
| in-sample Sharpe, major costs | 1.77 [1.70, 1.84] (0.27) |
| in-sample Sharpe, EM costs | -0.44 [-0.50, -0.38] (0.24) |

The carry-flip generator gives the base leg an **8%** deposit rate against
**1%** on the hedge leg, with the high-yielder's spot drifting down at half
the differential. The signal is mostly long the high-yielder. Synthetic
seed 4:

| | total P&L | spot | carry | Sharpe |
|---|---|---|---|---|
| spot-only backtest | **-0.1431** | -0.1426 | 0 | -0.98 |
| carry-inclusive | **+0.0844** | -0.1426 | **+0.2275** | 0.58 |
| carry-inclusive, carry entry veto | +0.0765 | -0.1546 | +0.2313 | 0.53 |

Across 60 synthetic seeds the spot-only P&L is -0.143 [-0.153, -0.133] and
the carry-inclusive P&L is +0.120 [0.110, 0.129]: ignoring carry flips the
sign. The carry entry veto (6 half-lives of expected hold) keeps every long
entry bar and vetoes 21 of 26 short entry bars on seed 4. Across seeds it
leaves the mean P&L at +0.120 [0.110, 0.129], because on this generator
nearly all trades are already long the high-yielder.

## 5. Failure modes (documented AND simulated)

### 5.1 SNB-style floor-then-break (the central case study)

Generator: 750 days of tight OU around a floor (EURCHF 2011–2015 analogue),
spread pinned below its mean at the floor edge, then a one-day **−15%** gap
and a high-vol regime. Synthetic pipeline numbers (seed 3):

* Formation scan on the pegged period: EG stat **−5.82** → "cointegrated",
  the best-looking spread a scan could find. That is the trap.
* Pre-break: cumulative P&L **+0.053** over 750 days, hit rate 0.75. The
  steady gains build false confidence (and, on a vol-targeted book, maximal
  leverage, because realised spread vol was collapsing).
* Break day: **-0.150** in one day, **2.8×** everything the strategy had
  made; full-sample P&L -0.065.
  market gaps straight through any stop (asserted in
  `test_scenarios.py::test_stop_cannot_prevent_the_gap_loss`).
* Position on break eve: +1, long the pegged pair, with the crowd.

Lesson encoded in tests: statistical excellence of a spread is **not**
evidence of economic stability; policy-maintained cointegrations carry the
policy's event risk.

### 5.2 Policy divergence (slow version)

Central-bank divergence that does not gap but trends (Fed hiking vs ECB on
hold, 2014–15) drifts the hedge ratio and the spread mean. Controls: RLS beta
tracking (tested), walk-forward re-fitting with an EG gate that de-selects the
pair once the formation window stops cointegrating (tested: correlated-walk
sample → windows skipped), time stops.

### 5.3 EM devaluations and wide costs

Step devaluations (CNY 2015, ARS repeatedly) are the A3 jump failure, same
shape as 5.1. Even without events, EM pip spreads of 30–150 turn the
strategy's gross edge negative (table above); the EM-vs-major cost test
asserts the sign flip at identical signals.

### 5.4 Crowded carry unwinds

Risk-off episodes flip the commodity-bloc/safe-haven correlation structure:
in the synthetic two-block panel, intra-block return correlation stays ~0.85
while cross-block correlation flips from **+0.09 to −0.57** in risk-off
windows (detectable on a 40-day rolling correlation, tested). Pairs that
diversified in calm regimes become one trade in stress; carry longs and RV
spreads that share the same high-yielder unwind together. Control: per-block
exposure limits (`DESK_GUIDE.md`), regime monitoring on rolling correlations.

## 6. Edge cases (each documented and unit-tested)

| Edge case | Behaviour | Test |
|---|---|---|
| Pegged / zero-vol instrument in universe | dropped from screen with `UserWarning`; OU fit on its spread raises `ValueError` | `test_universe`, `test_scenarios` |
| Triangular identity spread | flagged degenerate, never cointegrated, ADF skipped (NaN stat) | `test_cointegration`, `test_scenarios` |
| Zero-trade backtest | all-zero ledgers, NaN Sharpe, 0 trades, no crashes | `test_backtest` |
| Missing days / calendar gaps | engine runs; carry accrues actual calendar days (Fri→Mon = 3/365; gap of 10 days = 10/365) | `test_backtest`, `test_carry` |
| Weekend/holiday accrual total | sums exactly to calendar days/365 | `test_carry` |
| Rate spike on last day | cannot affect prior accruals (lagged rates) | `test_carry` |
| Series too short / NaNs / mismatched lengths / bad thresholds | informative `ValueError`s | all modules |
| **±Inf in a series** (logged zero/missing price) | rejected by `adf_test`, `engle_granger`, OU fitters and `run_backtest`; previously accepted by `isnan`-only guards and returned "not cointegrated" | `test_nan_guards` (§6.1) |
| **NaN threshold / parameter** (`stop`, `sigma`, `notional`, `min_abs_corr`, `ann_factor`) | rejected; previously disabled the hard stop, muted the strategy or emptied the screen silently | `test_nan_guards` (§6.1) |
| Zero or non-positive price level | rejected before `log`, with a message naming the series | `test_nan_guards` (§6.1) |
| Vol-target on collapsing vol | scale capped at max leverage (the SNB lesson) | `test_signals` |
| Sub-5% EG rejection on pure noise | size control holds | `test_cointegration` |

### 6.1 Non-finite inputs (`tests/test_nan_guards.py`)

Every public entry point rejects NaN and ±Inf with a named `ValueError`.
Thresholds and parameters (`stop`, `sigma`, `notional`, `min_abs_corr`,
`ann_factor`, deposit rates) are checked with `np.isfinite`, because an
inequality guard such as `if sigma <= 0` passes NaN. Series inputs to
`adf_test`, `engle_granger`, the OU fitters and `run_backtest` are checked
with `np.isfinite`, because a zero or missing fixing becomes `-inf` after
`log`, and an `isnan` check would let it through. Without these checks a NaN
`stop` disables the hard stop, a NaN `sigma` produces an all-flat strategy
reported as a zero-P&L run, and an Inf makes `engle_granger` report "not
cointegrated" on data it could not test. Price levels must be strictly
positive.

Companion tests check that the guards do not over-reject: the planted pair is
still detected, a finite stop still fires, the vol-target cap still binds at
10×, RLS at `lam=1` still converges to batch OLS, and the clean backtest still
satisfies `total = spot + carry + costs` to 1e-12.

`RLSHedge.update` validates before it mutates `theta` and `P`, because the
filter cannot recover from a non-finite state. The metrics layer tolerates
NaN in a P&L series (it drops non-finite entries) because a warm-up window
produces NaN P&L before the first position; the pricing, signal and backtest
layers reject.

## 7. Metrics and research statistics

Sharpe, Sortino, MDD, hit rate and turnover are hand-computed on toy ledgers.
The Sharpe standard error is the Lo (2002) delta-method estimator with a
Newey-West (Bartlett) long-run variance truncated at `floor(4 (T/100)^(2/9))`
lags (`research_stats.lo_sharpe_se`). With zero lags it equals the Mertens
iid formula to 1e-10; on AR(1) returns with phi 0.4 it matches the Monte
Carlo sd of the Sharpe estimate within 10% (`test_research_stats.py`).

## 8. Walk-forward headline over many seeds (synthetic)

Walk-forward 252/63, entry 2.0, exit 0.5, stop 4.0, EG gate at 10%, major
costs and carry, default t+1 execution. 60 synthetic seeds (1000 to 1059),
command `python examples/run_pipeline.py`:

| quantity | mean [95% CI] (sd) |
|---|---|
| walk-forward Sharpe | 1.49 [1.39, 1.59] (0.38) |
| walk-forward Sharpe, same-close fill, no auction cost | 1.51 [1.41, 1.61] (0.39) |
| walk-forward Sharpe, vol-targeted to 10% | 1.46 [1.36, 1.56] (0.38) |
| PBO over the 6-config entry/exit grid | 0.32 [0.26, 0.38] (0.25) |

The modelled t+1 fill costs nothing measurable on this generator (half-life
8.7 days). On real data with shorter-lived deviations it has to be measured.

Reference seed 9, trading days only: Sharpe 0.99, circular block-bootstrap
95% CI [0.25, 1.74] (20-day blocks), Lo/NW SE 0.39. The pipeline logs 9
trials on that pair (two in-sample variants, the walk-forward, its
vol-targeted version, and five more points of the entry/exit grid). The
deflated Sharpe ratio at 9 trials is 0.83, below the usual 0.95 bar.
