# Validation — FX Statistical Pairs

How the implementation was validated, with the numbers. All figures below are
reproducible: `python examples/run_pipeline.py` (seeded, offline, ~1 s) and
`pytest -q` (246 tests, ~6 s, offline).

---

## 1. Cross-validation against statsmodels (from-scratch econometrics)

| Check | Result |
|---|---|
| ADF t-stat, fixed lags (0 and 4), `regression="c"` | matches `statsmodels.adfuller` to **< 1e-10** (observed ~8e-16) |
| ADF, no-constant (`"n"`) on residual-type series | matches to < 1e-10 |
| ADF auto-lag (AIC) | selected lag identical; stat matches to < 1e-8 |
| MacKinnon critical values (N=1, finite-sample) | match statsmodels-reported values to < 1e-10 |
| Engle–Granger statistic | matches `statsmodels.coint(trend='c', autolag='aic')` to **~2e-15**; 5% critical value matches to < 1e-8 |
| EG vs plain ADF critical values | N=2 table more negative by > 0.3 at every level (asymptotic 5%: −3.336 vs −2.862) — using the N=1 table on residuals would over-reject |

## 2. Statistical size and power

* **Spurious-regression size control**: 200 seeded independent random-walk
  pairs, EG at 5% → rejection rate **4.5%** (nominal 5%; asserted < 10%).
* **Power / recovery**: planted cointegrated pair (n=1500, true beta 1.0):
  EG stat **−7.76** vs 5% cv −3.34; recovered beta **0.999**, alpha −0.0004.
  At n=2000, beta 1.4 recovered within ±0.05 (unit test).
* **Correlated ≠ cointegrated**: walks with daily return correlation 0.9
  are *not* flagged (single-case test + the funnel below).

## 3. Model recovery and cross-model consistency

* **OU recovery** (true kappa=20/yr, sigma=0.05, half-life 8.7 bd; n=1500):
  OLS kappa **20.1**, sigma **0.051**, half-life **8.7 bd**. MLE matches OLS
  (kappa within 2%, as it must — conditional MLE ≡ OLS for AR(1)); this is a
  consistency check, not an independent estimate. Known caveat: kappa carries
  the upward AR(1) small-sample bias; recovery tolerance ±25% at n=8000.
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
  leaves P&L and positions through k **bit-identical** (detector test).

## 4. The carry-flip demonstration (why spot-only FX backtests lie)

Constructed pair: base leg deposits **8%** vs hedge leg **1%** (persistent),
spot of the high-yielder drifting down at half the differential (forward
premium puzzle calibration). Identical trades, two accountings:

| | total P&L | spot | carry | Sharpe |
|---|---|---|---|---|
| spot-only backtest | **−0.143** | −0.143 | — | −0.98 |
| carry-inclusive | **+0.085** | −0.143 | **+0.228** | +0.58 |

The signal is systematically long the 'cheap' high-yielder: on spot it bleeds
(the drift never stops), while the position earns the differential. Ignoring
carry does not just misstate the P&L — **it flips its sign**. The carry-aware
entry filter on the same data keeps all 874 carry-favourable long-entry bars
and vetoes 21 of 26 short-entry bars (shorts pay 7.0%/yr; veto threshold
|z| ≈ 2.26 at a 6-half-life expected hold).

Cost sensitivity on the planted pair (same signal path):

| costs | total P&L | costs paid | Sharpe |
|---|---|---|---|
| frictionless | +0.194 | 0 | +1.51 |
| major (0.7/1.0 pips) | +0.190 | −0.005 | +1.48 |
| EM (60/30 pips) | **−0.065** | −0.259 | −0.45 |

## 5. Failure modes (documented AND simulated)

### 5.1 SNB-style floor-then-break (the central case study)

Generator: 750 days of tight OU around a floor (EURCHF 2011–2015 analogue),
spread pinned below its mean at the floor edge, then a one-day **−15%** gap
and a high-vol regime. Pipeline numbers (seed 3):

* Formation scan on the pegged period: EG stat **−5.82** → "cointegrated",
  the best-looking spread a scan could find. That is the trap.
* Pre-break: cumulative P&L **+0.058** over 750 days, hit rate 0.75 — the
  classic steady grind that builds false confidence (and, on a vol-targeted
  book, maximal leverage: realised spread vol was collapsing).
* Break day: **−0.150** in one day — **2.6×** every franc the strategy ever
  made; full-sample P&L −0.060. A z-stop at |z|=25 changes nothing: the
  market gaps straight through any stop (asserted in
  `test_scenarios.py::test_stop_cannot_prevent_the_gap_loss`).
* Position on break eve: +1, long the pegged pair — with the crowd.

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

Step devaluations (CNY 2015, ARS repeatedly) are the A3 jump failure — same
shape as 5.1. Even without events, EM pip spreads of 30–150 turn the
strategy's gross edge negative (table above); the EM-vs-major cost test
asserts the sign flip at identical signals.

### 5.4 Crowded carry unwinds

Risk-off episodes flip the commodity-bloc/safe-haven correlation structure:
in the synthetic two-block panel, intra-block return correlation stays ~0.85
while cross-block correlation flips from **+0.09 to −0.57** in risk-off
windows (detectable on a 40-day rolling correlation — tested). Pairs that
diversified in calm regimes become one trade in stress; carry longs and RV
spreads that share the same high-yielder unwind together. Control: per-block
exposure limits (`DESK_GUIDE.md`), regime monitoring on rolling correlations.

## 6. Edge cases (each documented and unit-tested)

| Edge case | Behaviour | Test |
|---|---|---|
| Pegged / zero-vol instrument in universe | dropped from screen with `UserWarning`; OU fit on its spread raises `ValueError` | `test_universe`, `test_scenarios` |
| Triangular identity spread | flagged degenerate, never cointegrated, ADF skipped (NaN stat) | `test_cointegration`, `test_scenarios` |
| Zero-trade backtest | all-zero ledgers, NaN Sharpe, 0 trades — no crashes | `test_backtest` |
| Missing days / calendar gaps | engine runs; carry accrues actual calendar days (Fri→Mon = 3/365; gap of 10 days = 10/365) | `test_backtest`, `test_carry` |
| Weekend/holiday accrual total | sums exactly to calendar days/365 | `test_carry` |
| Rate spike on last day | cannot affect prior accruals (lagged rates) | `test_carry` |
| Series too short / NaNs / mismatched lengths / bad thresholds | informative `ValueError`s | all modules |
| **±Inf in a series** (logged zero/missing price) | rejected by `adf_test`, `engle_granger`, OU fitters and `run_backtest`; previously accepted by `isnan`-only guards and returned "not cointegrated" | `test_nan_guards` (§6.1) |
| **NaN threshold / parameter** (`stop`, `sigma`, `notional`, `min_abs_corr`, `ann_factor`) | rejected; previously disabled the hard stop, muted the strategy or emptied the screen silently | `test_nan_guards` (§6.1) |
| Zero or non-positive price level | rejected before `log`, with a message naming the series | `test_nan_guards` (§6.1) |
| Vol-target on collapsing vol | scale capped at max leverage (the SNB lesson) | `test_signals` |
| Sub-5% EG rejection on pure noise | size control holds | `test_cointegration` |

### 6.1 Non-finite inputs: the guards that silently muted the strategy
(`tests/test_nan_guards.py`)

Two distinct defect patterns were found and closed; both produced a
*plausible-looking* result instead of an error, which is why neither had
been noticed.

**Pattern 1 — inequality-only guards.** `if sigma <= 0`, `if stop <= entry`,
`if notional <= 0`, `if target_vol <= 0`, `if delta <= 0`. Every comparison
against NaN is False, so NaN passed every one of them. Specific
consequences reproduced in tests:

* NaN `sigma` in `zscore(mu=…, sigma=…)` → all-NaN z-score → the state
  machine stays flat for the entire sample → the backtest reports a clean
  zero-P&L, zero-trade run. A strategy that never traded, presented as a
  result.
* NaN `stop` in `generate_positions` → **the hard stop is disabled**, because
  `state * z <= -nan` is always False. That stop is the single control that
  exists to limit a regime break (§5, the SNB floor case). It would have been
  switched off with no message.
* NaN `min_abs_corr` in `correlation_screen` → every `abs(rho) >= threshold`
  False → an empty candidate table, indistinguishable from "the universe
  contains no tradable pairs".
* A non-finite deposit rate in `run_backtest(rates=…)` → NaN carry accrual →
  NaN carry P&L → a NaN Sharpe in the summary. The error now names the
  offending dict key (`rates['rq2']`).
* NaN `ann_factor` in the metrics → NaN Sharpe/Sortino/turnover.

**Pattern 2 — `isnan`-only guards on series.** `adf_test`, `engle_granger`,
`_validate_spread` (OU fitting) and `run_backtest` all tested
`np.isnan(x).any()` and therefore **accepted ±Inf**. Inf, not NaN, is the
realistic corruption in this package: a zero or missing fixing becomes
`-inf` the instant it passes through `log`. The consequence was the most
serious defect found here — an Inf in the input made `lstsq` return NaN
coefficients, the ADF tau NaN, and `bool(nan < critical_value)` **False**,
so `engle_granger` reported **"not cointegrated"** on data it could not test.
A false negative identical in appearance to a genuine one. All four
validators now use `np.isfinite`, and price-level inputs are additionally
required to be strictly positive so the zero-price case raises *before* the
`log`.

Positive companions in the same file guard against over-rejection: the
planted cointegrated pair is still detected (`cointegrated=True`,
beta ≈ 1.0); a finite stop still fires when z walks to −6 and the position is
flat afterwards; the vol-target cap still binds at 10× on a near-pegged
spread; `half_life_days` still returns `inf` for `kappa <= 0` and
`ln2/(kappa·dt)` otherwise; RLS at `lam=1` still converges to the batch OLS
beta to 1e-6; the correlation screen still finds the EURUSD/GBPUSD block; and
the clean backtest still satisfies `total = spot + carry + costs` to 1e-12.

**RLS state integrity.** `RLSHedge.update` now validates *before* mutating
`theta` and `P`. This matters because the filter has no recovery mechanism: a
single non-finite tick would have made every subsequent hedge ratio NaN for
the rest of the run. A test asserts `(alpha, beta, n_obs)` are bit-identical
after a rejected update.

**Deliberate asymmetry.** The metrics layer still *tolerates* NaN in a P&L
series (it drops non-finite entries) because a warmup window legitimately
produces NaN P&L before the first position exists; the pricing, signal and
backtest layers reject. This split is now documented in `metrics._clean` and
asserted in `test_metrics_still_tolerate_warmup_nans_in_pnl`.

## 7. Metrics validation

Sharpe/Sortino/MDD/hit-rate/turnover hand-computed on toy ledgers; Lo (2002)
SE: collapses to the classical `sqrt((1+SR²/2)/T)` for iid returns (q=0,
match to 1e-10) and inflates by > 1.5× on AR(1)-correlated P&L with rho=0.6 —
the honest error bar for autocorrelated mean-reversion books (walk-forward
Sharpe above: **0.77 ± 0.41** — barely 2 SEs from zero, which is exactly the
kind of humility the Lo SE is there to enforce).
