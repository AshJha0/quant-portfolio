# Validation — MA-Crossover Signal Backtest

How the implementation was validated (contract items 3, 4, 6). All numbers
below are reproduced by `python examples/run_pipeline.py` (seeded, offline,
~3 s) and the test suite (`pytest -q`, 99 tests, ~3 s, offline).

---

## 1. No-look-ahead: structural proof, not inspection

`tests/test_engine.py::TestNoLookAhead` verifies the execution lag four
ways:

1. **Direct structural equality.** For a hand-built signal, `position.iloc[t]
   == signal.iloc[t-1]` is asserted for every `t` in the sample
   (`test_position_equals_prior_day_signal_structurally`) — not "the code
   looks right", but every single day checked.
2. **Leading-fill correctness.** `position.iloc[0] == 0.0` (never `NaN`,
   regardless of what `signal.iloc[0]` was) —
   `test_first_day_position_is_zero_not_nan`.
3. **A detector test.** `test_cheat_profits_from_a_jump_honest_engine_does_not`
   builds a price series that is flat, then jumps 30% overnight, with a
   signal that fires exactly on the jump day. Same-day ("cheat")
   execution captures the jump (`cheat_equity.iloc[-1] > 1.25`); the
   engine's t-1 lag means the position on the jump day was decided
   *before* the jump happened, so the honest engine's equity is unchanged
   (`== 1.0` to `1e-9`). This is the same style of proof used for the
   `eq_pairs` project's no-lookahead test: construct a case where cheating
   is profitable, and show the code under test does not take it.
4. **Unreachability.** `test_position_never_uses_same_day_signal_value`:
   a signal that is 1 only on the *final* day of the sample can never
   produce a nonzero position anywhere, because there is no `t+1` for the
   engine to trade it on.

## 2. Exact transaction-cost accounting

`tests/test_engine.py::TestTransactionCosts::test_position_change_incurs_exact_cost_bps_drag`
constructs a 3-day series where the position moves 0→1 on day 2, runs the
backtest once with `cost_bps=5.0` and once with `cost_bps=0.0`, and asserts
the difference in that day's return is **exactly** `5 / 10_000 = 0.0005`
(`abs=1e-12`). `test_no_position_change_no_cost` asserts the costed and
cost-free equity curves are bit-for-bit identical when the position never
changes. `test_round_trip_pays_cost_on_each_leg` checks a 3-trade round
trip charges exactly 3 separate costs. `test_cost_free_always_long_matches_buy_and_hold_exactly`
proves `cost_bps=0` with an always-on signal reproduces buy-and-hold
exactly (`pd.testing.assert_series_equal`, not `approx`) — both day-0
returns are independently forced to `0.0` by `pct_change().fillna(0.0)`,
so the two curves are identical, not merely close.

## 3. performance_stats against hand-computable equity curves

`tests/test_engine.py::TestPerformanceStats`:

| Statistic | Construction | Check |
|---|---|---|
| CAGR | 252 days of a constant daily return solved from a target 20% CAGR | `stats["cagr"] == 0.20` to `rel=1e-9` |
| Max drawdown | Hand-built equity `[1.0, 1.2, 0.9, 1.1]` → drawdown `-0.25` at day 2 | `stats["max_drawdown"] == -0.25` to `1e-9` |
| Sharpe | Returns alternating `mu+d` / `mu-d` (closed-form mean = `mu` exactly, closed-form `ddof=1` std = `d*sqrt(n/(n-1))`) | `stats["sharpe"]` matches the closed-form value to `rel=1e-9` |
| Sharpe (degenerate) | All-zero returns (exact IEEE-754 zero variance) and a single-observation series | both give `NaN`, never `0` or `±inf` |

A subtlety documented and deliberately *not* over-engineered: a
**repeated non-zero decimal** (e.g. `[0.001] * 20`) does **not** produce
exactly zero variance in floating point — `pandas`' two-pass mean/variance
picks up ~1e-19-scale rounding noise, which is enough to make the
`std > 0` guard evaluate `True` and return a very large (not `NaN`, not
`inf`) Sharpe. The zero-variance test therefore uses literal `0.0` returns
(the case that actually occurs in this codebase — an all-flat signal
produces exact zero returns, since `0.0 * anything == 0.0` exactly), which
is both the realistic degenerate case and the one that is exactly
representable.

## 4. Parameter grid: shape and exclusion

`tests/test_sensitivity.py` checks the pivoted grid's index/column sets
match the requested ranges, that a cell for an excluded `fast >= slow`
pair is present-but-`NaN` when other cells in that row are valid (pivot
fills the rectangle) and *absent as a row entirely* when every combination
for that `fast` is invalid, that an all-invalid range returns an empty
`DataFrame` rather than raising, and that a grid cell's value matches an
independent direct call to `run_backtest` for the same `(fast, slow)`.

## 5. Walk-forward hygiene

`tests/test_split.py::TestWalkForwardWindows` and `TestWalkForwardBacktest`:

- Formation and trading windows **cannot overlap**
  (`WalkForwardWindow.__post_init__` raises `ValueError` on construction;
  asserted over full schedules too).
- Default `step = trading` produces **contiguous** trading windows (no
  gap, no overlap) so the stitched curve covers every out-of-sample day
  exactly once.
- **Parameters are frozen during the trading window**:
  `test_frozen_parameters_reproduce_window_returns_exactly` reconstructs
  the first trading window's exact daily strategy *returns* from the
  recorded `(fast, slow)` pair and the formation+trading price context
  alone, and matches the walk-forward engine's stitched output to
  `atol=1e-10`. If parameters were being re-estimated inside the trading
  window, or if the formation history leaked into the trading return
  computation, this would diverge.
- A too-short sample raises an informative `ValueError` ("too short")
  rather than silently returning an empty or ambiguous result, and the
  message names the offending sizes (`test_error_message_names_the_offending_sizes`).
  `TestWalkForwardWindowsLargerThanData` covers the whole family:
  formation alone longer than the sample, trading alone longer, the two
  together overshooting by one observation, and the exact boundary where
  `formation + trading == n` fits precisely one window while one fewer
  observation fits none.
- **Stitching is compounding, not splicing**
  (`TestWalkForwardStitching`): the stitched out-of-sample index is
  unique, monotonic and exactly `trading * n_windows` days long (every
  out-of-sample day counted once, none dropped), and the final equity
  equals the running product of each window's own gross return to
  `rel=1e-12`. A naive splice of per-window equity curves — each
  restarting at 1.0 — would fail this.
- **The formation window can be too short to select anything.** If no
  candidate `(fast, slow)` pair warms up inside the formation window,
  every grid cell is a flat, zero-variance strategy with a `NaN` Sharpe.
  `select_best_params` raises rather than "selecting" an arbitrary corner
  of an all-`NaN` grid
  (`test_formation_window_too_short_for_the_slow_ma_fails_loudly`).

## 6. Headline numbers: bundled synthetic data, seed=32 (`examples/run_pipeline.py`)

10 years of synthetic daily closes (2016-12-19 .. 2026-08-14, two-regime
generator, seed 32), 70/30 train/test split, grid search
`fast ∈ {10,...,70}`, `slow ∈ {100,...,250}`, `cost_bps=5`.

**About the bundled data.** The generator's regime parameters are set so
its *stationary* behaviour looks like a single large-cap equity — ~8%/yr
long-run drift, ~21% annualised volatility, ~17% of days in a stressed
regime, and 30-50% peak drawdowns — rather than a trend-follower's dream.
The seed matters more than one might like: 10-year paths from this model
run from about -14% to +28% CAGR depending on the draw, so seed 32 was
picked because its path sits near the model's central case (buy & hold:
+8.66% CAGR, 21.1% vol, Sharpe 0.50, -46.9% max drawdown), not because it
flatters the strategy. It does not.

Parameters selected on the training window: **fast=10, slow=125**.

| | strategy | buy & hold |
|---|---:|---:|
| In-sample CAGR / Sharpe / maxDD | +12.39% / 0.87 / -30.06% | +15.73% / 0.82 / -46.57% |
| **Out-of-sample** CAGR / Sharpe / maxDD | **-2.27% / -0.12 / -26.50%** | -7.25% / -0.21 / -46.93% |
| Full-period CAGR / Sharpe / maxDD (32 trades) | +7.39% / 0.54 / -34.06% | +8.66% / 0.50 / -46.93% |
| Full-period, zero costs | +7.57% / 0.55 / -33.67% | — |

**Key number 1 — why the in-sample Sharpe should be distrusted.**
In-sample Sharpe **0.87** → out-of-sample Sharpe **-0.12**. That is not
decay, it is disappearance: the strategy that looked like a 0.87-Sharpe
trend follower on the window its parameters were chosen from lost money
on the window they weren't. By the pipeline's own printed threshold
(`Sharpe_oos < 0.5 * Sharpe_is` → "substantial decay"), this is the
unambiguous case, and it is what a working evaluation harness is supposed
to produce when handed a strategy without a real edge.

A reviewer should read this as the project's *result*, not its
embarrassment. The alternative — a bundled dataset tuned until the
strategy worked — would demonstrate nothing except that synthetic data can
be made to say anything.

**Key number 2 — strategy vs. buy & hold: drawdowns, not returns.** The
one property that reproduces in both windows is drawdown control: -30.06%
(in-sample) and -26.50% (out-of-sample) against buy & hold's -46.57% and
-46.93%. Out-of-sample, over a period when the asset itself lost 7.25%/yr,
the strategy lost only 2.27%/yr and had the better (less negative) Sharpe
of the two. That is the textbook trend-following signature described in
`docs/METHODOLOGY.md` — step aside during sustained declines — and it
survives out-of-sample even though the alpha does not. It is also not, on
its own, a reason to trade the strategy: a cash allocation delivers
drawdown control more cheaply and more reliably.

**Key number 3 — transaction cost drag.** At the selected parameters
(32 trades over 10 years) the drag is small: full-period CAGR falls from
7.57% (zero cost) to 7.39% (5 bps), **0.17 percentage points**. Section 7
shows how quickly that grows with turnover.

**Walk-forward, 7 rolling windows (formation 756d / trading 252d):**

| metric | walk-forward stitched OOS | buy & hold (same dates) |
|---|---:|---:|
| CAGR | -2.26% | +0.73% |
| Sharpe | -0.06 | 0.15 |
| Max drawdown | -43.93% | -46.93% |
| Trades | 30 | — |

Per-window selected parameters and realised Sharpe swing violently — the
seven windows produced Sharpes of -0.61, +1.76, +0.54, -1.55, +0.48,
-0.57 and -0.91, with the selected `slow` window jumping between 100 and
250 as the formation data changed. "The strategy" is really seven
different re-fitted strategies, four of which lost money in their trading
window. The stitched out-of-sample Sharpe (-0.06) is the number that best
answers "what would an investor who mechanically re-optimised every year
actually have earned", and it agrees with the single-split out-of-sample
result (-0.12) rather than rescuing it. Two independent evaluation
protocols reaching the same negative conclusion is considerably stronger
evidence than either alone (Assumption A6/A7 in `docs/METHODOLOGY.md`).

**Key number 4 — parameter sensitivity, and why a plateau proves less
than it looks.** In-sample grid Sharpe ranges from 0.19 to 0.87, and
**39% of grid cells sit within 25% of the best Sharpe** — a broad plateau,
which is conventionally read as evidence *against* curve-fitting (a
genuinely overfit result is supposed to look like one green cell in a red
sea). Here that reading would be wrong: the plateau is real and the
out-of-sample Sharpe is still negative. A plateau says the in-sample
result does not depend on one lucky parameter cell; it says nothing about
whether the whole neighbourhood is fitted to the same noise. The pipeline
prints this reconciliation in its own output rather than leaving the
heatmap to imply robustness it cannot support (see
`output/figures/backtest_overview.png`, bottom-right panel).

## 7. Cost sensitivity: unrealistically low vs. realistic

Same data, varying `(fast, slow)` and `cost_bps` together (turnover rises
sharply as the windows shorten):

| (fast, slow) | trades (10y) | Sharpe @ 0bps | Sharpe @ 5bps | Sharpe @ 20bps | CAGR @ 0bps | CAGR @ 5bps | CAGR @ 20bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| (10, 50)  | 55 | 0.719 | 0.702 | 0.648 | 10.54% | 10.24% | 9.33% |
| (10, 100) | 40 | 0.587 | 0.574 | 0.535 | 8.11% | 7.89% | 7.25% |
| **(10, 125) — selected** | **32** | **0.550** | **0.540** | **0.509** | **7.57%** | **7.39%** | **6.88%** |
| (60, 250) | 12 | 0.078 | 0.074 | 0.063 | 0.01% | -0.05% | -0.23% |

Two things to read off this table, and they pull in opposite directions —
which is the honest version of the story:

1. **Cost drag scales with turnover, exactly as expected.** Going from 0
   to 20 bps costs the 55-trade (10, 50) pair **1.21 percentage points**
   of CAGR and 0.071 of Sharpe; it costs the 12-trade (60, 250) pair
   0.24pp and 0.015. A backtest run at `cost_bps=0` overstates a fast
   pair's edge by roughly five times as much as a slow pair's, and the
   error is systematic, not noise: it always favours over-trading.
2. **On this data, costs do not reverse the ranking.** The fastest pair
   (10, 50) has the highest Sharpe at every cost level tested, including
   20 bps. The selection step still did not choose it, because
   `select_best_params` searches the *training* window only, where a
   different pair won. That is a useful illustration in itself: which
   parameters look best is not stable across sub-samples, which is the
   same instability the walk-forward windows show and the same
   instability that produces the negative out-of-sample result.

The general lesson stands and is why `parameter_grid`/`select_best_params`
always run *with* `cost_bps` set (default 5.0), never cost-free: a
cost-free grid search is biased toward over-trading parameter pairs, and
whether that bias happens to flip the winner on one particular sample is
luck, not a reason to omit costs. On the choppy path in §8, where turnover
is much higher relative to the moves captured, the same 5 bps is the
difference between -8.38% and -8.59% CAGR — still not the deciding factor
there either, because that failure is a signal failure, not a cost
failure.

## 8. Failure mode: choppy / range-bound regime (whipsaw losses)

The synthetic generator in `data/synthetic.py` is regime-switching
between an uptrend and a drawdown regime, so its default output has
persistent directional moves in it — enough that a trend-following signal
has something to aim at, even though on this seed it fails to convert
that into out-of-sample performance (§6 above). To reproduce the failure mode this strategy class is
documented to have — **choppy, range-bound markets, where every crossover
is a whipsaw that pays costs and captures no move** — a genuinely
mean-reverting synthetic path is needed instead. `tests/test_edge_cases.py::TestChoppyRegimeFailureMode`
builds one: an Ornstein-Uhlenbeck-style log-price
(`x_t = x_{t-1} + kappa*(0 - x_{t-1}) + sigma*eps_t`, `kappa=0.05,
sigma=0.012`, seed 99, 750 days) that oscillates in a band around its
starting level with no persistent drift, then runs the standard
`ma_crossover_signal(fast=20, slow=100)` on it:

| | strategy (5bps costs) | buy & hold |
|---|---:|---:|
| Trades | 14 | — |
| CAGR | **-8.59%** | +1.31% |
| Sharpe | **-0.60** | 0.16 |
| Max drawdown | -24.93% | (range-bound; smaller) |

The strategy loses money outright while buy & hold is roughly flat-to-
slightly-positive: 14 crossovers over 3 years, each one entering just as
the "trend" reverses (because there is no trend — mean reversion, by
construction). `test_costs_make_the_whipsaw_regime_strictly_worse` further
checks the cost-free variant of the same signal is *less bad*
(CAGR -8.38% vs. -8.59%) but still loses — the whipsaw loss is a signal
problem, not merely a cost problem, in this regime. This reproduces, with
an exact seeded example, the regime-dependence statement in
`docs/METHODOLOGY.md` Assumption A3/§1.

## 9. Failure modes summary (contract item 4)

- **Selection bias in the grid search (§6, quantified):** the in-sample
  Sharpe is optimistic — on this data, entirely so (0.87 in-sample,
  -0.12 out-of-sample). Walk-forward re-selection does not eliminate the
  bias, it re-applies it every window (Assumption A5), and reaches the
  same negative conclusion.
- **A plateau in the parameter grid is weaker evidence than it looks
  (§6):** 39% of cells within 25% of the best Sharpe, and still no
  out-of-sample edge. Plateau vs. spike distinguishes "one lucky cell"
  from "a lucky neighbourhood"; it cannot distinguish either from
  "the whole surface is fitted to the same noise".
- **Regime dependence (§8, tested):** loses in range-bound/mean-reverting
  regimes; the same mechanism that limits drawdown in a bear market
  (stepping aside) produces repeated small losses in a choppy one.
- **Cost sensitivity (§7, tested via `TestTransactionCosts`):** fast
  parameter pairs are cost-fragile — 20 bps costs the fastest grid pair
  1.21pp of CAGR against 0.24pp for the slowest — and a zero-cost
  assumption is systematically biased toward over-trading.
- **Single asset, single simulated history (Assumption A7):** every
  number above describes one seed's synthetic path (or, via `data/live.py`,
  one real ticker's one realised history). None of it is evidence about
  how the effect behaves across assets — and the seed dispersion noted in
  §6 (-14% to +28% CAGR across draws) is a direct measure of how little a
  single path can tell you.

## 10. Data-quality and degenerate-input guards (contract item 6)

The engine used to accept several kinds of bad input and return a
plausible-looking number. Each is now rejected at the door, and each
rejection is a test in `tests/test_edge_cases.py`:

| Input | Old behaviour | Now |
|---|---|---|
| `NaN` in prices | `pct_change().fillna(0.0)` recorded the gap as a **flat day**, silently swallowing the move across it | `ValueError`, message explains the gap must be resolved deliberately (`TestNonFinitePrices`) |
| `inf` in prices | propagated into `NaN` equity | `ValueError` |
| A price of exactly `0.0` | next day's `pct_change` is `inf`; equity `NaN` from that point on, with only a `RuntimeWarning` | `ValueError` ("strictly positive") |
| Negative price | drawdown/return arithmetic silently meaningless | `ValueError` |
| Signal index ≠ price index | pandas outer-joined them into `NaN` positions and reported a wrong curve | `ValueError` ("share the exact index") |
| Signal containing `NaN` | `NaN` positions, `NaN` equity | `ValueError` |
| Signal of `0.5` / `-1.0` / `2.0` | multiplied straight through, as if fractional sizing and shorting were supported and costed | `ValueError` (long/flat only) |
| `cost_bps` negative or `NaN` | negative costs *paid* the strategy to trade | `ValueError` |
| `fast`/`slow` below 1, or non-integer | pandas' own rolling-window error, or silent nonsense | `ValueError` naming the window |

**Extreme transaction costs** get their own class
(`TestExtremeTransactionCosts`) because the old failure was quantitative
rather than categorical. With a one-way cost above 100% of traded value
(`cost_bps > 10_000`), the daily strategy return fell below -100%, so
`(1 + r)` went negative: the equity curve flipped sign on every subsequent
trade and produced a **finite, recovered-looking CAGR** out of what should
have been a total loss. The engine now floors the daily strategy return at
-1.0, so:

- a 200%-per-leg cost wipes equity to exactly zero, and zero is absorbing
  (`test_zero_is_an_absorbing_state`) — no later gain resurrects it;
- a 50%-per-leg cost (100% round trip) is survivable and merely brutal:
  equity stays strictly positive and decays geometrically;
- final equity is monotonically non-increasing in `cost_bps` across
  0 → 5 → 50 → 500 → 5,000 → 20,000 bps
  (`test_costs_are_monotonically_worse`).

**Windows longer than the sample** are *not* an error: both moving
averages are `NaN` throughout, `NaN > NaN` is `False`, so the signal is
flat everywhere, the strategy never trades and equity stays at 1.0
(`test_windows_longer_than_the_sample_give_an_all_flat_signal`). A
strategy that never warms up simply never trades, which is the correct
answer rather than an exception.

**Other degenerate edges (all unit-tested):** an all-flat signal produces
exactly zero trades and a constant equity curve (`TestAllFlatSignal`);
zero-variance returns give `NaN` Sharpe, never 0 or ±inf; a
single-observation series does not crash `run_backtest` or
`ma_crossover_signal` (`TestSingleDaySeries`); `cost_bps=0` with an
always-on signal reproduces buy & hold bit-for-bit.
