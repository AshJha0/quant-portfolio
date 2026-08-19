# Validation — MA-Crossover Signal Backtest

How the implementation was validated (contract items 3, 4, 6). All numbers
below are reproduced by `python examples/run_pipeline.py` (seeded, offline,
~3 s) and the test suite (`pytest -q`, 64 tests, ~2 s, offline).

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
  rather than silently returning an empty or ambiguous result.

## 6. Headline numbers: bundled synthetic data, seed=2 (`examples/run_pipeline.py`)

10 years of synthetic daily closes (2016-12-19 .. 2026-08-14, two-regime
generator, seed 2), 70/30 train/test split, grid search
`fast ∈ {10,...,70}`, `slow ∈ {100,...,250}`, `cost_bps=5`:

Parameters selected on the training window: **fast=20, slow=200**.

| | strategy | buy & hold |
|---|---:|---:|
| In-sample CAGR / Sharpe / maxDD | +7.12% / 0.70 / -11.74% | +7.92% / 0.62 / -26.68% |
| **Out-of-sample** CAGR / Sharpe / maxDD | **+6.20% / 0.64 / -9.77%** | +4.21% / 0.37 / -27.11% |
| Full-period CAGR / Sharpe / maxDD (10 trades) | +8.70% / 0.81 / -11.74% | — |
| Full-period, zero costs | +8.75% / 0.81 / -11.74% | — |

**Key number 1 — why the in-sample Sharpe should be distrusted.**
In-sample Sharpe 0.70 → out-of-sample Sharpe 0.64: a **~9% relative
decay**, i.e. the honest number retains ~91% of the in-sample estimate on
this run. This is a *moderate* decay by the project's own
`Sharpe_oos < 0.5 * Sharpe_is` threshold for "substantial decay" (see
`run_pipeline.py`'s printed "honest read"), and moderate decay from a
single split is still weak evidence on its own — which is exactly why
walk-forward validation (below) exists.

**Key number 2 — strategy vs. buy & hold, drawdowns not just CAGR.** The
strategy's full-period max drawdown (**-11.74%**) is well under half of
buy & hold's (**-26.68% to -27.11%**) across every reported window, while
CAGR is close to or below buy & hold's. This is the textbook trend-
following signature described in `docs/METHODOLOGY.md`: the strategy earns
its keep by avoiding the worst of the drawdown, not by out-compounding a
rising market.

**Key number 3 — transaction cost drag.** At the selected (slow-turnover,
10-trade) parameters, cost drag is small: full-period CAGR falls from
8.75% (zero cost) to 8.70% (5 bps) — a **0.05 percentage-point** drag.
This looks reassuring in isolation, but section 7 below shows it is
parameter-dependent: faster pairs pay far more.

**Walk-forward, 7 rolling windows (formation 756d / trading 252d):**

| metric | walk-forward stitched OOS | buy & hold (same dates) |
|---|---:|---:|
| CAGR | +8.05% | +10.56% |
| Sharpe | 0.69 | 0.79 |
| Max drawdown | -18.53% | -27.11% |
| Trades | 22 | — |

Per-window selected parameters and realised Sharpe swing considerably
(e.g. window 4: fast=10, slow=250, Sharpe **-0.09**; window 5: fast=20,
slow=175, Sharpe **+2.11**) — a reminder that "the strategy" is really a
sequence of seven different re-fitted strategies here, and any one
window's result is noisy. The **stitched** out-of-sample Sharpe (0.69) is
the number that best answers "what would an investor who mechanically
re-optimised every year actually have earned", and it sits close to (in
this run, slightly below) the single-split out-of-sample Sharpe (0.64),
which is a mild reassurance that the single-split result is not a fluke
of that particular boundary — but seven correlated, overlapping-regime
windows on one simulated history is still evidence, not proof (Assumption
A6/A7 in `docs/METHODOLOGY.md`).

**Key number 4 — parameter sensitivity.** In-sample grid Sharpe ranges
from 0.16 to 0.70; **37% of grid cells sit within 25% of the best Sharpe**
— a broad plateau, not a lone spike. Consistent with a real, if modest,
trend-following effect on this data rather than a curve-fit accident: a
genuinely overfit result typically looks like a single green cell in a
red sea (compare `output/figures/backtest_overview.png`, bottom-right
panel).

## 7. Cost sensitivity: unrealistically low vs. realistic

Same data, varying `(fast, slow)` and `cost_bps` together (turnover rises
sharply as the windows shorten):

| (fast, slow) | trades (10y) | Sharpe @ 0bps | Sharpe @ 5bps | Sharpe @ 20bps | CAGR @ 0bps | CAGR @ 5bps | CAGR @ 20bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| (10, 50)  | 76 | 0.368 | 0.331 | 0.222 | 3.36% | 2.96% | 1.79% |
| (10, 100) | 44 | 0.449 | 0.428 | 0.365 | 4.29% | 4.06% | 3.37% |
| **(20, 200) — selected** | **10** | **0.813** | **0.809** | **0.795** | **8.75%** | **8.70%** | **8.53%** |
| (60, 250) | 8  | 0.639 | 0.635 | 0.625 | 6.75% | 6.70% | 6.58% |

The lesson the legacy README called out is reproduced exactly: **at
`cost_bps=0` the fast (10, 50) pair (Sharpe 0.368) already trails the
selected (20, 200) pair (Sharpe 0.813)**, and the gap widens further as
costs rise to a realistic 5-20 bps, because 76 round-trip-adjacent trades
over 10 years pay costs far more often than 10. Setting `cost_bps` to an
unrealistically low value (0-1 bp) would make the fast pair look
competitive with the selected one; at realistic costs it is not close.
This is why `parameter_grid`/`select_best_params` always run *with*
`cost_bps` set (default 5.0), never cost-free — a cost-free grid search
would systematically favour over-trading parameter pairs.

## 8. Failure mode: choppy / range-bound regime (whipsaw losses)

The synthetic generator in `data/synthetic.py` is regime-switching
between a calm uptrend and a stressed downtrend, so its default output is
directional enough that a trend-following signal has something to catch
(§6 above). To reproduce the failure mode this strategy class is
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

- **Regime dependence (§8, tested):** loses in range-bound/mean-reverting
  regimes; the same mechanism that limits drawdown in a bear market
  (stepping aside) produces repeated small losses in a choppy one.
- **Cost sensitivity (§7, tested via `TestTransactionCosts`):** fast
  parameter pairs are cost-fragile; an unrealistic zero-cost assumption
  masks this and would select the wrong parameters.
- **Selection bias in the grid search (§6, quantified):** the in-sample
  Sharpe is optimistic; walk-forward re-selection does not eliminate this,
  it re-applies it every window (Assumption A5).
- **Single asset, single simulated history (Assumption A7):** every
  number above describes one seed's synthetic path (or, via `data/live.py`,
  one real ticker's one realised history). None of it is evidence the
  effect generalises across assets.
- **Degenerate numeric edges (all unit-tested):** an all-flat signal
  produces exactly zero trades and a constant equity curve
  (`TestAllFlatSignal`); zero-variance returns give `NaN` Sharpe, never 0
  or ±inf; a single-observation series does not crash `run_backtest` or
  `ma_crossover_signal` (`TestSingleDaySeries`); `cost_bps=0` with an
  always-on signal reproduces buy & hold bit-for-bit.
