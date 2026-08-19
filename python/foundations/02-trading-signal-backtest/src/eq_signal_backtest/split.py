"""In-sample/out-of-sample splitting and walk-forward parameter selection.

The legacy analysis script chose parameters by grid search "on the first
70% of history only" as an inline computation
(``split = int(len(prices) * 0.7)``). That is the single most important
piece of evaluation discipline in this project -- and inline code that
important is untested and easy to silently break during a refactor. This
module factors it into two testable, reusable pieces:

* :func:`train_test_split` -- one static 70/30-style split.
* :func:`select_best_params` -- grid search *restricted to the training
  set*, returned alongside the full grid (so the sensitivity heatmap is
  still available for inspection).
* :func:`walk_forward_windows` / :func:`walk_forward_backtest` -- the
  "what I would improve" item from the legacy README, promoted to a real
  function: re-select parameters on a rolling formation window, trade the
  following (disjoint) window with those frozen parameters, and stitch
  together only the out-of-sample trading segments. A single train/test
  split answers "did it work once"; walk-forward answers it several times
  over history, which is the more convincing (though still not
  conclusive -- see ``docs/METHODOLOGY.md``) form of evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import pandas as pd

from .engine import performance_stats
from .sensitivity import parameter_grid
from .signals import ma_crossover_signal

__all__ = [
    "TrainTestSplit",
    "train_test_split",
    "select_best_params",
    "WalkForwardWindow",
    "walk_forward_windows",
    "WalkForwardResult",
    "walk_forward_backtest",
]


@dataclass(frozen=True)
class TrainTestSplit:
    """Result of :func:`train_test_split`.

    Attributes
    ----------
    train, test : pandas.Series
        Contiguous, non-overlapping slices of the input price series;
        ``train`` is the earlier (first ``train_frac``) portion.
    split_date : pandas.Timestamp
        Last date in ``train`` (the boundary the two segments share only
        as a label -- no observation is duplicated in both).
    train_frac : float
        The fraction requested, echoed back for bookkeeping.
    """

    train: pd.Series
    test: pd.Series
    split_date: Any
    train_frac: float


def train_test_split(prices: pd.Series, train_frac: float = 0.7) -> TrainTestSplit:
    """Split a price series into a contiguous train/test pair, in order.

    This is a *time-ordered* split, not a random one: shuffling a price
    history would destroy the autocorrelation structure a trading signal
    is trying to exploit and would leak future information into the
    training set via adjacent, highly-correlated observations.

    Parameters
    ----------
    prices : pandas.Series
        Close prices, ascending date index.
    train_frac : float, default 0.7
        Fraction of observations (by count, not by calendar time) placed
        in ``train``. Must be strictly between 0 and 1.

    Returns
    -------
    TrainTestSplit

    Raises
    ------
    ValueError
        If ``train_frac`` is not in ``(0, 1)``, or ``prices`` has fewer
        than 2 observations (there would be nothing to put in one of the
        two segments).
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if len(prices) < 2:
        raise ValueError(f"need at least 2 observations to split, got {len(prices)}")
    split = int(len(prices) * train_frac)
    split = min(max(split, 1), len(prices) - 1)  # keep both sides non-empty
    train, test = prices.iloc[:split], prices.iloc[split:]
    return TrainTestSplit(
        train=train, test=test, split_date=train.index[-1], train_frac=train_frac
    )


def select_best_params(
    train_prices: pd.Series,
    fast_range: Iterable[int],
    slow_range: Iterable[int],
    cost_bps: float = 5.0,
) -> tuple[int, int, pd.DataFrame]:
    """Grid-search (fast, slow) Sharpe on ``train_prices`` ONLY.

    This is where selection bias enters the pipeline: picking the argmax
    of a grid is itself a mild form of overfitting (the highest Sharpe in
    a grid of noisy estimates is, in expectation, biased upward relative
    to the true Sharpe of that parameter pair). It is not eliminated by
    restricting the search to a training set -- it is *contained*: the
    resulting parameters are then judged on data this function never
    sees. Callers must never call this on test/out-of-sample data.

    Parameters
    ----------
    train_prices : pandas.Series
        In-sample close prices (e.g. ``TrainTestSplit.train``).
    fast_range, slow_range : iterable of int
        Candidate windows, passed through to
        :func:`eq_signal_backtest.sensitivity.parameter_grid`.
    cost_bps : float, default 5.0
        Transaction cost assumption used during selection.

    Returns
    -------
    (best_fast, best_slow, grid) : (int, int, pandas.DataFrame)
        The Sharpe-maximising pair and the full sensitivity grid it was
        chosen from (for plotting / reporting).

    Raises
    ------
    ValueError
        If the grid is empty or entirely ``NaN`` (e.g. every candidate
        pair had ``fast >= slow``, or the training set is too short for
        any slow window to produce a defined Sharpe ratio).
    """
    grid = parameter_grid(train_prices, fast_range, slow_range, cost_bps)
    stacked = grid.stack() if not grid.empty else grid
    if stacked.empty:
        raise ValueError(
            "parameter grid is empty or all-NaN; widen fast_range/slow_range "
            "or lengthen train_prices"
        )
    best_fast, best_slow = stacked.idxmax()
    return int(best_fast), int(best_slow), grid


@dataclass(frozen=True)
class WalkForwardWindow:
    """One (formation, trading) pair of integer row positions in a series.

    Formation is the window parameters are *selected* on; trading is the
    disjoint, later window they are *frozen and applied* to. Positions
    are inclusive on both ends.
    """

    formation_start: int
    formation_end: int
    trading_start: int
    trading_end: int

    def __post_init__(self) -> None:
        ok = (
            0 <= self.formation_start <= self.formation_end < self.trading_start
            and self.trading_start <= self.trading_end
        )
        if not ok:
            raise ValueError(
                "malformed walk-forward window: formation and trading must be "
                f"ordered and non-overlapping, got {self}"
            )


def walk_forward_windows(
    n: int, formation: int, trading: int, step: Optional[int] = None
) -> list[WalkForwardWindow]:
    """Tile ``[0, n)`` into non-overlapping (formation, trading) windows.

    Parameters
    ----------
    n : int
        Total number of observations available.
    formation : int
        Length in observations of each formation (parameter-selection)
        window. Must be at least 2 (grid search needs a nonzero-length
        return series).
    trading : int
        Length in observations of each trading (out-of-sample) window.
        Must be at least 1.
    step : int, optional
        Number of observations to advance the whole (formation, trading)
        block between windows. Defaults to ``trading``, which produces
        contiguous, non-overlapping trading windows that -- when stitched
        -- cover every out-of-sample day exactly once. A smaller step
        re-tests more often at the cost of overlapping trading windows
        (not used by :func:`walk_forward_backtest`'s stitching, which
        assumes ``step >= trading``, but available for research).

    Returns
    -------
    list of WalkForwardWindow
        Empty if ``n`` is too short to fit even one window.

    Raises
    ------
    ValueError
        If ``formation < 2``, ``trading < 1``, or ``step <= 0``.
    """
    if formation < 2:
        raise ValueError(f"formation must be >= 2, got {formation}")
    if trading < 1:
        raise ValueError(f"trading must be >= 1, got {trading}")
    step = trading if step is None else step
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    windows: list[WalkForwardWindow] = []
    start = 0
    while True:
        formation_start = start
        formation_end = formation_start + formation - 1
        trading_start = formation_end + 1
        trading_end = trading_start + trading - 1
        if trading_end >= n:
            break
        windows.append(
            WalkForwardWindow(formation_start, formation_end, trading_start, trading_end)
        )
        start += step
    return windows


@dataclass
class WalkForwardResult:
    """Output of :func:`walk_forward_backtest`.

    Attributes
    ----------
    equity : pandas.Series
        Stitched out-of-sample strategy equity curve, restarting at 1.0
        (it is the cumulative product of the concatenated out-of-sample
        daily returns from every trading window -- NOT a splice of each
        window's own equity curve, which would double-count compounding
        at the seams).
    benchmark : pandas.Series
        Buy & hold equity curve over the same stitched out-of-sample
        dates, for comparison.
    windows : pandas.DataFrame
        One row per window: formation/trading date boundaries, the
        (fast, slow) pair selected on that formation window, trades
        executed and Sharpe realised within that trading window.
    stats : dict
        :func:`eq_signal_backtest.engine.performance_stats` computed on
        the full stitched out-of-sample return series, plus a nested
        ``"benchmark"`` entry.
    n_trades : int
        Total trades across all trading windows.
    """

    equity: pd.Series
    benchmark: pd.Series
    windows: pd.DataFrame
    stats: dict[str, Any]
    n_trades: int


def walk_forward_backtest(
    prices: pd.Series,
    fast_range: Iterable[int],
    slow_range: Iterable[int],
    formation: int,
    trading: int,
    cost_bps: float = 5.0,
    step: Optional[int] = None,
) -> WalkForwardResult:
    """Walk-forward validation: re-select parameters on a rolling window.

    For each :class:`WalkForwardWindow`: grid-search (fast, slow) on the
    formation slice only (:func:`select_best_params`), compute the signal
    over formation+trading combined so the moving averages have a proper
    warm-up (no look-ahead is introduced by this -- the signal still only
    uses prices up to and including the day it is computed on, and is
    still executed with a one-day lag), then keep only the trading-window
    portion of the resulting daily returns. Those out-of-sample-only
    segments are concatenated across all windows into one continuous
    equity curve.

    Parameters selected for window *k* are frozen for the entirety of
    trading window *k*: they are never re-fit using any day inside that
    trading window, which is the property that makes the stitched curve
    an honest out-of-sample track record rather than a sequence of
    in-sample fits.

    Parameters
    ----------
    prices : pandas.Series
        Full close-price history.
    fast_range, slow_range : iterable of int
        Candidate windows for the per-formation-window grid search.
    formation : int
        Length in observations of each formation window.
    trading : int
        Length in observations of each trading window.
    cost_bps : float, default 5.0
        Transaction cost assumption, used identically in selection and
        in the frozen-parameter trading backtest.
    step : int, optional
        See :func:`walk_forward_windows`. Defaults to ``trading``
        (contiguous, gapless stitching).

    Returns
    -------
    WalkForwardResult

    Raises
    ------
    ValueError
        If the sample is too short to form even one window (propagated
        as an explicit, informative error rather than returning an
        ambiguous empty result).
    """
    n = len(prices)
    windows = walk_forward_windows(n, formation, trading, step)
    if not windows:
        raise ValueError(
            f"sample too short for formation={formation}, trading={trading} "
            f"(n={n} rows); need at least formation + trading observations"
        )

    strat_pieces: list[pd.Series] = []
    bench_pieces: list[pd.Series] = []
    records: list[dict[str, Any]] = []

    for w in windows:
        context = prices.iloc[w.formation_start : w.trading_end + 1]
        train = prices.iloc[w.formation_start : w.formation_end + 1]
        best_fast, best_slow, _ = select_best_params(
            train, fast_range, slow_range, cost_bps
        )

        signal = ma_crossover_signal(context, best_fast, best_slow)
        position = signal.shift(1).fillna(0.0)
        rets = context.pct_change().fillna(0.0)
        trades = position.diff().abs().fillna(0.0)
        costs = trades * cost_bps / 10_000
        strat_rets_full = position * rets - costs

        local_trading_start = w.trading_start - w.formation_start
        window_strat = strat_rets_full.iloc[local_trading_start:]
        window_bench = rets.iloc[local_trading_start:]
        window_trades = trades.iloc[local_trading_start:]

        strat_pieces.append(window_strat)
        bench_pieces.append(window_bench)

        window_equity = (1 + window_strat).cumprod()
        window_stats = performance_stats(window_strat, window_equity)
        records.append(
            {
                "formation_start": prices.index[w.formation_start],
                "formation_end": prices.index[w.formation_end],
                "trading_start": prices.index[w.trading_start],
                "trading_end": prices.index[w.trading_end],
                "fast": best_fast,
                "slow": best_slow,
                "n_trades": int(window_trades.sum()),
                "window_sharpe": window_stats["sharpe"],
            }
        )

    stitched_strat = pd.concat(strat_pieces)
    stitched_bench = pd.concat(bench_pieces)
    equity = (1 + stitched_strat).cumprod()
    benchmark = (1 + stitched_bench).cumprod()

    stats = performance_stats(stitched_strat, equity)
    stats["benchmark"] = performance_stats(stitched_bench, benchmark)

    windows_df = pd.DataFrame(records)
    n_trades = int(windows_df["n_trades"].sum())

    return WalkForwardResult(
        equity=equity,
        benchmark=benchmark,
        windows=windows_df,
        stats=stats,
        n_trades=n_trades,
    )
