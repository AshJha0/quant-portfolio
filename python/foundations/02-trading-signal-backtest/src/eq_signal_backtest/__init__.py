"""eq_signal_backtest -- long/flat moving-average crossover signal backtest.

Pipeline: signal (:func:`ma_crossover_signal`) -> one-day execution lag +
transaction costs (:func:`run_backtest`) -> summary statistics
(:func:`performance_stats`) -> parameter sensitivity
(:func:`parameter_grid`) -> in-sample/out-of-sample and walk-forward
evaluation discipline (:mod:`eq_signal_backtest.split`).

The strategy is deliberately simple. The point of this project is the
*evaluation discipline* around it: no-look-ahead execution, explicit
transaction costs, a train/test split, a walk-forward variant, and a
parameter-sensitivity map that exposes overfitting -- see
``docs/METHODOLOGY.md`` and ``docs/VALIDATION.md``.
"""

from .engine import (
    TRADING_DAYS,
    BacktestResult,
    performance_stats,
    run_backtest,
    strategy_returns,
)
from .sensitivity import parameter_grid
from .signals import ma_crossover_signal
from .split import (
    TrainTestSplit,
    WalkForwardResult,
    WalkForwardWindow,
    select_best_params,
    train_test_split,
    walk_forward_backtest,
    walk_forward_windows,
)

__version__ = "1.0.0"

__all__ = [
    "TRADING_DAYS",
    "BacktestResult",
    "ma_crossover_signal",
    "strategy_returns",
    "run_backtest",
    "performance_stats",
    "parameter_grid",
    "TrainTestSplit",
    "train_test_split",
    "select_best_params",
    "WalkForwardWindow",
    "walk_forward_windows",
    "WalkForwardResult",
    "walk_forward_backtest",
]
