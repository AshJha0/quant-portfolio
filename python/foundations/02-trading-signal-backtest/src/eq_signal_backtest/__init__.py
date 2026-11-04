"""eq_signal_backtest -- long/flat moving-average crossover signal backtest.

Pipeline: signal (:func:`ma_crossover_signal`) -> modelled t+1 fill +
transaction costs (:func:`run_backtest`, :class:`ExecutionConfig`) -> summary statistics
(:func:`performance_stats`) -> parameter sensitivity
(:func:`parameter_grid`) -> in-sample/out-of-sample and walk-forward
evaluation discipline (:mod:`eq_signal_backtest.split`).

The strategy is simple on purpose. The project is about the evaluation
around it: a tested fill convention, explicit transaction costs, a
train/test split, a walk-forward variant, a parameter-sensitivity map, and
the statistics in :mod:`eq_signal_backtest.research_stats`. See
``docs/METHODOLOGY.md`` and ``docs/VALIDATION.md``.
"""

from .engine import (
    TRADING_DAYS,
    BacktestResult,
    ExecutionConfig,
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

__version__ = "3.0.0"

__all__ = [
    "TRADING_DAYS",
    "BacktestResult",
    "ExecutionConfig",
    "TrainTestSplit",
    "WalkForwardResult",
    "WalkForwardWindow",
    "ma_crossover_signal",
    "parameter_grid",
    "performance_stats",
    "run_backtest",
    "select_best_params",
    "strategy_returns",
    "train_test_split",
    "walk_forward_backtest",
    "walk_forward_windows",
]
