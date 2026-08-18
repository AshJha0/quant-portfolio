"""eq_algo — equity algorithmic trading & execution modeling.

Alpha layer (daily): point-in-time features -> signal combination -> long-
short backtest with linear + square-root costs -> IC / deflated-Sharpe /
capacity evaluation.

Execution layer (intraday): seeded market simulator (U-shaped volume,
temporary + permanent impact) -> VWAP / TWAP / POV schedulers -> Almgren-
Chriss optimal trajectories -> Perold implementation-shortfall TCA.
"""

from .features import (momentum, short_term_reversal, realized_vol, ma_crossover,
                       rsi, turnover_zscore, cs_rank, cs_zscore, winsorize)
from .signals import (forward_returns, information_coefficient,
                      combine_equal_weight, combine_ic_weighted, signal_decay,
                      decile_portfolios, freeze_signal, apply_rebalance_band,
                      turnover)
from .backtest import BacktestConfig, BacktestResult, long_short_weights, run_backtest
from .evaluation import (newey_west_se, newey_west_tstat, ic_summary,
                         quantile_monotonicity, sharpe_ratio, sortino_ratio,
                         max_drawdown, perf_summary, probabilistic_sharpe_ratio,
                         expected_max_sharpe, deflated_sharpe_ratio, capacity_curve)
from .intraday import u_shaped_profile, IntradayConfig, ExecutionResult, IntradayMarket
from .benchmarks import (vwap, twap, arrival_price, slippage_bps, benchmark_slippage,
                         twap_schedule, vwap_schedule, pov_schedule)
from .almgren_chriss import (ACParams, ac_kappa, ac_trajectory, ac_trades,
                             ac_cost_moments, efficient_frontier, evaluate_schedules)
from .tca import ISReport, is_decomposition, tca_report, slippage_attribution, aggregate_tca
from .data.synthetic import DailyPanel, generate_daily_panel

__version__ = "1.0.0"

__all__ = [
    # features
    "momentum", "short_term_reversal", "realized_vol", "ma_crossover", "rsi",
    "turnover_zscore", "cs_rank", "cs_zscore", "winsorize",
    # signals
    "forward_returns", "information_coefficient", "combine_equal_weight",
    "combine_ic_weighted", "signal_decay", "decile_portfolios",
    "freeze_signal", "apply_rebalance_band", "turnover",
    # backtest
    "BacktestConfig", "BacktestResult", "long_short_weights", "run_backtest",
    # evaluation
    "newey_west_se", "newey_west_tstat", "ic_summary", "quantile_monotonicity",
    "sharpe_ratio", "sortino_ratio", "max_drawdown", "perf_summary",
    "probabilistic_sharpe_ratio", "expected_max_sharpe", "deflated_sharpe_ratio",
    "capacity_curve",
    # intraday
    "u_shaped_profile", "IntradayConfig", "ExecutionResult", "IntradayMarket",
    # benchmarks
    "vwap", "twap", "arrival_price", "slippage_bps", "benchmark_slippage",
    "twap_schedule", "vwap_schedule", "pov_schedule",
    # almgren-chriss
    "ACParams", "ac_kappa", "ac_trajectory", "ac_trades", "ac_cost_moments",
    "efficient_frontier", "evaluate_schedules",
    # tca
    "ISReport", "is_decomposition", "tca_report", "slippage_attribution",
    "aggregate_tca",
    # data
    "DailyPanel", "generate_daily_panel",
]
