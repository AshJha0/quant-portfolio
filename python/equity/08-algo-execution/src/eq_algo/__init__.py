"""eq_algo, equity algorithmic trading & execution modeling.

Alpha layer (daily): point-in-time features -> signal combination -> long-
short backtest with linear + square-root costs -> IC / deflated-Sharpe /
capacity evaluation.

Execution layer (intraday): seeded market simulator (U-shaped volume,
temporary + permanent impact) -> VWAP / TWAP / POV schedulers -> Almgren-
Chriss optimal trajectories -> Perold implementation-shortfall TCA.
"""

from .almgren_chriss import (
                       ACParams,
                       ac_cost_moments,
                       ac_kappa,
                       ac_trades,
                       ac_trajectory,
                       efficient_frontier,
                       evaluate_schedules,
)
from .backtest import (
                       BacktestConfig,
                       BacktestResult,
                       ExecutionConfig,
                       capacity_backtest,
                       long_short_weights,
                       run_backtest,
)
from .benchmarks import (
                       arrival_price,
                       benchmark_slippage,
                       pov_schedule,
                       slippage_bps,
                       twap,
                       twap_schedule,
                       vwap,
                       vwap_schedule,
)
from .data.synthetic import DailyPanel, generate_daily_panel
from .evaluation import (
                       capacity_curve,
                       deflated_sharpe_ratio,
                       expected_max_sharpe,
                       ic_summary,
                       max_drawdown,
                       newey_west_se,
                       newey_west_tstat,
                       perf_summary,
                       probabilistic_sharpe_ratio,
                       quantile_monotonicity,
                       sharpe_ratio,
                       sortino_ratio,
)
from .features import (
                       cs_rank,
                       cs_zscore,
                       ma_crossover,
                       momentum,
                       realized_vol,
                       rsi,
                       short_term_reversal,
                       turnover_zscore,
                       winsorize,
)
from .intraday import ExecutionResult, IntradayConfig, IntradayMarket, u_shaped_profile
from .signals import (
                       apply_rebalance_band,
                       combine_equal_weight,
                       combine_ic_weighted,
                       decile_portfolios,
                       forward_returns,
                       freeze_signal,
                       information_coefficient,
                       signal_decay,
                       turnover,
)
from .tca import ISReport, aggregate_tca, is_decomposition, slippage_attribution, tca_report

__version__ = "1.0.0"

__all__ = [
                       # almgren-chriss
                       "ACParams",
                       "BacktestConfig",
                       "BacktestResult",
                       # data
                       "DailyPanel",
                       # backtest
                       "ExecutionConfig",
                       "ExecutionResult",
                       # tca
                       "ISReport",
                       "IntradayConfig",
                       "IntradayMarket",
                       "ac_cost_moments",
                       "ac_kappa",
                       "ac_trades",
                       "ac_trajectory",
                       "aggregate_tca",
                       "apply_rebalance_band",
                       "arrival_price",
                       "benchmark_slippage",
                       "capacity_backtest",
                       "capacity_curve",
                       "combine_equal_weight",
                       "combine_ic_weighted",
                       "cs_rank",
                       "cs_zscore",
                       "decile_portfolios",
                       "deflated_sharpe_ratio",
                       "efficient_frontier",
                       "evaluate_schedules",
                       "expected_max_sharpe",
                       # signals
                       "forward_returns",
                       "freeze_signal",
                       "generate_daily_panel",
                       "ic_summary",
                       "information_coefficient",
                       "is_decomposition",
                       "long_short_weights",
                       "ma_crossover",
                       "max_drawdown",
                       # features
                       "momentum",
                       # evaluation
                       "newey_west_se",
                       "newey_west_tstat",
                       "perf_summary",
                       "pov_schedule",
                       "probabilistic_sharpe_ratio",
                       "quantile_monotonicity",
                       "realized_vol",
                       "rsi",
                       "run_backtest",
                       "sharpe_ratio",
                       "short_term_reversal",
                       "signal_decay",
                       "slippage_attribution",
                       "slippage_bps",
                       "sortino_ratio",
                       "tca_report",
                       "turnover",
                       "turnover_zscore",
                       "twap",
                       "twap_schedule",
                       # intraday
                       "u_shaped_profile",
                       # benchmarks
                       "vwap",
                       "vwap_schedule",
                       "winsorize",
]
