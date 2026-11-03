"""eq_pairs: equity statistical pairs trading.

Pipeline: pair selection (correlation on returns or SSD), Engle-Granger
cointegration with MacKinnon critical values, Ornstein-Uhlenbeck spread
model, z-score signals, backtest with a modelled t+1 fill and costs,
walk-forward validation, research statistics (Lo SE, deflated Sharpe, PBO).
"""

from . import research_stats
from .backtest import (
    COST_COLUMNS,
    ZERO_COSTS,
    CostModel,
    ExecutionConfig,
    PairResult,
    PortfolioResult,
    align_pair,
    backtest_pair,
    backtest_portfolio,
    walk_forward_pair,
    walk_forward_portfolio,
    walk_forward_windows,
)
from .cointegration import (
    ADFResult,
    EGResult,
    adf_test,
    engle_granger,
    hedge_ratio,
    mackinnon_crit,
)
from .kalman import KalmanHedgeResult, kalman_hedge_ratio
from .metrics import (
    avg_holding_period,
    cost_drag,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sharpe_se,
    sortino_ratio,
    summary,
    turnover,
)
from .signals import (
    SignalRules,
    generate_signals,
    size_positions,
    time_stop_bars,
    zscore_ou,
    zscore_rolling,
)
from .spread import (
    OUFit,
    compute_spread,
    fit_ou_mle,
    fit_ou_ols,
    half_life_from_kappa,
    rls_hedge_ratio,
    rolling_ols_hedge_ratio,
)
from .universe import (
    candidate_pairs,
    correlation_screen,
    log_returns,
    pair_correlations,
    ssd_distances,
    ssd_screen,
)

__version__ = "3.0.0.dev0"

__all__ = [
    # backtest
    "COST_COLUMNS",
    "ZERO_COSTS",
    # cointegration
    "ADFResult",
    "CostModel",
    "EGResult",
    "ExecutionConfig",
    "KalmanHedgeResult",
    # spread
    "OUFit",
    "PairResult",
    "PortfolioResult",
    # signals
    "SignalRules",
    "adf_test",
    "align_pair",
    # metrics
    "avg_holding_period",
    "backtest_pair",
    "backtest_portfolio",
    # universe
    "candidate_pairs",
    "compute_spread",
    "correlation_screen",
    "cost_drag",
    "engle_granger",
    "fit_ou_mle",
    "fit_ou_ols",
    "generate_signals",
    "half_life_from_kappa",
    "hedge_ratio",
    "hit_rate",
    "kalman_hedge_ratio",
    "log_returns",
    "mackinnon_crit",
    "max_drawdown",
    "pair_correlations",
    # research statistics (module)
    "research_stats",
    "rls_hedge_ratio",
    "rolling_ols_hedge_ratio",
    "sharpe_ratio",
    "sharpe_se",
    "size_positions",
    "sortino_ratio",
    "ssd_distances",
    "ssd_screen",
    "summary",
    "time_stop_bars",
    "turnover",
    "walk_forward_pair",
    "walk_forward_portfolio",
    "walk_forward_windows",
    "zscore_ou",
    "zscore_rolling",
]
