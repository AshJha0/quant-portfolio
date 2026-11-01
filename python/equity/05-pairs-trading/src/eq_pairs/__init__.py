"""eq_pairs: equity statistical pairs trading.

Pipeline: pair selection (correlation on returns or SSD), Engle-Granger
cointegration with MacKinnon critical values, Ornstein-Uhlenbeck spread
model, z-score signals, backtest with a modelled t+1 fill and costs,
walk-forward validation, research statistics (Lo SE, deflated Sharpe, PBO).
"""

from . import research_stats
from .backtest import (
    COST_COLUMNS,
    CostModel,
    ExecutionConfig,
    PairResult,
    PortfolioResult,
    ZERO_COSTS,
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

__version__ = "1.0.0"

__all__ = [
    # universe
    "candidate_pairs",
    "correlation_screen",
    "log_returns",
    "pair_correlations",
    "ssd_distances",
    "ssd_screen",
    # cointegration
    "ADFResult",
    "EGResult",
    "adf_test",
    "engle_granger",
    "hedge_ratio",
    "mackinnon_crit",
    # spread
    "OUFit",
    "compute_spread",
    "fit_ou_mle",
    "fit_ou_ols",
    "half_life_from_kappa",
    "rls_hedge_ratio",
    "rolling_ols_hedge_ratio",
    # signals
    "SignalRules",
    "generate_signals",
    "size_positions",
    "time_stop_bars",
    "zscore_ou",
    "zscore_rolling",
    # backtest
    "COST_COLUMNS",
    "CostModel",
    "ExecutionConfig",
    "ZERO_COSTS",
    "PairResult",
    "PortfolioResult",
    "align_pair",
    "backtest_pair",
    "backtest_portfolio",
    "walk_forward_pair",
    "walk_forward_portfolio",
    "walk_forward_windows",
    # metrics
    "avg_holding_period",
    "cost_drag",
    "hit_rate",
    "max_drawdown",
    "sharpe_ratio",
    "sharpe_se",
    "sortino_ratio",
    "summary",
    "turnover",
    # research statistics (module)
    "research_stats",
]
