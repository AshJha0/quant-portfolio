"""fx_pairs: FX statistical pairs / cross-rate mean-reversion trading.

Pipeline: universe & cross construction -> correlation screen -> Engle-Granger
cointegration -> OU spread modelling -> z-score signals -> carry-adjusted,
cost-aware backtesting -> metrics with carry/spot decomposition.
"""

from . import data as data
from .backtest import (
    BacktestResult,
    ExecutionConfig,
    WalkForwardResult,
    WalkForwardWindow,
    run_backtest,
    walk_forward_backtest,
)
from .carry import (
    carry_accrual,
    carry_adjusted_log_price,
    carry_ledger,
    daily_roll_yield,
    day_count_fractions,
    forward_outright,
    swap_points,
)
from .cointegration import (
    ADFResult,
    EngleGrangerResult,
    adf_test,
    engle_granger,
    is_degenerate_spread,
    mackinnon_crit,
)
from .metrics import (
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sharpe_se_lo,
    sortino_ratio,
    summarize,
    trade_pnls,
    turnover,
)
from .signals import (
    Trade,
    carry_entry_veto,
    generate_positions,
    hedged_vol_scale,
    vol_target_scale,
    zscore,
)
from .spread import (
    OUFit,
    RLSHedge,
    fit_ou_mle,
    fit_ou_ols,
    half_life_days,
    log_spread,
)
from .universe import (
    DEFAULT_PIP_SPREADS,
    EM_CURRENCIES,
    G10_CURRENCIES,
    cointegration_funnel,
    correlation_screen,
    enumerate_candidate_pairs,
    make_cross,
    market_pair,
    market_price_from_legs,
    pip_size,
    pip_spread,
    triangular_spread,
)

__all__ = [
    "DEFAULT_PIP_SPREADS",
    "EM_CURRENCIES",
    # universe
    "G10_CURRENCIES",
    # cointegration
    "ADFResult",
    # backtest
    "BacktestResult",
    "EngleGrangerResult",
    "ExecutionConfig",
    # spread
    "OUFit",
    "RLSHedge",
    # signals
    "Trade",
    "WalkForwardResult",
    "WalkForwardWindow",
    "adf_test",
    # carry
    "carry_accrual",
    "carry_adjusted_log_price",
    "carry_entry_veto",
    "carry_ledger",
    "cointegration_funnel",
    "correlation_screen",
    "daily_roll_yield",
    "day_count_fractions",
    "engle_granger",
    "enumerate_candidate_pairs",
    "fit_ou_mle",
    "fit_ou_ols",
    "forward_outright",
    "generate_positions",
    "half_life_days",
    "hedged_vol_scale",
    # metrics
    "hit_rate",
    "is_degenerate_spread",
    "log_spread",
    "mackinnon_crit",
    "make_cross",
    "market_pair",
    "market_price_from_legs",
    "max_drawdown",
    "pip_size",
    "pip_spread",
    "run_backtest",
    "sharpe_ratio",
    "sharpe_se_lo",
    "sortino_ratio",
    "summarize",
    "swap_points",
    "trade_pnls",
    "triangular_spread",
    "turnover",
    "vol_target_scale",
    "walk_forward_backtest",
    "zscore",
]

__version__ = "3.0.0.dev0"
