"""fx_pairs: FX statistical pairs / cross-rate mean-reversion trading.

Pipeline: universe & cross construction -> correlation screen -> Engle-Granger
cointegration -> OU spread modelling -> z-score signals -> carry-adjusted,
cost-aware backtesting -> metrics with carry/spot decomposition.
"""

from . import data
from .universe import (
    G10_CURRENCIES,
    EM_CURRENCIES,
    DEFAULT_PIP_SPREADS,
    market_pair,
    pip_size,
    pip_spread,
    make_cross,
    market_price_from_legs,
    triangular_spread,
    enumerate_candidate_pairs,
    correlation_screen,
)
from .cointegration import (
    ADFResult,
    EngleGrangerResult,
    adf_test,
    engle_granger,
    mackinnon_crit,
    is_degenerate_spread,
)
from .spread import (
    OUFit,
    RLSHedge,
    fit_ou_ols,
    fit_ou_mle,
    half_life_days,
    log_spread,
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
from .signals import (
    Trade,
    carry_entry_veto,
    generate_positions,
    vol_target_scale,
    zscore,
)
from .backtest import (
    BacktestResult,
    WalkForwardResult,
    WalkForwardWindow,
    run_backtest,
    walk_forward_backtest,
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

__all__ = [
    # universe
    "G10_CURRENCIES", "EM_CURRENCIES", "DEFAULT_PIP_SPREADS", "market_pair",
    "pip_size", "pip_spread", "make_cross", "market_price_from_legs",
    "triangular_spread", "enumerate_candidate_pairs", "correlation_screen",
    # cointegration
    "ADFResult", "EngleGrangerResult", "adf_test", "engle_granger",
    "mackinnon_crit", "is_degenerate_spread",
    # spread
    "OUFit", "RLSHedge", "fit_ou_ols", "fit_ou_mle", "half_life_days",
    "log_spread",
    # carry
    "carry_accrual", "carry_adjusted_log_price", "carry_ledger",
    "daily_roll_yield", "day_count_fractions", "forward_outright",
    "swap_points",
    # signals
    "Trade", "carry_entry_veto", "generate_positions", "vol_target_scale",
    "zscore",
    # backtest
    "BacktestResult", "WalkForwardResult", "WalkForwardWindow",
    "run_backtest", "walk_forward_backtest",
    # metrics
    "hit_rate", "max_drawdown", "sharpe_ratio", "sharpe_se_lo",
    "sortino_ratio", "summarize", "trade_pnls", "turnover",
]

__version__ = "1.0.0"
