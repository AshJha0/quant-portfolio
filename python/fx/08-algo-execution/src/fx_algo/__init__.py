"""fx_algo: FX algorithmic trading & execution modeling for OTC dealer markets.

Pipeline: signal generation -> feature engineering -> backtesting ->
transaction costs -> slippage -> TWAP/fix benchmarks -> optimal
execution -> performance analysis, specialised for the 24h OTC FX
market (session liquidity, dealer last-look, WM/R fix, no volume tape).
"""

from .sessions import (
    EURUSD,
    GBPUSD,
    USDMXN,
    FIX_HOUR_LONDON,
    FIX_WINDOW_MINUTES,
    PairProfile,
    SESSION_BOUNDS,
    SESSION_NAMES,
    fix_window_mask,
    make_time_grid,
    session_of_hour,
    weekend_mask,
)
from .features import (
    build_bars,
    carry_feature,
    feature_matrix,
    london_open_breakout,
    momentum,
    reversion_to_session_mean,
)
from .signals import (
    carry_gate,
    combine_signals,
    rolling_zscore,
    session_filter,
    vol_target_positions,
)
from .backtest import BacktestConfig, IntradayBacktester, information_coefficient
from .data.synthetic import generate_daily_panel, generate_ticks
from .execution.simulator import (
    ExecutionResult,
    FirmVenue,
    LastLookVenue,
    MarketSimulator,
    last_look_reject_prob,
)
from .execution.schedulers import (
    fix_schedule,
    liquidity_weighted_schedule,
    pov_schedule,
    twap_schedule,
)
from .execution.optimal import (
    ac_closed_form_schedule,
    ac_expected_cost,
    eta_from_depth,
    piecewise_ac_schedule,
)
from .execution.tca import (
    decompose_implementation_shortfall,
    fix_benchmark,
    rejection_cost_pips,
    slippage_vs_benchmark,
    twap_benchmark,
    venue_comparison,
)

__all__ = [
    # sessions
    "PairProfile",
    "EURUSD",
    "GBPUSD",
    "USDMXN",
    "SESSION_NAMES",
    "SESSION_BOUNDS",
    "FIX_HOUR_LONDON",
    "FIX_WINDOW_MINUTES",
    "session_of_hour",
    "make_time_grid",
    "weekend_mask",
    "fix_window_mask",
    # features
    "build_bars",
    "momentum",
    "reversion_to_session_mean",
    "london_open_breakout",
    "carry_feature",
    "feature_matrix",
    # signals
    "rolling_zscore",
    "combine_signals",
    "vol_target_positions",
    "session_filter",
    "carry_gate",
    # backtest
    "BacktestConfig",
    "IntradayBacktester",
    "information_coefficient",
    # data
    "generate_ticks",
    "generate_daily_panel",
    # execution
    "MarketSimulator",
    "ExecutionResult",
    "FirmVenue",
    "LastLookVenue",
    "last_look_reject_prob",
    "twap_schedule",
    "liquidity_weighted_schedule",
    "pov_schedule",
    "fix_schedule",
    "ac_closed_form_schedule",
    "piecewise_ac_schedule",
    "ac_expected_cost",
    "eta_from_depth",
    # tca
    "decompose_implementation_shortfall",
    "twap_benchmark",
    "fix_benchmark",
    "slippage_vs_benchmark",
    "rejection_cost_pips",
    "venue_comparison",
]

__version__ = "1.0.0"
