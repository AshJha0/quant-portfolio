"""fx_algo: FX algorithmic trading & execution modeling for OTC dealer markets.

Pipeline: signal generation -> feature engineering -> backtesting ->
transaction costs -> slippage -> TWAP/fix benchmarks -> optimal
execution -> performance analysis, specialised for the 24h OTC FX
market (session liquidity, dealer last-look, WM/R fix, no volume tape).
"""

from .backtest import (
    BARS_PER_YEAR,
    BacktestConfig,
    ExecutionConfig,
    IntradayBacktester,
    information_coefficient,
)
from .data.synthetic import generate_daily_panel, generate_ticks
from .execution.optimal import (
    ac_closed_form_schedule,
    ac_expected_cost,
    eta_from_depth,
    piecewise_ac_schedule,
)
from .execution.schedulers import (
    fix_schedule,
    liquidity_weighted_schedule,
    pov_schedule,
    twap_schedule,
)
from .execution.simulator import (
    ExecutionResult,
    FirmVenue,
    LastLookVenue,
    MarketSimulator,
    last_look_reject_prob,
)
from .execution.tca import (
    decompose_implementation_shortfall,
    fix_benchmark,
    fix_print,
    fix_self_impact_pips,
    rejection_cost_pips,
    slippage_vs_benchmark,
    twap_benchmark,
    venue_comparison,
)
from .features import (
    build_bars,
    carry_feature,
    feature_matrix,
    london_open_breakout,
    momentum,
    reversion_to_session_mean,
)
from .sessions import (
    EURUSD,
    FIX_HOUR_LONDON,
    FIX_WINDOW_MINUTES,
    GBPUSD,
    SESSION_BOUNDS,
    SESSION_NAMES,
    USDMXN,
    PairProfile,
    fix_window_mask,
    make_time_grid,
    session_of_hour,
    weekend_mask,
)
from .signals import (
    carry_gate,
    combine_signals,
    rolling_zscore,
    session_filter,
    vol_target_positions,
)

__all__ = [
    "BARS_PER_YEAR",
    "EURUSD",
    "FIX_HOUR_LONDON",
    "FIX_WINDOW_MINUTES",
    "GBPUSD",
    "SESSION_BOUNDS",
    "SESSION_NAMES",
    "USDMXN",
    # backtest
    "BacktestConfig",
    "ExecutionConfig",
    "ExecutionResult",
    "FirmVenue",
    "IntradayBacktester",
    "LastLookVenue",
    # execution
    "MarketSimulator",
    # sessions
    "PairProfile",
    "ac_closed_form_schedule",
    "ac_expected_cost",
    # features
    "build_bars",
    "carry_feature",
    "carry_gate",
    "combine_signals",
    # tca
    "decompose_implementation_shortfall",
    "eta_from_depth",
    "feature_matrix",
    "fix_benchmark",
    "fix_print",
    "fix_schedule",
    "fix_self_impact_pips",
    "fix_window_mask",
    "generate_daily_panel",
    # data
    "generate_ticks",
    "information_coefficient",
    "last_look_reject_prob",
    "liquidity_weighted_schedule",
    "london_open_breakout",
    "make_time_grid",
    "momentum",
    "piecewise_ac_schedule",
    "pov_schedule",
    "rejection_cost_pips",
    "reversion_to_session_mean",
    # signals
    "rolling_zscore",
    "session_filter",
    "session_of_hour",
    "slippage_vs_benchmark",
    "twap_benchmark",
    "twap_schedule",
    "venue_comparison",
    "vol_target_positions",
    "weekend_mask",
]

__version__ = "3.0.0.dev0"
