"""eq_var — Equity market risk engine: VaR, Expected Shortfall, backtesting,
Basel traffic light and stress testing.

Pipeline: portfolio & factor mapping -> historical / parametric / Monte Carlo
VaR -> Expected Shortfall -> Kupiec / Christoffersen backtests -> Basel
traffic light -> stress & reverse stress testing.
"""

from .backtesting import (
    BacktestResult,
    acerbi_szekely_z2,
    basel_traffic_light,
    basel_zone_probabilities,
    christoffersen_cc,
    christoffersen_independence,
    exception_cluster_table,
    exceptions_from_pnl,
    kupiec_pof,
    rolling_var_backtest,
)
from .expected_shortfall import (
    es_standard_error_bootstrap,
    expected_shortfall,
    normal_es,
    parametric_es,
    student_t_es,
)
from .historical_var import (
    age_weighted_var,
    brw_weights,
    ewma_volatility,
    filtered_historical_var,
    historical_var,
    overlapping_horizon_pnl,
    scale_var_sqrt_time,
)
from .monte_carlo_var import (
    monte_carlo_pnl,
    monte_carlo_var,
    safe_cholesky,
    simulate_factor_returns,
    var_confidence_interval,
    var_standard_error_bootstrap,
)
from .parametric_var import (
    cornish_fisher_domain_ok,
    cornish_fisher_var,
    cornish_fisher_z,
    ewma_covariance,
    parametric_var,
    portfolio_sigma,
    sample_covariance,
)
from .portfolio import (
    EquityPosition,
    FuturePosition,
    OptionPosition,
    Portfolio,
    Position,
    RiskFactor,
    bs_greeks,
    bs_price,
)
from .stress_testing import (
    HISTORICAL_SCENARIOS,
    StressScenario,
    apply_scenario,
    reverse_stress_delta,
    reverse_stress_delta_gamma,
    scenario_shock_vector,
    scenario_table,
    sensitivity_ladder,
)

__all__ = [
    # portfolio
    "RiskFactor",
    "Position",
    "EquityPosition",
    "FuturePosition",
    "OptionPosition",
    "Portfolio",
    "bs_price",
    "bs_greeks",
    # historical
    "historical_var",
    "age_weighted_var",
    "brw_weights",
    "ewma_volatility",
    "filtered_historical_var",
    "scale_var_sqrt_time",
    "overlapping_horizon_pnl",
    # parametric
    "sample_covariance",
    "ewma_covariance",
    "portfolio_sigma",
    "parametric_var",
    "cornish_fisher_z",
    "cornish_fisher_domain_ok",
    "cornish_fisher_var",
    # monte carlo
    "safe_cholesky",
    "simulate_factor_returns",
    "monte_carlo_pnl",
    "monte_carlo_var",
    "var_standard_error_bootstrap",
    "var_confidence_interval",
    # expected shortfall
    "expected_shortfall",
    "normal_es",
    "student_t_es",
    "parametric_es",
    "es_standard_error_bootstrap",
    # backtesting
    "exceptions_from_pnl",
    "kupiec_pof",
    "christoffersen_independence",
    "christoffersen_cc",
    "basel_traffic_light",
    "basel_zone_probabilities",
    "acerbi_szekely_z2",
    "exception_cluster_table",
    "rolling_var_backtest",
    "BacktestResult",
    # stress
    "StressScenario",
    "HISTORICAL_SCENARIOS",
    "scenario_shock_vector",
    "apply_scenario",
    "scenario_table",
    "sensitivity_ladder",
    "reverse_stress_delta",
    "reverse_stress_delta_gamma",
]

__version__ = "1.0.0"
