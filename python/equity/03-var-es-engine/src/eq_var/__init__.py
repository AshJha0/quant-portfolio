"""eq_var: Equity market risk engine: VaR, Expected Shortfall, backtesting,
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
    age_weighted_es,
    age_weighted_var,
    brw_weights,
    ewma_forecast,
    ewma_volatility,
    factor_filtered_historical_var,
    filtered_historical_scenarios,
    filtered_historical_var,
    historical_var,
    overlapping_horizon_pnl,
    scale_var_sqrt_time,
)
from .monte_carlo_var import (
    MonteCarloRisk,
    cholesky,
    higham_nearest_correlation,
    monte_carlo_pnl,
    monte_carlo_risk,
    monte_carlo_var,
    repair_covariance,
    safe_cholesky,
    simulate_factor_returns,
    var_confidence_interval,
    var_standard_error_bootstrap,
    var_standard_error_kde,
)
from .parametric_var import (
    cornish_fisher_domain_ok,
    cornish_fisher_valid,
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
    "HISTORICAL_SCENARIOS",
    "BacktestResult",
    "EquityPosition",
    "FuturePosition",
    "MonteCarloRisk",
    "OptionPosition",
    "Portfolio",
    "Position",
    # portfolio
    "RiskFactor",
    # stress
    "StressScenario",
    "acerbi_szekely_z2",
    "age_weighted_es",
    "age_weighted_var",
    "apply_scenario",
    "basel_traffic_light",
    "basel_zone_probabilities",
    "brw_weights",
    "bs_greeks",
    "bs_price",
    # monte carlo
    "cholesky",
    "christoffersen_cc",
    "christoffersen_independence",
    "cornish_fisher_domain_ok",
    "cornish_fisher_valid",
    "cornish_fisher_var",
    "cornish_fisher_z",
    "es_standard_error_bootstrap",
    "ewma_covariance",
    "ewma_forecast",
    "ewma_volatility",
    "exception_cluster_table",
    # backtesting
    "exceptions_from_pnl",
    # expected shortfall
    "expected_shortfall",
    "factor_filtered_historical_var",
    "filtered_historical_scenarios",
    "filtered_historical_var",
    "higham_nearest_correlation",
    # historical
    "historical_var",
    "kupiec_pof",
    "monte_carlo_pnl",
    "monte_carlo_risk",
    "monte_carlo_var",
    "normal_es",
    "overlapping_horizon_pnl",
    "parametric_es",
    "parametric_var",
    "portfolio_sigma",
    "repair_covariance",
    "reverse_stress_delta",
    "reverse_stress_delta_gamma",
    "rolling_var_backtest",
    "safe_cholesky",
    # parametric
    "sample_covariance",
    "scale_var_sqrt_time",
    "scenario_shock_vector",
    "scenario_table",
    "sensitivity_ladder",
    "simulate_factor_returns",
    "student_t_es",
    "var_confidence_interval",
    "var_standard_error_bootstrap",
    "var_standard_error_kde",
]

__version__ = "3.0.0.dev0"
