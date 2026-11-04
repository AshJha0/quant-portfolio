"""fx_var: FX market-risk VaR & Expected Shortfall engine.

Pipeline: historical VaR -> parametric VaR -> Monte Carlo VaR -> ES ->
backtesting (Kupiec / Christoffersen / Basel traffic light) -> stress
testing, specialised for a multi-currency FX book (USD triangulation,
CIP-consistent forwards, Garman-Kohlhagen options, peg-break add-ons).
"""

from .backtesting import (
    BacktestResult,
    TrafficLight,
    acerbi_szekely_z2,
    basel_traffic_light,
    christoffersen_independence,
    conditional_coverage,
    evaluate_var_backtest,
    kupiec_pof,
    rolling_backtest,
)
from .book import Book, Cash, Forward, Market, Option, Position, Spot
from .common import (
    PEG_VOL_THRESHOLD,
    TRADING_DAYS_PER_YEAR,
    NumericalWarning,
    PegBlindnessWarning,
    fx_factor,
    ir_factor,
    split_pair,
    vol_factor,
)
from .expected_shortfall import (
    empirical_es,
    empirical_var,
    empirical_var_es,
    normal_es,
    normal_var,
    t_es,
    t_var,
)
from .gk import gk_delta, gk_gamma, gk_price, gk_vega
from .historical_var import HistoricalVaRResult, ewma_volatility, historical_var
from .monte_carlo_var import (
    JumpSpec,
    MonteCarloVaRResult,
    monte_carlo_var,
    nearest_correlation_higham,
    robust_cholesky,
    simulate_factor_returns,
    var_standard_error,
    var_standard_error_bootstrap,
)
from .parametric_var import (
    ParametricVaRResult,
    cornish_fisher_valid,
    cornish_fisher_var,
    cornish_fisher_z,
    ewma_cov,
    parametric_var,
    sample_cov,
    var_covar,
)
from .stress_testing import (
    Scenario,
    historical_scenarios,
    peg_break_scenario,
    reverse_stress_linear,
    reverse_stress_numerical,
    run_stress,
    sensitivity_ladder,
    usd_broad_move,
)

__version__ = "3.0.0"

__all__ = [
    "PEG_VOL_THRESHOLD",
    # conventions
    "TRADING_DAYS_PER_YEAR",
    "BacktestResult",
    "Book",
    "Cash",
    "Forward",
    "HistoricalVaRResult",
    "JumpSpec",
    # book
    "Market",
    "MonteCarloVaRResult",
    "NumericalWarning",
    "Option",
    "ParametricVaRResult",
    "PegBlindnessWarning",
    "Position",
    # stress
    "Scenario",
    "Spot",
    "TrafficLight",
    "acerbi_szekely_z2",
    "basel_traffic_light",
    "christoffersen_independence",
    "conditional_coverage",
    "cornish_fisher_valid",
    "cornish_fisher_var",
    "cornish_fisher_z",
    "empirical_es",
    # ES
    "empirical_var",
    "empirical_var_es",
    "evaluate_var_backtest",
    "ewma_cov",
    "ewma_volatility",
    "fx_factor",
    "gk_delta",
    "gk_gamma",
    # GK
    "gk_price",
    "gk_vega",
    "historical_scenarios",
    # historical
    "historical_var",
    "ir_factor",
    # backtesting
    "kupiec_pof",
    # monte carlo
    "monte_carlo_var",
    "nearest_correlation_higham",
    "normal_es",
    "normal_var",
    # parametric
    "parametric_var",
    "peg_break_scenario",
    "reverse_stress_linear",
    "reverse_stress_numerical",
    "robust_cholesky",
    "rolling_backtest",
    "run_stress",
    "sample_cov",
    "sensitivity_ladder",
    "simulate_factor_returns",
    "split_pair",
    "t_es",
    "t_var",
    "usd_broad_move",
    "var_covar",
    "var_standard_error",
    "var_standard_error_bootstrap",
    "vol_factor",
]
