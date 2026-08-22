"""fx_var: FX market-risk VaR & Expected Shortfall engine.

Pipeline: historical VaR -> parametric VaR -> Monte Carlo VaR -> ES ->
backtesting (Kupiec / Christoffersen / Basel traffic light) -> stress
testing, specialised for a multi-currency FX book (USD triangulation,
CIP-consistent forwards, Garman-Kohlhagen options, peg-break add-ons).
"""

from .backtesting import (
    BacktestResult,
    TrafficLight,
    basel_traffic_light,
    christoffersen_independence,
    conditional_coverage,
    es_backtest_acerbi_szekely,
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
    robust_cholesky,
    simulate_factor_returns,
    var_standard_error,
    var_standard_error_bootstrap,
)
from .parametric_var import (
    ParametricVaRResult,
    cornish_fisher_domain_ok,
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

__version__ = "1.0.0"

__all__ = [
    # book
    "Market", "Book", "Cash", "Spot", "Forward", "Option", "Position",
    # conventions
    "TRADING_DAYS_PER_YEAR", "PEG_VOL_THRESHOLD", "PegBlindnessWarning",
    "NumericalWarning", "split_pair", "fx_factor", "ir_factor", "vol_factor",
    # GK
    "gk_price", "gk_delta", "gk_gamma", "gk_vega",
    # historical
    "historical_var", "HistoricalVaRResult", "ewma_volatility",
    # parametric
    "parametric_var", "ParametricVaRResult", "var_covar", "sample_cov",
    "ewma_cov", "cornish_fisher_var", "cornish_fisher_z",
    "cornish_fisher_domain_ok",
    # monte carlo
    "monte_carlo_var", "MonteCarloVaRResult", "simulate_factor_returns",
    "JumpSpec", "robust_cholesky", "var_standard_error", "var_standard_error_bootstrap",
    # ES
    "empirical_var", "empirical_es", "empirical_var_es", "normal_var",
    "normal_es", "t_var", "t_es",
    # backtesting
    "kupiec_pof", "christoffersen_independence", "conditional_coverage",
    "basel_traffic_light", "TrafficLight", "BacktestResult",
    "evaluate_var_backtest", "rolling_backtest", "es_backtest_acerbi_szekely",
    # stress
    "Scenario", "historical_scenarios", "usd_broad_move",
    "peg_break_scenario", "run_stress", "sensitivity_ladder",
    "reverse_stress_linear", "reverse_stress_numerical",
]
