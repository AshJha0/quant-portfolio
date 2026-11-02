"""eq_risk_metrics: single-asset equity risk metrics on real data.

Pipeline: returns -> volatility (3 estimators) -> VaR (3 methods) +
Expected Shortfall -> drawdown -> Sharpe/Sortino -> normality
diagnostics -> VaR backtest (Kupiec proportion-of-failures).

Conventions (portfolio-wide, see ``CONVENTIONS.md``): 252 trading days
per year; volatility annualised on log-returns-equivalent scale via
``sqrt(252)`` scaling of simple-return variance; VaR/ES reported as
**positive loss fractions** at a given confidence and a 1-day horizon.
Every stochastic routine (synthetic data generation) takes an explicit
seed. This is a **single-asset** toolkit -- portfolio-level risk
(correlation, diversification) is out of scope; see
``python/equity/03-var-es-engine`` in this portfolio for the
multi-asset extension.
"""

from .backtest import KupiecResult, count_var_exceptions, kupiec_pof_test
from .diagnostics import NormalityReport, normality_report
from .performance import DrawdownResult, max_drawdown, sharpe_ratio, sortino_ratio
from .var_es import (
    cornish_fisher_valid,
    expected_shortfall,
    var_cornish_fisher,
    var_historical,
    var_parametric,
)
from .volatility import (
    TRADING_DAYS,
    annualised_volatility,
    ewma_volatility,
    log_returns,
    rolling_volatility,
    simple_returns,
)

__version__ = "1.0.0"

__all__ = [
    # volatility
    "TRADING_DAYS",
    # performance
    "DrawdownResult",
    # backtest
    "KupiecResult",
    # diagnostics
    "NormalityReport",
    "__version__",
    "annualised_volatility",
    # var_es
    "cornish_fisher_valid",
    "count_var_exceptions",
    "ewma_volatility",
    "expected_shortfall",
    "kupiec_pof_test",
    "log_returns",
    "max_drawdown",
    "normality_report",
    "rolling_volatility",
    "sharpe_ratio",
    "simple_returns",
    "sortino_ratio",
    "var_cornish_fisher",
    "var_historical",
    "var_parametric",
]
