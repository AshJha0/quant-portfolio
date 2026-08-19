"""eq_risk_metrics: single-asset equity risk metrics on real data.

Pipeline: returns -> volatility (3 estimators) -> VaR (3 methods) +
Expected Shortfall -> drawdown -> Sharpe/Sortino -> normality
diagnostics.

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

from .diagnostics import NormalityReport, normality_report
from .performance import DrawdownResult, max_drawdown, sharpe_ratio, sortino_ratio
from .var_es import (
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
    "__version__",
    # volatility
    "TRADING_DAYS",
    "simple_returns",
    "log_returns",
    "annualised_volatility",
    "rolling_volatility",
    "ewma_volatility",
    # var_es
    "var_historical",
    "var_parametric",
    "var_cornish_fisher",
    "expected_shortfall",
    # performance
    "DrawdownResult",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    # diagnostics
    "NormalityReport",
    "normality_report",
]
