"""eq_vol — equity volatility modeling & forecasting.

Pipeline: historical/range estimators -> EWMA -> GARCH(1,1) -> EGARCH(1,1)
-> GJR-GARCH(1,1) -> multi-step forecasting -> forecast evaluation.

All models are implemented from scratch (numpy/scipy only); the ``arch``
package is used exclusively as an independent cross-validation benchmark in
the test suite. Returns are daily log-returns in decimal units; vols are
annualised with sqrt(252).
"""

from ._results import VolatilityFitResult
from ._utils import TRADING_DAYS, ConvergenceError
from .ewma import (
    ewma_forecast,
    ewma_variance,
    ewma_variance_recursive,
    halflife_to_lambda,
    lambda_to_halflife,
)
from .historical import (
    garman_klass_var,
    parkinson_var,
    range_vol,
    realized_vol,
    rogers_satchell_var,
    window_sensitivity,
)
from .garch import (
    fit_garch,
    garch_loglik,
    garch_recursion,
    persistence,
    unconditional_variance,
    vol_halflife,
)
from .egarch import egarch_recursion, fit_egarch
from .egarch import news_impact_curve as egarch_news_impact_curve
from .gjr import fit_gjr, gjr_persistence, gjr_recursion, gjr_unconditional_variance
from .gjr import news_impact_curve as gjr_news_impact_curve
from .forecasting import (
    RollingForecastResult,
    forecast,
    forecast_egarch,
    forecast_garch,
    forecast_gjr,
    forecast_historical,
    rolling_one_step_forecasts,
    term_structure,
)
from .evaluation import (
    DMResult,
    MZResult,
    arch_lm_test,
    diebold_mariano,
    forecast_race_table,
    ljung_box_squared,
    mincer_zarnowitz,
    mse_loss,
    qlike_loss,
    sign_bias_test,
)
from .data import synthetic

__version__ = "1.0.0"

__all__ = [
    "TRADING_DAYS",
    "ConvergenceError",
    "VolatilityFitResult",
    # historical
    "realized_vol",
    "parkinson_var",
    "garman_klass_var",
    "rogers_satchell_var",
    "range_vol",
    "window_sensitivity",
    # ewma
    "ewma_variance",
    "ewma_variance_recursive",
    "ewma_forecast",
    "lambda_to_halflife",
    "halflife_to_lambda",
    # garch family
    "fit_garch",
    "garch_recursion",
    "garch_loglik",
    "persistence",
    "unconditional_variance",
    "vol_halflife",
    "fit_egarch",
    "egarch_recursion",
    "egarch_news_impact_curve",
    "fit_gjr",
    "gjr_recursion",
    "gjr_persistence",
    "gjr_unconditional_variance",
    "gjr_news_impact_curve",
    # forecasting
    "forecast",
    "forecast_garch",
    "forecast_gjr",
    "forecast_egarch",
    "forecast_historical",
    "term_structure",
    "rolling_one_step_forecasts",
    "RollingForecastResult",
    # evaluation
    "qlike_loss",
    "mse_loss",
    "mincer_zarnowitz",
    "MZResult",
    "diebold_mariano",
    "DMResult",
    "ljung_box_squared",
    "arch_lm_test",
    "sign_bias_test",
    "forecast_race_table",
    # data
    "synthetic",
]
