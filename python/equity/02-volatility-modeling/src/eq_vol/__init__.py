"""eq_vol: equity volatility modeling & forecasting.

Pipeline: historical/range estimators -> EWMA -> GARCH(1,1) -> EGARCH(1,1)
-> GJR-GARCH(1,1) -> multi-step forecasting -> forecast evaluation.

All models are implemented from scratch (numpy/scipy only); the ``arch``
package is used exclusively as an independent cross-validation benchmark in
the test suite. Returns are daily log-returns in decimal units; vols are
annualised with sqrt(252).
"""

from ._results import VolatilityFitResult
from ._utils import TRADING_DAYS, ConvergenceError
from .data import synthetic
from .egarch import egarch_recursion, egarch_unconditional_variance, fit_egarch
from .egarch import news_impact_curve as egarch_news_impact_curve
from .evaluation import (
    DMResult,
    MZResult,
    arch_lm_test,
    default_hac_lags,
    diebold_mariano,
    forecast_race_table,
    ljung_box_squared,
    mincer_zarnowitz,
    mse_loss,
    qlike_loss,
    sign_bias_test,
)
from .ewma import (
    ewma_forecast,
    ewma_variance,
    ewma_variance_recursive,
    halflife_to_lambda,
    lambda_to_halflife,
)
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
from .garch import (
    fit_garch,
    garch_loglik,
    garch_recursion,
    persistence,
    unconditional_variance,
    vol_halflife,
)
from .gjr import fit_gjr, gjr_persistence, gjr_recursion, gjr_unconditional_variance
from .gjr import news_impact_curve as gjr_news_impact_curve
from .historical import (
    garman_klass_var,
    parkinson_var,
    range_vol,
    realized_vol,
    rogers_satchell_var,
    window_sensitivity,
    yang_zhang_var,
)

__version__ = "3.0.0.dev0"

__all__ = [
    "TRADING_DAYS",
    "ConvergenceError",
    "DMResult",
    "MZResult",
    "RollingForecastResult",
    "VolatilityFitResult",
    "arch_lm_test",
    "default_hac_lags",
    "diebold_mariano",
    "egarch_news_impact_curve",
    "egarch_recursion",
    "egarch_unconditional_variance",
    "ewma_forecast",
    # ewma
    "ewma_variance",
    "ewma_variance_recursive",
    "fit_egarch",
    # garch family
    "fit_garch",
    "fit_gjr",
    # forecasting
    "forecast",
    "forecast_egarch",
    "forecast_garch",
    "forecast_gjr",
    "forecast_historical",
    "forecast_race_table",
    "garch_loglik",
    "garch_recursion",
    "garman_klass_var",
    "gjr_news_impact_curve",
    "gjr_persistence",
    "gjr_recursion",
    "gjr_unconditional_variance",
    "halflife_to_lambda",
    "lambda_to_halflife",
    "ljung_box_squared",
    "mincer_zarnowitz",
    "mse_loss",
    "parkinson_var",
    "persistence",
    # evaluation
    "qlike_loss",
    "range_vol",
    # historical
    "realized_vol",
    "rogers_satchell_var",
    "rolling_one_step_forecasts",
    "sign_bias_test",
    # data
    "synthetic",
    "term_structure",
    "unconditional_variance",
    "vol_halflife",
    "window_sensitivity",
    "yang_zhang_var",
]
