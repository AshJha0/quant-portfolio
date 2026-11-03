"""fx_vol: FX volatility modeling & forecasting.

Pipeline: FX return construction -> historical/realized estimators -> EWMA ->
GARCH(1,1)/GARCH-X -> EGARCH -> GJR-GARCH -> forecasting -> evaluation ->
vol risk premium; all models implemented from scratch (the `arch` package is
used only for cross-validation in the test suite).

Conventions: pairs quoted BASE/QUOTE, log returns, daily decimal units,
annualization factor 252 by default (260 = FX 24h5d alternative, see
fx_vol.historical).
"""

from ._mle import ConvergenceError, FitResult
from .egarch import egarch_filter, egarch_unconditional_variance, fit_egarch
from .evaluation import (
    arch_lm,
    default_hac_lags,
    diebold_mariano,
    ljung_box,
    mincer_zarnowitz,
    mse_loss,
    newey_west_variance,
    qlike_loss,
    sign_bias_test,
)
from .ewma import ewma_forecast, ewma_variance, ewma_weights
from .forecasting import forecast_egarch_simulated, forecast_variance, rolling_one_step
from .garch import fit_garch, garch_filter
from .gjr import fit_gjr, gjr_filter
from .historical import (
    close_to_close_vol,
    day_of_week_vol_factors,
    garman_klass_vol,
    parkinson_vol,
    rolling_close_vol,
)
from .returns import (
    cross_pair_signs,
    cross_volatility,
    invert_prices,
    invert_returns,
    log_returns,
    pair_currencies,
    triangulate_prices,
    triangulate_returns,
)
from .vol_premium import (
    premium_summary,
    realized_vol_forward,
    variance_swap_pnl,
    vol_risk_premium,
)

__version__ = "3.0.0.dev0"

__all__ = [
    "ConvergenceError",
    "FitResult",
    "arch_lm",
    # historical
    "close_to_close_vol",
    "cross_pair_signs",
    "cross_volatility",
    "day_of_week_vol_factors",
    "default_hac_lags",
    "diebold_mariano",
    "egarch_filter",
    "egarch_unconditional_variance",
    "ewma_forecast",
    # ewma
    "ewma_variance",
    "ewma_weights",
    "fit_egarch",
    "fit_garch",
    "fit_gjr",
    "forecast_egarch_simulated",
    # forecasting
    "forecast_variance",
    # models
    "garch_filter",
    "garman_klass_vol",
    "gjr_filter",
    "invert_prices",
    "invert_returns",
    "ljung_box",
    # returns
    "log_returns",
    "mincer_zarnowitz",
    "mse_loss",
    "newey_west_variance",
    "pair_currencies",
    "parkinson_vol",
    "premium_summary",
    # evaluation
    "qlike_loss",
    # vol premium
    "realized_vol_forward",
    "rolling_close_vol",
    "rolling_one_step",
    "sign_bias_test",
    "triangulate_prices",
    "triangulate_returns",
    "variance_swap_pnl",
    "vol_risk_premium",
]
