"""fx_vol -- FX volatility modeling & forecasting.

Pipeline: FX return construction -> historical/realized estimators -> EWMA ->
GARCH(1,1)/GARCH-X -> EGARCH -> GJR-GARCH -> forecasting -> evaluation ->
vol risk premium; all models implemented from scratch (the `arch` package is
used only for cross-validation in the test suite).

Conventions: pairs quoted BASE/QUOTE, log returns, daily decimal units,
annualization factor 252 by default (260 = FX 24h5d alternative, see
fx_vol.historical).
"""

from ._mle import FitResult
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
from .historical import (
    close_to_close_vol,
    day_of_week_vol_factors,
    garman_klass_vol,
    parkinson_vol,
    rolling_close_vol,
)
from .ewma import ewma_forecast, ewma_variance, ewma_weights
from .garch import fit_garch, garch_filter
from .egarch import egarch_filter, fit_egarch
from .gjr import fit_gjr, gjr_filter
from .forecasting import forecast_egarch_simulated, forecast_variance, rolling_one_step
from .evaluation import (
    arch_lm,
    diebold_mariano,
    ljung_box,
    mincer_zarnowitz,
    mse_loss,
    newey_west_variance,
    qlike_loss,
    sign_bias_test,
)
from .vol_premium import (
    premium_summary,
    realized_vol_forward,
    variance_swap_pnl,
    vol_risk_premium,
)

__version__ = "1.0.0"

__all__ = [
    "FitResult",
    # returns
    "log_returns", "invert_prices", "invert_returns", "pair_currencies",
    "cross_pair_signs", "triangulate_prices", "triangulate_returns", "cross_volatility",
    # historical
    "close_to_close_vol", "rolling_close_vol", "parkinson_vol", "garman_klass_vol",
    "day_of_week_vol_factors",
    # ewma
    "ewma_variance", "ewma_forecast", "ewma_weights",
    # models
    "garch_filter", "fit_garch", "egarch_filter", "fit_egarch", "gjr_filter", "fit_gjr",
    # forecasting
    "forecast_variance", "forecast_egarch_simulated", "rolling_one_step",
    # evaluation
    "qlike_loss", "mse_loss", "mincer_zarnowitz", "newey_west_variance",
    "diebold_mariano", "ljung_box", "arch_lm", "sign_bias_test",
    # vol premium
    "realized_vol_forward", "vol_risk_premium", "variance_swap_pnl", "premium_summary",
]
