"""eq_surface: equity volatility surface & Heston stochastic volatility.

Pipeline: implied vol -> SVI smiles -> total-variance surface -> Heston
(Fourier + Monte Carlo) -> calibration -> Greeks -> model validation.
"""

from .black_scholes import (
    ImpliedVolWarning,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vega,
    implied_vol,
    implied_vol_vector,
)
from .calibration import CalibrationResult, calibrate_heston, heston_model_ivs
from .greeks import HestonGreeks, bs_equivalent_greeks, heston_greeks, smile_adjusted_delta
from .heston import (
    FellerWarning,
    HestonParams,
    feller_condition,
    heston_call,
    heston_call_damped,
    heston_call_gl,
    heston_call_p1p2,
    heston_cf,
    heston_put,
)
from .heston_mc import MCResult, heston_mc_price, simulate_heston_terminal
from .smile import (
    QuadraticDeltaFit,
    SVIFitResult,
    SVIParams,
    check_butterfly,
    durrleman_g,
    fit_quadratic_delta,
    fit_svi,
    svi_d2w_dk2,
    svi_dw_dk,
    svi_implied_vol,
    svi_total_variance,
)
from .surface import CalendarCheck, VolSurface, check_calendar

__all__ = [
    "CalendarCheck",
    "CalibrationResult",
    "FellerWarning",
    "HestonGreeks",
    "HestonParams",
    "ImpliedVolWarning",
    "MCResult",
    "QuadraticDeltaFit",
    "SVIFitResult",
    "SVIParams",
    "VolSurface",
    "bs_delta",
    "bs_equivalent_greeks",
    "bs_gamma",
    "bs_price",
    "bs_vega",
    "calibrate_heston",
    "check_butterfly",
    "check_calendar",
    "durrleman_g",
    "feller_condition",
    "fit_quadratic_delta",
    "fit_svi",
    "heston_call",
    "heston_call_damped",
    "heston_call_gl",
    "heston_call_p1p2",
    "heston_cf",
    "heston_greeks",
    "heston_mc_price",
    "heston_model_ivs",
    "heston_put",
    "implied_vol",
    "implied_vol_vector",
    "simulate_heston_terminal",
    "smile_adjusted_delta",
    "svi_d2w_dk2",
    "svi_dw_dk",
    "svi_implied_vol",
    "svi_total_variance",
]

__version__ = "3.0.0.dev0"
