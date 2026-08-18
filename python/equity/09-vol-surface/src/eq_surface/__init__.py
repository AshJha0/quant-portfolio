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
    "ImpliedVolWarning",
    "bs_delta",
    "bs_gamma",
    "bs_price",
    "bs_vega",
    "implied_vol",
    "implied_vol_vector",
    "SVIParams",
    "SVIFitResult",
    "QuadraticDeltaFit",
    "svi_total_variance",
    "svi_dw_dk",
    "svi_d2w_dk2",
    "svi_implied_vol",
    "durrleman_g",
    "check_butterfly",
    "fit_svi",
    "fit_quadratic_delta",
    "VolSurface",
    "CalendarCheck",
    "check_calendar",
    "HestonParams",
    "FellerWarning",
    "feller_condition",
    "heston_cf",
    "heston_call",
    "heston_call_p1p2",
    "heston_call_damped",
    "heston_call_gl",
    "heston_put",
    "MCResult",
    "simulate_heston_terminal",
    "heston_mc_price",
    "CalibrationResult",
    "calibrate_heston",
    "heston_model_ivs",
    "HestonGreeks",
    "heston_greeks",
    "bs_equivalent_greeks",
    "smile_adjusted_delta",
]

__version__ = "1.0.0"
