"""fx_surface: FX volatility surface and stochastic volatility toolkit.

Pipeline: broker quotes (ATM/RR/BF in delta space) -> five-point smiles
-> strike solving under four delta conventions -> SVI and vanna-volga
smile fits -> delta-space surface -> Heston (GK) calibration -> pricing,
Greeks and Monte Carlo validation.
"""

from .calibration import (
    CalibrationResult,
    CalibrationSlice,
    calibrate_heston,
    heston_smile_vols,
)
from .garman_kohlhagen import (
    DELTA_CONVENTIONS,
    gk_delta,
    gk_digital,
    gk_forward,
    gk_gamma,
    gk_price,
    gk_rho_domestic,
    gk_rho_foreign,
    gk_theta,
    gk_vanna,
    gk_vega,
    gk_volga,
    implied_vol,
)
from .greeks import gk_greeks, heston_greeks_fd
from .heston import (
    HestonParams,
    heston_cf,
    heston_digital,
    heston_price,
    price_cos,
    price_gil_pelaez,
)
from .heston_mc import mc_price, simulate_terminal
from .smile import SVIParams, SVISmile, VannaVolgaSmile, durrleman_g, smile_digital
from .smile_from_quotes import (
    PILLAR_LABELS,
    SmileQuotes,
    atm_dns_strike,
    pa_call_delta_max,
    quotes_from_vols,
    solve_pillar_strikes,
    strike_from_delta,
    strike_from_delta_pa_candidates,
    vols_from_quotes,
)
from .surface import FXVolSurface, SmileSlice, build_slice, build_surface

__all__ = [
    "DELTA_CONVENTIONS",
    "PILLAR_LABELS",
    "CalibrationResult",
    "CalibrationSlice",
    "FXVolSurface",
    "HestonParams",
    "SVIParams",
    "SVISmile",
    "SmileQuotes",
    "SmileSlice",
    "VannaVolgaSmile",
    "atm_dns_strike",
    "build_slice",
    "build_surface",
    "calibrate_heston",
    "durrleman_g",
    "gk_delta",
    "gk_digital",
    "gk_forward",
    "gk_gamma",
    "gk_greeks",
    "gk_price",
    "gk_rho_domestic",
    "gk_rho_foreign",
    "gk_theta",
    "gk_vanna",
    "gk_vega",
    "gk_volga",
    "heston_cf",
    "heston_digital",
    "heston_greeks_fd",
    "heston_price",
    "heston_smile_vols",
    "implied_vol",
    "mc_price",
    "pa_call_delta_max",
    "price_cos",
    "price_gil_pelaez",
    "quotes_from_vols",
    "simulate_terminal",
    "smile_digital",
    "solve_pillar_strikes",
    "strike_from_delta",
    "strike_from_delta_pa_candidates",
    "vols_from_quotes",
]

__version__ = "1.0.0"
