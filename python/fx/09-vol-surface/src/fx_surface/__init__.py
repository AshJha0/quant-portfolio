"""fx_surface: FX volatility surface and stochastic volatility toolkit.

Pipeline: broker quotes (ATM/RR/BF in delta space) -> five-point smiles
-> strike solving under four delta conventions -> SVI and vanna-volga
smile fits -> delta-space surface -> Heston (GK) calibration -> pricing,
Greeks and Monte Carlo validation.
"""

from .bates import JumpParams, bates_call, bates_cf_x, bates_put, merton_call
from .calibration import (
    CalibrationResult,
    CalibrationSlice,
    calibrate_heston,
    heston_smile_vols,
)
from .dupire import (
    LocalVolArbitrageError,
    local_vol_at_delta,
    local_vol_from_prices,
    local_vol_from_total_variance,
    local_vol_mc_price,
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
    broker_strangle_from_smile,
    pa_call_delta_max,
    quotes_from_vols,
    smile_quotes_from_broker,
    solve_pillar_strikes,
    strike_from_delta,
    strike_from_delta_pa_candidates,
    vols_from_quotes,
)
from .ssvi import (
    ESSVIParams,
    SSVIFitResult,
    SSVIParams,
    calendar_arbitrage_free,
    fit_ssvi,
    fit_ssvi_from_pillars,
    ssvi_slice_as_svi,
    ssvi_strike_at_delta,
    ssvi_total_variance,
)
from .surface import (
    FXVolSurface,
    SmileSlice,
    SurfaceConvergenceError,
    build_slice,
    build_surface,
)

__all__ = [
    "DELTA_CONVENTIONS",
    "PILLAR_LABELS",
    "CalibrationResult",
    "CalibrationSlice",
    "ESSVIParams",
    "FXVolSurface",
    "HestonParams",
    "JumpParams",
    "LocalVolArbitrageError",
    "SSVIFitResult",
    "SSVIParams",
    "SVIParams",
    "SVISmile",
    "SmileQuotes",
    "SmileSlice",
    "SurfaceConvergenceError",
    "VannaVolgaSmile",
    "atm_dns_strike",
    "bates_call",
    "bates_cf_x",
    "bates_put",
    "broker_strangle_from_smile",
    "build_slice",
    "build_surface",
    "calendar_arbitrage_free",
    "calibrate_heston",
    "durrleman_g",
    "fit_ssvi",
    "fit_ssvi_from_pillars",
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
    "local_vol_at_delta",
    "local_vol_from_prices",
    "local_vol_from_total_variance",
    "local_vol_mc_price",
    "mc_price",
    "merton_call",
    "pa_call_delta_max",
    "price_cos",
    "price_gil_pelaez",
    "quotes_from_vols",
    "simulate_terminal",
    "smile_digital",
    "smile_quotes_from_broker",
    "solve_pillar_strikes",
    "ssvi_slice_as_svi",
    "ssvi_strike_at_delta",
    "ssvi_total_variance",
    "strike_from_delta",
    "strike_from_delta_pa_candidates",
    "vols_from_quotes",
]

__version__ = "3.0.0.dev0"
