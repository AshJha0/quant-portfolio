"""Adjoint algorithmic differentiation (plan W19).

``tape`` is the reverse-mode engine; ``black_scholes``, ``monte_carlo`` and
``swap`` apply it to a Black-Scholes price, a pathwise Monte Carlo pricer and
a swap PV off the dated OIS bootstrap. The Rust (``rust/aad``) and C++
(``cpp/aad``) ports implement the same tape and the same three pricers.
"""

from .black_scholes import AadGreeks, bs_greeks_aad, bs_price
from .monte_carlo import McGreeks, mc_greeks_aad, mc_price, normals
from .swap import (
    PillarGreeks,
    QuoteGreeks,
    Swap,
    pillar_greeks,
    pv_float,
    quote_greeks,
    quote_residual,
    residual_jacobian,
    swap_pv,
)
from .tape import Tape, Var, exp, log, ncdf, sqrt, value

__all__ = [
    "AadGreeks", "McGreeks", "PillarGreeks", "QuoteGreeks", "Swap", "Tape", "Var",
    "bs_greeks_aad", "bs_price", "exp", "log", "mc_greeks_aad", "mc_price", "ncdf",
    "normals", "pillar_greeks", "pv_float", "quote_greeks", "quote_residual",
    "residual_jacobian", "sqrt", "swap_pv", "value",
]
