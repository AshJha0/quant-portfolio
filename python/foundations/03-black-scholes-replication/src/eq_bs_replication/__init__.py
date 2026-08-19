"""eq_bs_replication -- a from-scratch, zero-scipy Black-Scholes replication.

Public API re-exports. See the package README and ``docs/`` for the
methodology, validation evidence and desk-usage framing behind this
project. This is a minimal, pedagogical, cross-validated replication of
the Black-Scholes-Merton model -- not a production pricing library. The
production, vectorised, multi-model equivalent lives at
``python/equity/01-options-pricing`` (package ``eq_options``).
"""
from .black_scholes import (
    Greeks,
    call_greeks,
    call_price,
    implied_volatility,
    put_greeks,
    put_price,
)
from .monte_carlo import mc_call_price

__all__ = [
    "Greeks",
    "call_price",
    "put_price",
    "call_greeks",
    "put_greeks",
    "implied_volatility",
    "mc_call_price",
]
