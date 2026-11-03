"""Dupire local volatility: a re-export shim over ``qp_core.vol.dupire``.

The maths (Gatheral 2006, eq. 1.10 and 1.4; Dupire 1994) lives once in
:mod:`qp_core.vol.dupire`, shared with ``fx_surface.dupire`` (plan W20).
"""

from __future__ import annotations

from qp_core.vol.dupire import (
    LocalVolArbitrageError,
    TotalVariance,
    local_vol_from_prices,
    local_vol_from_total_variance,
    local_vol_mc_price,
)

__all__ = [
    "LocalVolArbitrageError",
    "TotalVariance",
    "local_vol_from_prices",
    "local_vol_from_total_variance",
    "local_vol_mc_price",
]
