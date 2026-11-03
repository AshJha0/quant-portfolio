"""SSVI and eSSVI total-variance surfaces: a shim over ``qp_core.vol.ssvi``.

The maths (Gatheral and Jacquier 2014; Hendriks and Martini 2019) lives once
in :mod:`qp_core.vol.ssvi`, shared with ``fx_surface.ssvi`` (plan W20). Only
:func:`ssvi_slice_as_svi` is local, because it returns this package's
:class:`eq_surface.smile.SVIParams`.
"""

from __future__ import annotations

from qp_core.vol.ssvi import (
    ESSVIParams,
    SSVIFitResult,
    SSVIParams,
    calendar_arbitrage_free,
    fit_ssvi,
    ssvi_slice_raw_svi,
    ssvi_total_variance,
)

from .smile import SVIParams

__all__ = [
    "ESSVIParams",
    "SSVIFitResult",
    "SSVIParams",
    "calendar_arbitrage_free",
    "fit_ssvi",
    "ssvi_slice_as_svi",
    "ssvi_total_variance",
]


def ssvi_slice_as_svi(theta: float, rho: float, phi: float) -> SVIParams:
    """The raw SVI parameters of one SSVI slice."""
    a, b, r, m, sigma = ssvi_slice_raw_svi(theta, rho, phi)
    return SVIParams(a=a, b=b, rho=r, m=m, sigma=sigma)
