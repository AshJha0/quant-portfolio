"""Standard normal CDF and density for scalar floats.

One definition for the option packages, which used to carry their own
copies. ``norm_cdf`` goes through ``erfc`` rather than ``0.5 (1 + erf)``,
so the lower tail keeps full relative precision (``norm_cdf(-30)`` is about
4.9e-198, not 0). These are the same expressions the packages used before,
so results are bit-for-bit unchanged.
"""
from __future__ import annotations

import math

__all__ = ["norm_cdf", "norm_pdf"]

_SQRT2 = 1.4142135623730951
_INV_SQRT_2PI = 0.3989422804014327


def norm_cdf(x: float) -> float:
    """N(x) = 0.5 erfc(-x / sqrt 2)."""
    return 0.5 * math.erfc(-x / _SQRT2)


def norm_pdf(x: float) -> float:
    """n(x) = exp(-x^2 / 2) / sqrt(2 pi)."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)
