"""Smile-consistent strike for a quoted FX delta.

FX quotes a smile point as "the 25-delta call", not as a strike. The strike
``K`` must satisfy ``delta(K, sigma(K)) = target``, where ``sigma(K)`` is the
smile's own vol at ``K`` (Clark 2011, "Foreign Exchange Option Pricing",
Wiley, s.3.5; Reiswich and Wystup 2010, "A guide to FX options quoting
conventions", J. Derivatives 18(2), s.3). We solve it by the same fixed point
the surface uses: strike -> vol -> strike, each step solved by
:func:`fx_surface.smile_from_quotes.strike_from_delta` under the chosen
convention (spot, forward, spot_pa, forward_pa). Premium-adjusted calls take
the market-standard high-strike branch.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .garman_kohlhagen import gk_forward
from .smile_from_quotes import strike_from_delta


def strike_at_smile_delta(
    vol_of_k: Callable[[float], float],
    delta: float,
    cp: int,
    S: float,
    T: float,
    r_d: float,
    r_f: float,
    convention: str = "spot",
    tol: float = 1e-12,
    max_iter: int = 500,
) -> float:
    """Strike whose delta, at the smile vol of that strike, equals ``delta``.

    ``vol_of_k`` maps log-moneyness ``k = ln(K/F)`` to implied vol.
    ``delta`` is unsigned, as quoted. Raises ``RuntimeError`` if the damped
    fixed point does not converge within ``max_iter`` steps.
    """
    F = gk_forward(S, T, r_d, r_f)
    sigma = float(vol_of_k(0.0))
    K = strike_from_delta(delta, cp, sigma, S, T, r_d, r_f, convention)
    for _ in range(max_iter):
        sigma_new = float(vol_of_k(math.log(K / F)))
        if not math.isfinite(sigma_new) or sigma_new <= 0.0:
            raise ValueError(f"smile vol must be finite and positive, got {sigma_new}")
        K_new = strike_from_delta(delta, cp, 0.5 * (sigma + sigma_new), S, T, r_d, r_f,
                                  convention)
        if abs(K_new - K) <= tol * F and abs(sigma_new - sigma) <= 1e-10:
            return K_new
        K, sigma = K_new, 0.5 * (sigma + sigma_new)
    raise RuntimeError("smile-consistent delta strike did not converge")
