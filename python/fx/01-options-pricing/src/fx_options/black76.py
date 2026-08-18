"""Black-76 pricing off the FX forward.

FX desks think in forwards: the smile is marked against the forward, and
Black-76 prices directly off it,

    call = e^{-r_d T} [F N(d1) - K N(d2)],
    d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T).

With the covered-interest-parity forward ``F = S e^{(r_d - r_f) T}``,
Black-76 is *algebraically identical* to Garman-Kohlhagen — substituting F
recovers the GK d1/d2 and price exactly.  The practical difference is the
input: quoting off F removes the need to know the rate *pair*, only the
domestic discount factor, which is how forward-space market data arrives.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from ._common import validate_option_type
from .forwards import cip_forward

__all__ = ["black76_price", "black76_from_spot"]


def black76_price(F: float, K: float, T: float, r_d: float, sigma: float,
                  option_type: str) -> float:
    """Black-76 price of a European FX option on the forward.

    Parameters
    ----------
    F : float
        Outright forward rate (domestic per unit foreign), > 0.
    K : float
        Strike, > 0.
    T : float
        Time to expiry in years, >= 0.
    r_d : float
        Domestic continuously compounded rate (discounting only).
    sigma : float
        Annualised volatility, >= 0.
    option_type : str
        ``"call"`` or ``"put"``.

    Returns
    -------
    float
        Premium in domestic currency per unit foreign notional.
    """
    phi = validate_option_type(option_type)
    for name, value in (("F", F), ("K", K), ("T", T), ("r_d", r_d),
                        ("sigma", sigma)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if F <= 0.0:
        raise ValueError(f"Forward F must be positive, got {F}")
    if K <= 0.0:
        raise ValueError(f"Strike K must be positive, got {K}")
    if T < 0.0:
        raise ValueError(f"Time to expiry T must be non-negative, got {T}")
    if sigma < 0.0:
        raise ValueError(f"Volatility sigma must be non-negative, got {sigma}")

    df = math.exp(-r_d * T)
    v = sigma * math.sqrt(T)
    if T == 0.0 or v <= 1e-12:
        return df * max(phi * (F - K), 0.0)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    return phi * df * (F * norm.cdf(phi * d1) - K * norm.cdf(phi * d2))


def black76_from_spot(S: float, K: float, T: float, r_d: float, r_f: float,
                      sigma: float, option_type: str) -> float:
    """Black-76 with the forward built from spot via CIP.

    Equals :func:`fx_options.garman_kohlhagen.gk_price` to machine
    precision — tested to 1e-10 in the suite.
    """
    F = cip_forward(S, T, r_d, r_f)
    return black76_price(F, K, T, r_d, sigma, option_type)
