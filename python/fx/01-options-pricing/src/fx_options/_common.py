"""Shared input validation and conventions for the fx_options package.

Conventions (used everywhere in this package)
---------------------------------------------
* FX pairs are quoted BASE/QUOTE: EURUSD = number of USD per 1 EUR.
* ``S`` is the spot rate in domestic (quote) currency per unit of foreign
  (base) currency.  All option prices are in domestic currency per unit of
  foreign notional.
* ``r_d`` is the domestic (quote-currency) continuously compounded rate,
  ``r_f`` the foreign (base-currency) rate, both annualised, ACT/365F.
* ``T`` is time to expiry in years, ``sigma`` the annualised lognormal vol.
* Negative rates are legal (EUR/CHF era); negative ``T`` or ``sigma`` is not.
"""

from __future__ import annotations

import math

__all__ = ["validate_inputs", "validate_option_type", "OTHER_TYPE", "PHI"]

#: Map option type to its opposite (used by foreign-domestic symmetry).
OTHER_TYPE = {"call": "put", "put": "call"}

#: Map option type to the sign convention phi (+1 call, -1 put).
PHI = {"call": 1.0, "put": -1.0}


def validate_option_type(option_type: str) -> float:
    """Validate ``option_type`` and return phi (+1 for call, -1 for put).

    Parameters
    ----------
    option_type : str
        ``"call"`` or ``"put"`` (case-insensitive).

    Returns
    -------
    float
        +1.0 for a call, -1.0 for a put.

    Raises
    ------
    ValueError
        If ``option_type`` is not ``"call"`` or ``"put"``.
    """
    if not isinstance(option_type, str) or option_type.lower() not in PHI:
        raise ValueError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        )
    return PHI[option_type.lower()]


def validate_inputs(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
) -> None:
    """Validate common pricing inputs, raising ``ValueError`` on bad data.

    Parameters
    ----------
    S : float
        Spot FX rate (domestic per unit foreign), must be > 0 and finite.
    K : float
        Strike, must be > 0 and finite.
    T : float
        Time to expiry in years, must be >= 0 and finite.
    r_d, r_f : float
        Domestic / foreign continuously compounded rates.  May be negative
        (EUR, CHF, JPY regimes) but must be finite.
    sigma : float
        Annualised volatility, must be >= 0 and finite.

    Raises
    ------
    ValueError
        With an informative message identifying the offending input.
    """
    for name, value in (("S", S), ("K", K), ("T", T), ("r_d", r_d),
                        ("r_f", r_f), ("sigma", sigma)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a real number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if S <= 0.0:
        raise ValueError(f"Spot S must be positive, got {S}")
    if K <= 0.0:
        raise ValueError(f"Strike K must be positive, got {K}")
    if T < 0.0:
        raise ValueError(f"Time to expiry T must be non-negative, got {T}")
    if sigma < 0.0:
        raise ValueError(f"Volatility sigma must be non-negative, got {sigma}")
