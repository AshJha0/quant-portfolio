"""FX forwards via covered interest parity (CIP).

Covered interest parity:  ``F = S * exp((r_d - r_f) * T)``.
A domestic investor can replicate the forward by borrowing domestic cash,
buying spot foreign currency and depositing it at ``r_f``; absence of
arbitrage forces the forward to the CIP level (abstracting from the
cross-currency basis — see docs/METHODOLOGY.md assumption register).

Forward points are quoted as ``(F - S)`` scaled by the pair's pip factor
(1e4 for most pairs, 1e2 for JPY-quoted pairs).
"""

from __future__ import annotations

import math

from ._common import validate_inputs

__all__ = [
    "cip_forward",
    "forward_points",
    "synthetic_forward_from_options",
    "PIP_FACTORS",
]

#: Standard pip scaling: price decimal places quoted by convention.
PIP_FACTORS = {"default": 1e4, "jpy": 1e2}


def cip_forward(S: float, T: float, r_d: float, r_f: float) -> float:
    """Covered-interest-parity forward rate.

    Parameters
    ----------
    S : float
        Spot rate, domestic per unit foreign, > 0.
    T : float
        Time to delivery in years, >= 0.
    r_d, r_f : float
        Domestic / foreign continuously compounded rates (ACT/365F).

    Returns
    -------
    float
        Forward rate ``F = S * exp((r_d - r_f) T)``.  ``F > S`` when the
        domestic rate exceeds the foreign rate (forward premium on the
        base currency), the classic carry relationship.
    """
    validate_inputs(S, S, T, r_d, r_f, 0.0)
    return S * math.exp((r_d - r_f) * T)


def forward_points(S: float, T: float, r_d: float, r_f: float,
                   pip_factor: float = 1e4) -> float:
    """Forward points ``(F - S) * pip_factor``.

    Parameters
    ----------
    S, T, r_d, r_f : float
        As in :func:`cip_forward`.
    pip_factor : float
        1e4 for e.g. EURUSD (pip = 0.0001), 1e2 for USDJPY (pip = 0.01).

    Returns
    -------
    float
        Forward points in pips; positive when the base currency trades at
        a forward premium.
    """
    if not math.isfinite(pip_factor) or pip_factor <= 0:
        raise ValueError(
            f"pip_factor must be positive and finite, got {pip_factor!r}"
        )
    return (cip_forward(S, T, r_d, r_f) - S) * pip_factor


def synthetic_forward_from_options(call_price: float, put_price: float,
                                   K: float, T: float, r_d: float) -> float:
    """Forward implied by put-call parity (a 'synthetic forward').

    Conversion/parity: ``C - P = e^{-r_d T} (F - K)``, hence
    ``F = K + (C - P) e^{r_d T}``.  Long call + short put at the same
    strike replicates a forward purchase of the base currency; desks call
    the position a *synthetic forward* (or a 'conversion' when run against
    an actual forward to lock in the mispricing).

    Parameters
    ----------
    call_price, put_price : float
        European premiums (domestic ccy per unit foreign) at strike ``K``.
    K : float
        Common strike, > 0.
    T : float
        Time to expiry in years, >= 0.
    r_d : float
        Domestic continuously compounded rate.

    Returns
    -------
    float
        The option-implied forward rate.
    """
    validate_inputs(K, K, T, r_d, 0.0, 0.0)
    for name, p in (("call_price", call_price), ("put_price", put_price)):
        if not math.isfinite(p):
            raise ValueError(f"{name} must be finite, got {p!r}")
    return K + (call_price - put_price) * math.exp(r_d * T)
